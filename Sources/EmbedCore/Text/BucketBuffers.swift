// Reusable input buffers for one packing length, and a pool of them.
//
// Building an S x S bias from scratch costs S^2 writes (4M fp16 values at 2048) per call. A buffer
// instead remembers the layout it was last filled with and only rewrites the blocks that change:
// resetting the old blocks to -1e4 and opening the new ones costs the sum of len^2 over the texts.
// Steady state allocates nothing.
//
// The sliding-window mask is `local_bias` [S, S] for embed_S and `band_bias` [S, 1536] for the windowed
// winp functions (WindowContract): the same open keys, stored at column key for the first and at column
// key - (row / 512) * 512 + 512 for the second.
import Foundation

public final class BucketBuffers: @unchecked Sendable {
    public let length: Int
    public let slots: Int
    /// Whether this is a winp (band) buffer.
    public let windowed: Bool
    public let inputsEmbeds: UnsafeMutablePointer<Float16>
    public let globalBias: UnsafeMutablePointer<Float16>
    /// local_bias [length, length], or band_bias [length, 1536] when `windowed`. Same memory as `globalBias`
    /// when every key is inside the window anyway (full attention and S <= window + 1).
    public let slidingBias: UnsafeMutablePointer<Float16>
    /// Values per row of `slidingBias`.
    public let slidingColumns: Int
    public let pool: UnsafeMutablePointer<Float16>
    public var localSharesGlobal: Bool { slidingBias == globalBias }
    /// Inputs for the model wrapped around this memory (set once by the runner; the memory never moves).
    public var modelInput: AnyObject?

    /// Layout of the last fill.
    private var segments: [PackedSegment] = []
    private var used: Int
    /// Rows below this may hold old token data (rows >= used are padding).
    private var dirtyRows = 0

    public static let alignment = 16_384  // page aligned

    public init(length: Int) {
        self.length = length
        slots = BucketContract.slots(forLength: length)
        windowed = BucketContract.isWindowed(length)
        slidingColumns = windowed ? WindowContract.bandColumns : length
        let hidden = BucketContract.hidden
        func allocate(_ count: Int, fill: Float16) -> UnsafeMutablePointer<Float16> {
            let raw = UnsafeMutableRawPointer.allocate(byteCount: count * 2, alignment: Self.alignment)
            let pointer = raw.bindMemory(to: Float16.self, capacity: count)
            pointer.initialize(repeating: fill, count: count)
            return pointer
        }
        inputsEmbeds = allocate(length * hidden, fill: 0)
        globalBias = allocate(length * length, fill: BucketContract.blocked)
        slidingBias = !windowed && length <= BucketContract.window + 1
            ? globalBias : allocate(length * slidingColumns, fill: BucketContract.blocked)
        pool = allocate(slots * length, fill: 0)
        used = length  // no padding diagonal is open yet
    }

    deinit {
        inputsEmbeds.deallocate()
        globalBias.deallocate()
        if slidingBias != globalBias { slidingBias.deallocate() }
        pool.deallocate()
    }

    /// Row `position` of inputs_embeds (512 fp16 values), for callers that write rows themselves
    /// (image soft tokens).
    public func embedsRow(_ position: Int) -> UnsafeMutablePointer<Float16> {
        inputsEmbeds + position * BucketContract.hidden
    }

    /// Fills everything for a call whose texts are token ids: rows, biases and pool.
    public func prepare(_ call: PackedCall, ids: [[Int32]], rows: some TokenRowSource) {
        prepare(call, sequences: ids.map { ModelSequence(ids: $0) }, rows: rows)
    }

    /// Fills everything for a call: each sequence's rows (token rows or given rows), biases and pool.
    public func prepare(_ call: PackedCall, sequences: [ModelSequence], rows: some TokenRowSource) {
        precondition(call.length == length)
        for segment in call.segments {
            let sequence = sequences[segment.item]
            precondition(sequence.count == segment.length)
            var position = segment.offset
            for part in sequence.parts {
                switch part {
                case .ids(let ids):
                    rows.gather(ids, into: embedsRow(position))
                    position += ids.count
                case .rows(let values):
                    let count = values.count / BucketContract.hidden
                    values.withUnsafeBufferPointer { embedsRow(position).update(from: $0.baseAddress!, count: values.count) }
                    position += count
                }
            }
        }
        setLayout(call.segments)
    }

    /// Sets biases and pool for `newSegments` and zeroes stale padding rows of inputs_embeds.
    public func setLayout(_ newSegments: [PackedSegment]) {
        precondition(newSegments.count <= slots)
        var newUsed = 0
        for segment in newSegments {
            precondition(segment.offset == newUsed && segment.length > 0, "segments must be contiguous from row 0")
            newUsed += segment.length
        }
        precondition(newUsed <= length)
        // Same layout as the last call (repeated batch shapes, a warm buffer): biases and pool are already right.
        if newSegments == segments && newUsed == used { return }
        let blocked = BucketContract.blocked

        // 1. Close the previous blocks and padding diagonal.
        for segment in segments {
            for row in segment.offset..<(segment.offset + segment.length) {
                (globalBias + row * length + segment.offset).update(repeating: blocked, count: segment.length)
                if !localSharesGlobal {
                    let (lower, upper) = band(row: row, segment: segment)
                    (slidingBias + row * slidingColumns + lower).update(repeating: blocked, count: upper - lower)
                }
            }
        }
        for row in used..<length {
            globalBias[row * length + row] = blocked
            if !localSharesGlobal { slidingBias[row * slidingColumns + diagonal(row)] = blocked }
        }
        for (slot, segment) in segments.enumerated() {
            (pool + slot * length + segment.offset).update(repeating: 0, count: segment.length)
        }

        // 2. Open the new blocks (local: inside the window) and the padding diagonal.
        for (slot, segment) in newSegments.enumerated() {
            for row in segment.offset..<(segment.offset + segment.length) {
                (globalBias + row * length + segment.offset).update(repeating: 0, count: segment.length)
                if !localSharesGlobal {
                    let (lower, upper) = band(row: row, segment: segment)
                    (slidingBias + row * slidingColumns + lower).update(repeating: 0, count: upper - lower)
                }
            }
            let weight = Float16(1 / Float(segment.length))
            (pool + slot * length + segment.offset).update(repeating: weight, count: segment.length)
        }
        for row in newUsed..<length {
            globalBias[row * length + row] = 0
            if !localSharesGlobal { slidingBias[row * slidingColumns + diagonal(row)] = 0 }
        }

        // 3. Padding rows must be finite; zero the ones that held tokens before.
        if dirtyRows > newUsed {
            embedsRow(newUsed).update(repeating: 0, count: (dirtyRows - newUsed) * BucketContract.hidden)
        }
        dirtyRows = newUsed
        segments = newSegments
        used = newUsed
    }

    /// Columns [lower, upper) of `slidingBias` row `row` whose keys are inside both the segment and the window.
    @inline(__always)
    private func band(row: Int, segment: PackedSegment) -> (Int, Int) {
        let lower = max(segment.offset, row - BucketContract.window)
        let upper = min(segment.offset + segment.length, row + BucketContract.window + 1)
        guard windowed else { return (lower, upper) }
        return (WindowContract.column(row: row, key: lower), WindowContract.column(row: row, key: upper))
    }

    /// Column of the diagonal (key == row) in `slidingBias`.
    @inline(__always)
    private func diagonal(_ row: Int) -> Int { windowed ? WindowContract.column(row: row, key: row) : row }
}

/// What one pooled sequence is made of: token ids (looked up and scaled) and/or raw model rows
/// (image soft tokens, 512 fp16 values per row, not scaled).
public enum SequencePart: Sendable {
    case ids([Int32])
    case rows([Float16])
}

public struct ModelSequence: Sendable {
    public var parts: [SequencePart]

    public init(parts: [SequencePart]) {
        self.parts = parts
    }

    public init(ids: [Int32]) {
        parts = [.ids(ids)]
    }

    /// `<bos> <|image> rows <image|> <eos>`
    public static func image(rows: [Float16]) -> ModelSequence {
        ModelSequence(parts: [
            .ids([SpecialTokens.bos, SpecialTokens.imageStart]), .rows(rows),
            .ids([SpecialTokens.imageEnd, SpecialTokens.eos]),
        ])
    }

    /// Rows this sequence occupies.
    public var count: Int {
        parts.reduce(0) { total, part in
            switch part {
            case .ids(let ids): return total + ids.count
            case .rows(let rows): return total + rows.count / BucketContract.hidden
            }
        }
    }
}

/// Idle buffers per bucket length. A call checks one out, fills it, runs, and returns it.
public final class BucketBufferPool: @unchecked Sendable {
    private let lock = NSLock()
    private var idle: [Int: [BucketBuffers]] = [:]
    private let maxIdlePerLength: Int
    private var outstanding = 0
    private var peakOutstanding = 0

    public init(maxIdlePerLength: Int = 4) {
        self.maxIdlePerLength = maxIdlePerLength
    }

    public func checkout(length: Int) -> BucketBuffers {
        let reused = lock.withLock { () -> BucketBuffers? in
            outstanding += 1
            peakOutstanding = max(peakOutstanding, outstanding)
            return idle[length]?.popLast()
        }
        return reused ?? BucketBuffers(length: length)
    }

    public func checkin(_ buffers: BucketBuffers) {
        lock.withLock {
            outstanding -= 1
            var list = idle[buffers.length, default: []]
            if list.count < maxIdlePerLength { list.append(buffers) }
            idle[buffers.length] = list
        }
    }

    /// Buffers checked out and not returned yet.
    public var checkedOut: Int { lock.withLock { outstanding } }
    /// The most buffers checked out at once since the pool was made.
    public var peakCheckedOut: Int { lock.withLock { peakOutstanding } }

    public func idleCount(length: Int) -> Int {
        lock.withLock { idle[length]?.count ?? 0 }
    }
}
