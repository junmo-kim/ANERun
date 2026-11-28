// Texts of 2049...8192 tokens: one winl{S}_b512 call per text, S = the smallest long length that holds it
// (3072, 4096, 6144 or 8192). See WindowContract for the inputs.
//
// The inputs other than the token rows depend only on the token count n, so a buffer remembers the n it
// was last filled for and rewrites only what changes: the band rows near the old and the new end of the
// text (|Δn| + 512 rows of 1536 values at most), key_bias and pool. Buffers are pooled per length; the
// steady state allocates nothing.
import Foundation

/// Reusable inputs of one winl length.
public final class LongBuffers: @unchecked Sendable {
    public let length: Int
    /// Token rows [length, 512]; rows >= tokens are zero.
    public let inputsEmbeds: UnsafeMutablePointer<Float16>
    /// [length, 1536] (WindowContract).
    public let bandBias: UnsafeMutablePointer<Float16>
    /// [length]: 0 for keys < tokens, else -1e4.
    public let keyBias: UnsafeMutablePointer<Float16>
    /// [length]: 2^-10 for rows < tokens, else 0.
    public let pool: UnsafeMutablePointer<Float16>
    /// Token count of the current layout (0: none yet, every row is padding).
    public private(set) var tokens = 0
    /// Rows below this may hold token data.
    private var dirtyRows = 0
    /// Band rows rewritten by the last layout change (for tests).
    public private(set) var lastRewrittenRows = 0
    /// Inputs for the model wrapped around this memory (set once by the runner).
    public var modelInput: AnyObject?

    public init(length: Int) {
        precondition(length % WindowContract.block == 0)
        self.length = length
        func allocate(_ count: Int, fill: Float16) -> UnsafeMutablePointer<Float16> {
            let raw = UnsafeMutableRawPointer.allocate(byteCount: count * 2, alignment: BucketBuffers.alignment)
            let pointer = raw.bindMemory(to: Float16.self, capacity: count)
            pointer.initialize(repeating: fill, count: count)
            return pointer
        }
        inputsEmbeds = allocate(length * BucketContract.hidden, fill: 0)
        bandBias = allocate(length * WindowContract.bandColumns, fill: BucketContract.blocked)
        keyBias = allocate(length, fill: BucketContract.blocked)
        pool = allocate(length, fill: 0)
        for row in 0..<length { fillBandRow(row, tokens: 0) }
    }

    deinit {
        inputsEmbeds.deallocate()
        bandBias.deallocate()
        keyBias.deallocate()
        pool.deallocate()
    }

    /// Gathers the token rows and sets band, key bias and pool for `ids.count` tokens.
    public func prepare(ids: [Int32], rows: any TokenRowSource) {
        let n = ids.count, hidden = BucketContract.hidden
        precondition(n >= 1 && n <= length)
        rows.gather(ids, into: inputsEmbeds)
        if dirtyRows > n { (inputsEmbeds + n * hidden).update(repeating: 0, count: (dirtyRows - n) * hidden) }
        dirtyRows = n
        setTokens(n)
    }

    /// Band, key bias and pool for a text of `n` rows.
    public func setTokens(_ n: Int) {
        precondition(n >= 0 && n <= length)
        let old = tokens
        guard n != old else {
            lastRewrittenRows = 0
            return
        }
        // A row r sees keys r - 512 ... r + 512; rows below min(old, n) - 512 see only keys inside both texts,
        // rows from max(old, n) on are padding in both. Everything between is rewritten.
        let lower = max(0, min(old, n) - WindowContract.halo), upper = max(old, n)
        for row in lower..<upper { fillBandRow(row, tokens: n) }
        lastRewrittenRows = upper - lower
        if n > old {
            (keyBias + old).update(repeating: 0, count: n - old)
            (pool + old).update(repeating: WindowContract.poolWeight, count: n - old)
        } else {
            (keyBias + n).update(repeating: BucketContract.blocked, count: old - n)
            (pool + n).update(repeating: 0, count: old - n)
        }
        tokens = n
    }

    /// Writes band row `row` for a single text of `n` rows from row 0.
    private func fillBandRow(_ row: Int, tokens n: Int) {
        let columns = WindowContract.bandColumns
        let target = bandBias + row * columns
        target.update(repeating: BucketContract.blocked, count: columns)
        if row < n {
            let lower = max(0, row - BucketContract.window), upper = min(n, row + BucketContract.window + 1)
            (target + WindowContract.column(row: row, key: lower)).update(repeating: 0, count: upper - lower)
        } else {
            target[WindowContract.column(row: row, key: row)] = 0  // padding rows see themselves only
        }
    }

    /// The band value at (row, column), for tests.
    public subscript(band row: Int, column: Int) -> Float16 { bandBias[row * WindowContract.bandColumns + column] }
}

/// Runs prepared winl calls. Implementations must be safe to call concurrently with different buffers.
public protocol LongFunctionRunner: Sendable {
    var lengths: [Int] { get }
    /// One L2-normalized embedding (768 values).
    func run(_ buffers: LongBuffers) async throws -> [Float]
}

/// Long texts over a `LongFunctionRunner`, with buffers pooled per length.
public final class LongTextEngine: LongTextRunner, @unchecked Sendable {
    public let runner: any LongFunctionRunner
    public let lengths: [Int]
    private let rows: any TokenRowSource
    private let lock = NSLock()
    private var idle: [Int: [LongBuffers]] = [:]
    private let maxIdlePerLength: Int

    public init(runner: any LongFunctionRunner, rows: any TokenRowSource, maxIdlePerLength: Int = 1) {
        self.runner = runner
        lengths = runner.lengths.sorted()
        self.rows = rows
        self.maxIdlePerLength = maxIdlePerLength
    }

    public var maxTokens: Int { lengths.last ?? 0 }

    public func warm() async throws {}

    /// The length a text of `tokens` runs in.
    public func length(forTokens tokens: Int) -> Int? { lengths.first { $0 >= tokens } }

    public func embed(ids: [Int32]) async throws -> [Float] {
        guard !ids.isEmpty else { throw RuntimeError.unsupported("a text with no tokens") }
        guard let length = length(forTokens: ids.count) else { throw RuntimeError.tooLong(tokens: ids.count, max: maxTokens) }
        let buffers = checkout(length: length)
        defer { checkin(buffers) }
        buffers.prepare(ids: ids, rows: rows)
        return try await runner.run(buffers)
    }

    func checkout(length: Int) -> LongBuffers {
        if let buffers = lock.withLock({ idle[length]?.popLast() }) { return buffers }
        return LongBuffers(length: length)
    }

    func checkin(_ buffers: LongBuffers) {
        lock.withLock {
            var list = idle[buffers.length, default: []]
            if list.count < maxIdlePerLength { list.append(buffers) }
            idle[buffers.length] = list
        }
    }
}
