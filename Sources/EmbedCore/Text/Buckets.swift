// The text functions of the model package (layout "layoutR") and the packing planner.
//
// Packing functions: one call embeds up to K texts packed back to back, all with the same outputs:
//
//   embed_S, S in {64, 128, 256, 512, 768, 1024, 1280}: full attention
//     inputs_embeds [1, S, 512] fp16   token rows (table row * sqrt(512)); padding rows are finite
//     global_bias   [1, 1, S, S] fp16  0 where query i and key j belong to the same text, else -1e4
//     local_bias    [1, 1, S, S] fp16  global_bias AND |i - j| <= 512 (Gemma's sliding window)
//     pool          [K, S] fp16        K = max(8, S / 32); row k = 1/len over text k's tokens, other rows 0
//     -> embedding  [K, 768] fp32      already L2-normalized
//
//   winp{S}_b512, S in {1536, 2048}: windowed attention (WindowContract), same inputs and output except
//     band_bias     [1, 1, S, 1536] fp16 instead of local_bias: row r, column c = key row (r/512)*512 - 512 + c
//
// Padding rows attend only to themselves, so no row is fully masked. Positions are not an input: RoPE
// is relative, and attention never crosses texts, so a packed text sees the same scores as alone.
// Masks are inputs rather than graph constants because a constant band miscompiles on the ANE.
//
// Texts above 2048 tokens use one winl{S}_b512 call each (WindowLong.swift).

public enum BucketContract {
    /// Every packing length: embed_S (full attention) then winp (windowed), ascending.
    public static let lengths = fullAttentionLengths + WindowContract.packingLengths
    /// embed_S functions.
    public static let fullAttentionLengths = [64, 128, 256, 512, 768, 1024, 1280]
    public static let hidden = TokenTable.width
    public static let dimension = 768
    public static let window = 512
    public static let blocked: Float16 = -10_000

    public static func slots(forLength length: Int) -> Int { max(8, length / 32) }
    public static func isWindowed(_ length: Int) -> Bool { WindowContract.packingLengths.contains(length) }
    public static func functionName(forLength length: Int) -> String {
        isWindowed(length) ? WindowContract.packingName(length) : "embed_\(length)"
    }

    /// Input and output feature names.
    public static let inputsEmbeds = "inputs_embeds"
    public static let globalBias = "global_bias"
    public static let localBias = "local_bias"
    public static let pool = "pool"
    public static let embedding = "embedding"
}

/// The windowed functions: sliding layers attend block by block (512 query rows against the 1536 key rows
/// around them) instead of over S x S, so one fixed-shape call reaches 8192 rows; the global layers
/// (5, 11, 17, 23) attend to every row.
///
///   band_bias [1, 1, S, 1536] fp16: row r, column c is query r against key k = (r / 512) * 512 - 512 + c;
///     0 if r and k are in the same text and |r - k| <= 512, or k == r; else -1e4. Columns whose key is
///     outside [0, S) are never read by the graph.
///   winp{S}_b512 (packing, 1536 / 2048): inputs_embeds, band_bias, global_bias, pool [K, S] -> [K, 768]
///   winl{S}_b512 (one text, 3072 / 4096 / 6144 / 8192): inputs_embeds (rows >= n zero), band_bias,
///     key_bias [1, 1, 1, S] (0 for k < n), pool [1, S] (2^-10 on the n rows, not 1/n: fp16 underflow)
///     -> [1, 768]
public enum WindowContract {
    public static let block = 512
    public static let halo = 512
    public static let bandColumns = block + 2 * halo
    public static let packingLengths = [1536, 2048]
    public static let longLengths = [3072, 4096, 6144, 8192]
    public static let poolWeight: Float16 = 0.000_976_562_5  // 2^-10
    public static let bandBias = "band_bias"
    public static let keyBias = "key_bias"

    public static func packingName(_ length: Int) -> String { "winp\(length)_b\(block)" }
    public static func longName(_ length: Int) -> String { "winl\(length)_b\(block)" }

    /// The smallest long length that holds `tokens`.
    public static func longLength(forTokens tokens: Int) -> Int? { longLengths.first { $0 >= tokens } }

    /// The long lengths a `maxTokens` limit loads: none up to 2048, else every one up to the smallest
    /// that holds `maxTokens`.
    public static func longLengths(forMaxTokens maxTokens: Int) -> [Int] {
        guard maxTokens > BucketContract.lengths.last!, let top = longLength(forTokens: maxTokens) else { return [] }
        return longLengths.filter { $0 <= top }
    }

    /// Key row of band column `column` for query `row`.
    @inline(__always) public static func key(row: Int, column: Int) -> Int { (row / block) * block - halo + column }
    /// Band column of key `key` for query `row` (valid when |row - key| <= halo).
    @inline(__always) public static func column(row: Int, key: Int) -> Int { key - (row / block) * block + halo }
}

/// One text placed in a packed call: rows offset..<offset+length, pool row = its index in the call.
public struct PackedSegment: Sendable, Equatable {
    /// Index of the text in the caller's list.
    public let item: Int
    public let offset: Int
    public let length: Int

    public init(item: Int, offset: Int, length: Int) {
        self.item = item
        self.offset = offset
        self.length = length
    }
}

/// One function call: a bucket length and the texts packed into it.
public struct PackedCall: Sendable, Equatable {
    public let length: Int
    public let segments: [PackedSegment]

    public var slots: Int { BucketContract.slots(forLength: length) }
    public var usedTokens: Int { segments.reduce(0) { $0 + $1.length } }

    public init(length: Int, segments: [PackedSegment]) {
        self.length = length
        self.segments = segments
    }
}

/// Splits a list of texts (token counts) into calls of the available buckets, minimizing an
/// estimated total cost.
///
/// For every bucket length C it packs first-fit-decreasing into bins of C tokens and C's slot count
/// (a text longer than C gets a bin of its own), shrinks each bin to the smallest bucket that holds it,
/// and sums the per-call costs; the cheapest C wins. Small C means many cheap calls, large C fewer
/// expensive ones; which is better depends on the measured per-call latency of each bucket.
public struct PackingPlanner: Sendable {
    public let lengths: [Int]
    /// Estimated cost of one call per bucket length (any unit).
    public let costs: [Int: Double]

    /// Per-call latency in ms of each packing function, one text filling it (`anerun bench --buckets`,
    /// Apple M5 Ultra, macOS 27). Only the ratios matter to the planner; other chips scale them roughly
    /// together.
    public static let defaultCosts: [Int: Double] = [
        64: 3.16, 128: 5.19, 256: 10.71, 512: 26.47, 768: 44.97, 1024: 68.93, 1280: 93.47, 1536: 114.53, 2048: 170.34,
    ]

    public init(lengths: [Int] = BucketContract.lengths, costs: [Int: Double] = PackingPlanner.defaultCosts) {
        precondition(!lengths.isEmpty)
        self.lengths = lengths.sorted()
        self.costs = costs
    }

    public var maxLength: Int { lengths.last! }

    public func cost(ofLength length: Int) -> Double {
        costs[length] ?? Double(length)
    }

    /// The smallest bucket that holds `tokens` tokens in `texts` slots.
    public func bucket(tokens: Int, texts: Int = 1) -> Int? {
        lengths.first { $0 >= tokens && BucketContract.slots(forLength: $0) >= texts }
    }

    /// Calls covering every text exactly once. Precondition: every count is 1...maxLength.
    public func plan(_ counts: [Int]) -> [PackedCall] {
        guard !counts.isEmpty else { return [] }
        precondition(counts.allSatisfy { $0 >= 1 && $0 <= maxLength }, "token count outside 1...\(maxLength)")
        let order = counts.indices.sorted { counts[$0] != counts[$1] ? counts[$0] > counts[$1] : $0 < $1 }
        var best: [PackedCall] = []
        var bestCost = Double.infinity
        for capacity in lengths {
            let calls = pack(order: order, counts: counts, capacity: capacity)
            let total = calls.reduce(0) { $0 + cost(ofLength: $1.length) }
            if total < bestCost {
                best = calls
                bestCost = total
            }
        }
        return best
    }

    private func pack(order: [Int], counts: [Int], capacity: Int) -> [PackedCall] {
        let slots = BucketContract.slots(forLength: capacity)
        var bins: [[Int]] = []  // item indices
        var used: [Int] = []
        var open: [Int] = []  // bins that can still take texts
        for item in order {
            let count = counts[item]
            if count > capacity {
                bins.append([item])
                used.append(count)
                continue
            }
            var placed = false
            for (position, bin) in open.enumerated() where used[bin] + count <= capacity && bins[bin].count < slots {
                bins[bin].append(item)
                used[bin] += count
                if bins[bin].count == slots || used[bin] == capacity { open.remove(at: position) }
                placed = true
                break
            }
            if !placed {
                bins.append([item])
                used.append(count)
                open.append(bins.count - 1)
            }
        }
        return bins.indices.map { index in
            var offset = 0
            let segments = bins[index].map { item -> PackedSegment in
                defer { offset += counts[item] }
                return PackedSegment(item: item, offset: offset, length: counts[item])
            }
            return PackedCall(length: bucket(tokens: used[index], texts: segments.count)!, segments: segments)
        }
    }
}
