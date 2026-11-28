// Texts and images through the uniform bucket functions: plan -> fill reusable buffers -> run -> pick
// each text's pooled row. The runner is the only model-specific part.
import Foundation

public final class BucketTextRuntime: EmbeddingRuntime {
    public let planner: PackingPlanner
    let runner: any BucketFunctionRunner
    let rows: any TokenRowSource
    let long: (any LongTextRunner)?
    let vision: (any VisionEncoder)?
    let buffers: BucketBufferPool
    public let backend: String
    private let functions: [String]
    private let notServed: [String]
    private let selfTestSkipped: Bool
    /// Every function already ran once (the load-time self-test), so `warm()` has nothing left to do.
    private let warmed: Bool

    public init(runner: any BucketFunctionRunner, rows: any TokenRowSource, long: (any LongTextRunner)? = nil,
                vision: (any VisionEncoder)? = nil, costs: [Int: Double] = PackingPlanner.defaultCosts,
                maxIdleBuffers: Int = 4, backend: String, functions: [String] = [], notServed: [String] = [],
                selfTestSkipped: Bool = false, warmed: Bool = false)
    {
        self.functions = functions
        self.notServed = notServed
        self.selfTestSkipped = selfTestSkipped
        self.warmed = warmed
        planner = PackingPlanner(lengths: runner.lengths, costs: costs)
        self.runner = runner
        self.rows = rows
        self.long = long
        self.vision = vision
        let buffers = BucketBufferPool(maxIdlePerLength: maxIdleBuffers)
        self.buffers = buffers
        self.backend = backend
        coalescer = Coalescer { [planner, runner, rows] sequences in
            try await BucketTextRuntime.run(sequences: sequences, planner: planner, buffers: buffers, runner: runner, rows: rows)
        }
    }

    public var info: RuntimeInfo {
        RuntimeInfo(maxTokens: max(planner.maxLength, long?.maxTokens ?? 0), maxImageTokens: vision?.maxTokens ?? 0,
                    bucketLengths: planner.lengths, backend: backend, functions: functions, notServed: notServed,
                    selfTestSkipped: selfTestSkipped)
    }

    public func warm() async throws {
        guard !warmed else { return }
        try await runner.warm()
        try await long?.warm()
        try await vision?.warm()
    }

    public func embed(tokenized texts: [[Int32]]) async throws -> [[Float]] {
        if let empty = texts.firstIndex(where: \.isEmpty) { throw Self.emptySequence(empty) }
        var vectors = [[Float]](repeating: [], count: texts.count)
        let bucketed = texts.indices.filter { texts[$0].count <= planner.maxLength }
        let longer = texts.indices.filter { texts[$0].count > planner.maxLength }
        if !longer.isEmpty {
            guard let long else {
                throw RuntimeError.tooLong(tokens: texts[longer[0]].count, max: planner.maxLength)
            }
            for index in longer {
                guard texts[index].count <= long.maxTokens else {
                    throw RuntimeError.tooLong(tokens: texts[index].count, max: long.maxTokens)
                }
                vectors[index] = try await long.embed(ids: texts[index])
            }
        }
        let sequences = bucketed.map { ModelSequence(ids: texts[$0]) }
        let results = try await embed(sequences: sequences)
        for (position, index) in bucketed.enumerated() { vectors[index] = results[position] }
        return vectors
    }

    public func embed(images: [PreparedImage]) async throws -> [[Float]] {
        guard let vision else { throw RuntimeError.unsupported("this model has no vision encoder") }
        var sequences: [ModelSequence] = []
        for image in images {
            guard image.tokens <= vision.maxTokens else {
                throw RuntimeError.unsupported("\(image.tokens) image tokens exceed \(vision.maxTokens)")
            }
            sequences.append(.image(rows: try await vision.encode(image)))
        }
        return try await embed(sequences: sequences)
    }

    private let coalescer: Coalescer
    static let coalescedRows = 512
    /// Calls of one run in flight at once. The Neural Engine executes one call at a time, so more in flight only
    /// queue: each holds its own filled buffers (1 MiB at 512 rows, up to 16 MiB at 2048), and an unbounded number
    /// let host memory grow with the batch. Three keep the device busy: one call executing, the next one filled and
    /// queued behind it, and a third being filled or handing back its rows. Measured on an M4 with requests of 256
    /// texts of 500 tokens and 2048 of 30: one at a time is 1–4 % slower than unbounded, two to four are as fast;
    /// three leave one call of slack for a slow fill.
    static let callsInFlight = 3

    /// Plans the sequences into calls and runs them, `callsInFlight` at a time: the ANE serializes the work
    /// itself, and overlapping lets one call's host-side filling hide behind another's execution.
    ///
    /// Requests whose sequences are all short (<= 512 rows: queries, and every image, which takes at most
    /// 284 rows) are coalesced with other such requests that arrive while the Neural Engine is busy, so
    /// concurrent requests share calls (measured on an M4 with 8 clients: 209 -> 501 requests/s, median
    /// 38 -> 16 ms; one client unchanged). Longer texts run on their own so a short query never waits
    /// behind them.
    func embed(sequences: [ModelSequence]) async throws -> [[Float]] {
        guard !sequences.isEmpty else { return [] }
        let counts = sequences.map(\.count)
        if let empty = counts.firstIndex(of: 0) { throw Self.emptySequence(empty) }
        if let over = counts.first(where: { $0 > planner.maxLength }) {
            throw RuntimeError.tooLong(tokens: over, max: planner.maxLength)
        }
        if counts.allSatisfy({ $0 <= Self.coalescedRows }) { return try await coalescer.submit(sequences) }
        return try await Self.run(sequences: sequences, planner: planner, buffers: buffers, runner: runner, rows: rows)
    }

    /// A text with no token ids (a tokenizer without <bos>/<eos> framing turns "" into nothing).
    static func emptySequence(_ index: Int) -> RuntimeError {
        .unsupported("input[\(index)] has no tokens; the tokenizer must frame every text with <bos>/<eos>")
    }

    static func run(sequences: [ModelSequence], planner: PackingPlanner, buffers: BucketBufferPool,
                    runner: any BucketFunctionRunner, rows: any TokenRowSource) async throws -> [[Float]]
    {
        let counts = sequences.map(\.count)
        let calls = planner.plan(counts)
        let results = try await windowedMap(calls, width: callsInFlight) { [buffers, runner, rows] call in
            let buffer = buffers.checkout(length: call.length)
            defer { buffers.checkin(buffer) }
            buffer.prepare(call, sequences: sequences, rows: rows)
            let output = try await runner.run(buffer)
            guard output.count >= call.segments.count else {
                throw RuntimeError.model("embed_\(call.length) returned \(output.count) rows")
            }
            return call.segments.enumerated().map { slot, segment in (segment.item, output[slot]) }
        }
        var vectors = [[Float]](repeating: [], count: sequences.count)
        for pairs in results {
            for (item, vector) in pairs { vectors[item] = vector }
        }
        return vectors
    }
}

/// Merges the sequences of concurrent requests into shared calls. A request that arrives while the
/// Neural Engine is busy waits for the next round, which plans everything that has queued up together:
/// no added latency when idle, fewer and fuller calls under load.
final class Coalescer: @unchecked Sendable {
    private let lock = NSLock()
    private var pending: [(sequences: [ModelSequence], continuation: CheckedContinuation<[[Float]], Error>)] = []
    private var running = 0
    private let maxRunning: Int
    private let execute: @Sendable ([ModelSequence]) async throws -> [[Float]]

    init(maxRunning: Int = 2, execute: @escaping @Sendable ([ModelSequence]) async throws -> [[Float]]) {
        self.maxRunning = maxRunning
        self.execute = execute
    }

    func submit(_ sequences: [ModelSequence]) async throws -> [[Float]] {
        try await withCheckedThrowingContinuation { continuation in
            let start = lock.withLock { () -> Bool in
                pending.append((sequences, continuation))
                guard running < maxRunning else { return false }
                running += 1
                return true
            }
            if start { Task { await drain() } }
        }
    }

    private func drain() async {
        while true {
            let batch = lock.withLock { () -> [(sequences: [ModelSequence], continuation: CheckedContinuation<[[Float]], Error>)] in
                defer { pending.removeAll() }
                if pending.isEmpty { running -= 1 }
                return pending
            }
            guard !batch.isEmpty else { return }
            do {
                let vectors = try await execute(batch.flatMap(\.sequences))
                var offset = 0
                for item in batch {
                    item.continuation.resume(returning: Array(vectors[offset..<(offset + item.sequences.count)]))
                    offset += item.sequences.count
                }
            } catch {
                for item in batch { item.continuation.resume(throwing: error) }
            }
        }
    }
}
