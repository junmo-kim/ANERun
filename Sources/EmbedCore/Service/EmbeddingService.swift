// The one place requests are validated and scheduled; CLI, HTTP and MCP all call this.
//
// Scheduling: three lanes with separate concurrency limits, so a burst of long documents or images
// cannot make short queries wait behind them (the prototype's lesson from a shared CPU worker queue).
//   query: short texts   image: decode + vision + text   long: texts above the query lane limit
import Foundation

public struct ServiceLimits: Sendable, Equatable {
    /// Longest text accepted in model tokens, capped by the runtime.
    public var maxTokens = 2048
    public var maxInputs = 2048
    public var maxImages = 8
    public var maxImageBytes = 8 << 20
    public var queryConcurrency = 8
    public var imageConcurrency = 2
    public var longConcurrency = 2
    /// Texts up to this many tokens use the query lane. Config uses 64 for Qwen3.
    public var shortTokens = 512

    public init() {}
}

public struct EmbeddingResult: Sendable, Equatable {
    public var vectors: [[Float]]
    /// Per input: tokens the model saw (images: soft tokens + 4).
    public var tokenCounts: [Int]
    public var truncatedIndices: [Int]
    public var totalTokens: Int { tokenCounts.reduce(0, +) }

    public init(vectors: [[Float]], tokenCounts: [Int], truncatedIndices: [Int] = []) {
        self.truncatedIndices = truncatedIndices
        self.vectors = vectors
        self.tokenCounts = tokenCounts
    }
}

public final class EmbeddingService: Sendable {
    public let tokenizer: any TextTokenizer
    public let runtime: any EmbeddingRuntime
    public let limits: ServiceLimits
    public let modelName: String
    public let modelID: ModelID
    private let queryLane: AsyncSemaphore
    private let imageLane: AsyncSemaphore
    private let longLane: AsyncSemaphore
    private let shortTokenizerLane: AsyncSemaphore
    private let longTokenizerLane: AsyncSemaphore
    private let rawTokenizerLane: AsyncSemaphore
    private let longRawTokenizerLane: AsyncSemaphore

    public init(tokenizer: any TextTokenizer, runtime: any EmbeddingRuntime, limits: ServiceLimits = ServiceLimits(),
                modelName: String = "embeddinggemma-2", modelID: ModelID = .embeddingGemma2)
    {
        self.tokenizer = tokenizer
        self.runtime = runtime
        self.limits = limits
        self.modelName = modelName
        self.modelID = modelID
        queryLane = AsyncSemaphore(max(1, limits.queryConcurrency))
        imageLane = AsyncSemaphore(max(1, limits.imageConcurrency))
        longLane = AsyncSemaphore(max(1, limits.longConcurrency))
        shortTokenizerLane = AsyncSemaphore(max(1, limits.queryConcurrency))
        longTokenizerLane = AsyncSemaphore(max(1, limits.longConcurrency))
        rawTokenizerLane = AsyncSemaphore(max(1, limits.longConcurrency))
        longRawTokenizerLane = AsyncSemaphore(max(2, limits.longConcurrency) - 1)
    }

    /// Raw tokenization and detokenization share a cap. Reserve one slot for
    /// small requests when the configured cap allows more than one worker.
    public func withTokenizerLane<T: Sendable>(isShort: Bool = false, _ body: @escaping @Sendable () throws(ServiceError) -> T)
        async throws(ServiceError) -> T
    {
        do {
            let run: @Sendable () async throws -> T = { [rawTokenizerLane] in
                try await rawTokenizerLane.withPermit {
                    try Task.checkCancellation()
                    return try body()
                }
            }
            if isShort || limits.longConcurrency <= 1 { return try await run() }
            return try await longRawTokenizerLane.withPermit(run)
        } catch let error as ServiceError {
            throw error
        } catch {
            throw Self.map(error)
        }
    }

    func isShortTokenizerInput(bytes: Int) -> Bool { bytes / 4 <= limits.shortTokens }

    public var maxTokens: Int { min(limits.maxTokens, runtime.info.maxTokens) }
    public var maxImageTokens: Int { modelID.supportsImages ? runtime.info.maxImageTokens : 0 }

    public func tokenize(_ text: String, inputType: InputType? = nil) -> [Int32] {
        let prepared = inputType?.apply(text, modelID: modelID) ?? text
        return tokenizer.encode(prepared)
    }

    // MARK: Texts

    public func embedTexts(_ texts: [String], inputType: InputType? = nil, truncate: Bool = true) async throws(ServiceError) -> EmbeddingResult {
        guard !texts.isEmpty else { throw .badRequest("empty_input", "input must not be empty", param: "input") }
        guard texts.count <= limits.maxInputs else {
            throw .badRequest("too_many_inputs", "at most \(limits.maxInputs) inputs per request", param: "input")
        }
        return try await embedTextBatch(texts, inputType: inputType, truncate: truncate)
    }

    // A similarity request has two distinct operands, so it can validate them together without a text-list limit.
    func embedTextBatch(_ texts: [String], inputType: InputType? = nil, truncate: Bool = true) async throws(ServiceError) -> EmbeddingResult {
        let limit = maxTokens
        func tooLong(_ index: Int, _ tokens: String) -> ServiceError {
            ServiceError(.invalidInput, "context_length_exceeded",
                         "input[\(index)] has \(tokens) tokens; this model's maximum context length is \(limit) tokens",
                         param: "input",
                         hint: limit < runtime.info.maxTokens ? "raise --max-tokens (up to \(runtime.info.maxTokens))" : nil)
        }
        if case .decision = modelID { throw modelID.unsupportedEmbedding }
        if !truncate, modelID == .embeddingGemma2, let bytesPerToken = tokenizer.maxBytesPerToken {
            let prefixBytes = inputType?.prefix.utf8.count ?? 0
            if let over = texts.firstIndex(where: { $0.utf8.count + prefixBytes > limit * bytesPerToken }) {
                let bytes = texts[over].utf8.count + prefixBytes
                throw tooLong(over, "at least \((bytes + bytesPerToken - 1) / bytesPerToken)")
            }
        }
        var tokenized: [[Int32]] = []
        var truncatedIndices: [Int] = []
        tokenized.reserveCapacity(texts.count)
        for (index, text) in texts.enumerated() {
            let prepared = inputType?.apply(text, modelID: modelID) ?? text
            // Choose a lane before BPE. The byte threshold is a scheduling heuristic,
            // independent of the actual token count used for inference routing.
            let lane = isShortTokenizerInput(bytes: prepared.utf8.count) ? shortTokenizerLane : longTokenizerLane
            let encoded: BoundedTokenization
            do {
                encoded = try await lane.withPermit { [tokenizer] in
                    try Task.checkCancellation()
                    return tokenizer.encodeForEmbedding(prepared, limit: limit, truncate: truncate)
                }
            } catch {
                throw Self.map(error)
            }
            let ids: [Int32]
            switch encoded {
            case .ids(let tokens): ids = tokens
            case .clipped(let tokens):
                ids = tokens
                truncatedIndices.append(index)
            case .tooLong(let count): throw tooLong(index, "at least \(count)")
            }
            guard !ids.isEmpty else {
                throw ServiceError(.invalidInput, "empty_sequence", "input[\(index)] has no tokens", param: "input")
            }
            guard ids.count <= limit else { throw tooLong(index, "\(ids.count)") }
            tokenized.append(ids)
        }
        let short = tokenized.indices.filter { tokenized[$0].count <= limits.shortTokens }
        let long = tokenized.indices.filter { tokenized[$0].count > limits.shortTokens }
        var vectors = [[Float]](repeating: [], count: texts.count)
        do {
            try await withThrowingTaskGroup(of: [(Int, [Float])].self) { group in
                let runtime = self.runtime
                if !short.isEmpty {
                    let batch = short.map { tokenized[$0] }
                    group.addTask { [queryLane] in
                        let result = try await queryLane.withPermit { try await runtime.embed(tokenized: batch) }
                        return Array(zip(short, result))
                    }
                }
                for index in long {
                    let ids = tokenized[index]
                    group.addTask { [longLane] in
                        let result = try await longLane.withPermit { try await runtime.embed(tokenized: [ids]) }
                        return [(index, result[0])]
                    }
                }
                for try await pairs in group {
                    for (index, vector) in pairs { vectors[index] = vector }
                }
            }
        } catch {
            throw Self.map(error)
        }
        return EmbeddingResult(vectors: vectors, tokenCounts: tokenized.map(\.count), truncatedIndices: truncatedIndices)
    }

    // MARK: Images

    /// Images as encoded bytes (PNG, JPEG, anything ImageIO reads), at most `limits.maxImages`.
    public func embedImages(_ images: [Data]) async throws(ServiceError) -> EmbeddingResult {
        guard !images.isEmpty else { throw .badRequest("empty_input", "input must not be empty", param: "input") }
        guard images.count <= limits.maxImages else {
            throw .badRequest("too_many_images", "at most \(limits.maxImages) image items per request", param: "input")
        }
        guard maxImageTokens > 0 else { throw Self.imagesUnsupported(modelID) }
        if let big = images.firstIndex(where: { $0.count > limits.maxImageBytes }) {
            throw ServiceError(.payloadTooLarge, "image_too_large",
                               "input[\(big)] is \(images[big].count) bytes; the limit is \(limits.maxImageBytes)",
                               param: "input")
        }
        // Decode and resize everything first, so a bad image fails the request before any model call.
        var prepared = [PreparedImage?](repeating: nil, count: images.count)
        var failure: ServiceError?
        await withTaskGroup(of: (Int, Result<PreparedImage, Error>).self) { group in
            for (index, data) in images.enumerated() {
                group.addTask { [imageLane] in
                    await imageLane.wait()
                    let result = Result { try ImagePrep.prepare(data) }
                    await imageLane.signal()
                    return (index, result)
                }
            }
            for await (index, result) in group {
                switch result {
                case .success(let image): prepared[index] = image
                case .failure(let error):
                    if failure == nil {
                        failure = ServiceError(.invalidInput, "image_decode_failed",
                                               "input[\(index)] cannot be decoded as an image (\(error))", param: "input")
                    }
                }
            }
        }
        if let failure { throw failure }
        let canvases = prepared.map { $0! }
        do {
            let runtime = self.runtime
            let vectors = try await imageLane.withPermit { try await runtime.embed(images: canvases) }
            // <bos> <|image> soft tokens <image|> <eos>, as llama-server counts them
            return EmbeddingResult(vectors: vectors, tokenCounts: canvases.map { $0.tokens + 4 })
        } catch {
            throw Self.map(error)
        }
    }

    static func imagesUnsupported(_ modelID: ModelID) -> ServiceError {
        ServiceError(.invalidInput, "images_unsupported",
                     "image inputs are not supported: the model has no vision encoder", param: "input",
                     hint: modelID.supportsImages ? nil : "select --model embeddinggemma-2 for image inputs")
    }

    static func map(_ error: Error) -> ServiceError {
        switch error {
        case let error as ServiceError: return error
        case RuntimeError.tooLong(let tokens, let max):
            return ServiceError(.invalidInput, "context_length_exceeded",
                                "\(tokens) tokens; this model's maximum context length is \(max) tokens", param: "input")
        case RuntimeError.unsupported(let message):
            return ServiceError(.invalidInput, "unsupported_input", message, param: "input")
        default:
            return ServiceError(.internal, "embedding_failed", "embedding failed: \(error)")
        }
    }
}
