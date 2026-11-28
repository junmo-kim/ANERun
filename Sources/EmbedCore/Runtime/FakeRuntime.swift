// A model-free runtime for tests and for running the CLI/HTTP/MCP surface without a model package
// (ANEEMBED_RUNTIME=fake). Gemma uses the real planner and buffers, and both fake models pool
// deterministic token rows. A text gets the same vector alone or batched with others.
import Foundation

public enum FakeRuntime {
    public static func make(maxTokens: Int, modelID: ModelID = .embeddingGemma2) -> any EmbeddingRuntime {
        switch modelID {
        case .decision: preconditionFailure("Expected an embedding model")
        case .embeddingGemma2: return make(maxTokens: maxTokens)
        case .pplx, .qwen3Embedding: return FakePPLXRuntime(maxTokens: maxTokens)
        }
    }

    /// Keeps the concrete Gemma runtime available to callers that use its planner.
    public static func make(maxTokens: Int) -> BucketTextRuntime {
        make(maxTokens: maxTokens, images: true)
    }

    /// A runtime over the real planner/buffers with deterministic fake token rows and functions.
    public static func make(maxTokens: Int = 2048, images: Bool = true, callDelay: Duration = .zero)
        -> BucketTextRuntime
    {
        // The same functions the Core ML runtime loads for this limit.
        let lengths = CoreMLBucketRunner.lengths(forMaxTokens: maxTokens)
        return BucketTextRuntime(
            runner: PoolingFakeRunner(lengths: lengths, delay: callDelay),
            rows: HashedTokenRows(), vision: images ? FakeVisionEncoder() : nil, backend: "fake")
    }
}

private struct FakePPLXRuntime: EmbeddingRuntime {
    let info: RuntimeInfo

    init(maxTokens: Int) {
        let range = ModelID.pplx.tokenRange
        let limit = min(max(maxTokens, range.lowerBound), range.upperBound)
        info = RuntimeInfo(maxTokens: limit, maxImageTokens: 0, bucketLengths: [], backend: "fake")
    }

    func embed(tokenized texts: [[Int32]]) async throws -> [[Float]] {
        for (index, ids) in texts.enumerated() {
            guard !ids.isEmpty else { throw RuntimeError.unsupported("input[\(index)] has no tokens") }
            guard ids.count <= info.maxTokens else { throw RuntimeError.tooLong(tokens: ids.count, max: info.maxTokens) }
        }
        return try texts.map { ids in
            try Task.checkCancellation()
            var pooled = [Float](repeating: 0, count: Vectors.pplxDimension)
            for id in ids {
                for column in pooled.indices {
                    pooled[column] += HashedTokenRows.value(id: id, column: column)
                }
            }
            return pooled.map { $0 / Float(ids.count) }
        }
    }

    func embed(images: [PreparedImage]) async throws -> [[Float]] {
        throw RuntimeError.unsupported("image inputs are not supported: the model has no vision encoder")
    }

    func warm() async throws {}
}

/// Pseudo-random but fixed token rows (no 256 MB table needed).
public struct HashedTokenRows: TokenRowSource {
    public init() {}

    public func gather(_ ids: [Int32], into destination: UnsafeMutablePointer<Float16>) {
        for (position, id) in ids.enumerated() {
            let row = destination + position * BucketContract.hidden
            for column in 0..<BucketContract.hidden {
                row[column] = Float16(Self.value(id: id, column: column))
            }
        }
    }

    /// Uniform in [-1, 1) from SplitMix64(id, column).
    static func value(id: Int32, column: Int) -> Float {
        var z = UInt64(bitPattern: Int64(id)) &* 0x9E37_79B9_7F4A_7C15 &+ UInt64(column) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z ^= z >> 31
        return Float(z >> 40) / Float(1 << 23) - 1
    }
}

/// Each slot's output = normalize(pool row · inputs_embeds), widened from 512 to 768 values.
public struct PoolingFakeRunner: BucketFunctionRunner {
    public let lengths: [Int]
    public let delay: Duration

    public init(lengths: [Int] = BucketContract.lengths, delay: Duration = .zero) {
        self.lengths = lengths
        self.delay = delay
    }

    public func warm() async throws {}

    public func run(_ buffers: BucketBuffers) async throws -> [[Float]] {
        if delay > .zero { try await Task.sleep(for: delay) }
        let length = buffers.length, hidden = BucketContract.hidden
        var output: [[Float]] = []
        output.reserveCapacity(buffers.slots)
        for slot in 0..<buffers.slots {
            var pooled = [Float](repeating: 0, count: hidden)
            for position in 0..<length {
                let weight = Float(buffers.pool[slot * length + position])
                guard weight != 0 else { continue }
                let row = buffers.embedsRow(position)
                for column in 0..<hidden { pooled[column] += weight * Float(row[column]) }
            }
            output.append(Vectors.l2Normalized(pooled + pooled.prefix(BucketContract.dimension - hidden)))
        }
        return output
    }
}

/// Soft tokens from the mean colour of each 48 x 48 cell, so the same picture at two resolutions gives
/// nearly the same vector, and different pictures differ.
public struct FakeVisionEncoder: VisionEncoder {
    public let maxTokens = ImagePrep.maxSoftTokens

    public init() {}

    public func warm() async throws {}

    public func encode(_ image: PreparedImage) async throws -> [Float16] {
        let columns = image.width / 48, rows = image.height / 48, hidden = BucketContract.hidden
        var output = [Float16](repeating: 0, count: columns * rows * hidden)
        for cellY in 0..<rows {
            for cellX in 0..<columns {
                var mean: [Float] = [0, 0, 0]
                for y in (cellY * 48)..<(cellY * 48 + 48) {
                    for x in (cellX * 48)..<(cellX * 48 + 48) {
                        let pixel = (y * image.width + x) * 3
                        for c in 0..<3 { mean[c] += Float(image.rgb[pixel + c]) }
                    }
                }
                let token = cellY * columns + cellX
                for column in 0..<hidden {
                    var value: Float = 0
                    for c in 0..<3 {
                        value += mean[c] / (48 * 48 * 255) * HashedTokenRows.value(id: Int32(c), column: column)
                    }
                    output[token * hidden + column] = Float16(value)
                }
            }
        }
        return output
    }
}

/// Stand-in tokenizer used when no tokenizer.json exists. Each UTF-8 byte gets one id.
public struct ByteTokenizer: TextTokenizer {
    private let modelID: ModelID

    public init(modelID: ModelID = .embeddingGemma2) {
        self.modelID = modelID
    }

    public func encode(_ text: String) -> [Int32] {
        encode(text, addSpecial: true, parseSpecial: true)
    }

    public func encodeForEmbedding(_ text: String, limit: Int) -> BoundedTokenization {
        if modelID != .embeddingGemma2 {
            let count = text.utf8.count + (modelID == .qwen3Embedding ? 1 : 0)
            if count > limit { return .tooLong(atLeast: count) }
        }
        return .ids(encode(text))
    }

    public func encodeForEmbedding(_ text: String, limit: Int, truncate: Bool) -> BoundedTokenization {
        guard truncate else { return encodeForEmbedding(text, limit: limit) }
        let framing = prefixIDs.count + suffixIDs.count
        let count = text.utf8.count + framing
        guard count > limit else { return .ids(encode(text)) }
        guard limit > 0, limit >= framing else { return .tooLong(atLeast: count) }
        let prefix = text.utf8.prefix(limit - framing).map { Int32($0) + 3 }
        return .clipped(prefixIDs + prefix + suffixIDs)
    }

    public var maxBytesPerToken: Int? { 1 }
    public var prefixIDs: [Int32] { modelID == .embeddingGemma2 ? [SpecialTokens.bos] : [] }
    public var suffixIDs: [Int32] {
        switch modelID {
        case .embeddingGemma2: [SpecialTokens.eos]
        case .qwen3Embedding: [151643]
        default: []
        }
    }

    /// It has no added tokens, so `parseSpecial` changes nothing.
    public func encode(_ text: String, addSpecial: Bool, parseSpecial: Bool) -> [Int32] {
        let ids = text.utf8.map { Int32($0) + 3 }
        if addSpecial && modelID == .qwen3Embedding { return ids + [151643] }
        return addSpecial && modelID == .embeddingGemma2 ? [SpecialTokens.bos] + ids + [SpecialTokens.eos] : ids
    }

    public func pieceBytes(_ id: Int32) -> [UInt8]? {
        switch id {
        case 151643 where modelID == .qwen3Embedding: return Array("<|endoftext|>".utf8)
        case SpecialTokens.pad where modelID == .embeddingGemma2: return Array("<pad>".utf8)
        case SpecialTokens.eos where modelID == .embeddingGemma2: return Array("<eos>".utf8)
        case SpecialTokens.bos where modelID == .embeddingGemma2: return Array("<bos>".utf8)
        case 3..<259: return [UInt8(id - 3)]
        default: return nil
        }
    }
}
