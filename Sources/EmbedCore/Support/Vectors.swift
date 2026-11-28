// Output-side vector operations: L2 normalization, Matryoshka truncation, OpenAI base64 encoding, cosine.
import Accelerate
import Foundation

public enum Vectors {
    /// Matryoshka sizes EmbeddingGemma 2 was trained for (prefixes stay meaningful once re-normalized).
    public static let dimensions = [768, 512, 256, 128]
    static let pplxDimension = 1024
    static let pplxDimensionRange = 1...pplxDimension
    static let qwenDimensionRange = 32...1024
    private static let pplxDimensions = Array(pplxDimensionRange.reversed())
    private static let qwenDimensions = Array(qwenDimensionRange.reversed())

    static func supportedDimensions(for modelID: ModelID) -> [Int] {
        switch modelID {
        case .decision: preconditionFailure("Expected an embedding model")
        case .embeddingGemma2: dimensions
        case .pplx: pplxDimensions
        case .qwen3Embedding: qwenDimensions
        }
    }

    static func defaultDimension(for modelID: ModelID) -> Int {
        supportedDimensions(for: modelID)[0]
    }

    static func validateDimensions(_ value: Int?, modelID: ModelID) throws(ServiceError) -> Int {
        let allowed: String
        switch modelID {
        case .decision: throw modelID.unsupportedEmbedding
        case .embeddingGemma2: allowed = "one of \(dimensions.map(String.init).joined(separator: ", "))"
        case .pplx: allowed = "an integer from 1 to \(pplxDimension)"
        case .qwen3Embedding: allowed = "an integer from \(qwenDimensionRange.lowerBound) to \(qwenDimensionRange.upperBound)"
        }
        if let value, supportedDimensions(for: modelID).contains(value) { return value }
        throw .badRequest("invalid_dimensions", "dimensions must be \(allowed)", param: "dimensions")
    }

    /// The vector over its L2 norm; unchanged when the norm is 0 or not finite. The values are scaled by the largest
    /// magnitude first, so squares neither underflow (values below ~1e-19) nor overflow (above ~1e19) in float32.
    public static func l2Normalized(_ vector: [Float]) -> [Float] {
        var maxAbs: Float = 0
        vDSP_maxmgv(vector, 1, &maxAbs, vDSP_Length(vector.count))
        guard maxAbs > 0, maxAbs.isFinite else { return vector }
        let scaled = vDSP.divide(vector, maxAbs)
        var sumOfSquares: Float = 0
        vDSP_svesq(scaled, 1, &sumOfSquares, vDSP_Length(scaled.count))
        let norm = sumOfSquares.squareRoot()
        guard norm > 0, norm.isFinite else { return vector }
        return vDSP.divide(scaled, norm)
    }

    /// The first `dimensions` values, L2-normalized. The full size is normalized too, so every output keeps the
    /// norm-1 promise even if a model function's own final normalization is slightly off (the self-test compares
    /// directions only).
    public static func truncated(_ vector: [Float], to dimensions: Int) -> [Float] {
        l2Normalized(dimensions < vector.count ? Array(vector.prefix(dimensions)) : vector)
    }

    /// Applies the selected model's output transform to the requested prefix.
    public static func output(_ vector: [Float], to dimensions: Int, modelID: ModelID) -> [Float] {
        switch modelID {
        case .decision: preconditionFailure("Expected an embedding model")
        case .embeddingGemma2, .qwen3Embedding:
            return truncated(vector, to: dimensions)
        case .pplx:
            return vector.prefix(dimensions).map(PPLXQuantization.code)
        }
    }

    /// OpenAI's `encoding_format: "base64"`: little-endian float32 bytes, base64.
    public static func base64(_ vector: [Float]) -> String {
        vector.map { $0.bitPattern.littleEndian }.withUnsafeBytes { Data($0) }.base64EncodedString()
    }

    public static func decodeBase64(_ text: String) -> [Float]? {
        guard let data = Data(base64Encoded: text), data.count % 4 == 0 else { return nil }
        return data.withUnsafeBytes { raw in
            (0..<(data.count / 4)).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
        }
    }

    public static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        var dot: Float = 0, aa: Float = 0, bb: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(a.count))
        vDSP_svesq(a, 1, &aa, vDSP_Length(a.count))
        vDSP_svesq(b, 1, &bb, vDSP_Length(b.count))
        let denominator = (aa * bb).squareRoot()
        return denominator > 0 ? dot / denominator : 0
    }
}

/// Optional task prompts selected by `input_type`.
public enum InputType: String, Sendable, CaseIterable {
    case query
    case document

    public var prefix: String {
        switch self {
        case .query: return "task: search result | query: "
        case .document: return "title: none | text: "
        }
    }

    public func apply(_ text: String) -> String { prefix + text }

    public func apply(_ text: String, modelID: ModelID) -> String {
        switch modelID {
        case .decision: preconditionFailure("Expected an embedding model")
        case .embeddingGemma2: apply(text)
        case .pplx: text
        case .qwen3Embedding:
            self == .query ? "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:" + text : text
        }
    }
}
