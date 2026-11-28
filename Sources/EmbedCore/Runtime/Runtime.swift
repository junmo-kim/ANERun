// Model runtime interfaces. The service, CLI, HTTP and MCP layers only see `EmbeddingRuntime`, so the
// whole surface runs against `FakeRuntime` in tests and on machines without a model or a Neural Engine.
//
//   EmbeddingRuntime        texts (token ids) and images -> L2-normalized 768-d vectors
//   └─ BucketTextRuntime    packing planner + reusable buffers + a BucketFunctionRunner
//        ├─ CoreMLBucketRunner   embed_S and winp functions of the model package (Neural Engine)
//        └─ PoolingFakeRunner    deterministic stand-in that honours the bias/pool contract
//   LongTextRunner          2049...8192 tokens: LongTextEngine over CoreMLLongRunner, one winl call per text
//   VisionEncoder           prepared image -> soft-token rows for the text functions
//   SelfTestSuite           golden inputs from the package; a function that fails is not served
import Foundation

/// Static facts a runtime reports to the API (`/v1/models`, doctor).
public struct RuntimeInfo: Sendable, Equatable {
    public var maxTokens: Int
    /// 0 when images are not supported.
    public var maxImageTokens: Int
    public var bucketLengths: [Int]
    public var backend: String
    /// Model functions in use (empty for runtimes without named functions).
    public var functions: [String]
    /// Loaded functions that are not served (they failed the self-test, or the package has no case for them).
    public var notServed: [String]
    /// Loaded with `--selftest skip`: every function serves unchecked.
    public var selfTestSkipped: Bool

    /// Serving, but with less than was loaded or without the self-test (`/health` says "degraded").
    public var degraded: Bool { !notServed.isEmpty || selfTestSkipped }

    public init(maxTokens: Int, maxImageTokens: Int, bucketLengths: [Int], backend: String, functions: [String] = [],
                notServed: [String] = [], selfTestSkipped: Bool = false)
    {
        self.maxTokens = maxTokens
        self.maxImageTokens = maxImageTokens
        self.bucketLengths = bucketLengths
        self.backend = backend
        self.functions = functions
        self.notServed = notServed
        self.selfTestSkipped = selfTestSkipped
    }
}

public protocol EmbeddingRuntime: Sendable {
    var info: RuntimeInfo { get }
    /// One vector per text, in order. Every text is token ids with <bos>/<eos>, at most `info.maxTokens`.
    func embed(tokenized texts: [[Int32]]) async throws -> [[Float]]
    /// One vector per image, in order.
    func embed(images: [PreparedImage]) async throws -> [[Float]]
    /// Loads (and on first use compiles) every function this runtime will call.
    func warm() async throws
}

/// A complete or clipped encoding, or a safe lower bound that exceeds the request's cap.
public enum BoundedTokenization: Sendable, Equatable {
    case ids([Int32])
    case clipped([Int32])
    case tooLong(atLeast: Int)
}

/// Text to model-specific token ids and bytes.
public protocol TextTokenizer: Sendable {
    func encode(_ text: String) -> [Int32]
    func encodeForEmbedding(_ text: String, limit: Int) -> BoundedTokenization
    func encodeForEmbedding(_ text: String, limit: Int, truncate: Bool) -> BoundedTokenization
    var prefixIDs: [Int32] { get }
    var suffixIDs: [Int32] { get }
    func tokenLabel(_ id: Int32) -> String
    /// The most UTF-8 bytes of input text that one token can stand for; nil when there is no such bound. With it, a
    /// text of more than `limit × maxBytesPerToken` bytes is known to have more than `limit` tokens without
    /// encoding it.
    var maxBytesPerToken: Int? { get }
    /// For `POST /tokenize` (llama-server's contract): ids of `text`, <bos>/<eos> only with `addSpecial`; with
    /// `parseSpecial` false, control-token text such as "<bos>" is encoded as ordinary text.
    func encode(_ text: String, addSpecial: Bool, parseSpecial: Bool) -> [Int32]
    /// For `POST /detokenize`: the bytes of text one token stands for (special tokens as their text); nil for an id
    /// outside the vocabulary.
    func pieceBytes(_ id: Int32) -> [UInt8]?
}

extension TextTokenizer {
    public var prefixIDs: [Int32] { [] }
    public var suffixIDs: [Int32] { [] }
    public var maxBytesPerToken: Int? { nil }

    public func encodeForEmbedding(_ text: String, limit: Int) -> BoundedTokenization {
        .ids(encode(text))
    }

    public func encodeForEmbedding(_ text: String, limit: Int, truncate: Bool) -> BoundedTokenization {
        guard truncate else { return encodeForEmbedding(text, limit: limit) }
        let ids = encode(text)
        guard ids.count > limit else { return .ids(ids) }
        guard limit > 0, prefixIDs.count + suffixIDs.count <= limit else { return .tooLong(atLeast: ids.count) }
        return .clipped(Array(ids.prefix(limit - suffixIDs.count)) + suffixIDs)
    }

    public func tokenLabel(_ id: Int32) -> String {
        pieceBytes(id).map { String(decoding: $0, as: UTF8.self) } ?? ""
    }
}

extension GemmaTokenizer: TextTokenizer {
    public func tokenLabel(_ id: Int32) -> String { tokens([id]).first ?? "" }
    public func encode(_ text: String) -> [Int32] { encode(text, addSpecialTokens: true) }

    public func encode(_ text: String, addSpecial: Bool, parseSpecial: Bool) -> [Int32] {
        encode(text, addSpecialTokens: addSpecial, parseSpecial: parseSpecial)
    }
}

/// Source of scaled fp16 token rows.
public protocol TokenRowSource: Sendable {
    func gather(_ ids: [Int32], into destination: UnsafeMutablePointer<Float16>)
}

extension TokenTable: TokenRowSource {}

/// Runs one prepared bucket call and returns `slots` rows of 768 floats (rows of unused slots are
/// undefined). Implementations must be safe to call concurrently with different buffers.
public protocol BucketFunctionRunner: Sendable {
    var lengths: [Int] { get }
    func run(_ buffers: BucketBuffers) async throws -> [[Float]]
    func warm() async throws
}

/// Texts longer than the largest packing function (2049...8192 tokens).
public protocol LongTextRunner: Sendable {
    var maxTokens: Int { get }
    func embed(ids: [Int32]) async throws -> [Float]
    func warm() async throws
}

/// A prepared canvas -> its soft-token rows (tokens x 512 fp16, unscaled), which the text functions
/// then embed as `<bos> <|image> rows <image|> <eos>`.
public protocol VisionEncoder: Sendable {
    var maxTokens: Int { get }
    func encode(_ image: PreparedImage) async throws -> [Float16]
    func warm() async throws
}

public enum RuntimeError: Error, CustomStringConvertible, Sendable {
    case tooLong(tokens: Int, max: Int)
    case unsupported(String)
    case model(String)

    public var description: String {
        switch self {
        case .tooLong(let tokens, let max): return "\(tokens) tokens exceed the runtime limit of \(max)"
        case .unsupported(let message): return message
        case .model(let message): return message
        }
    }
}

/// Special token ids of EmbeddingGemma 2 used to wrap media rows.
public enum SpecialTokens {
    public static let bos: Int32 = 2
    public static let eos: Int32 = 1
    public static let pad: Int32 = 0
    public static let imageStart: Int32 = 255_999  // <|image>
    public static let imageEnd: Int32 = 258_882  // <image|>
}
