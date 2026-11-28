// OpenAI-compatible POST /v1/embeddings: request parsing and response encoding.
//
//   input            string | [string] | [image item]   (texts and images are not mixed)
//                    image item = llama-server's {"content": [{"type": "image_url",
//                                                 "image_url": {"url": "data:image/...;base64,..."}}]}
//   encoding_format  "float" (default) | "base64" (little-endian float32)
//   dimensions       model-specific Matryoshka prefix (full width by default)
//   input_type       optional extension (Voyage/Cohere style): "query" | "document" adds the model's
//                    task prefix when applicable; without it texts are embedded as given
//   model, user      accepted and ignored; the response names the configured model
import Foundation

public enum EncodingFormat: String, Sendable {
    case float
    case base64
}

public struct EmbeddingsRequest: Sendable, Equatable {
    public enum Input: Sendable, Equatable {
        case texts([String])
        case images([Data])
    }

    public var input: Input
    public var encodingFormat: EncodingFormat = .float
    public var dimensions = 768
    public var inputType: InputType?
    public var truncate: Bool

    public init(input: Input, encodingFormat: EncodingFormat = .float, dimensions: Int = 768, inputType: InputType? = nil,
                truncate: Bool = true) {
        self.truncate = truncate
        self.input = input
        self.encodingFormat = encodingFormat
        self.dimensions = dimensions
        self.inputType = inputType
    }

    public static let maxImageURLBytes = 8 << 20

    /// With `limits`, the number of inputs (texts, or images before any is decoded) is checked while parsing.
    public static func parse(_ body: Data, limits: ServiceLimits? = nil, modelID: ModelID = .embeddingGemma2)
        throws(ServiceError) -> EmbeddingsRequest
    {
        let root: JSONValue
        do {
            root = try JSONValue.parse(body)
        } catch {
            throw .badRequest("invalid_json", "request body is not valid JSON (\(error.description))")
        }
        guard let object = root.objectValue else { throw .badRequest("invalid_json", "expected a JSON object") }

        var request = EmbeddingsRequest(input: .texts([]), dimensions: Vectors.defaultDimension(for: modelID))
        switch object["input"] {
        case .string(let text)?:
            request.input = .texts([text])
        case .array(let items)? where !items.isEmpty && items.allSatisfy({ $0.stringValue != nil }):
            if let limits, items.count > limits.maxInputs {
                throw .badRequest("too_many_inputs", "at most \(limits.maxInputs) inputs per request", param: "input")
            }
            request.input = .texts(items.map { $0.stringValue! })
        case .array(let items)? where !items.isEmpty && items.allSatisfy({ $0.objectValue != nil }):
            if !modelID.supportsImages {
                for (index, item) in items.enumerated() { _ = try imageURL(item, index: index) }
                throw EmbeddingService.imagesUnsupported(modelID)
            }
            if let limits, items.count > limits.maxImages {
                throw .badRequest("too_many_images", "at most \(limits.maxImages) image items per request", param: "input")
            }
            var images: [Data] = []
            for (index, item) in items.enumerated() {
                images.append(try imageData(item, index: index))
            }
            request.input = .images(images)
        case .array(let items)? where items.isEmpty:
            throw .badRequest("empty_input", "input must not be empty", param: "input")
        default:
            throw .badRequest(
                "invalid_input", "input must be a string, a nonempty list of strings or a nonempty list of image items",
                param: "input")
        }

        switch object["encoding_format"] {
        case nil, .null?: break
        case .string(let value)?:
            guard let format = EncodingFormat(rawValue: value) else {
                throw .badRequest("invalid_encoding_format", "encoding_format must be float or base64",
                                  param: "encoding_format")
            }
            request.encodingFormat = format
        default:
            throw .badRequest("invalid_encoding_format", "encoding_format must be float or base64", param: "encoding_format")
        }

        switch object["dimensions"] {
        case nil, .null?: break
        case let value?:
            request.dimensions = try Vectors.validateDimensions(value.intValue, modelID: modelID)
        }

        switch object["input_type"] {
        case nil, .null?: break
        case .string(let value)?:
            guard let type = InputType(rawValue: value) else {
                throw .badRequest("invalid_input_type", "input_type must be query or document", param: "input_type")
            }
            guard case .texts = request.input else {
                throw .badRequest("invalid_input_type", "input_type applies to text inputs only", param: "input_type")
            }
            request.inputType = type
        default:
            throw .badRequest("invalid_input_type", "input_type must be query or document", param: "input_type")
        }
        if let value = object["truncate"] {
            guard case .bool(let truncate) = value else {
                throw .badRequest("invalid_truncate", "truncate must be a boolean", param: "truncate")
            }
            request.truncate = truncate
        }
        return request
    }

    /// The image bytes of one llama-server image item.
    static func imageData(_ item: JSONValue, index: Int) throws(ServiceError) -> Data {
        let url = try imageURL(item, index: index)
        guard url.utf8.count <= maxImageURLBytes else {
            throw ServiceError(.payloadTooLarge, "image_too_large", "input[\(index)]: image URLs are limited to 8 MiB",
                               param: "input")
        }
        return try decodeDataURL(url, index: index)
    }

    private static func imageURL(_ item: JSONValue, index: Int) throws(ServiceError) -> String {
        guard let parts = item["content"]?.arrayValue, parts.count == 1, parts[0]["type"]?.stringValue == "image_url",
            let url = parts[0]["image_url"]?["url"]?.stringValue
        else {
            throw .badRequest("invalid_image_item",
                              "input[\(index)]: an image item needs one content part of type image_url",
                              param: "input")
        }
        return url
    }

    /// `data:image/<type>;base64,<payload>`; remote URLs are not fetched.
    public static func decodeDataURL(_ url: String, index: Int) throws(ServiceError) -> Data {
        guard url.hasPrefix("data:image/"), let comma = url.firstIndex(of: ","), url[..<comma].hasSuffix(";base64") else {
            throw .badRequest("invalid_image_url", "input[\(index)]: image_url must be a data:image/...;base64 URL",
                              param: "input")
        }
        guard let data = Data(base64Encoded: String(url[url.index(after: comma)...]), options: .ignoreUnknownCharacters),
            !data.isEmpty
        else {
            throw ServiceError(.invalidInput, "image_decode_failed", "input[\(index)]: the data URL is not valid base64",
                               param: "input")
        }
        return data
    }
}

public enum EmbeddingsResponse {
    /// The OpenAI response body for `result`.
    public static func body(_ result: EmbeddingResult, request: EmbeddingsRequest, model: String,
                            modelID: ModelID = .embeddingGemma2, promptTokensPerSecond: Double? = nil) -> Data
    {
        var output: [UInt8] = []
        output.reserveCapacity(result.vectors.count * (request.encodingFormat == .float ? request.dimensions * 12 : request.dimensions * 6) + 128)
        output.append(contentsOf: #"{"object":"list","data":["#.utf8)
        let clipped = Set(result.truncatedIndices)
        for (index, vector) in result.vectors.enumerated() {
            if index > 0 { output.append(UInt8(ascii: ",")) }
            output.append(contentsOf: #"{"object":"embedding","index":\#(index),"embedding":"#.utf8)
            let embedding = Vectors.output(vector, to: request.dimensions, modelID: modelID)
            switch request.encodingFormat {
            case .float: writeJSONFloats(embedding, to: &output)
            case .base64: writeJSONString(Vectors.base64(embedding), to: &output)
            }
            if clipped.contains(index) { output.append(contentsOf: #","truncated":true"#.utf8) }
            output.append(UInt8(ascii: "}"))
        }
        output.append(contentsOf: #"],"model":"#.utf8)
        writeJSONString(model, to: &output)
        let tokens = result.totalTokens
        output.append(contentsOf: #","usage":{"prompt_tokens":\#(tokens),"total_tokens":\#(tokens)"#.utf8)
        if let rate = promptTokensPerSecond, rate.isFinite, rate > 0 {
            output.append(contentsOf: #","prompt_tokens_per_second":\#(rate)"#.utf8)
        }
        output.append(contentsOf: "}}".utf8)
        return Data(output)
    }
}
