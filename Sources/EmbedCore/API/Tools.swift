// Tools shared by MCP (stdio and POST /mcp), HTTP QUERY /{tool} and the CLI: embed_text, embed_image,
// similarity. Each returns a JSONValue that the caller renders as JSON or YAML.
import Foundation

public struct ToolContext: Sendable {
    public var service: EmbeddingService
    /// Whether `path` arguments may read local files. True for stdio MCP and the CLI; false over HTTP,
    /// where it would turn the server into a file-reading service for anyone who can reach it.
    public var allowPaths: Bool

    public init(service: EmbeddingService, allowPaths: Bool) {
        self.service = service
        self.allowPaths = allowPaths
    }
}

public enum Tools {
    public static let names = ["embed_text", "embed_image", "similarity"]

    private static let dimensionsSchema: JSONValue = [
        "type": "integer", "enum": [768, 512, 256, 128],
        "description": "Matryoshka size: the first N values, re-normalized. Default 768.",
    ]
    private static let encodingSchema: JSONValue = [
        "type": "string", "enum": ["float", "base64"],
        "description": "float (default) or base64 of little-endian float32, as in the OpenAI API.",
    ]
    private static let inputTypeSchema: JSONValue = [
        "type": "string", "enum": ["query", "document"],
        "description":
            "Adds EmbeddingGemma 2's task prefix: query -> 'task: search result | query: ', document -> 'title: none | text: '. Omit to embed the text as given.",
    ]
    private static let formatSchema: JSONValue = [
        "type": "string", "enum": ["json", "yaml"], "description": "Output format of the tool text. Default json.",
    ]
    private static let truncateSchema: JSONValue = [
        "type": "boolean", "default": true,
        "description": "Truncate text at the context limit. Set to false to reject overlong text.",
    ]

    /// MCP tool descriptors (tools/list).
    public static let definitions: [JSONValue] = [
        [
            "name": "embed_text",
            "description": """
                Embed text with EmbeddingGemma 2 on the Apple Neural Engine (768-d, L2-normalized). Pass `text` or `texts` \
                (overlong texts are truncated by default). For retrieval, embed \
                queries with input_type=query and documents with input_type=document. Over HTTP: \
                QUERY /embed_text?input_type=query with the raw UTF-8 text as the body.
                """,
            "inputSchema": [
                "type": "object",
                "properties": [
                    "text": ["type": "string", "description": "One text."],
                    "texts": ["type": "array", "items": ["type": "string"], "description": "Several texts (batched)."],
                    "input_type": inputTypeSchema, "dimensions": dimensionsSchema, "encoding_format": encodingSchema,
                    "format": formatSchema, "truncate": truncateSchema,
                ],
            ],
        ],
        [
            "name": "embed_image",
            "description": """
                Embed an image into the same space as text (official EmbeddingGemma 2 recipe: resized to at most 280 \
                soft tokens). Pass base64 image bytes in `data` (or `images` for several), or a local `path` when \
                running over stdio. Over HTTP: QUERY /embed_image with the raw image bytes as the body.
                """,
            "inputSchema": [
                "type": "object",
                "properties": [
                    "data": ["type": "string", "description": "Base64 of the image file (PNG, JPEG, HEIC, ...)."],
                    "images": ["type": "array", "items": ["type": "string"], "description": "Several base64 images."],
                    "path": ["type": "string", "description": "Local file path (stdio MCP only)."],
                    "dimensions": dimensionsSchema, "encoding_format": encodingSchema, "format": formatSchema,
                ],
            ],
        ],
        [
            "name": "similarity",
            "description": """
                Cosine similarity of two inputs. Each side is a text (`a` / `b`, optional `a_input_type` / \
                `b_input_type`) or an image (`a_image` / `b_image` base64, or `a_path` / `b_path` over stdio). \
                Over HTTP: QUERY /similarity with these arguments as a JSON body.
                """,
            "inputSchema": [
                "type": "object",
                "properties": [
                    "a": ["type": "string"], "b": ["type": "string"],
                    "a_input_type": inputTypeSchema, "b_input_type": inputTypeSchema,
                    "a_image": ["type": "string"], "b_image": ["type": "string"],
                    "a_path": ["type": "string"], "b_path": ["type": "string"],
                    "dimensions": dimensionsSchema, "format": formatSchema, "truncate": truncateSchema,
                ],
            ],
        ],
    ]

    private static let pplxDefinitions = textOnlyDefinitions(for: .pplx)
    private static let qwenDefinitions = textOnlyDefinitions(for: .qwen3Embedding)

    private static func textOnlyDefinitions(for modelID: ModelID) -> [JSONValue] {
        let range = modelID == .pplx ? Vectors.pplxDimensionRange : Vectors.qwenDimensionRange
        return definitions.compactMap { definition in
            guard var tool = definition.objectValue, let name = tool["name"]?.stringValue, name != "embed_image",
                  var schema = tool["inputSchema"]?.objectValue, var properties = schema["properties"]?.objectValue
            else { return nil }
            properties["dimensions"] = [
                "type": "integer", "minimum": .int(range.lowerBound), "maximum": .int(range.upperBound),
                "description": .string("Number of prefix values to return. Default \(range.upperBound)."),
            ]
            let prompt: JSONValue = [
                "type": "string", "enum": ["query", "document"],
                "description": modelID == .pplx ? "Optional compatibility field. The text is embedded as given."
                    : "Query adds Qwen3’s retrieval instruction. Document and omitted types use the original text.",
            ]
            if name == "embed_text" {
                properties["input_type"] = prompt
                tool["description"] = modelID == .pplx ? "Embed one text or a batch as quantized pplx-embed-v1-0.6b vectors."
                    : "Embed one text or a batch as normalized Qwen3 vectors."
            } else {
                for side in ["a", "b"] {
                    properties["\(side)_image"] = nil
                    properties["\(side)_path"] = nil
                    properties["\(side)_input_type"] = prompt
                }
                tool["description"] = "Cosine similarity of two text embeddings."
            }
            schema["properties"] = .object(properties)
            tool["inputSchema"] = .object(schema)
            return .object(tool)
        }
    }

    public static func definitions(for modelID: ModelID, maxContextLength: Int? = nil) -> [JSONValue] {
        let catalog: [JSONValue]
        switch modelID {
        case .decision: return []
        case .embeddingGemma2: catalog = definitions
        case .pplx: catalog = pplxDefinitions
        case .qwen3Embedding: catalog = qwenDefinitions
        }
        guard let maxContextLength else { return catalog }
        return catalog.map { definition in
            guard var tool = definition.objectValue, tool["name"]?.stringValue != "embed_image" else { return definition }
            var metadata = tool["_meta"]?.objectValue ?? JSONObject()
            metadata["aneembed/max_context_length"] = .int(maxContextLength)
            tool["_meta"] = .object(metadata)
            return .object(tool)
        }
    }

    public static func call(_ name: String, arguments: JSONObject, context: ToolContext) async throws(ServiceError)
        -> JSONValue
    {
        switch name {
        case "embed_text": return try await embedText(arguments, context: context)
        case "embed_image": return try await embedImage(arguments, context: context)
        case "similarity": return try await similarity(arguments, context: context)
        default: throw ServiceError(.notFound, "unknown_tool", "unknown tool \(name); tools: \(names.joined(separator: ", "))")
        }
    }

    // MARK: Tools

    static func embedText(_ arguments: JSONObject, context: ToolContext) async throws(ServiceError) -> JSONValue {
        let texts: [String]
        if let text = arguments["text"]?.stringValue {
            texts = [text]
        } else if let list = arguments["texts"]?.arrayValue, list.allSatisfy({ $0.stringValue != nil }) {
            texts = list.map { $0.stringValue! }
        } else {
            throw .badRequest("missing_input", "pass `text` (string) or `texts` (list of strings)", param: "text")
        }
        let options = try outputOptions(arguments, modelID: context.service.modelID)
        let inputType = try self.inputType(arguments["input_type"], param: "input_type")
        let truncate = try truncateOption(arguments)
        let result = try await context.service.embedTexts(texts, inputType: inputType, truncate: truncate)
        return embeddingOutput(result, options: options, model: context.service.modelName, modelID: context.service.modelID)
    }

    static func embedImage(_ arguments: JSONObject, context: ToolContext) async throws(ServiceError) -> JSONValue {
        if !context.service.modelID.supportsImages { throw EmbeddingService.imagesUnsupported(context.service.modelID) }
        var images: [Data] = []
        let limits = context.service.limits
        if let data = arguments["data"]?.stringValue {
            images = [try base64Image(data, param: "data", limits: limits)]
        } else if let list = arguments["images"]?.arrayValue, list.allSatisfy({ $0.stringValue != nil }) {
            // Counted before any is decoded, as /v1/embeddings does.
            guard list.count <= limits.maxImages else {
                throw .badRequest("too_many_images", "at most \(limits.maxImages) images per request", param: "images")
            }
            for (index, item) in list.enumerated() {
                images.append(try base64Image(item.stringValue!, param: "images[\(index)]", limits: limits))
            }
        } else if let path = arguments["path"]?.stringValue {
            images = [try readPath(path, param: "path", context: context)]
        } else {
            throw .badRequest("missing_input", "pass `data` (base64 image), `images` or `path`", param: "data")
        }
        let options = try outputOptions(arguments, modelID: context.service.modelID)
        let result = try await context.service.embedImages(images)
        return embeddingOutput(result, options: options, model: context.service.modelName, modelID: context.service.modelID)
    }

    static func similarity(_ arguments: JSONObject, context: ToolContext) async throws(ServiceError) -> JSONValue {
        let service = context.service
        let dimensions = try outputOptions(arguments, modelID: service.modelID).dimensions
        let truncate = try truncateOption(arguments)
        if !service.modelID.supportsImages {
            return try await textSimilarity(arguments, service: service, dimensions: dimensions, truncate: truncate)
        }
        var vectors: [[Float]] = []
        var tokens: [Int] = []
        var truncatedIndices: [Int] = []
        for (index, side) in ["a", "b"].enumerated() {
            let result: EmbeddingResult
            if let text = arguments[side]?.stringValue {
                let type = try inputType(arguments["\(side)_input_type"], param: "\(side)_input_type")
                result = try await service.embedTexts([text], inputType: type, truncate: truncate)
            } else if let data = arguments["\(side)_image"]?.stringValue {
                result = try await context.service.embedImages([try base64Image(data, param: "\(side)_image",
                                                                                limits: context.service.limits)])
            } else if let path = arguments["\(side)_path"]?.stringValue {
                result = try await context.service.embedImages([try readPath(path, param: "\(side)_path", context: context)])
            } else {
                throw .badRequest("missing_input", "pass `\(side)` (text), `\(side)_image` (base64) or `\(side)_path`",
                                  param: side)
            }
            vectors.append(result.vectors[0])
            tokens.append(result.tokenCounts[0])
            if !result.truncatedIndices.isEmpty { truncatedIndices.append(index) }
        }
        return similarityOutput(vectors, tokens: tokens, truncatedIndices: truncatedIndices,
                                dimensions: dimensions, service: service)
    }

    private static func textSimilarity(_ arguments: JSONObject, service: EmbeddingService, dimensions: Int, truncate: Bool)
        async throws(ServiceError) -> JSONValue
    {
        var texts: [String] = []
        for side in ["a", "b"] {
            if let text = arguments[side]?.stringValue {
                let type = try inputType(arguments["\(side)_input_type"], param: "\(side)_input_type")
                texts.append(type?.apply(text, modelID: service.modelID) ?? text)
            } else if arguments["\(side)_image"]?.stringValue != nil || arguments["\(side)_path"]?.stringValue != nil {
                throw EmbeddingService.imagesUnsupported(service.modelID)
            } else {
                throw .badRequest("missing_input", "pass `\(side)` (text)", param: side)
            }
        }
        let result = try await service.embedTextBatch(texts, truncate: truncate)
        return similarityOutput(result.vectors, tokens: result.tokenCounts, truncatedIndices: result.truncatedIndices,
                                dimensions: dimensions, service: service)
    }

    private static func similarityOutput(_ vectors: [[Float]], tokens: [Int], truncatedIndices: [Int],
                                         dimensions: Int, service: EmbeddingService) -> JSONValue
    {
        let output = vectors.map { Vectors.output($0, to: dimensions, modelID: service.modelID) }
        var result: JSONObject = [
            "cosine": .number(Double(Vectors.cosine(output[0], output[1]))), "dimensions": .int(dimensions),
            "tokens": [.int(tokens[0]), .int(tokens[1])], "model": .string(service.modelName),
        ]
        if !truncatedIndices.isEmpty { result["truncated_indices"] = .array(truncatedIndices.map(JSONValue.int)) }
        return .object(result)
    }

    // MARK: Shared

    static func truncateOption(_ arguments: JSONObject) throws(ServiceError) -> Bool {
        guard let value = arguments["truncate"] else { return true }
        guard let truncate = value.boolValue else {
            throw .badRequest("invalid_truncate", "truncate must be a boolean", param: "truncate")
        }
        return truncate
    }

    public struct OutputOptions: Sendable {
        public var dimensions = 768
        public var encoding = EncodingFormat.float

        public init(dimensions: Int = 768, encoding: EncodingFormat = .float) {
            self.dimensions = dimensions
            self.encoding = encoding
        }
    }

    public static func outputOptions(_ arguments: JSONObject, modelID: ModelID = .embeddingGemma2)
        throws(ServiceError) -> OutputOptions
    {
        var options = OutputOptions(dimensions: Vectors.defaultDimension(for: modelID))
        if let value = arguments["dimensions"], !value.isNull {
            let dimensions = value.intValue ?? value.stringValue.flatMap { Int($0) }
            options.dimensions = try Vectors.validateDimensions(dimensions, modelID: modelID)
        }
        if let value = arguments["encoding_format"], !value.isNull {
            guard let format = value.stringValue.flatMap(EncodingFormat.init(rawValue:)) else {
                throw .badRequest("invalid_encoding_format", "encoding_format must be float or base64",
                                  param: "encoding_format")
            }
            options.encoding = format
        }
        return options
    }

    static func inputType(_ value: JSONValue?, param: String) throws(ServiceError) -> InputType? {
        guard let value, !value.isNull else { return nil }
        guard let type = value.stringValue.flatMap(InputType.init(rawValue:)) else {
            throw .badRequest("invalid_input_type", "\(param) must be query or document", param: param)
        }
        return type
    }

    /// The image bytes of base64 `text` (or a data URL). Its size is checked against `limits.maxImageBytes` before
    /// it is decoded, from the base64 symbols it holds (the decoder ignores everything else).
    static func base64Image(_ text: String, param: String, limits: ServiceLimits) throws(ServiceError) -> Data {
        let payload = text.hasPrefix("data:") ? text[(text.firstIndex(of: ",").map { text.index(after: $0) } ?? text.endIndex)...] : text[...]
        let size = decodedSize(payload)
        guard size <= limits.maxImageBytes else {
            throw ServiceError(.payloadTooLarge, "image_too_large", "\(param) is \(size) bytes; the limit is \(limits.maxImageBytes)",
                               param: param)
        }
        guard let data = Data(base64Encoded: String(payload), options: .ignoreUnknownCharacters), !data.isEmpty else {
            throw .badRequest("invalid_base64", "\(param) is not valid base64", param: param)
        }
        return data
    }

    /// Bytes that base64 `payload` decodes to: 3 per 4 symbols of the base64 alphabet, padding and other characters
    /// not counted.
    static func decodedSize(_ payload: Substring) -> Int {
        var symbols = 0
        for byte in payload.utf8 {
            switch byte {
            case UInt8(ascii: "A")...UInt8(ascii: "Z"), UInt8(ascii: "a")...UInt8(ascii: "z"), UInt8(ascii: "0")...UInt8(ascii: "9"),
                UInt8(ascii: "+"), UInt8(ascii: "/"):
                symbols += 1
            default: break
            }
        }
        return symbols * 3 / 4
    }

    static func readPath(_ path: String, param: String, context: ToolContext) throws(ServiceError) -> Data {
        guard context.allowPaths else {
            throw .badRequest("path_not_allowed", "\(param) is only accepted over stdio; send the image bytes instead",
                              param: param)
        }
        do {
            return try Data(contentsOf: URL(fileURLWithPath: (path as NSString).expandingTildeInPath))
        } catch {
            throw .badRequest("unreadable_path", "cannot read \(path): \(error.localizedDescription)", param: param)
        }
    }

    /// The common embedding output of every surface except the OpenAI endpoint.
    public static func embeddingOutput(_ result: EmbeddingResult, options: OutputOptions, model: String,
                                       modelID: ModelID = .embeddingGemma2) -> JSONValue
    {
        let truncated = Set(result.truncatedIndices)
        let data: [JSONValue] = result.vectors.enumerated().map { index, vector in
            let output = Vectors.output(vector, to: options.dimensions, modelID: modelID)
            let embedding: JSONValue = options.encoding == .base64 ? .string(Vectors.base64(output)) : .floats(output)
            var item: JSONObject = ["index": .int(index), "tokens": .int(result.tokenCounts[index]), "embedding": embedding]
            if truncated.contains(index) { item["truncated"] = true }
            return .object(item)
        }
        return [
            "model": .string(model), "dimensions": .int(options.dimensions),
            "encoding_format": .string(options.encoding.rawValue), "data": .array(data),
            "usage": ["prompt_tokens": .int(result.totalTokens), "total_tokens": .int(result.totalTokens)],
        ]
    }
}
