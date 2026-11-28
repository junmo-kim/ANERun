// HTTP routes of `anerun serve`, independent of the socket layer:
//
//   POST  /v1/embeddings (/embeddings)   OpenAI-compatible embeddings
//   GET   /v1/models (/models)           the model id and its limits
//   GET   /health                        200 ready ("ok", or "degraded" when loaded functions are not served
//                                        or the self-test was skipped), 503 loading or failed
//   POST  /tokenize, /detokenize         llama-server's tokenizer API (text <-> the model's token ids)
//   POST  /mcp                           MCP JSON-RPC (one message per request)
//   QUERY /{tool}?<options>              one-shot tool call (RFC 10008 QUERY, as in macvis): the body is the
//                                        raw input (UTF-8 text for embed_text, image bytes for embed_image)
//                                        or, with Content-Type application/json, the tool arguments
import DecisionProfiles
import Foundation

public struct Router: Sendable {
    public let state: ServerState
    public let modelName: String
    public let modelID: ModelID
    private let configuredMaxTokens: Int
    /// Server log: the details of internal errors, which responses do not carry.
    public let log: @Sendable (String) -> Void

    public init(state: ServerState, modelName: String, modelID: ModelID = .embeddingGemma2,
                configuredMaxTokens: Int = ServiceLimits().maxTokens,
                log: @escaping @Sendable (String) -> Void = { _ in })
    {
        self.state = state
        self.modelName = modelName
        self.modelID = modelID
        self.configuredMaxTokens = configuredMaxTokens
        self.log = log
    }

    /// The error as a client sees it (generic for 500/503), with an internal one's detail logged.
    func published(_ error: ServiceError, _ request: HTTPRequest) -> ServiceError {
        if error.kind == .internal { log("\(request.method) \(request.path): \(error.description)") }
        return error.publicFacing
    }

    /// Takes the request over, so a route can free its body as soon as the body is parsed.
    public func handle(_ request: consuming HTTPRequest) async -> HTTPResponse {
        switch (request.method, request.path) {
        case ("GET", "/health"), ("HEAD", "/health"):
            return health()
        case ("GET", "/v1/models"), ("GET", "/models"):
            return models()
        case ("POST", "/v1/embeddings"), ("POST", "/embeddings"):
            return await embeddings(consume request)
        case ("POST", "/tokenize"):
            return await tokenize(consume request)
        case ("POST", "/detokenize"):
            return await detokenize(consume request)
        case ("POST", "/mcp"):
            let handler = MCPHandler(state: state, allowPaths: false, waitForModel: false, publicErrors: true,
                                     modelID: modelID, configuredMaxTokens: configuredMaxTokens, log: log)
            guard let response = await handler.handle(raw: request.body) else { return HTTPResponse(status: 202) }
            return .json(200, response)
        case ("QUERY", let path):
            return await query(request, tool: String(path.dropFirst()))
        case (_, "/health"), (_, "/v1/models"), (_, "/models"), (_, "/v1/embeddings"), (_, "/embeddings"), (_, "/mcp"),
             (_, "/tokenize"), (_, "/detokenize"):
            return .openAIError(ServiceError(.badRequest, "method_not_allowed", "\(request.method) is not allowed on \(request.path)"))
                .with(status: 405)
        default:
            return .openAIError(ServiceError(.notFound, "not_found", "no route for \(request.method) \(request.path)"))
        }
    }

    func health() -> HTTPResponse {
        switch state.status {
        case .ready(let service):
            let info = service.runtime.info
            guard info.degraded else { return .json(200, ["status": "ok"]) }
            var body: JSONObject = ["status": "degraded", "max_context_length": .int(service.maxTokens),
                                    "max_image_tokens": .int(service.maxImageTokens)]
            if !info.notServed.isEmpty { body["not_served"] = .array(info.notServed.map(JSONValue.string)) }
            if info.selfTestSkipped { body["selftest"] = "skipped" }
            return .json(200, .object(body))
        case .loading: return .json(503, ["status": "loading"])
        case .failed(let error):
            return .json(503, ["status": "error", "reason": .string(error.reason), "detail": .string(error.publicFacing.message)])
        }
    }

    /// `created` of the model object: 2026-11-15T00:00:00Z, the creation date in the manifest of the model package
    /// this build pins. OpenAI's model object always has it, and strictly typed clients expect it.
    static let modelCreated = 1_794_700_800

    func models() -> HTTPResponse {
        var model: JSONObject = [
            "id": .string(modelName), "object": "model", "created": .int(Self.modelCreated), "owned_by": "local",
            "dimensions": .array(Vectors.supportedDimensions(for: modelID).map { .int($0) }),
            "max_context_length": .int(configuredMaxTokens),
        ]
        if !modelID.supportsImages {
            model["max_image_tokens"] = 0
        } else {
            model["image_preprocessing"] = "hf-gemma4-280"
        }
        if case .ready(let service) = state.status {
            model["max_context_length"] = .int(service.maxTokens)
            if modelID == .embeddingGemma2 { model["max_image_tokens"] = .int(service.maxImageTokens) }
            model["max_inputs"] = .int(service.limits.maxInputs)
            model["max_images"] = .int(modelID == .qwen3Embedding ? 0 : service.limits.maxImages)
        }
        return .json(200, ["object": "list", "data": [.object(model)]])
    }

    func embeddings(_ request: consuming HTTPRequest) async -> HTTPResponse {
        do throws(ServiceError) {
            // The service first: its limits bound the parse (no image is decoded beyond --max-images), and a server
            // that is still loading answers 503 without parsing.
            let service = try state.service()
            // The raw body (up to --max-body) is freed once parsed: only the decoded images stay while they are
            // embedded.
            let parsed = try EmbeddingsRequest.parse(request.takeBody(), limits: service.limits, modelID: modelID)
            let timer = InputProcessingTimer()
            let result: EmbeddingResult
            switch parsed.input {
            case .texts(let texts): result = try await service.embedTexts(texts, inputType: parsed.inputType, truncate: parsed.truncate)
            case .images(let images): result = try await service.embedImages(images)
            }
            let rate = timer.tokensPerSecond(result.totalTokens)
            return .json(200, raw: EmbeddingsResponse.body(result, request: parsed, model: modelName,
                                                           modelID: modelID, promptTokensPerSecond: rate))
        } catch {
            return .openAIError(published(error, request))
        }
    }

    func tokenize(_ request: consuming HTTPRequest) async -> HTTPResponse {
        do throws(ServiceError) {
            let service = try state.service()
            // The raw body is freed once parsed, as for /v1/embeddings.
            let parsed = try TokenizeRequest.parse(request.takeBody())
            let tokenizer = service.tokenizer
            let body = try await service.withTokenizerLane(isShort: service.isShortTokenizerInput(bytes: parsed.content.utf8.count)) { () throws(ServiceError) -> Data in
                let ids = tokenizer.encode(parsed.content, addSpecial: parsed.addSpecial, parseSpecial: parsed.parseSpecial)
                return TokenizerResponse.tokens(ids, pieces: parsed.withPieces ? tokenizer : nil)
            }
            return .json(200, raw: body)
        } catch {
            return .openAIError(published(error, request))
        }
    }

    func detokenize(_ request: consuming HTTPRequest) async -> HTTPResponse {
        do throws(ServiceError) {
            let service = try state.service()
            let ids = try DetokenizeRequest.parse(request.takeBody())
            let tokenizer = service.tokenizer
            let content = try await service.withTokenizerLane(isShort: ids.count <= service.limits.shortTokens) { () throws(ServiceError) -> String in
                try TokenizerResponse.content(ids, tokenizer: tokenizer)
            }
            return .json(200, ["content": .string(content)])
        } catch {
            return .openAIError(published(error, request))
        }
    }

    func query(_ request: HTTPRequest, tool: String) async -> HTTPResponse {
        let format = request.queryValue("format").flatMap(OutputFormat.init(rawValue:)) ?? .yaml
        func reply(_ status: Int, _ value: JSONValue) -> HTTPResponse {
            HTTPResponse(status: status,
                         headers: [("Content-Type", format == .json ? "application/json" : "application/yaml; charset=utf-8")],
                         body: Data((value.render(as: format) + "\n").utf8))
        }
        guard Tools.names.contains(tool) else {
            let error = ServiceError(.notFound, "unknown_tool", "unknown tool \(tool); tools: \(Tools.names.joined(separator: ", "))")
            return reply(404, error.structured)
        }
        guard let contentType = request.header("Content-Type")?.lowercased() else {
            return reply(400, ServiceError.badRequest("missing_content_type", "QUERY needs a Content-Type header").structured)
        }
        do throws(ServiceError) {
            var arguments = JSONObject()
            if contentType.hasPrefix("application/json") {
                let parsed: JSONValue
                do {
                    parsed = try JSONValue.parse(request.body)
                } catch {
                    throw ServiceError.badRequest("invalid_json", error.description)
                }
                guard let object = parsed.objectValue else {
                    throw ServiceError.badRequest("invalid_json", "the JSON body must be an object of tool arguments")
                }
                arguments = object
            } else {
                switch tool {
                case "embed_text":
                    guard let text = String(validating: request.body, as: UTF8.self) else {
                        throw ServiceError.badRequest("invalid_utf8", "the embed_text body must be UTF-8 text")
                    }
                    arguments["text"] = .string(text)
                case "embed_image":
                    if !modelID.supportsImages {
                        _ = try state.service()
                        throw EmbeddingService.imagesUnsupported(modelID)
                    }
                    arguments["data"] = .string(request.body.base64EncodedString())
                default:
                    throw ServiceError.badRequest("json_required", "\(tool) takes its arguments as a JSON body")
                }
            }
            for (name, value) in request.query where name != "format" {
                if name == "truncate", let flag = Bool(value) {
                    arguments[name] = .bool(flag)
                } else {
                    arguments[name] = .string(value)
                }
            }
            let service = try state.service()
            let output = try await Tools.call(tool, arguments: arguments, context: ToolContext(service: service, allowPaths: false))
            return reply(200, output)
        } catch {
            return reply(error.httpStatus, published(error, request).structured)
        }
    }
}

extension HTTPResponse {
    func with(status: Int) -> HTTPResponse {
        var copy = self
        copy.status = status
        return copy
    }
}
