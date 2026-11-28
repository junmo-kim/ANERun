#if ANERUN_REQUIRE_COREAI && !canImport(CoreAI)
#error("The release binary requires the CoreAI SDK")
#endif

#if canImport(CoreAI)
import CoreAI
#endif
import DecisionRuntime
import EmbedCore
import Foundation

struct DecideCommand: Command {
    let spec = ArgumentSpec(values: ["format"], flags: ["stdin"])

    func run(_ arguments: ParsedArguments, config: Config, format: OutputFormat) async throws(ServiceError) -> ExitCode {
        throw .badRequest("unsupported_model", "decide requires a decision model", param: "model")
    }
}

enum DecisionCommands {
    static func run(_ command: String, arguments: ParsedArguments, config: Config, model: ModelID,
                    format: OutputFormat) async throws(ServiceError) -> ExitCode {
        if command.hasPrefix("models "), arguments.positionals.count != 1 {
            throw .badRequest("invalid_arguments", "models takes one action")
        }
        if command == "models path" {
            Output.data(config.modelDirectory.path)
            return .success
        }
        guard !config.useFakeRuntime else {
            throw .badRequest("fake_runtime", "decision models require CoreAI", param: "runtime")
        }
        #if canImport(CoreAI)
        if #available(macOS 27.0, *) {
            switch command {
            case "models install":
                let store = ModelStore(dataDirectory: config.dataDirectory, model: model)
                let target: URL
                if let from = arguments.value("from") {
                    guard !from.isEmpty, arguments.value("repo") == nil, arguments.value("revision") == nil else {
                        throw .badRequest("invalid_model_source", "supply --from DIR without --repo or --revision")
                    }
                    let source = URL(fileURLWithPath: (from as NSString).expandingTildeInPath).standardizedFileURL.resolvingSymlinksInPath()
                    target = try await DecisionInstall.install(source, model: model, store: store, architecture: AIModel.deviceArchitectureName, prepare: prepareForInstall)
                } else {
                    let architecture = AIModel.deviceArchitectureName
                    let source = try DecisionDownload.source(model: model, architecture: architecture,
                                                             repo: arguments.value("repo"), revision: arguments.value("revision"))
                    target = try await DecisionDownload.install(source, model: model, architecture: architecture, store: store,
                                                                fetch: ModelDownloader(source: source).fetch, prepare: prepareForInstall)
                }
                Output.data(["model": .string(model.rawValue), "model_dir": .string(target.path), "prepared": true], format: format)
            case "models verify", "models warm":
                try await prepare(config.modelDirectory)
                Output.data(["model_dir": .string(config.modelDirectory.path), "prepared": true], format: format)
            case "serve":
                if model == .decision(.valen4B) {
                    do { try DecisionEngine.validateManifest(at: config.modelDirectory) }
                    catch { throw ServiceError(.invalidInput, "invalid_bundle", "Valen manifest: \(error)") }
                }
                let timeout = config.int("idle-timeout")
                let engine = DecisionEngine(bundleURL: config.modelDirectory, idleTimeout: timeout > 0 ? .seconds(timeout) : nil)
                let server = HTTPServer(host: config.host, port: config.port, maxBody: config.maxBodyBytes,
                                        maxConnections: config.maxConnections,
                                        router: DecisionRouter(engine: engine, model: config.modelName))
                try server.start()
                return await server.wait()
            case "decide":
                guard arguments.has("stdin"), arguments.positionals.isEmpty else {
                    throw .badRequest("missing_input", "use decide --stdin with one JSON request")
                }
                let router = DecisionRouter(engine: DecisionEngine(bundleURL: config.modelDirectory), model: config.modelName)
                let data = try await router.decide(readStandardInput())
                if format == .json {
                    Output.write(data + Data([10]), to: .standardOutput)
                } else {
                    do { Output.data(try JSONValue.parse(data), format: format) }
                    catch { throw ServiceError(.internal, "invalid_response", error.description) }
                }
            default:
                throw model.unsupportedEmbedding
            }
            return .success
        }
        #endif
        throw ServiceError(.unavailable, "coreai_unavailable", "decision models require macOS 27 and a build with CoreAI")
    }

    #if canImport(CoreAI)
    @available(macOS 27.0, *)
    static func prepareForInstall(_ directory: URL) async throws {
        do { try await DecisionEngine(bundleURL: directory).prepare() }
        catch let failure as CompiledFunctionLoadFailure { throw failure }
        catch { throw ServiceError(.unavailable, "prepare_failed", "cannot prepare \(directory.path): \(error)") }
    }

    @available(macOS 27.0, *)
    static func prepare(_ directory: URL) async throws(ServiceError) {
        do { try await DecisionEngine(bundleURL: directory).prepare() }
        catch { throw ServiceError(.unavailable, "prepare_failed", "cannot prepare \(directory.path): \(error)") }
    }
    #endif
}

#if canImport(CoreAI)
@available(macOS 27.0, *)
struct DecisionRouter: HTTPRouting {
    let engine: DecisionEngine
    let model: String

    func decide(_ data: Data) async throws(ServiceError) -> Data {
        struct Request: Decodable {
            enum CodingKeys: String, CodingKey {
                case model, state, questions, image, truncate
            }

            let model: String?
            let state: String
            let questions: [DecisionQuestion]
            let image: Data?
            let truncate: Bool

            init(from decoder: Decoder) throws {
                let values = try decoder.container(keyedBy: CodingKeys.self)
                model = try values.decodeIfPresent(String.self, forKey: .model)
                state = try values.decode(String.self, forKey: .state)
                questions = try values.decode([DecisionQuestion].self, forKey: .questions)
                image = try values.decodeIfPresent(Data.self, forKey: .image)
                truncate = values.contains(.truncate) ? try values.decode(Bool.self, forKey: .truncate) : true
            }
        }
        let request: Request
        do { request = try JSONDecoder().decode(Request.self, from: data) }
        catch {
            throw .badRequest("invalid_decision_request",
                              "expected state, questions, optional base64 image and optional boolean truncate: \(error)")
        }
        if let requested = request.model, requested != model {
            throw .badRequest("model_mismatch", "requested \(requested), this server serves \(model)", param: "model")
        }
        do {
            let answer = try await engine.decide(.init(state: request.state, questions: request.questions,
                                                       image: request.image, truncate: request.truncate))
            return try JSONEncoder().encode(answer.reportingModel(model))
        } catch let error as DecisionError where error.isInputError {
            throw ServiceError(.invalidInput, "invalid_decision_input", error.description)
        } catch {
            throw ServiceError(.unavailable, "decision_failed", "\(error)")
        }
    }

    func handle(_ request: consuming HTTPRequest) async -> HTTPResponse {
        switch (request.method, request.path) {
        case ("GET", "/health"), ("HEAD", "/health"):
            return .json(200, ["status": "ok", "loaded": .bool(await engine.isLoaded)])
        case ("GET", "/v1/models"), ("GET", "/models"):
            return .json(200, ["object": "list", "data": [["object": "model", "id": .string(model)]]])
        case ("POST", "/v1/decisions"):
            do { return .json(200, raw: try await decide(request.takeBody())) }
            catch {
                if error.httpStatus >= 500 { Log.info(error.description) }
                return .openAIError(error.publicFacing)
            }
        case (_, "/health"), (_, "/v1/models"), (_, "/models"), (_, "/v1/decisions"):
            return .json(405, ServiceError.badRequest("method_not_allowed", "unsupported method on \(request.path)").openAIBody)
        default:
            return .openAIError(ServiceError(.notFound, "not_found", "no decision route for \(request.method) \(request.path)"))
        }
    }
}
#endif
