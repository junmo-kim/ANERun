import DecisionRuntime
import EmbedCore
import Foundation

struct PlaygroundCommand: Command {
    let spec = ArgumentSpec(values: ["format"], flags: ["no-open"])

    func run(_ arguments: ParsedArguments, config: Config, format: OutputFormat) async throws(ServiceError) -> ExitCode {
        guard arguments.positionals.isEmpty else {
            throw .badRequest("invalid_arguments", "playground does not take positional arguments")
        }
        guard config.host == "127.0.0.1" else {
            throw .badRequest("invalid_option", "playground requires --host 127.0.0.1", param: "host")
        }
        guard let model = config.modelID else {
            throw .badRequest("missing_model", "select a model with --model or --model-dir", param: "model")
        }

        let seconds = config.sources["idle-timeout"] == .default ? 300 : config.int("idle-timeout")
        let lifetime = PlaygroundLifetime(timeout: seconds > 0 ? .seconds(seconds) : nil)
        let router = PlaygroundRouter(config: config, model: model, lifetime: lifetime)
        let server = HTTPServer(host: "127.0.0.1", port: config.port, maxBody: config.maxBodyBytes,
                                maxConnections: config.maxConnections, router: router)
        let noOpen = arguments.has("no-open")
        let address = "http://127.0.0.1:\(config.port)/playground"
        try server.start {
            Task {
                guard await lifetime.listenerReady(), !noOpen else { return }
                let browser = Process()
                browser.executableURL = URL(fileURLWithPath: "/usr/bin/open")
                browser.arguments = [address]
                do { try browser.run() }
                catch { Log.info("cannot open browser: \(error). Open \(address) manually.") }
            }
        }
        defer { server.stop() }
        return await withTaskGroup(of: ExitCode.self) { group in
            group.addTask { await server.wait() }
            group.addTask { await lifetime.wait() }
            let code = await group.next() ?? .runtimeError
            group.cancelAll()
            return code
        }
    }
}

/// A response stays active from admission until the transport reports completion or failure.
private actor PlaygroundLifetime {
    private let timeout: Duration?
    private let stopped = AsyncStream<Void>.makeStream()
    private var ready = false
    private var stopping = false
    private var active = 0
    private var timer: Task<Void, Never>?
    private var generation: UInt64 = 0

    init(timeout: Duration?) { self.timeout = timeout }

    func listenerReady() -> Bool {
        guard !ready, !stopping else { return false }
        ready = true
        if active == 0 { armTimer() }
        return true
    }

    func beginResponse() -> Bool {
        guard !stopping else { return false }
        active += 1
        cancelTimer()
        return true
    }

    func finishResponse() {
        active -= 1
        guard active == 0 else { return }
        if stopping { stopped.continuation.finish() }
        else if ready { armTimer() }
    }

    func requestStop() {
        stopping = true
        cancelTimer()
        if active == 0 { stopped.continuation.finish() }
    }

    func wait() async -> ExitCode {
        for await _ in stopped.stream {}
        return .success
    }

    private func cancelTimer() {
        generation &+= 1
        timer?.cancel()
        timer = nil
    }

    private func armTimer() {
        guard let timeout else { return }
        let current = generation
        timer = Task { [weak self] in
            do { try await Task.sleep(for: timeout) } catch { return }
            await self?.expire(current)
        }
    }

    private func expire(_ current: UInt64) {
        guard current == generation, active == 0, !stopping else { return }
        Log.info("playground idle timeout")
        requestStop()
    }
}

private actor PlaygroundInference {
    let config: Config
    let model: ModelID
    private var loading: Task<Result<any HTTPRouting, ServiceError>, Never>?

    init(config: Config, model: ModelID) {
        self.config = config
        self.model = model
    }

    func router() async throws(ServiceError) -> any HTTPRouting {
        if let loading { return try await loading.value.get() }
        let task = Task<Result<any HTTPRouting, ServiceError>, Never> { [config, model] in
            if model.isDecision {
                guard !config.useFakeRuntime else {
                    return .failure(.badRequest("fake_runtime", "decision models require CoreAI", param: "runtime"))
                }
                #if canImport(CoreAI)
                if #available(macOS 27.0, *) {
                    // The engine serializes the first session load and owns model residency.
                    return .success(DecisionRouter(engine: DecisionEngine(bundleURL: config.modelDirectory),
                                                   model: config.modelName))
                }
                #endif
                return .failure(ServiceError(.unavailable, "coreai_unavailable",
                                             "decision models require macOS 27 and a build with CoreAI"))
            }
            let state = ServerState()
            await load(into: state, config: config)
            return .success(Router(state: state, modelName: config.modelName, modelID: model,
                                   configuredMaxTokens: config.maxTokens, log: { Log.info($0) }))
        }
        loading = task
        return try await task.value.get()
    }
}

private struct PlaygroundRouter: HTTPConnectionRouting {
    let lifetime: PlaygroundLifetime
    private let inference: PlaygroundInference
    private let modelName: String
    private let modelID: String
    private let decision: Bool
    private let images: Bool
    private let hosts: Set<String>

    init(config: Config, model: ModelID, lifetime: PlaygroundLifetime) {
        self.lifetime = lifetime
        inference = PlaygroundInference(config: config, model: model)
        modelName = config.modelName
        modelID = model.rawValue
        decision = model.isDecision
        images = model.supportsImages || (decision && ["valen-4b", "elephant-1"].contains(model.rawValue))
        var hosts: Set<String> = ["127.0.0.1:\(config.port)", "localhost:\(config.port)"]
        if config.port == 80 { hosts.formUnion(["127.0.0.1", "localhost"]) }
        self.hosts = hosts
    }

    func accept(_ transport: NetworkTransport, limiter: ConnectionLimiter, budget: BodyBudget,
                limits: HTTPConnectionLimits) async {
        let connection = PlaygroundConnection(transport: transport, router: self)
        await HTTPConnection.accept(connection, limiter: limiter, budget: budget, limits: limits) { request in
            await connection.handle(consume request)
        }
    }

    /// Reject ambiguous or nonlocal authorities before admitting any activity.
    func rejection(_ request: borrowing HTTPRequest) -> HTTPResponse? {
        let hosts = request.headers.filter { $0.name.lowercased() == "host" }
        guard hosts.count == 1, let host = hosts.first?.value.lowercased(), self.hosts.contains(host) else {
            return error(403, "invalid_host", "playground requires a loopback Host on its listening port")
        }
        guard request.method == "POST" else { return nil }
        let origins = request.headers.filter { $0.name.lowercased() == "origin" }
        let securitySite = request.header("Sec-Fetch-Site")
        // Non-browser clients may omit Origin. Browsers must be same-origin and
        // cross-origin preflights never receive CORS permission.
        guard (origins.isEmpty || (origins.count == 1 && origins[0].value == "http://\(host)")),
              securitySite != "cross-site", securitySite != "same-site" else {
            return error(403, "invalid_origin", "playground requires a same-origin request")
        }
        let contentTypes = request.headers.filter { $0.name.lowercased() == "content-type" }
        guard contentTypes.count == 1,
              contentTypes[0].value.split(separator: ";", maxSplits: 1, omittingEmptySubsequences: false)
                .first?.trimmingCharacters(in: .whitespaces).lowercased() == "application/json" else {
            return error(415, "invalid_content_type", "playground POST requests require application/json")
        }
        return nil
    }

    func handle(_ request: consuming HTTPRequest) async -> HTTPResponse {
        var response: HTTPResponse
        switch (request.method, request.path) {
        case ("GET", "/"), ("GET", "/playground"):
            response = HTTPResponse(status: 200, headers: [("Content-Type", "text/html; charset=utf-8")],
                                    body: Data(PlaygroundPage.html.utf8))
        case ("GET", "/playground/config"):
            response = .json(200, ["model": .string(modelName), "id": .string(modelID),
                                   "kind": decision ? "decision" : "embedding", "images": .bool(images)])
        case ("POST", "/playground/stop"):
            guard let payload = try? JSONValue.parse(request.takeBody()), payload.objectValue != nil else {
                return error(400, "invalid_json", "expected a JSON object")
            }
            await lifetime.requestStop()
            response = .json(200, ["status": "stopping"])
        case ("POST", "/v1/decisions") where decision, ("POST", "/v1/embeddings") where !decision:
            do {
                let router = try await inference.router()
                response = await router.handle(consume request)
            } catch {
                Log.info(error.description)
                response = .openAIError(error.publicFacing)
            }
        default:
            return error(404, "not_found", "no playground route for this request")
        }
        response.headers += [("Cache-Control", "no-store"), ("X-Frame-Options", "DENY"),
                             ("X-Content-Type-Options", "nosniff")]
        return response
    }

    private func error(_ status: Int, _ code: String, _ message: String) -> HTTPResponse {
        .json(status, ServiceError.badRequest(code, message).openAIBody)
    }
}

/// HTTPConnection sends each response before it reads the next request.
private actor PlaygroundConnection: HTTPTransport {
    private nonisolated let transport: NetworkTransport
    private let router: PlaygroundRouter
    private var pendingResponse = false

    init(transport: NetworkTransport, router: PlaygroundRouter) {
        self.transport = transport
        self.router = router
    }

    func handle(_ request: consuming HTTPRequest) async -> HTTPResponse {
        if let rejection = router.rejection(request) { return rejection }
        guard await router.lifetime.beginResponse() else {
            return .openAIError(ServiceError(.unavailable, "playground_stopping", "playground is stopping"))
        }
        pendingResponse = true
        return await router.handle(consume request)
    }

    nonisolated func receive() async -> Data? { await transport.receive() }
    nonisolated func cancel() { transport.cancel() }

    func send(_ data: Data) async -> Bool {
        let sent = await transport.send(data)
        if pendingResponse {
            pendingResponse = false
            await router.lifetime.finishResponse()
        }
        return sent
    }
}
