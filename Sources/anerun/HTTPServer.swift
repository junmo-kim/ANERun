// The only listening socket: HTTP/1.1 over Network.framework with keep-alive. Parsing and routing live
// in EmbedCore (HTTPParser, Router); this file moves bytes.
import EmbedCore
import Foundation
import Network

protocol HTTPRouting: Sendable {
    func handle(_ request: consuming HTTPRequest) async -> HTTPResponse
}

extension Router: HTTPRouting {}

/// An opt-in connection path for routers that need to track response delivery.
protocol HTTPConnectionRouting: HTTPRouting {
    func accept(_ transport: NetworkTransport, limiter: ConnectionLimiter, budget: BodyBudget,
                limits: HTTPConnectionLimits) async
}

final class HTTPServer: @unchecked Sendable {
    let host: String
    let port: Int
    let maxBody: Int
    let router: any HTTPRouting
    let connections: ConnectionLimiter
    /// Large request bodies buffered at once, across connections: one of --max-body.
    let bodies: BodyBudget
    /// Waiting for a request and receiving its head (not extended by progress).
    static let headTimeout: Duration = .seconds(60)
    /// Longest pause while a body arrives (restarted by every received chunk).
    static let bodyIdleTimeout: Duration = .seconds(60)
    /// Sending a response: the same 60 s, plus one second per 64 KiB (about 0.5 Mbit/s, slower than any link a client
    /// of this server is likely to read over). The largest response, 2048 vectors of 768 floats as JSON (about
    /// 20 MB), gets about 6 minutes; a client that stops reading holds its connection slot no longer than that.
    static let sendTimeout: Duration = .seconds(60)
    static let sendMinimumRate = 64 << 10
    private var listener: NWListener?
    private let stopped = AsyncStream<Int32>.makeStream()

    init(host: String, port: Int, maxBody: Int, maxConnections: Int, router: any HTTPRouting) {
        self.host = host
        self.port = port
        self.maxBody = maxBody
        self.router = router
        connections = ConnectionLimiter(limit: maxConnections)
        bodies = BodyBudget(capacity: maxBody)
    }

    func start(onReady: (@Sendable () -> Void)? = nil) throws(ServiceError) {
        let parameters = NWParameters.tcp
        guard let endpointPort = NWEndpoint.Port(rawValue: UInt16(port)) else {
            throw .badRequest("invalid_option", "invalid port \(port)", param: "port")
        }
        parameters.requiredLocalEndpoint = .hostPort(host: NWEndpoint.Host(host), port: endpointPort)
        parameters.allowLocalEndpointReuse = true
        let listener: NWListener
        do { listener = try NWListener(using: parameters) } catch {
            throw ServiceError(.unavailable, "listen_failed", "cannot listen on \(host):\(port): \(error)")
        }
        listener.newConnectionHandler = { [self] connection in
            connection.start(queue: .global())
            Task { await self.serve(connection) }
        }
        listener.stateUpdateHandler = { [self] update in
            switch update {
            case .ready:
                Log.info("listening on http://\(host):\(port)")
                onReady?()
            case .failed(let error):
                Log.info("listener failed: \(error)")
                stopped.continuation.yield(ExitCode.unavailable.rawValue)
            default: break
            }
        }
        listener.start(queue: .global())
        self.listener = listener
    }

    /// Returns when the listener fails.
    func wait() async -> ExitCode {
        for await code in stopped.stream { return ExitCode(rawValue: code) ?? .runtimeError }
        return .runtimeError
    }

    func stop() { listener?.cancel() }

    // MARK: Connections

    private func serve(_ connection: NWConnection) async {
        let limits = HTTPConnectionLimits(maxBody: maxBody, headTimeout: Self.headTimeout, bodyIdleTimeout: Self.bodyIdleTimeout,
                                          sendTimeout: Self.sendTimeout, sendMinimumRate: Self.sendMinimumRate)
        let transport = NetworkTransport(connection)
        if let connectionRouter = router as? any HTTPConnectionRouting {
            await connectionRouter.accept(transport, limiter: connections, budget: bodies, limits: limits)
            return
        }
        await HTTPConnection.accept(transport, limiter: connections, budget: bodies,
                                    limits: limits) { [router] request in
            await router.handle(consume request)  // without `consume` the closure would keep the body to the end
        }
    }
}

/// An NWConnection as an HTTPTransport.
struct NetworkTransport: HTTPTransport {
    let connection: NWConnection

    init(_ connection: NWConnection) {
        self.connection = connection
    }

    func receive() async -> Data? {
        while true {
            let chunk: Data? = await withCheckedContinuation { continuation in
                connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { data, _, isComplete, error in
                    if let data, !data.isEmpty {
                        continuation.resume(returning: data)
                    } else if isComplete || error != nil {
                        continuation.resume(returning: nil)
                    } else {
                        continuation.resume(returning: Data())
                    }
                }
            }
            guard let chunk else { return nil }
            if !chunk.isEmpty { return chunk }
        }
    }

    func send(_ data: Data) async -> Bool {
        await withCheckedContinuation { continuation in
            connection.send(content: data, completion: .contentProcessed { error in continuation.resume(returning: error == nil) })
        }
    }

    func cancel() {
        connection.cancel()
    }
}
