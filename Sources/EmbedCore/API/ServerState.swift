// Model lifecycle shared by the HTTP server and the MCP loop. Loading (and the first ANE compile) can
// take minutes, so the listener starts first and reports `loading` until the service is ready.
import Foundation

public final class ServerState: @unchecked Sendable {
    public enum Status: Sendable {
        case loading
        case ready(EmbeddingService)
        case failed(ServiceError)
    }

    private let lock = NSLock()
    private var current: Status = .loading
    private var waiters: [CheckedContinuation<Void, Never>] = []

    public init(_ status: Status = .loading) {
        current = status
    }

    public var status: Status { lock.withLock { current } }

    public func set(_ status: Status) {
        let resumed = lock.withLock { () -> [CheckedContinuation<Void, Never>] in
            current = status
            if case .loading = status { return [] }
            defer { waiters = [] }
            return waiters
        }
        resumed.forEach { $0.resume() }
    }

    /// The service now, or the error to report (loading or failed).
    public func service() throws(ServiceError) -> EmbeddingService {
        switch status {
        case .ready(let service): return service
        case .loading:
            throw ServiceError(.loading, "model_loading", "the model is loading (the first start compiles it for the Neural Engine)",
                               hint: "retry when GET /health returns 200")
        case .failed(let error): throw error
        }
    }

    /// Waits until loading has finished, then returns the service or its failure.
    public func waitForService() async throws(ServiceError) -> EmbeddingService {
        while true {
            if case .loading = status {
                await withCheckedContinuation { continuation in
                    let stillLoading = lock.withLock { () -> Bool in
                        if case .loading = current {
                            waiters.append(continuation)
                            return true
                        }
                        return false
                    }
                    if !stillLoading { continuation.resume() }
                }
                continue
            }
            return try service()
        }
    }
}
