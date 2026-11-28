// One HTTP/1.1 connection: frames requests from the byte stream (keep-alive, pipelining), answers each through a
// handler and enforces the connection's time limits. Independent of the socket layer: `anerun serve` passes a
// transport over its Network.framework connection, the tests an in-memory one.
import Foundation

/// The byte stream of one connection.
public protocol HTTPTransport: Sendable {
    /// The next bytes from the peer; nil once the peer closed or the connection was cancelled.
    func receive() async -> Data?
    /// Sends `data`; false when it could not be sent.
    func send(_ data: Data) async -> Bool
    /// Closes the connection; a pending `receive` returns nil.
    func cancel()
}

public struct HTTPConnectionLimits: Sendable {
    public var maxBody: Int
    /// Waiting for the next request and receiving its head: one deadline that progress does not extend, so a head
    /// trickled byte by byte (or an idle kept-alive connection) cannot hold the connection longer.
    public var headTimeout: Duration
    /// While a body arrives: the longest pause between received bytes. Every chunk restarts it, so a slow upload
    /// that keeps arriving is not cut, and a stalled one is.
    public var bodyIdleTimeout: Duration
    /// Sending a response: `sendTimeout` plus one second per `sendMinimumRate` bytes of it, from the start of the
    /// send. The transport reports only when everything was sent, not progress, so this is one deadline that grows
    /// with the response rather than a limit on pauses. A client that stops reading is cut when it passes, which
    /// releases its connection slot.
    public var sendTimeout: Duration
    public var sendMinimumRate: Int

    public init(maxBody: Int, headTimeout: Duration = .seconds(60), bodyIdleTimeout: Duration = .seconds(60),
                sendTimeout: Duration = .seconds(60), sendMinimumRate: Int = 64 << 10)
    {
        self.maxBody = maxBody
        self.headTimeout = headTimeout
        self.bodyIdleTimeout = bodyIdleTimeout
        self.sendTimeout = sendTimeout
        self.sendMinimumRate = max(1, sendMinimumRate)
    }

    /// The deadline for sending `bytes`.
    public func sendDeadline(for bytes: Int) -> Duration {
        sendTimeout + .milliseconds(bytes / sendMinimumRate * 1000 + bytes % sendMinimumRate * 1000 / sendMinimumRate)
    }
}

/// Request framing over one connection's bytes: a head, then a Content-Length body; pipelined requests stay buffered.
public struct HTTPRequestReader: Sendable {
    public let maxBody: Int
    /// Received bytes that are not part of a body: the head being received, then whatever follows the body.
    private var buffer = Data()
    private var head: HTTPHead?
    /// The body of `head`: one allocation of its Content-Length that the received bytes are copied into.
    private var body: BodyBuffer?
    /// Bytes of `buffer` already searched for the end of the head (minus 3, for a terminator split across reads).
    private var scanned = 0
    /// The size of `buffer` at the last search that did not find the end of the head.
    private var searchedLength = 0
    /// Total bytes searched for heads (the cost of slow, byte-by-byte heads stays linear).
    private(set) var searchedBytes = 0

    public init(maxBody: Int) {
        self.maxBody = maxBody
    }

    /// Bytes received and not yet returned as part of a request, outside the body being filled.
    public var bufferedBytes: Int { buffer.count }

    /// The head of the next request has arrived and its body has not (yet) completely.
    public var isReadingBody: Bool { head != nil }

    public mutating func append(_ data: Data) {
        guard let body, !body.isFull else {
            buffer.append(data)
            return
        }
        let taken = body.fill(from: data)
        if taken < data.count { buffer.append(data[(data.startIndex + taken)...]) }
    }

    /// Parses the head once it has arrived. Returns the Content-Length of a body that is not admitted yet: the caller
    /// decides when its bytes may be read (`admitBody`) and reads none meanwhile.
    public mutating func bodyAwaitingAdmission() throws(HTTPParseError) -> Int? {
        try parseHead()
        guard let head, head.contentLength > 0, body == nil else { return nil }
        return head.contentLength
    }

    /// Admits the pending body: its buffer is allocated now and holds `reservation` until its bytes are freed.
    public mutating func admitBody(_ reservation: BodyReservation?) {
        guard let head, head.contentLength > 0, body == nil else { return }
        startBody(head, reservation: reservation)
    }

    /// The next complete request; nil while more bytes are needed. A body not admitted yet is admitted without a
    /// reservation.
    public mutating func next() throws(HTTPParseError) -> HTTPRequest? {
        try parseHead()
        guard let current = head else { return nil }
        var request = current.request
        if current.contentLength > 0 {
            if body == nil { startBody(current, reservation: nil) }
            guard let body, body.isFull else { return nil }
            request.body = body.handOver()
            self.body = nil
        } else {
            buffer = buffer.subdata(in: (buffer.startIndex + current.headLength)..<buffer.endIndex)
        }
        head = nil
        return request
    }

    private mutating func parseHead() throws(HTTPParseError) {
        guard head == nil, buffer.count > searchedLength else { return }  // nothing new since the last search
        searchedBytes += buffer.count - scanned
        guard let parsed = try HTTPParser.parseHead(buffer, maxBody: maxBody, scanFrom: scanned) else {
            scanned = max(buffer.count - 3, 0)
            searchedLength = buffer.count
            return
        }
        head = parsed
        scanned = 0
        searchedLength = 0
    }

    /// Allocates the body and moves the body bytes that arrived with the head into it; what follows them stays.
    private mutating func startBody(_ current: HTTPHead, reservation: BodyReservation?) {
        let body = BodyBuffer(count: current.contentLength, reservation: reservation)
        let from = buffer.startIndex + current.headLength
        let taken = body.fill(from: buffer[from..<buffer.endIndex])
        buffer = buffer.subdata(in: (from + taken)..<buffer.endIndex)
        self.body = body
    }
}

/// One request body, allocated at its Content-Length (no regrowing while it arrives). The bytes become a Data that
/// frees them when the last reference goes; if the body is never completed, they are freed with the buffer. Either
/// way the body's budget reservation is released when its bytes are freed, not before.
final class BodyBuffer: @unchecked Sendable {
    let count: Int
    private(set) var filled = 0
    private let bytes: UnsafeMutableRawPointer
    private let reservation: BodyReservation?
    private var handedOver = false

    init(count: Int, reservation: BodyReservation?) {
        self.count = count
        self.reservation = reservation
        bytes = .allocate(byteCount: count, alignment: 16)
    }

    var isFull: Bool { filled == count }

    /// Copies as much of `data` as still fits; returns how many bytes that was.
    func fill(from data: Data) -> Int {
        let taken = min(count - filled, data.count)
        data.copyBytes(to: (bytes + filled).assumingMemoryBound(to: UInt8.self), count: taken)
        filled += taken
        return taken
    }

    /// The full body as Data, which owns the bytes from now on.
    func handOver() -> Data {
        handedOver = true
        return Data(bytesNoCopy: bytes, count: count, deallocator: .custom { [reservation] pointer, _ in
            pointer.deallocate()
            reservation?.release()
        })
    }

    deinit {
        guard !handedOver else { return }
        bytes.deallocate()
        reservation?.release()
    }
}

/// Counts open connections against a limit.
public final class ConnectionLimiter: @unchecked Sendable {
    public let limit: Int
    private let lock = NSLock()
    private var count = 0

    public init(limit: Int) {
        self.limit = limit
    }

    public var active: Int { lock.withLock { count } }

    /// Takes a slot; false when all `limit` slots are taken.
    public func tryAcquire() -> Bool {
        lock.withLock {
            guard count < limit else { return false }
            count += 1
            return true
        }
    }

    public func release() {
        lock.withLock { count -= 1 }
    }
}

public enum HTTPConnection {
    /// Serves `transport` while fewer than `limiter.limit` connections are open; beyond that answers
    /// `503 too_many_connections` (Retry-After: 1) and closes, so open connections and their buffers stay bounded.
    /// Bodies over `budget.unreservedLimit` share `budget` with every other connection's.
    public static func accept(_ transport: some HTTPTransport, limiter: ConnectionLimiter, budget: BodyBudget? = nil,
                              limits: HTTPConnectionLimits,
                              scheduler: any DeadlineScheduler = DispatchDeadlineScheduler(),
                              handler: @Sendable (consuming HTTPRequest) async -> HTTPResponse) async
    {
        guard limiter.tryAcquire() else {
            let timer = ConnectionTimer(transport, scheduler: scheduler)
            _ = await send(retryLater(ServiceError(
                .unavailable, "too_many_connections", "the server has \(limiter.limit) open connections, its limit; retry shortly")),
                on: transport, timer: timer, limits: limits)
            transport.cancel()
            return
        }
        defer { limiter.release() }
        await serve(transport, limits: limits, budget: budget, scheduler: scheduler, handler: handler)
    }

    /// Serves requests on `transport` until the peer closes, a request asks to close, or a limit is hit. A body over
    /// `budget.unreservedLimit` is read only once `budget` has room for it (no budget: every body is read at once);
    /// after waiting `limits.bodyIdleTimeout` for that, it is answered `503 body_budget_busy` (Retry-After: 1).
    /// Every response is sent within `limits.sendDeadline` of its size, or the connection is cancelled.
    /// `scheduler` runs the deadlines; tests pass a clock they advance by hand. `handler` owns the request: a closure
    /// that passes it on should `consume` it, or the closure keeps the body alive until it returns.
    public static func serve(_ transport: some HTTPTransport, limits: HTTPConnectionLimits, budget: BodyBudget? = nil,
                             scheduler: any DeadlineScheduler = DispatchDeadlineScheduler(),
                             handler: @Sendable (consuming HTTPRequest) async -> HTTPResponse) async
    {
        var reader = HTTPRequestReader(maxBody: limits.maxBody)
        let timer = ConnectionTimer(transport, scheduler: scheduler)
        defer {
            timer.stop()
            transport.cancel()
        }
        while true {
            timer.start(limits.headTimeout)
            let request: HTTPRequest
            // Also released once the handler returns, in case the body outlived it.
            var reservation: BodyReservation?
            do {
                while true {
                    if let length = try reader.bodyAwaitingAdmission() {
                        // A large body waits for room in the budget before any of it is read. The socket is not read
                        // meanwhile, so TCP flow control holds the client back. The wait ends when earlier bodies are
                        // freed, which is soon: a body is freed once its request is parsed. One that does not end
                        // within the body idle timeout means a body is held for long (a stalled upload, say): the
                        // client is told to retry rather than kept waiting, so waiting connections cannot pile up.
                        timer.stop()
                        if let budget {
                            do {
                                reservation = try await budget.reserve(length, waitingAtMost: limits.bodyIdleTimeout,
                                                                         scheduler: scheduler)
                            } catch is BodyBudget.WaitTimedOut {
                                _ = await send(retryLater(ServiceError(
                                    .unavailable, "body_budget_busy",
                                    "the server is receiving other large request bodies; retry shortly")),
                                    on: transport, timer: timer, limits: limits)
                                return
                            } catch {
                                return
                            }
                        }
                        reader.admitBody(reservation)
                    }
                    if let next = try reader.next() {
                        request = next
                        break
                    }
                    // Head: the deadline set above. Body: restarted on every chunk.
                    if reader.isReadingBody { timer.start(limits.bodyIdleTimeout) }
                    guard let chunk = await transport.receive() else { return }
                    reader.append(chunk)
                }
            } catch {
                let failure = ServiceError(error.status == 413 ? .payloadTooLarge : .badRequest, "bad_http", error.message)
                _ = await send(HTTPResponse.openAIError(failure).with(status: error.status).serialized(keepAlive: false),
                               on: transport, timer: timer, limits: limits)
                return
            }
            timer.stop()
            let keepAlive = request.keepAlive, includeBody = request.method != "HEAD"
            // The handler owns the request from here, so it can free the body before it is done.
            let response = await handler(consume request)
            reservation?.release()
            let bytes = response.serialized(keepAlive: keepAlive, includeBody: includeBody)
            guard await send(bytes, on: transport, timer: timer, limits: limits), keepAlive else { return }
        }
    }

    /// Sends `bytes` within `limits.sendDeadline(for:)`; past it `timer` cancels the transport, which ends the send.
    static func send(_ bytes: Data, on transport: some HTTPTransport, timer: ConnectionTimer,
                     limits: HTTPConnectionLimits) async -> Bool
    {
        timer.start(limits.sendDeadline(for: bytes.count))
        defer { timer.stop() }
        return await transport.send(bytes)
    }

    /// `error` with `Retry-After: 1`, for a connection that is closed after it.
    static func retryLater(_ error: ServiceError) -> Data {
        var response = HTTPResponse.openAIError(error)
        response.headers.append(("Retry-After", "1"))
        return response.serialized(keepAlive: false)
    }
}

/// Cancels a connection when a deadline passes; `start` replaces the running deadline.
///
/// A deadline is a scheduled piece of work, not a task: the timer restarts on every received chunk and every send, and
/// a Swift task per restart is costly and aborted the process in `swift_task_dealloc` on macOS 26. Each `start` and
/// `stop` begins a new generation; a deadline whose work runs after it was replaced (its cancellation came too late)
/// finds its generation stale and leaves the connection alone.
final class ConnectionTimer: @unchecked Sendable {
    private let transport: any HTTPTransport
    private let scheduler: any DeadlineScheduler
    private let lock = NSLock()
    private var generation = 0
    private var deadline: (any ScheduledDeadline)?

    init(_ transport: any HTTPTransport, scheduler: any DeadlineScheduler) {
        self.transport = transport
        self.scheduler = scheduler
    }

    func start(_ timeout: Duration) {
        let current = lock.withLock { () -> Int in
            generation += 1
            return generation
        }
        // Scheduled outside the lock: the scheduler takes its own lock, and the work takes this one.
        let next = scheduler.schedule(after: timeout) { [weak self] in self?.expire(current) }
        let superseded = lock.withLock { () -> (any ScheduledDeadline)? in
            guard generation == current else { return next }  // a later start or stop came first
            defer { deadline = next }
            return deadline
        }
        superseded?.cancel()
    }

    func stop() {
        let superseded = lock.withLock { () -> (any ScheduledDeadline)? in
            generation += 1
            defer { deadline = nil }
            return deadline
        }
        superseded?.cancel()
    }

    private func expire(_ generation: Int) {
        guard lock.withLock({ self.generation == generation }) else { return }
        transport.cancel()
    }
}
