import Foundation
import Testing

@testable import EmbedCore

/// An in-memory connection: the test writes the client's bytes and reads what the server sent.
final class MemoryTransport: HTTPTransport, @unchecked Sendable {
    private let lock = NSLock()
    private var incoming: [Data] = []
    private var waiting: CheckedContinuation<Data?, Never>?
    private var closedByClient = false
    private var output = Data()
    private(set) var cancelled = false
    private var holdingSends = false
    /// Sends of a client that does not read: each waits, with its size, until the connection is cancelled.
    private var heldSends: [(bytes: Int, continuation: CheckedContinuation<Bool, Never>)] = []

    /// Bytes from the client.
    func write(_ data: Data) {
        let resumed = lock.withLock { () -> CheckedContinuation<Data?, Never>? in
            guard !cancelled else { return nil }
            if let waiting {
                self.waiting = nil
                return waiting
            }
            incoming.append(data)
            return nil
        }
        resumed?.resume(returning: data)
    }

    func write(_ text: String) { write(Data(text.utf8)) }

    /// The client half-closes: receive returns nil once the buffered bytes are read.
    func closeInput() {
        let resumed = lock.withLock { () -> CheckedContinuation<Data?, Never>? in
            closedByClient = true
            defer { waiting = nil }
            return waiting
        }
        resumed?.resume(returning: nil)
    }

    func receive() async -> Data? {
        await withCheckedContinuation { continuation in
            let result = lock.withLock { () -> Data?? in
                if !incoming.isEmpty { return .some(incoming.removeFirst()) }
                if cancelled || closedByClient { return .some(nil) }
                waiting = continuation
                return .none
            }
            if let result { continuation.resume(returning: result) }
        }
    }

    /// The client stops reading: from now on a send does not complete until the connection is cancelled, and then
    /// fails, as a send into a full TCP window does.
    func stopReading() {
        lock.withLock { holdingSends = true }
    }

    func send(_ data: Data) async -> Bool {
        await withCheckedContinuation { continuation in
            let result = lock.withLock { () -> Bool? in
                guard !cancelled else { return false }
                if holdingSends {
                    heldSends.append((data.count, continuation))
                    return nil
                }
                output.append(data)
                return true
            }
            if let result { continuation.resume(returning: result) }
        }
    }

    func cancel() {
        let (resumed, held) = lock.withLock { () -> (CheckedContinuation<Data?, Never>?, [CheckedContinuation<Bool, Never>]) in
            cancelled = true
            defer {
                waiting = nil
                heldSends = []
            }
            return (waiting, heldSends.map(\.continuation))
        }
        resumed?.resume(returning: nil)
        held.forEach { $0.resume(returning: false) }
    }

    /// The size of the send that waits for the client, once there is one.
    var heldSendBytes: Int? { lock.withLock { heldSends.first?.bytes } }

    var sent: Data { lock.withLock { output } }
    var isCancelled: Bool { lock.withLock { cancelled } }
    /// Chunks the client wrote that the server has not received yet.
    var pendingChunks: Int { lock.withLock { incoming.count } }
}

/// A clock the test advances by hand, so timeout tests do not depend on how fast the machine schedules work.
final class ManualClock: DeadlineScheduler, @unchecked Sendable {
    private let lock = NSLock()
    private var now: Duration = .zero
    private var nextID = 0
    private var pending: [Int: (deadline: Duration, work: @Sendable () -> Void)] = [:]
    private var armedCount = 0

    /// Deadlines scheduled so far (waiting, run or cancelled).
    var armed: Int { lock.withLock { armedCount } }
    /// Deadlines waiting to run now.
    var sleeping: Int { lock.withLock { pending.count } }

    func schedule(after delay: Duration, _ work: @escaping @Sendable () -> Void) -> any ScheduledDeadline {
        let id = lock.withLock { () -> Int in
            nextID += 1
            armedCount += 1
            pending[nextID] = (now + delay, work)
            return nextID
        }
        return Deadline { [weak self] in self?.cancel(id) }
    }

    private func cancel(_ id: Int) {
        _ = lock.withLock { pending.removeValue(forKey: id) }
    }

    /// Moves time forward and runs every deadline that has passed, earliest first, outside the lock.
    func advance(by duration: Duration) {
        let due = lock.withLock { () -> [@Sendable () -> Void] in
            now += duration
            let ids = pending.filter { $0.value.deadline <= now }
                .sorted { ($0.value.deadline, $0.key) < ($1.value.deadline, $1.key) }.map(\.key)
            return ids.compactMap { pending.removeValue(forKey: $0)?.work }
        }
        due.forEach { $0() }
    }

    /// Waits until more than `count` deadlines are armed. Polling only paces the wait; no result depends on its timing.
    func waitUntilArmed(moreThan count: Int) async throws {
        while armed <= count { try await Task.sleep(for: .milliseconds(1)) }
    }

    private struct Deadline: ScheduledDeadline {
        let cancelling: @Sendable () -> Void
        init(_ cancelling: @escaping @Sendable () -> Void) { self.cancelling = cancelling }
        func cancel() { cancelling() }
    }
}

/// `clock`, except that cancelling a deadline has no effect: the work runs anyway, as work that was already running
/// when its deadline was replaced does.
struct UncancellableDeadlines: DeadlineScheduler {
    let clock: ManualClock

    func schedule(after delay: Duration, _ work: @escaping @Sendable () -> Void) -> any ScheduledDeadline {
        _ = clock.schedule(after: delay, work)
        return Ignored()
    }

    private struct Ignored: ScheduledDeadline {
        func cancel() {}
    }
}

/// The result of `task`, failing instead of hanging if it has not finished within 10 s. Awaiting the task itself
/// could outlast the test's time limit: the limit cancels the test, and that does not reach the task.
@discardableResult
func finished<Success: Sendable, Failure: Error>(_ task: Task<Success, Failure>,
                                                  sourceLocation: SourceLocation = #_sourceLocation) async throws
    -> Result<Success, Failure>
{
    let done = Shared(false)
    Task {
        _ = await task.result
        done.value = true
    }
    for _ in 0..<100 where !done.value { await Task.yield() }  // most tasks are about to finish
    for _ in 0..<10_000 where !done.value { try await Task.sleep(for: .milliseconds(1)) }
    try #require(done.value, "the task did not finish", sourceLocation: sourceLocation)
    return await task.result
}

/// A client-side reading of the server's bytes: responses in order, with HEAD responses carrying no body.
struct ParsedResponse {
    var status: Int
    var headers: [String: String]
    var body: Data
}

func parseResponses(_ data: Data, methods: [String]) -> (responses: [ParsedResponse], rest: Data) {
    var rest = data
    var responses: [ParsedResponse] = []
    for method in methods {
        guard let end = rest.range(of: Data("\r\n\r\n".utf8)) else { break }
        let head = String(decoding: rest[rest.startIndex..<end.lowerBound], as: UTF8.self).components(separatedBy: "\r\n")
        let statusLine = head[0].split(separator: " ")
        guard statusLine.count >= 2, statusLine[0] == "HTTP/1.1", let status = Int(statusLine[1]) else { break }
        var headers: [String: String] = [:]
        for line in head.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            headers[line[..<colon].lowercased()] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }
        let length = method == "HEAD" ? 0 : Int(headers["content-length"] ?? "0") ?? 0
        guard rest.distance(from: end.upperBound, to: rest.endIndex) >= length else { break }
        let bodyEnd = rest.index(end.upperBound, offsetBy: length)
        responses.append(ParsedResponse(status: status, headers: headers, body: Data(rest[end.upperBound..<bodyEnd])))
        rest = Data(rest[bodyEnd...])
    }
    return (responses, rest)
}

@Suite("HTTP connection (in-memory transport)")
struct HTTPConnectionTests {
    static let router = Router(state: ServerState(.ready(TestSupport.service())), modelName: "embeddinggemma-2")

    /// Runs a connection until it ends; `client` drives the transport meanwhile.
    func connection(limits: HTTPConnectionLimits = HTTPConnectionLimits(maxBody: 1 << 20), clock: ManualClock? = nil,
                    _ client: @Sendable (MemoryTransport) async throws -> Void) async throws -> MemoryTransport
    {
        let transport = MemoryTransport()
        let scheduler: any DeadlineScheduler = clock ?? DispatchDeadlineScheduler()
        let server = Task {
            await HTTPConnection.serve(transport, limits: limits, scheduler: scheduler) { await Self.router.handle($0) }
        }
        try await client(transport)
        try await finished(server)
        return transport
    }

    @Test func keepAliveAndPipelinedRequests() async throws {
        let body = #"{"input":"hello"}"#
        let transport = try await connection { client in
            client.write("GET /health HTTP/1.1\r\nHost: x\r\n\r\nPOST /v1/embeddings HTTP/1.1\r\nContent-Length: \(body.utf8.count)\r\n\r\n")
            client.write(body + "GET /health HTTP/1.1\r\nConnection: close\r\n\r\n")
        }
        let (responses, rest) = parseResponses(transport.sent, methods: ["GET", "POST", "GET"])
        #expect(responses.map(\.status) == [200, 200, 200] && rest.isEmpty)
        #expect(responses[2].headers["connection"] == "close")
        #expect(try JSONValue.parse(responses[1].body)["data"]?.arrayValue?.count == 1)
    }

    @Test func aBadRequestIsAnsweredAndClosed() async throws {
        let transport = try await connection { client in
            client.write("GARBAGE\r\n\r\nGET /health HTTP/1.1\r\n\r\n")
        }
        let (responses, _) = parseResponses(transport.sent, methods: ["GET", "GET"])
        #expect(responses.map(\.status) == [400] && responses[0].headers["connection"] == "close")
        #expect(transport.isCancelled)
    }

    @Test func thePeerClosingEndsTheConnection() async throws {
        let transport = try await connection { client in
            client.write("GET /health HTTP/1.1\r\n")
            client.closeInput()
        }
        #expect(transport.sent.isEmpty && transport.isCancelled)
    }

    /// A HEAD response has the GET response's headers (Content-Length included) and no body, so the next
    /// response on the connection starts where the client expects it.
    @Test func headResponsesCarryNoBody() async throws {
        let transport = try await connection { client in
            client.write("HEAD /health HTTP/1.1\r\n\r\nHEAD /v1/embeddings HTTP/1.1\r\n\r\nHEAD /nope HTTP/1.1\r\n\r\n")
            client.write("GET /health HTTP/1.1\r\nConnection: close\r\n\r\n")
        }
        let (responses, rest) = parseResponses(transport.sent, methods: ["HEAD", "HEAD", "HEAD", "GET"])
        try #require(responses.map(\.status) == [200, 405, 404, 200])
        #expect(rest.isEmpty, "bytes after the last response: \(String(decoding: rest, as: UTF8.self))")
        #expect(responses[0].headers["content-length"] == "15" && responses[0].headers["content-type"] == "application/json")
        #expect(try JSONValue.parse(responses[3].body) == ["status": "ok"])
    }

    /// A CL.CL request whose second length would smuggle a request inside the body gets one 400 and a close.
    @Test func conflictingContentLengthsCloseTheConnection() async throws {
        let smuggled = "GET /health HTTP/1.1\r\nHost: x\r\n\r\n"
        let transport = try await connection { client in
            client.write("POST /v1/embeddings HTTP/1.1\r\nContent-Length: 0\r\nContent-Length: \(smuggled.utf8.count)\r\n\r\n"
                         + smuggled)
            client.closeInput()
        }
        let (responses, rest) = parseResponses(transport.sent, methods: ["POST", "GET"])
        #expect(responses.map(\.status) == [400] && rest.isEmpty)
        #expect(responses.first?.headers["connection"] == "close" && transport.isCancelled)
    }

    /// A head that arrives one byte at a time is searched for its end in linear, not quadratic, time.
    @Test func headSearchResumesWhereItStopped() throws {
        var reader = HTTPRequestReader(maxBody: 10)
        let head = "GET /health HTTP/1.1\r\nX-Pad: " + String(repeating: "a", count: 32_000) + "\r\n\r\n"
        var requests: [HTTPRequest] = []
        for byte in head.utf8 {
            reader.append(Data([byte]))
            if let request = try reader.next() { requests.append(request) }
        }
        #expect(requests.count == 1 && requests.first?.path == "/health")
        #expect(reader.searchedBytes < 5 * head.utf8.count, "searched \(reader.searchedBytes) bytes for a \(head.utf8.count)-byte head")
    }

    static let shortTimeouts = HTTPConnectionLimits(maxBody: 1 << 20, headTimeout: .seconds(1), bodyIdleTimeout: .seconds(1))

    /// A body that keeps arriving, slowly, is not cut by the timeout, however long it takes in total.
    /// These tests run on a ManualClock: the timeouts fire when the test moves time, not when a busy machine gets
    /// around to it. A connection that should end but does not fails the test after 10 s.
    @Test(.timeLimit(.minutes(1))) func aSlowButActiveUploadIsNotCut() async throws {
        let body = Array(#"{"input":"slow"}"#.utf8)
        let clock = ManualClock()
        let transport = try await connection(limits: Self.shortTimeouts, clock: clock) { client in
            client.write("POST /v1/embeddings HTTP/1.1\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n")
            try await clock.waitUntilArmed(moreThan: 1)  // the head deadline, then the body's idle timeout
            for (index, byte) in body.enumerated() {
                // 16 bytes 0.9 s apart: 14.4 s in total, far past the 1 s head deadline, never 1 s idle.
                clock.advance(by: .milliseconds(900))
                let armed = clock.armed
                client.write(Data([byte]))
                if index < body.count - 1 { try await clock.waitUntilArmed(moreThan: armed) }
            }
        }
        let (responses, _) = parseResponses(transport.sent, methods: ["POST"])
        #expect(responses.map(\.status) == [200])
    }

    @Test(.timeLimit(.minutes(1))) func aStalledBodyTimesOut() async throws {
        let clock = ManualClock()
        let transport = try await connection(limits: Self.shortTimeouts, clock: clock) { client in
            client.write("POST /v1/embeddings HTTP/1.1\r\nContent-Length: 100\r\n\r\n{\"input\"")
            try await clock.waitUntilArmed(moreThan: 1)  // the head deadline, then the body's idle timeout
            clock.advance(by: .milliseconds(999))
            #expect(!client.isCancelled)
            clock.advance(by: .milliseconds(1))
        }
        #expect(transport.isCancelled && transport.sent.isEmpty)
    }

    /// The head has one deadline that progress does not extend: a head trickled byte by byte, or a kept-alive
    /// connection that sends nothing, cannot hold the connection for longer.
    @Test(.timeLimit(.minutes(1))) func aTrickledHeadAndAnIdleConnectionTimeOut() async throws {
        let head = Array("GET /health HTTP/1.1\r\nX-Slow: abcdefghijklmnopqrstuvwxyz\r\n\r\n".utf8)
        let clock = ManualClock()
        let trickled = try await connection(limits: Self.shortTimeouts, clock: clock) { client in
            try await clock.waitUntilArmed(moreThan: 0)  // the head deadline
            for byte in head.prefix(9) {  // a byte every 0.1 s keeps arriving until the deadline passes
                client.write(Data([byte]))
                clock.advance(by: .milliseconds(100))
            }
            #expect(!client.isCancelled)
            clock.advance(by: .milliseconds(100))
            while !client.isCancelled { try await Task.sleep(for: .milliseconds(1)) }
            client.write(Data(head.dropFirst(9)))  // too late: the connection is closed
        }
        #expect(trickled.isCancelled && trickled.sent.isEmpty)

        let idleClock = ManualClock()
        let idle = try await connection(limits: Self.shortTimeouts, clock: idleClock) { client in
            client.write("GET /health HTTP/1.1\r\n\r\n")
            try await idleClock.waitUntilArmed(moreThan: 1)  // the first request's head deadline, then the next one's
            idleClock.advance(by: .seconds(1))
        }
        #expect(idle.isCancelled && parseResponses(idle.sent, methods: ["GET"]).responses.map(\.status) == [200])
    }

    /// A client that stops reading cannot hold its connection: the send is cut once its deadline passes, which grows
    /// with the size of the response, and the connection's slot is free again.
    @Test(.timeLimit(.minutes(1))) func aClientThatStopsReadingIsCutAndFreesItsSlot() async throws {
        let clock = ManualClock()
        let limiter = ConnectionLimiter(limit: 1)
        let limits = HTTPConnectionLimits(maxBody: 1 << 20, headTimeout: .seconds(1), bodyIdleTimeout: .seconds(1),
                                          sendTimeout: .seconds(1), sendMinimumRate: 100)
        let body = #"{"input":["a","b","c","d","e","f","g","h"]}"#
        let transport = MemoryTransport()
        transport.stopReading()
        transport.write("POST /v1/embeddings HTTP/1.1\r\nContent-Length: \(body.utf8.count)\r\n\r\n\(body)")
        let server = Task {
            await HTTPConnection.accept(transport, limiter: limiter, limits: limits, scheduler: clock) {
                await Self.router.handle($0)
            }
        }
        while transport.heldSendBytes == nil { try await Task.sleep(for: .milliseconds(1)) }
        let bytes = try #require(transport.heldSendBytes)
        let deadline = limits.sendDeadline(for: bytes)
        #expect(bytes > 10_000 && deadline == .seconds(1) + .milliseconds(bytes * 10))  // 100 bytes per second
        try await clock.waitUntilArmed(moreThan: 1)  // the head deadline, then the send deadline
        clock.advance(by: deadline - .milliseconds(1))
        #expect(!transport.isCancelled && limiter.active == 1)
        clock.advance(by: .milliseconds(1))
        try await Self.untilCancelled(transport)
        try await finished(server)
        #expect(limiter.active == 0)
    }

    /// The refusal of a connection beyond the limit is bounded the same way.
    @Test(.timeLimit(.minutes(1))) func aRefusedClientThatDoesNotReadIsCut() async throws {
        let clock = ManualClock()
        let limiter = ConnectionLimiter(limit: 0)
        let limits = HTTPConnectionLimits(maxBody: 1 << 20, sendTimeout: .seconds(1), sendMinimumRate: 100)
        let transport = MemoryTransport()
        transport.stopReading()
        let refusal = Task {
            await HTTPConnection.accept(transport, limiter: limiter, limits: limits, scheduler: clock) {
                await Self.router.handle($0)
            }
        }
        while transport.heldSendBytes == nil { try await Task.sleep(for: .milliseconds(1)) }
        try await clock.waitUntilArmed(moreThan: 0)  // the send deadline
        clock.advance(by: limits.sendDeadline(for: try #require(transport.heldSendBytes)))
        try await Self.untilCancelled(transport)
        try await finished(refusal)
    }

    /// Waits for the deadline to cancel `transport`, failing (instead of hanging) if it does not within 10 s.
    static func untilCancelled(_ transport: MemoryTransport) async throws {
        for _ in 0..<10_000 where !transport.isCancelled { try await Task.sleep(for: .milliseconds(1)) }
        try #require(transport.isCancelled, "the send outlived its deadline")
    }

    /// Connections beyond the limit get a 503 and are closed; a slot frees when a connection ends.
    @Test func connectionsBeyondTheLimitAreRefused() async throws {
        let limiter = ConnectionLimiter(limit: 2)
        let limits = HTTPConnectionLimits(maxBody: 1 << 20)
        func accept(_ transport: MemoryTransport) -> Task<Void, Never> {
            Task { await HTTPConnection.accept(transport, limiter: limiter, limits: limits) { await Self.router.handle($0) } }
        }
        let first = MemoryTransport(), second = MemoryTransport()
        let held = [accept(first), accept(second)]
        while limiter.active < 2 { try await Task.sleep(for: .milliseconds(10)) }

        let refused = MemoryTransport()
        await accept(refused).value
        let (responses, _) = parseResponses(refused.sent, methods: ["GET"])
        #expect(responses.map(\.status) == [503] && refused.isCancelled)
        #expect(responses.first?.headers["retry-after"] == "1" && responses.first?.headers["connection"] == "close")
        #expect(try JSONValue.parse(try #require(responses.first).body)["error"]?["code"] == "too_many_connections")

        first.closeInput()
        await held[0].value
        #expect(limiter.active == 1)
        let next = MemoryTransport()
        next.write("GET /health HTTP/1.1\r\nConnection: close\r\n\r\n")
        await accept(next).value
        #expect(parseResponses(next.sent, methods: ["GET"]).responses.map(\.status) == [200])
        second.closeInput()
        await held[1].value
        #expect(limiter.active == 0)
    }

    /// Many refusals at once, on the default Dispatch deadlines: each sends its 503 well within its deadline and
    /// ends, and refused clients that do not read are cut once their (short) send deadline passes.
    @Test(.timeLimit(.minutes(1))) func manyRefusedConnectionsEachEndCleanly() async throws {
        let limiter = ConnectionLimiter(limit: 0)
        func refuse(_ transports: [MemoryTransport], _ limits: HTTPConnectionLimits) async throws {
            let refusals = transports.map { transport in
                Task { await HTTPConnection.accept(transport, limiter: limiter, limits: limits) { await Self.router.handle($0) } }
            }
            for refusal in refusals { try await finished(refusal) }
        }
        let reading = (0..<400).map { _ in MemoryTransport() }
        try await refuse(reading, HTTPConnectionLimits(maxBody: 1 << 20))
        let statuses = reading.map { parseResponses($0.sent, methods: ["GET"]).responses.map(\.status) }
        #expect(statuses.allSatisfy { $0 == [503] } && reading.allSatisfy(\.isCancelled))

        let stalled = (0..<100).map { _ in MemoryTransport() }
        stalled.forEach { $0.stopReading() }
        try await refuse(stalled, HTTPConnectionLimits(maxBody: 1 << 20, sendTimeout: .milliseconds(20)))
        #expect(stalled.allSatisfy { $0.isCancelled && $0.sent.isEmpty })
        #expect(limiter.active == 0)
    }

    /// A deadline that was replaced (restarted or stopped) never cancels the connection, even if its work still
    /// runs: here cancelling a deadline has no effect, so only the timer's own check stops it.
    @Test func aReplacedDeadlineNeverCancelsTheConnection() {
        let clock = ManualClock(), transport = MemoryTransport()
        let timer = ConnectionTimer(transport, scheduler: UncancellableDeadlines(clock: clock))
        timer.start(.seconds(1))
        timer.start(.seconds(2))  // restarted: the first deadline is stale
        clock.advance(by: .seconds(1))
        #expect(!transport.isCancelled && clock.sleeping == 1)
        timer.stop()  // the second one is stale too
        clock.advance(by: .seconds(1))
        #expect(!transport.isCancelled && clock.sleeping == 0)
        timer.start(.seconds(1))
        clock.advance(by: .seconds(1))
        #expect(transport.isCancelled)
    }

    /// A body is handed over without a copy, and nothing of a finished request stays buffered.
    @Test func readerReleasesWhatItReturned() throws {
        var reader = HTTPRequestReader(maxBody: 1 << 20)
        let body = Data(repeating: 7, count: 300_000)
        reader.append(Data("POST /a HTTP/1.1\r\nContent-Length: \(body.count)\r\n\r\n".utf8) + body.prefix(1000))
        #expect(try reader.next() == nil && reader.isReadingBody)
        reader.append(body.dropFirst(1000))
        #expect(try reader.next()?.body == body)
        #expect(reader.bufferedBytes == 0 && !reader.isReadingBody)
    }

    @Test func readerKeepsPipelinedBytes() throws {
        var reader = HTTPRequestReader(maxBody: 100)
        reader.append(Data("POST /a HTTP/1.1\r\nContent-Length: 3\r\n\r\nabcGET /b HTTP/1.1\r\n\r\nGET /c".utf8))
        let first = try #require(try reader.next())
        #expect(first.path == "/a" && first.body == Data("abc".utf8))
        #expect(try reader.next()?.path == "/b")
        #expect(try reader.next() == nil)
        reader.append(Data(" HTTP/1.1\r\n\r\n".utf8))
        #expect(try reader.next()?.path == "/c")
        #expect(reader.bufferedBytes == 0)
    }
}

@Suite("Dispatch deadlines")
struct DispatchDeadlineTests {
    @Test func durationsBecomeWholeNanosecondsRoundedUp() {
        #expect(DispatchDeadlineScheduler.nanoseconds(.seconds(61) + .milliseconds(5)) == 61_005_000_000)
        #expect(DispatchDeadlineScheduler.nanoseconds(Duration(secondsComponent: 0, attosecondsComponent: 1)) == 1)
        #expect(DispatchDeadlineScheduler.nanoseconds(.zero) == 0)
        #expect(DispatchDeadlineScheduler.nanoseconds(.seconds(-1)) == 0)
        #expect(DispatchDeadlineScheduler.nanoseconds(.seconds(Int64.max)) == .max)
    }

    @Test(.timeLimit(.minutes(1))) func aDeadlineRunsItsWorkOnce() async throws {
        let runs = Shared(0)
        let deadline = DispatchDeadlineScheduler().schedule(after: .milliseconds(5)) { runs.update { $0 += 1 } }
        for _ in 0..<10_000 where runs.value == 0 { try await Task.sleep(for: .milliseconds(1)) }
        deadline.cancel()  // after it ran: nothing to withdraw
        deadline.cancel()
        try await Task.sleep(for: .milliseconds(20))
        #expect(runs.value == 1)
    }

    /// The queue is held while the first deadline is cancelled, so the test does not depend on how fast it gets there.
    @Test(.timeLimit(.minutes(1))) func aCancelledDeadlineNeverRuns() async throws {
        let queue = DispatchQueue(label: "aCancelledDeadlineNeverRuns")
        queue.suspend()
        let scheduler = DispatchDeadlineScheduler(queue: queue)
        let cancelled = Shared(false), later = Shared(false)
        scheduler.schedule(after: .milliseconds(1)) { cancelled.value = true }.cancel()
        _ = scheduler.schedule(after: .milliseconds(30)) { later.value = true }
        queue.resume()
        for _ in 0..<10_000 where !later.value { try await Task.sleep(for: .milliseconds(1)) }
        queue.sync {}
        #expect(later.value && !cancelled.value)
    }
}
