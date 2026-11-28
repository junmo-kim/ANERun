import Foundation
import Testing

@testable import EmbedCore

/// A value several tasks write and the test reads.
final class Shared<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Value

    init(_ value: Value) { stored = value }

    var value: Value {
        get { lock.withLock { stored } }
        set { lock.withLock { stored = newValue } }
    }

    func update(_ change: (inout Value) -> Void) { lock.withLock { change(&stored) } }
}

/// The fake vision encoder, which first calls `observe` for every image it encodes.
struct ObservingVision: VisionEncoder {
    let maxTokens = ImagePrep.maxSoftTokens
    let observe: @Sendable () -> Void

    func warm() async throws {}

    func encode(_ image: PreparedImage) async throws -> [Float16] {
        observe()
        return try await FakeVisionEncoder().encode(image)
    }
}

enum BodyTestSupport {
    /// A router over the fake runtime whose vision encoder calls `observe` for every image.
    static func router(observe: @escaping @Sendable () -> Void) -> Router {
        let runtime = BucketTextRuntime(runner: PoolingFakeRunner(lengths: CoreMLBucketRunner.lengths(forMaxTokens: 2048)),
                                        rows: HashedTokenRows(), vision: ObservingVision(observe: observe), backend: "fake")
        let service = EmbeddingService(tokenizer: ByteTokenizer(), runtime: runtime, limits: ServiceLimits())
        return Router(state: ServerState(.ready(service)), modelName: "embeddinggemma-2")
    }

    /// An embeddings request body with one 8 x 8 PNG.
    static let imageRequest: Data = {
        let item: JSONValue = ["content": [["type": "image_url", "image_url": ["url": .string(TestSupport.dataURL(TestSupport.png(width: 8, height: 8)))]]]]
        return (["input": [item]] as JSONValue).jsonData()
    }()

    /// `bytes` in memory of their own, which calls `onFree` when the last reference to it goes.
    static func tracked(_ bytes: Data, onFree: @escaping @Sendable () -> Void) -> Data {
        let memory = UnsafeMutableRawPointer.allocate(byteCount: bytes.count, alignment: 16)
        bytes.copyBytes(to: memory.assumingMemoryBound(to: UInt8.self), count: bytes.count)
        return Data(bytesNoCopy: memory, count: bytes.count, deallocator: .custom { pointer, _ in
            pointer.deallocate()
            onFree()
        })
    }
}

@Suite("Request bodies")
struct RequestBodyTests {
    /// The raw body of an embeddings request is freed once parsed, before its images are embedded, rather than
    /// after the response.
    @Test func theEmbeddingsBodyIsFreedBeforeTheImagesAreEmbedded() async throws {
        let freed = Shared(false), seen = Shared<[Bool]>([])
        let router = BodyTestSupport.router { seen.update { $0.append(freed.value) } }
        let response = await router.handle(HTTPRequest(method: "POST", path: "/v1/embeddings",
                                                       body: BodyTestSupport.tracked(BodyTestSupport.imageRequest) { freed.value = true }))
        #expect(response.status == 200, "\(String(decoding: response.body, as: UTF8.self))")
        #expect(seen.value == [true])
    }

    /// The reader allocates a body once, at its Content-Length, and hands exactly those bytes over.
    @Test func aBodyArrivingInPiecesIsHandedOverWhole() throws {
        var reader = HTTPRequestReader(maxBody: 1 << 20)
        let body = Data((0..<5000).map { UInt8(truncatingIfNeeded: $0) })
        reader.append(Data("POST /a HTTP/1.1\r\nContent-Length: 5000\r\n\r\n".utf8) + body.prefix(10))
        #expect(try reader.next() == nil)
        for start in stride(from: 10, to: 5000, by: 997) { reader.append(body[start..<min(start + 997, 5000)]) }
        reader.append(Data("GET /b HTTP/1.1\r\n\r\n".utf8))
        #expect(try reader.next()?.body == body)
        #expect(try reader.next()?.path == "/b" && reader.bufferedBytes == 0)
    }
}

/// Large bodies share one budget across connections (BodyBudget); small ones do not wait for it. The connection
/// tests use bodies of 4096 bytes as the "maximum size" and 64 bytes as the unreserved limit.
@Suite("Body budget")
struct BodyBudgetTests {
    static let maxBody = 4096
    static let router = Router(state: ServerState(.ready(TestSupport.service())), modelName: "embeddinggemma-2")

    static func budget(capacity: Int = maxBody) -> BodyBudget { BodyBudget(capacity: capacity, unreservedLimit: 64) }

    /// A valid embeddings body of exactly `length` bytes (JSON padded with spaces).
    static func body(_ length: Int) -> Data {
        let json = Data(#"{"input":"a"}"#.utf8)
        return json + Data(repeating: 0x20, count: length - json.count)
    }

    static func head(_ length: Int, close: Bool = true) -> Data {
        Data("POST /v1/embeddings HTTP/1.1\r\nContent-Length: \(length)\r\n\(close ? "Connection: close\r\n" : "")\r\n".utf8)
    }

    /// `data` in `count` chunks.
    static func chunks(_ data: Data, _ count: Int) -> [Data] {
        let size = (data.count + count - 1) / count
        return stride(from: 0, to: data.count, by: size).map { Data(data[$0..<min($0 + size, data.count)]) }
    }

    /// A connection being served; `done` turns true when it ends.
    struct Served {
        let task: Task<Void, Never>
        let done: Shared<Bool>
    }

    static func serve(_ transport: MemoryTransport, _ budget: BodyBudget, limits: HTTPConnectionLimits? = nil,
                      clock: ManualClock? = nil, router: Router = router, started: Shared<[String]>? = nil,
                      name: String = "") -> Served
    {
        let done = Shared(false)
        let scheduler: any DeadlineScheduler = clock ?? DispatchDeadlineScheduler()
        let task = Task {
            await HTTPConnection.serve(transport, limits: limits ?? HTTPConnectionLimits(maxBody: maxBody), budget: budget,
                                       scheduler: scheduler) { request in
                started?.update { $0.append(name) }
                return await router.handle(consume request)
            }
            done.value = true
        }
        return Served(task: task, done: done)
    }

    /// Waits until `condition` holds. Polling only paces the wait; no result depends on its timing.
    static func until(_ condition: () -> Bool) async throws {
        while !condition() { try await Task.sleep(for: .milliseconds(1)) }
    }

    static func statuses(_ transport: MemoryTransport, _ methods: [String] = ["POST"]) -> [Int] {
        parseResponses(transport.sent, methods: methods).responses.map(\.status)
    }

    /// Starts a connection that reserves `length` bytes and holds them: its head and the first bytes of its body
    /// arrive, the rest only when the test calls `finish`.
    static func holder(_ budget: BodyBudget, length: Int = maxBody, started: Shared<[String]>? = nil,
                       name: String = "") async throws -> (MemoryTransport, Served, finish: () -> Void)
    {
        let transport = MemoryTransport()
        let body = Self.body(length)
        let before = budget.reservedBytes
        transport.write(head(length) + body.prefix(100))
        let served = serve(transport, budget, started: started, name: name)
        try await until { budget.reservedBytes == before + length || served.done.value }
        return (transport, served, { transport.write(Data(body.dropFirst(100))) })
    }

    // MARK: connections

    @Test(.timeLimit(.minutes(1))) func twoMaxSizeBodiesAreNeverBufferedAtOnce() async throws {
        let budget = Self.budget()
        let (first, firstServed, finishFirst) = try await Self.holder(budget)
        #expect(budget.reservedBytes == Self.maxBody)
        let second = MemoryTransport()
        second.write(Self.head(Self.maxBody))
        Self.chunks(Self.body(Self.maxBody), 4).forEach(second.write)
        let secondServed = Self.serve(second, budget)
        try await Self.until { budget.waiting == 1 || secondServed.done.value }
        #expect(budget.waiting == 1 && second.pendingChunks == 4, "the second body was read while the first held the budget")

        finishFirst()  // the waiter proceeds once the first body is freed
        try await finished(firstServed.task)
        try await finished(secondServed.task)
        #expect(Self.statuses(first) == [200] && Self.statuses(second) == [200])
        #expect(budget.reservedBytes == 0 && budget.waiting == 0)
    }

    @Test(.timeLimit(.minutes(1))) func aSmallRequestDoesNotWaitBehindALargeOne() async throws {
        let budget = Self.budget()
        let (_, firstServed, finishFirst) = try await Self.holder(budget)
        let large = MemoryTransport()
        large.write(Self.head(Self.maxBody) + Self.body(Self.maxBody))
        let largeServed = Self.serve(large, budget)
        try await Self.until { budget.waiting == 1 || largeServed.done.value }

        let small = MemoryTransport()
        small.write(Self.head(40) + Self.body(40))
        try await finished(Self.serve(small, budget).task)
        #expect(Self.statuses(small) == [200])
        #expect(budget.waiting == 1 && !largeServed.done.value)

        finishFirst()
        try await finished(firstServed.task)
        try await finished(largeServed.task)
        #expect(Self.statuses(large) == [200])
    }

    /// First come, first served: a later body that would fit waits behind an earlier one that does not, so a
    /// large upload cannot be starved by smaller ones.
    @Test(.timeLimit(.minutes(1))) func largeBodiesAreAdmittedInArrivalOrder() async throws {
        // A body is freed once parsed, before its response, so connections may finish in any order; the order in
        // which their requests reach the handler is the order in which their bodies were admitted.
        let budget = Self.budget(), started = Shared<[String]>([])
        let (_, firstServed, finishFirst) = try await Self.holder(budget, length: 2048, started: started, name: "first")
        let full = MemoryTransport(), quarter = MemoryTransport()
        full.write(Self.head(Self.maxBody) + Self.body(Self.maxBody))
        let fullServed = Self.serve(full, budget, started: started, name: "full")
        try await Self.until { budget.waiting == 1 || fullServed.done.value }
        quarter.write(Self.head(1024))
        quarter.write(Self.body(1024))
        let quarterServed = Self.serve(quarter, budget, started: started, name: "quarter")
        try await Self.until { budget.waiting == 2 || quarterServed.done.value }
        #expect(budget.waiting == 2 && quarter.pendingChunks == 1, "the 1024-byte body overtook the waiting one")

        finishFirst()
        for served in [firstServed, fullServed, quarterServed] { try await finished(served.task) }
        #expect(started.value == ["first", "full", "quarter"])
        #expect([full, quarter].map { Self.statuses($0) } == [[200], [200]])
    }

    static let shortTimeouts = HTTPConnectionLimits(maxBody: maxBody, headTimeout: .seconds(1), bodyIdleTimeout: .seconds(1))

    /// Starts a connection whose maximum-size body waits for the budget, on `clock`; returns once it waits and its
    /// deadline is armed.
    static func waiter(_ budget: BodyBudget, clock: ManualClock, started: Shared<[String]>? = nil,
                       name: String = "") async throws -> (MemoryTransport, Served)
    {
        let transport = MemoryTransport()
        let before = budget.waiting
        transport.write(head(maxBody))
        transport.write(body(maxBody))
        let served = serve(transport, budget, limits: shortTimeouts, clock: clock, started: started, name: name)
        try await until { (budget.waiting > before && clock.sleeping == 1) || served.done.value }
        return (transport, served)
    }

    /// A body waits for the budget at most the body idle timeout; then its connection is told to retry and closed,
    /// so connections waiting behind a body that is held for long do not pile up without end.
    @Test(.timeLimit(.minutes(1))) func aBodyThatWaitsTooLongForTheBudgetIsAskedToRetry() async throws {
        let budget = Self.budget(), clock = ManualClock()
        let (_, firstServed, finishFirst) = try await Self.holder(budget)
        let (waiter, waiterServed) = try await Self.waiter(budget, clock: clock)
        #expect(budget.waiting == 1)
        clock.advance(by: .milliseconds(999))
        #expect(!waiter.isCancelled)
        clock.advance(by: .milliseconds(1))
        try await finished(waiterServed.task)

        let response = try #require(parseResponses(waiter.sent, methods: ["POST"]).responses.first)
        #expect(response.status == 503 && waiter.isCancelled && waiter.pendingChunks == 1)
        #expect(response.headers["retry-after"] == "1" && response.headers["connection"] == "close")
        #expect(try JSONValue.parse(response.body)["error"]?["code"] == "body_budget_busy")
        #expect(budget.waiting == 0 && budget.reservedBytes == Self.maxBody)
        finishFirst()
        try await finished(firstServed.task)
        #expect(budget.reservedBytes == 0)
    }

    /// A waiter that timed out leaves the queue: it is never granted, and the one behind it is.
    @Test(.timeLimit(.minutes(1))) func aTimedOutWaiterMakesWayForTheNext() async throws {
        let budget = Self.budget(), started = Shared<[String]>([])
        let (_, firstServed, finishFirst) = try await Self.holder(budget)
        let firstClock = ManualClock(), nextClock = ManualClock()
        let (timedOut, timedOutServed) = try await Self.waiter(budget, clock: firstClock, started: started, name: "timed out")
        let (next, nextServed) = try await Self.waiter(budget, clock: nextClock, started: started, name: "next")
        #expect(budget.waiting == 2)
        firstClock.advance(by: .seconds(1))
        try await finished(timedOutServed.task)
        #expect(Self.statuses(timedOut) == [503] && budget.waiting == 1)

        finishFirst()
        try await finished(firstServed.task)
        try await finished(nextServed.task)
        #expect(Self.statuses(next) == [200] && started.value == ["next"])
        #expect(budget.reservedBytes == 0 && budget.waiting == 0)
    }

    /// A waiter granted before its deadline reads its body as usual, and its deadline is gone.
    @Test(.timeLimit(.minutes(1))) func aWaiterGrantedInTimeProceeds() async throws {
        let budget = Self.budget(), clock = ManualClock()
        let (_, firstServed, finishFirst) = try await Self.holder(budget)
        let (waiter, waiterServed) = try await Self.waiter(budget, clock: clock)
        clock.advance(by: .milliseconds(900))
        finishFirst()
        try await finished(firstServed.task)
        try await finished(waiterServed.task)
        #expect(Self.statuses(waiter) == [200])
        #expect(clock.sleeping == 0 && budget.reservedBytes == 0 && budget.waiting == 0)
    }

    /// A kept-alive connection releases a request's reservation before its next request reserves; otherwise the
    /// second of two pipelined maximum-size bodies would wait for itself.
    @Test(.timeLimit(.minutes(1))) func pipelinedLargeBodiesReleaseBeforeTheNextReserves() async throws {
        let budget = Self.budget()
        let connection = MemoryTransport()
        let request = Self.head(Self.maxBody, close: false) + Self.body(Self.maxBody)
        connection.write(request + request + Data("GET /health HTTP/1.1\r\nConnection: close\r\n\r\n".utf8))
        try await finished(Self.serve(connection, budget).task)
        #expect(Self.statuses(connection, ["POST", "POST", "GET"]) == [200, 200, 200])
        #expect(budget.reservedBytes == 0)
    }

    @Test(.timeLimit(.minutes(1))) func aConnectionThatClosesMidBodyReleasesItsReservation() async throws {
        let budget = Self.budget()
        let (transport, served, _) = try await Self.holder(budget)
        #expect(budget.reservedBytes == Self.maxBody)
        transport.closeInput()
        try await finished(served.task)
        #expect(budget.reservedBytes == 0 && transport.sent.isEmpty)
    }

    @Test(.timeLimit(.minutes(1))) func aStalledBodyReleasesItsReservationWhenItTimesOut() async throws {
        let budget = Self.budget(), clock = ManualClock()
        let transport = MemoryTransport()
        transport.write(Self.head(Self.maxBody) + Self.body(Self.maxBody).prefix(100))
        let limits = HTTPConnectionLimits(maxBody: Self.maxBody, headTimeout: .seconds(1), bodyIdleTimeout: .seconds(1))
        let served = Self.serve(transport, budget, limits: limits, clock: clock)
        try await clock.waitUntilArmed(moreThan: 1)  // the head deadline, then the body's idle timeout
        #expect(budget.reservedBytes == Self.maxBody)
        clock.advance(by: .seconds(1))
        try await finished(served.task)
        #expect(transport.isCancelled && budget.reservedBytes == 0)
    }

    @Test(.timeLimit(.minutes(1))) func aCancelledWaiterLeavesTheQueue() async throws {
        let budget = Self.budget()
        let (first, firstServed, finishFirst) = try await Self.holder(budget)
        let waiter = MemoryTransport()
        waiter.write(Self.head(Self.maxBody))
        let waiterServed = Self.serve(waiter, budget)
        try await Self.until { budget.waiting == 1 || waiterServed.done.value }
        waiterServed.task.cancel()
        try await finished(waiterServed.task)
        #expect(budget.waiting == 0 && waiter.isCancelled && budget.reservedBytes == Self.maxBody)

        finishFirst()
        try await finished(firstServed.task)
        #expect(Self.statuses(first) == [200] && budget.reservedBytes == 0)
    }

    /// The body's share of the budget is free once the request is parsed, while its images are still embedded.
    @Test(.timeLimit(.minutes(1))) func theReservationEndsWhenTheParsedBodyIsFreed() async throws {
        let budget = BodyBudget(capacity: 1 << 20, unreservedLimit: 64)
        let seen = Shared<[Int]>([])
        let router = BodyTestSupport.router { seen.update { $0.append(budget.reservedBytes) } }
        let body = BodyTestSupport.imageRequest
        let transport = MemoryTransport()
        transport.write(Self.head(body.count) + body.prefix(10))
        let served = Self.serve(transport, budget, limits: HTTPConnectionLimits(maxBody: 1 << 20), router: router)
        try await Self.until { budget.reservedBytes == body.count || served.done.value }
        #expect(budget.reservedBytes == body.count)
        transport.write(Data(body.dropFirst(10)))
        try await finished(served.task)
        #expect(Self.statuses(transport) == [200])
        #expect(seen.value == [0])
    }

    // MARK: the budget itself

    @Test func smallBodiesDoNotReserve() async throws {
        let budget = Self.budget()
        #expect(try await budget.reserve(64) == nil)
        let clamped = try #require(try await budget.reserve(1 << 30))
        #expect(clamped.bytes == Self.maxBody && budget.reservedBytes == Self.maxBody)
        #expect(try await budget.reserve(1) == nil)  // small bodies do not wait even when the budget is spent
        clamped.release()
        clamped.release()
        #expect(budget.reservedBytes == 0)
    }

    @Test(.timeLimit(.minutes(1))) func reservationsAreGrantedInOrderAndReleasedWhenDropped() async throws {
        let budget = BodyBudget(capacity: 100, unreservedLimit: 10)
        var first = try await budget.reserve(60)
        let order = Shared<[String]>([])
        let held = Shared<[BodyReservation]>([])
        func waiter(_ bytes: Int, _ name: String) -> Task<Void, Error> {
            Task {
                let reservation = try #require(try await budget.reserve(bytes))
                order.update { $0.append(name) }
                held.update { $0.append(reservation) }
            }
        }
        let large = waiter(60, "large")
        try await Self.until { budget.waiting == 1 }
        let small = waiter(30, "small")  // fits now, but queues behind the larger one
        try await Self.until { budget.waiting == 2 }
        #expect(order.value.isEmpty && budget.reservedBytes == 60)
        first = nil  // dropping a reservation releases it; then both fit
        try await finished(large).get()
        try await finished(small).get()
        #expect(order.value.sorted() == ["large", "small"] && budget.reservedBytes == 90)
        held.value = []
        #expect(budget.reservedBytes == 0 && first == nil)
    }

    @Test(.timeLimit(.minutes(1))) func cancellingTheFirstWaiterLetsTheNextOneIn() async throws {
        let budget = BodyBudget(capacity: 100, unreservedLimit: 10)
        let holding = try await budget.reserve(60)
        let blocked = Task { try await budget.reserve(60) }
        try await Self.until { budget.waiting == 1 }
        let next = Task { try await budget.reserve(30) }
        try await Self.until { budget.waiting == 2 }
        blocked.cancel()
        let cancelled = try await finished(blocked)
        #expect(throws: CancellationError.self) { try cancelled.get() }
        let granted = try await finished(next).get()
        #expect(granted?.bytes == 30 && budget.reservedBytes == 90 && budget.waiting == 0)
        _ = holding
    }

    @Test(.timeLimit(.minutes(1))) func aWaitThatTimesOutLetsTheNextOneIn() async throws {
        let budget = BodyBudget(capacity: 100, unreservedLimit: 10), clock = ManualClock()
        let holding = try await budget.reserve(60)
        let blocked = Task { try await budget.reserve(60, waitingAtMost: .seconds(1), scheduler: clock) }
        try await Self.until { budget.waiting == 1 && clock.sleeping == 1 }
        let next = Task { try await budget.reserve(30) }
        try await Self.until { budget.waiting == 2 }
        clock.advance(by: .seconds(1))
        let timedOut = try await finished(blocked)
        #expect(throws: BodyBudget.WaitTimedOut.self) { try timedOut.get() }
        let granted = try await finished(next).get()
        #expect(granted?.bytes == 30 && budget.reservedBytes == 90 && budget.waiting == 0)
        _ = holding
    }

    /// A grant that lands just before the deadline runs is kept: the late deadline finds nothing to take back.
    @Test(.timeLimit(.minutes(1))) func aDeadlineAfterTheGrantChangesNothing() async throws {
        let budget = BodyBudget(capacity: 100, unreservedLimit: 10), clock = ManualClock()
        var holding = try await budget.reserve(60)
        let blocked = Task {
            try await budget.reserve(60, waitingAtMost: .seconds(1), scheduler: UncancellableDeadlines(clock: clock))
        }
        try await Self.until { budget.waiting == 1 && clock.sleeping == 1 }
        holding = nil
        let granted = try await finished(blocked).get()
        clock.advance(by: .seconds(1))  // the deadline's work runs although the wait is over
        #expect(granted?.bytes == 60 && budget.reservedBytes == 60 && budget.waiting == 0 && holding == nil)
    }

    /// Cancellations that race grants, before, during and after the wait, leave the budget exactly balanced: every
    /// reservation either never happened or is released, and nobody is left in the queue.
    @Test(.timeLimit(.minutes(1))) func cancellationsRacingGrantsLeaveTheBudgetBalanced() async throws {
        let budget = BodyBudget(capacity: 100, unreservedLimit: 10)
        for round in 0..<300 {
            let holding = try await budget.reserve(60)
            let tasks = (0..<4).map { i in Task { _ = try await budget.reserve(30 + i) } }
            if round % 3 == 0 { await Task.yield() }
            tasks[round % 4].cancel()
            holding?.release()  // grants race the cancellation
            tasks[(round + 1) % 4].cancel()
            for task in tasks { try await finished(task) }
            #expect(budget.reservedBytes == 0 && budget.waiting == 0, "round \(round)")
        }
    }

    /// Room in the budget is granted at once, without arming a deadline.
    @Test func aReservationWithRoomDoesNotWait() async throws {
        let budget = BodyBudget(capacity: 100, unreservedLimit: 10), clock = ManualClock()
        let granted = try await budget.reserve(60, waitingAtMost: .seconds(1), scheduler: clock)
        #expect(granted?.bytes == 60 && clock.armed == 0)
    }
}
