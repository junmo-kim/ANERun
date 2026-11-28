// One budget for the request bodies that all HTTP connections buffer at once. Without it every connection may hold
// up to --max-body (72 MiB by default), about 4.6 GB with 64 connections. A large body reserves its Content-Length
// before any of it is read and waits, first come first served, while the budget is spent; small bodies do not
// reserve at all.
import Foundation

public final class BodyBudget: @unchecked Sendable {
    /// Bodies of at most this many bytes are read without a reservation, so a short text request never queues
    /// behind an image upload. 1 MiB is far above any text request (8192 tokens of text is well under 100 KB), and
    /// their memory is already bounded by the connection limit: 64 connections x 1 MiB = 64 MiB at the defaults.
    public static let unreservedBodyBytes = 1 << 20

    /// Bytes that reserved bodies may hold at once: one body of the maximum size.
    public let capacity: Int
    public let unreservedLimit: Int
    private let lock = NSLock()
    private var reserved = 0
    private var queue: [(id: Int, bytes: Int, continuation: CheckedContinuation<BodyReservation, Error>)] = []
    private var nextID = 0

    /// One reservation's progress, read and written under the budget's lock.
    private final class RequestState: @unchecked Sendable {
        var entered = false
        /// Why the request ended before it was queued (cancelled, or its deadline passed), if it did.
        var abandoned: (any Error)?
    }

    public init(capacity: Int, unreservedLimit: Int = BodyBudget.unreservedBodyBytes) {
        self.capacity = capacity
        self.unreservedLimit = unreservedLimit
    }

    /// Bytes currently reserved.
    public var reservedBytes: Int { lock.withLock { reserved } }
    /// Reservations waiting for their turn.
    public var waiting: Int { lock.withLock { queue.count } }

    /// A reservation for a body of `bytes`, once the budget has room for it and every earlier request has had its
    /// turn; nil without waiting for a body of at most `unreservedLimit`. Throws CancellationError if the task is
    /// cancelled while it waits.
    public func reserve(_ bytes: Int) async throws -> BodyReservation? {
        guard bytes > unreservedLimit else { return nil }
        let (id, state) = request()
        return try await wait(min(bytes, capacity), id: id, state: state)  // the parser already refuses bodies over --max-body
    }

    /// Thrown by `reserve(_:waitingAtMost:scheduler:)` when the wait ran out; the request has left the queue.
    public struct WaitTimedOut: Error {}

    /// `reserve(_:)` that waits at most `timeout` (`scheduler` runs the deadline; tests pass a clock they advance by
    /// hand), then throws WaitTimedOut. A reservation the budget has room for is granted at once, with no deadline armed.
    public func reserve(_ bytes: Int, waitingAtMost timeout: Duration,
                        scheduler: any DeadlineScheduler) async throws -> BodyReservation?
    {
        guard bytes > unreservedLimit else { return nil }
        let amount = min(bytes, capacity)
        let grantedAtOnce = lock.withLock { () -> Bool in
            guard queue.isEmpty && reserved + amount <= capacity else { return false }
            reserved += amount
            return true
        }
        if grantedAtOnce { return BodyReservation(budget: self, bytes: amount) }
        // The deadline takes the request out of the queue the way a cancellation does. Whichever of a grant and the
        // deadline takes the request first decides; the other then finds nothing to do.
        let (id, state) = request()
        let deadline = scheduler.schedule(after: timeout) { [self] in abandon(id, state, with: WaitTimedOut()) }
        defer { deadline.cancel() }
        return try await wait(amount, id: id, state: state)
    }

    private func request() -> (Int, RequestState) {
        let id = lock.withLock { () -> Int in
            nextID += 1
            return nextID
        }
        return (id, RequestState())
    }

    /// Waits for `amount` in the queue as request `id`. Cancellation can arrive before the request is queued, while
    /// it waits, or after it was granted. The state lives with this call (guarded by `lock`), so a late cancellation
    /// leaves nothing behind in the budget.
    private func wait(_ amount: Int, id: Int, state: RequestState) async throws -> BodyReservation {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<BodyReservation, Error>) in
                enum Outcome { case granted, abandoned(any Error), queued }
                let outcome = lock.withLock { () -> Outcome in
                    if let reason = state.abandoned { return .abandoned(reason) }
                    state.entered = true
                    if queue.isEmpty && reserved + amount <= capacity {
                        reserved += amount
                        return .granted
                    }
                    queue.append((id, amount, continuation))
                    return .queued
                }
                switch outcome {
                case .granted: continuation.resume(returning: BodyReservation(budget: self, bytes: amount))
                case .abandoned(let reason): continuation.resume(throwing: reason)
                case .queued: break
                }
            }
        } onCancel: {
            abandon(id, state, with: CancellationError())
        }
    }

    /// Ends request `id` with `reason` unless it was already granted: out of the queue if it waits there, or before it
    /// enters if it has not yet.
    private func abandon(_ id: Int, _ state: RequestState, with reason: any Error) {
        let removed = lock.withLock { () -> CheckedContinuation<BodyReservation, Error>? in
            guard let index = queue.firstIndex(where: { $0.id == id }) else {
                if !state.entered && state.abandoned == nil { state.abandoned = reason }  // a granted one needs nothing
                return nil
            }
            return queue.remove(at: index).continuation
        }
        removed?.resume(throwing: reason)
        grantWaiting()  // the request may have been first in line
    }

    fileprivate func release(_ bytes: Int) {
        lock.withLock { reserved -= bytes }
        grantWaiting()
    }

    /// Grants waiting reservations in order while the first one fits; a later, smaller one never overtakes it.
    private func grantWaiting() {
        let granted = lock.withLock { () -> [(Int, CheckedContinuation<BodyReservation, Error>)] in
            var granted: [(Int, CheckedContinuation<BodyReservation, Error>)] = []
            while let first = queue.first, reserved + first.bytes <= capacity {
                reserved += first.bytes
                granted.append((first.bytes, first.continuation))
                queue.removeFirst()
            }
            return granted
        }
        for (bytes, continuation) in granted { continuation.resume(returning: BodyReservation(budget: self, bytes: bytes)) }
    }
}

/// A body's share of the budget. Released once, explicitly or when the reservation is dropped.
public final class BodyReservation: @unchecked Sendable {
    public let bytes: Int
    private let budget: BodyBudget
    private let lock = NSLock()
    private var released = false

    fileprivate init(budget: BodyBudget, bytes: Int) {
        self.budget = budget
        self.bytes = bytes
    }

    public func release() {
        let first = lock.withLock { () -> Bool in
            defer { released = true }
            return !released
        }
        if first { budget.release(bytes) }
    }

    deinit { release() }
}
