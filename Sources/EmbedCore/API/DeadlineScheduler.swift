// Runs a piece of work once a deadline passes, unless the deadline is cancelled first. Connection deadlines (head,
// body idle, send, body budget wait) are armed and replaced many times per connection, so they run on Dispatch timers
// rather than on a Swift task per deadline. Tests pass a clock they advance by hand.
import Foundation

public protocol DeadlineScheduler: Sendable {
    /// Runs `work` once, `delay` from now, unless the returned deadline is cancelled before then. `work` never runs
    /// synchronously inside this call.
    func schedule(after delay: Duration, _ work: @escaping @Sendable () -> Void) -> any ScheduledDeadline
}

public protocol ScheduledDeadline: Sendable {
    /// Withdraws the deadline. Idempotent; a deadline whose work already started is not interrupted.
    func cancel()
}

/// Deadlines on one-shot Dispatch timer sources. A cancelled timer releases its work at once, so a deadline that is
/// replaced on every received chunk leaves nothing queued behind it.
public struct DispatchDeadlineScheduler: DeadlineScheduler {
    private let queue: DispatchQueue

    public init(queue: DispatchQueue = .global()) {
        self.queue = queue
    }

    public func schedule(after delay: Duration, _ work: @escaping @Sendable () -> Void) -> any ScheduledDeadline {
        let source = DispatchSource.makeTimerSource(queue: queue)
        let nanoseconds = Self.nanoseconds(delay)
        source.schedule(deadline: nanoseconds == .max ? .distantFuture : .now() + .nanoseconds(nanoseconds))
        // The handler holds the source until it fires or is cancelled; cancelling releases the handler, and the cycle.
        source.setEventHandler {
            work()
            source.cancel()
        }
        source.resume()
        return DispatchDeadline(source: source)
    }

    /// `duration` in whole nanoseconds, rounded up so a deadline never fires early: 0 for a negative duration, `.max`
    /// for one too long to count in nanoseconds.
    static func nanoseconds(_ duration: Duration) -> Int {
        guard duration > .zero else { return 0 }
        let (seconds, attoseconds) = duration.components
        let (whole, overflow) = seconds.multipliedReportingOverflow(by: 1_000_000_000)
        let (total, carry) = whole.addingReportingOverflow((attoseconds + 999_999_999) / 1_000_000_000)
        return overflow || carry || total > Int64(Int.max) ? .max : Int(total)
    }
}

private final class DispatchDeadline: ScheduledDeadline, @unchecked Sendable {
    // DispatchSource is thread-safe; cancel may be called from any thread, any number of times.
    private let source: any DispatchSourceTimer

    init(source: any DispatchSourceTimer) {
        self.source = source
    }

    func cancel() {
        source.cancel()
    }
}
