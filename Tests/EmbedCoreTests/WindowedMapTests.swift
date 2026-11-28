import Foundation
import Testing

@testable import EmbedCore

@Suite("Windowed map")
struct WindowedMapTests {
    final class Gauge: @unchecked Sendable {
        private let lock = NSLock()
        private var running = 0
        private(set) var peak = 0
        private(set) var started = 0

        func enter() { lock.withLock { running += 1; started += 1; peak = max(peak, running) } }
        func leave() { lock.withLock { running -= 1 } }
        var startedCount: Int { lock.withLock { started } }
        var peakCount: Int { lock.withLock { peak } }
    }

    struct Failure: Error {}

    @Test(arguments: [1, 3, 8])
    func keepsTheOrderAndTheWidth(_ width: Int) async throws {
        let gauge = Gauge()
        let results = try await windowedMap(Array(0..<40), width: width) { value in
            gauge.enter()
            defer { gauge.leave() }
            try await Task.sleep(for: .microseconds(200 * (value % 5)))  // finish out of order
            return value * 2
        }
        #expect(results == (0..<40).map { $0 * 2 })
        #expect(gauge.peakCount <= width && gauge.startedCount == 40)
    }

    @Test func theFirstErrorStopsTheRest() async {
        let gauge = Gauge()
        await #expect(throws: Failure.self) {
            _ = try await windowedMap(Array(0..<100), width: 2) { value in
                gauge.enter()
                defer { gauge.leave() }
                if value == 3 { throw Failure() }
                return value
            }
        }
        #expect(gauge.startedCount < 100)
    }

    @Test func noElementsIsNoWork() async throws {
        #expect(try await windowedMap([Int](), width: 3) { $0 }.isEmpty)
    }
}
