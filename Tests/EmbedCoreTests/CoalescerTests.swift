import Foundation
import Testing

@testable import EmbedCore

@Suite("Coalescing concurrent requests")
struct CoalescerTests {
    final class Recorder: @unchecked Sendable {
        let lock = NSLock()
        var rounds: [Int] = []
        func record(_ count: Int) { lock.withLock { rounds.append(count) } }
    }

    @Test func requestsArrivingWhileBusyShareARoundAndKeepTheirOrder() async throws {
        let recorder = Recorder()
        let coalescer = Coalescer(maxRunning: 1) { sequences in
            recorder.record(sequences.count)
            try await Task.sleep(for: .milliseconds(30))
            return sequences.map { [Float($0.count)] }
        }
        let results = try await withThrowingTaskGroup(of: (Int, [[Float]]).self) { group in
            for request in 0..<12 {
                group.addTask {
                    if request > 0 { try await Task.sleep(for: .milliseconds(5)) }
                    let sequences = (0..<(request % 3 + 1)).map { ModelSequence(ids: [Int32](repeating: 1, count: request + $0 + 1)) }
                    return (request, try await coalescer.submit(sequences))
                }
            }
            var all: [Int: [[Float]]] = [:]
            for try await (request, vectors) in group { all[request] = vectors }
            return all
        }
        for request in 0..<12 {
            let expected: [[Float]] = (0..<(request % 3 + 1)).map { [Float(request + $0 + 1)] }
            #expect(results[request] == expected)
        }
        let total: Int = (0..<12).map { $0 % 3 + 1 }.reduce(0, +)
        #expect(recorder.rounds.count < 12)  // later requests were merged
        #expect(recorder.rounds.reduce(0, +) == total)
    }

    @Test func aFailedRoundFailsItsRequestsOnly() async throws {
        let coalescer = Coalescer(maxRunning: 1) { sequences in
            if sequences.contains(where: { $0.count == 7 }) { throw RuntimeError.model("boom") }
            return sequences.map { _ in [1] }
        }
        await #expect(throws: RuntimeError.self) { _ = try await coalescer.submit([ModelSequence(ids: [Int32](repeating: 1, count: 7))]) }
        #expect(try await coalescer.submit([ModelSequence(ids: [1, 2])]) == [[1]])
    }
}
