import Foundation
import Testing

@testable import EmbedCore

/// The calls a batch is planned into run a few at a time, so the buffers they fill stay bounded however many inputs
/// a request has.
@Suite("Bucket calls in flight")
struct CallWindowTests {
    /// Rows of zeros: the gather costs nothing, so a large batch stays quick in a debug build.
    struct ZeroRows: TokenRowSource {
        func gather(_ ids: [Int32], into destination: UnsafeMutablePointer<Float16>) {
            destination.update(repeating: 0, count: ids.count * BucketContract.hidden)
        }
    }

    /// Answers every call after a short sleep, as the Neural Engine would be busy, and fails call `failing` if set.
    final class SlowRunner: BucketFunctionRunner, @unchecked Sendable {
        let lengths = BucketContract.lengths
        let failing: Int?
        private let lock = NSLock()
        private var started = 0

        init(failing: Int? = nil) {
            self.failing = failing
        }

        var calls: Int { lock.withLock { started } }

        func warm() async throws {}

        func run(_ buffers: BucketBuffers) async throws -> [[Float]] {
            let call = lock.withLock { () -> Int in
                started += 1
                return started
            }
            try await Task.sleep(for: .milliseconds(2))
            if call == failing { throw RuntimeError.model("call \(call) failed") }
            return [[Float]](repeating: [Float](repeating: 0, count: BucketContract.dimension), count: buffers.slots)
        }
    }

    @Test func aLargeBatchChecksOutAFewBuffersAtATime() async throws {
        let runner = SlowRunner()
        let runtime = BucketTextRuntime(runner: runner, rows: ZeroRows(), backend: "test")
        let texts: [[Int32]] = (0..<2048).map { (i: Int) -> [Int32] in
            let body = [Int32](repeating: 1000 + Int32(i % 7), count: 120 + i % 300)
            return [SpecialTokens.bos] + body + [SpecialTokens.eos]
        }
        let vectors = try await runtime.embed(tokenized: texts)
        #expect(vectors.count == 2048 && vectors.allSatisfy { $0.count == BucketContract.dimension })
        #expect(runner.calls > 10 * BucketTextRuntime.callsInFlight)
        #expect(runtime.buffers.peakCheckedOut <= BucketTextRuntime.callsInFlight)
        #expect(runtime.buffers.checkedOut == 0)
    }

    /// Many more calls than the window: every text still gets its own vector, the one it gets alone.
    @Test func vectorsStayInInputOrder() async throws {
        let runtime = FakeRuntime.make()
        let texts: [[Int32]] = (0..<300).map { index in
            let body: [Int32] = (0..<(5 + index % 60)).map { Int32(1000 + index * 61 + $0) }
            return [SpecialTokens.bos] + body + [SpecialTokens.eos]
        }
        let together = try await BucketTextRuntime.run(
            sequences: texts.map { ModelSequence(ids: $0) }, planner: runtime.planner, buffers: runtime.buffers,
            runner: runtime.runner, rows: runtime.rows)
        #expect(runtime.planner.plan(texts.map(\.count)).count > BucketTextRuntime.callsInFlight)
        for (index, text) in texts.enumerated() {
            #expect(try await runtime.embed(tokenized: [text])[0] == together[index])
        }
    }

    @Test func aFailedCallFailsTheBatchAndReturnsItsBuffers() async throws {
        let runner = SlowRunner(failing: 5)
        let runtime = BucketTextRuntime(runner: runner, rows: ZeroRows(), backend: "test")
        let texts = (0..<2048).map { _ in [SpecialTokens.bos] + [Int32](repeating: 1000, count: 400) + [SpecialTokens.eos] }
        await #expect(throws: RuntimeError.self) { _ = try await runtime.embed(tokenized: texts) }
        #expect(runner.calls < 2048)
        #expect(runtime.buffers.checkedOut == 0)
    }
}
