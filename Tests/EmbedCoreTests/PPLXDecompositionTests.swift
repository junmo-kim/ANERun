@preconcurrency import CoreML
import Foundation
import Synchronization
import Testing

@testable import EmbedCore

@Suite("PPLX host decomposition")
struct PPLXDecompositionTests {
    @Test func floatTableDoesNotScalePPLXRows() throws {
        let width = PPLXTokenTable.width
        let values = (0..<(width * 2)).map { Float($0) / 1024 }
        let bytes = values.map { $0.bitPattern.littleEndian }.withUnsafeBytes { Data($0) }
        let table = try PPLXTokenTable(data: bytes, rows: 2)
        var gathered = [Float](repeating: 0, count: width * 2)
        try gathered.withUnsafeMutableBufferPointer { try table.gatherFloat([1, 0], into: $0.baseAddress!) }
        let expected = Array(values[width...]) + Array(values[..<width])
        #expect(gathered == expected)
        var half = [Float16](repeating: 0, count: width)
        try half.withUnsafeMutableBufferPointer { try table.gatherHalf([1], into: $0.baseAddress!) }
        #expect(half == values[width...].map(Float16.init))
        #expect(throws: RuntimeError.self) { try half.withUnsafeMutableBufferPointer { try table.gatherHalf([2], into: $0.baseAddress!) } }
    }

    @Test func rotaryPreservesThePrototypeOrderAndPositions() {
        let rope = PPLXRotary(rows: 3)
        let start = rope.rotate(first: 1.5, second: -0.5, row: 0, frequency: 0, scale: 1)
        #expect(start.0 == 1.5 && start.1 == -0.5)
        let rotated = rope.rotate(first: 1.5, second: -0.5, row: 2, frequency: 0, scale: 0.25)
        #expect(abs(Float(rotated.0) - Float(0.25 * (1.5 * cos(2.0) + 0.5 * sin(2.0)))) < 0.0002)
        #expect(abs(Float(rotated.1) - Float(0.25 * (-0.5 * cos(2.0) + 1.5 * sin(2.0)))) < 0.0002)
        let expected: [Float] = [1, 1, 1 / 8, 1 / 8, 1 / 64, 1 / 64]
        #expect([0, 9, 10, 18, 19, 27].map(PPLXDecomposition.alpha) == expected)
    }

    @Test func causalAttentionEncodesValidKeyRowsAndPadding() async throws {
        let runtime = try makeRuntime(CausalMaskProbe(), length: 2048, causal: true)
        await #expect(throws: CausalMaskProbe.Finished.self) {
            try await runtime.profile([0, 1], length: 2048)
        }
    }

    private struct CausalMaskProbe: PPLXPredicting {
        enum Finished: Error { case checkedAttention }

        func predict(_ function: String, inputs: [String: MLMultiArray], outputs: [String: MLMultiArray])
            async throws -> MLFeatureProvider
        {
            if function.hasPrefix("attn_") {
                let bias = PPLXHostMath.halfPointer(try #require(inputs["bias"]))
                #expect(bias[0] == 0 && bias[1] == 1)
                #expect(bias[2] == 2048 && bias[2047] == 2048)
                throw Finished.checkedAttention
            }
            return try await IntermediateFixture().predict(function, inputs: inputs, outputs: outputs)
        }
    }

    @Test func finalPoolingUsesOnlyTheProvidedValidRows() {
        let width = PPLXTokenTable.width
        var residual = [Float](repeating: 2, count: 3 * width)
        residual.replaceSubrange(width..<(2 * width), with: repeatElement(Float(-4), count: width))
        residual.replaceSubrange((2 * width)..., with: repeatElement(Float.nan, count: width))
        let pooled = residual.withUnsafeBufferPointer {
            PPLXDecomposition.pool($0.baseAddress!, validRows: 2, norm: [Float](repeating: 3, count: width))
        }
        #expect(pooled.count == width)
        #expect(pooled.allSatisfy { $0.isFinite && abs($0) < 1e-5 })
    }

    @Test func vectorizedTileLayoutMatchesReferenceAndPreservesPadding() async throws {
        let runtime = try makeRuntime(IntermediateFixture())
        let result = try await runtime.profile([0, 1], length: 16384)
        #expect(result.vector.count == width && result.vector.allSatisfy { abs($0) < 1e-5 })
        #expect(result.layers.count == 28)
    }

    @Test(arguments: ["q", "o", "a"])
    func rejectsShortFallbackOutputs(_ name: String) async throws {
        let runtime = try makeRuntime(IntermediateFixture(shortenedOutput: name))
        await #expect(throws: RuntimeError.self) { try await runtime.profile([0, 1], length: 16384) }
    }

    @Test func validatesFallbackRowsBeforeBorrowingTheirStorage() throws {
        let storage = UnsafeMutablePointer<Float16>.allocate(capacity: 8)
        storage.initialize(repeating: 1, count: 8)
        defer { storage.deallocate() }
        let shape = [1, 2, 3]
        let overlapping = try MLMultiArray(dataPointer: storage, shape: [1, 2, 3], dataType: .float16,
                                           strides: [8, 2, 1], deallocator: nil)
        #expect(throws: RuntimeError.self) {
            try PPLXHostMath.withHalfOutput(overlapping, shape: shape) { _, _ in
                Issue.record("overlapping rows must not be read")
            }
        }

        let padded = try MLMultiArray(dataPointer: storage, shape: [1, 2, 3], dataType: .float16,
                                      strides: [8, 4, 1], deallocator: nil)
        try PPLXHostMath.withHalfOutput(padded, shape: shape) { pointer, strides in
            #expect(strides[1] == 4 && pointer[6] == 1)
        }
    }

    @Test(arguments: [false, true]) func predictionFailureHandsTheWorkspaceToTheNextRequest(cancelled: Bool) async throws {
        let runner = InterruptedPrediction(cancelled: cancelled)
        let runtime = try makeRuntime(runner)
        let task = Task { try await runtime.profile([0, 1], length: 16384) }
        if cancelled {
            await #expect(throws: CancellationError.self) { try await task.value }
        } else {
            await #expect(throws: RuntimeError.self) { try await task.value }
        }
        let next = try await runtime.profile([0, 1], length: 16384)
        #expect(next.vector.count == width && next.vector.allSatisfy { abs($0) < 1e-5 })
    }

    @Test(.timeLimit(.minutes(1))) func cancelledPredictionKeepsWorkspaceUntilCompletion() async throws {
        let (started, reportStart) = AsyncStream<Void>.makeStream()
        let (trying, reportAttempt) = AsyncStream<Void>.makeStream()
        let hold = AsyncSemaphore(1)
        await hold.wait()
        let runner = SuspendedPrediction(hold: hold, started: reportStart)
        let runtime = try makeRuntime(runner)
        let first = Task { try await runtime.profile([0, 1], length: 16384) }
        var starts = started.makeAsyncIterator()
        _ = await starts.next()
        first.cancel()
        let second = Task {
            reportAttempt.yield(())
            return try await runtime.profile([0, 1], length: 16384)
        }
        var attempts = trying.makeAsyncIterator()
        _ = await attempts.next()
        // The simulated device still owns the first request's tensors after task cancellation.
        // A second prediction would be admitted during this window if the long permit were removed.
        try? await Task.sleep(for: .milliseconds(250))
        #expect(runner.admissions.withLock { $0 } == 1)
        await hold.signal()
        await #expect(throws: CancellationError.self) { try await first.value }
        let next = try await second.value
        #expect(next.vector.count == width)
        #expect(runner.admissions.withLock { $0 } == 2)
    }

    private var width: Int { PPLXTokenTable.width }

    private func makeRuntime(_ runner: any PPLXPredicting, length: Int = 16384, causal: Bool = false)
        throws -> PPLXTextRuntime
    {
        let values = [Float](repeating: 2, count: width) + [Float](repeating: -4, count: width)
        let raw = values.map { $0.bitPattern.littleEndian }.withUnsafeBytes { Data($0) }
        let table = try PPLXTokenTable(data: raw, rows: 2)
        return PPLXTextRuntime(runner: runner, table: table, norm: [Float](repeating: 1, count: width),
                               lengths: [length], maxTokens: length, functions: [], notServed: [], skipped: true, causal: causal)
    }

    private final class InterruptedPrediction: PPLXPredicting {
        let cancelled: Bool
        let interrupted = Mutex(false)

        init(cancelled: Bool) { self.cancelled = cancelled }

        func predict(_ function: String, inputs: [String: MLMultiArray], outputs: [String: MLMultiArray])
            async throws -> MLFeatureProvider
        {
            if function.hasPrefix("attn_"), interrupted.withLock({ wasInterrupted in
                if wasInterrupted { return false }
                wasInterrupted = true
                return true
            }) {
                if cancelled { withUnsafeCurrentTask { $0?.cancel() } }
                try Task.checkCancellation()
                throw RuntimeError.model("injected attention prediction failure")
            }
            return try await IntermediateFixture().predict(function, inputs: inputs, outputs: outputs)
        }
    }

    private final class SuspendedPrediction: PPLXPredicting {
        let hold: AsyncSemaphore
        let started: AsyncStream<Void>.Continuation
        let admissions = Mutex(0)

        init(hold: AsyncSemaphore, started: AsyncStream<Void>.Continuation) {
            self.hold = hold
            self.started = started
        }

        func predict(_ function: String, inputs: [String: MLMultiArray], outputs: [String: MLMultiArray])
            async throws -> MLFeatureProvider
        {
            if function == "pre0" {
                let order = admissions.withLock { count in
                    count += 1
                    return count
                }
                if order == 1 {
                    started.yield(())
                    await hold.wait()
                    try Task.checkCancellation()
                }
            }
            return try await IntermediateFixture().predict(function, inputs: inputs, outputs: outputs)
        }
    }

    private struct IntermediateFixture: PPLXPredicting {
        var shortenedOutput: String? = nil

        func predict(_ function: String, inputs: [String: MLMultiArray], outputs: [String: MLMultiArray])
            async throws -> MLFeatureProvider
        {
            let width = PPLXTokenTable.width
            let qWidth = 2048
            if function.hasPrefix("pre") {
                let layer = try #require(Int(function.dropFirst(3)))
                let xs = PPLXHostMath.halfPointer(try #require(inputs["xs"]))
                let alpha = Float16(PPLXDecomposition.alpha(layer))
                #expect(xs[0] == 2 * alpha && xs[width] == -4 * alpha && xs[2 * width] == 0)
                let q = PPLXHostMath.halfPointer(try #require(outputs["q"]))
                let k = PPLXHostMath.halfPointer(try #require(outputs["k"]))
                let v = PPLXHostMath.halfPointer(try #require(outputs["v"]))
                q.update(repeating: 0, count: 2 * qWidth)
                k.update(repeating: 0, count: 2 * width)
                v.update(repeating: 0, count: 2 * width)
                q[qWidth] = 1.5
                q[qWidth + 64] = -0.5
                k[width + 128] = 1.5
                k[width + 128 + 64] = -0.5
                v[width + 128] = 3
            } else if function.hasPrefix("attn_") {
                let q = PPLXHostMath.halfPointer(try #require(inputs["q"]))
                let k = PPLXHostMath.halfPointer(try #require(inputs["k"]))
                let v = PPLXHostMath.halfPointer(try #require(inputs["v"]))
                let mask = PPLXHostMath.halfPointer(try #require(inputs["bias"]))
                // Float16 values from the reference rotary at position one.
                let q0 = Float16((1.5 * cos(1.0) + 0.5 * sin(1.0)) / sqrt(128.0))
                let q64 = Float16((-0.5 * cos(1.0) + 1.5 * sin(1.0)) / sqrt(128.0))
                let k0 = Float16(1.5 * cos(1.0) + 0.5 * sin(1.0))
                let k64 = Float16(-0.5 * cos(1.0) + 1.5 * sin(1.0))
                #expect(q[128] == q0 && q[128 + 64] == q64 && q[256] == 0)
                let keyHeadStride = 16384 * 128
                #expect(k[keyHeadStride + 128] == k0 && k[keyHeadStride + 128 + 64] == k64)
                #expect(v[keyHeadStride + 128] == 3 && v[2 * keyHeadStride - 1] == 0)
                #expect(mask[0] == 0 && mask[1] == 0 && mask[2] == -10_000 && mask[16383] == -10_000)
                let attention = PPLXHostMath.halfPointer(try #require(outputs["o"]))
                for head in 0..<16 {
                    (attention + head * 2048 * 128).update(repeating: 0, count: 2 * 128)
                }
                attention[2048 * 128 + 128] = 4
            } else {
                let layer = try #require(Int(function.dropFirst(4)))
                let xs = PPLXHostMath.halfPointer(try #require(inputs["xs"]))
                let o = PPLXHostMath.halfPointer(try #require(inputs["o"]))
                #expect(xs[width] == -4 * Float16(PPLXDecomposition.alpha(layer)))
                #expect(o[qWidth + 128] == 4 && o[2 * qWidth] == 0)
                let a = PPLXHostMath.halfPointer(try #require(outputs["a"]))
                let d = PPLXHostMath.halfPointer(try #require(outputs["d"]))
                a.update(repeating: 0, count: 2 * width)
                d.update(repeating: 0, count: 2 * width)
            }
            var returned = outputs
            if let name = shortenedOutput, let original = outputs[name] {
                var shape = original.shape.map(\.intValue)
                let row = shape.count - 2
                let originalRows = shape[row]
                shape[row] = 2
                let short = try PPLXTensor.half(shape)
                let heads = row == 2 ? shape[1] : 1
                let width = shape[row + 1]
                for head in 0..<heads {
                    (PPLXHostMath.halfPointer(short) + head * 2 * width)
                        .update(from: PPLXHostMath.halfPointer(original) + head * originalRows * width, count: 2 * width)
                }
                returned[name] = short
            }
            return try MLDictionaryFeatureProvider(dictionary: returned.mapValues { MLFeatureValue(multiArray: $0) })
        }
    }
}
