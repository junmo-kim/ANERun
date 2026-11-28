@preconcurrency import CoreML
import Foundation
import Synchronization
import Testing

@testable import EmbedCore

@Suite("PPLX fused host")
struct PPLXFusedTests {
    @Test func packedIsolatesSequencesAndMasksPadding() async throws {
        let runner = FakePredictor { function, inputs, outputs in
            #expect(function == "embed_64")
            #expect(Set(inputs.keys) == ["inputs_embeds", "bias", "pool"])
            let embeddings = try tensor(inputs, "inputs_embeds", [1, 64, 1024], .float16)
            let bias = try tensor(inputs, "bias", [1, 1, 64, 64], .float16)
            let pool = try tensor(inputs, "pool", [8, 64], .float16)
            #expect(Set(outputs.keys) == ["pooled"])
            _ = try tensor(outputs, "pooled", [8, 1024], .float32)

            #expect(embeddings[[0, 0, 0]].floatValue == 2)
            #expect(embeddings[[0, 1, 0]].floatValue == 6)
            #expect(embeddings[[0, 2, 0]].floatValue == 6)
            #expect(embeddings[[0, 2, 1023]].floatValue == 6)
            #expect(embeddings[[0, 3, 0]].floatValue == 0)
            #expect(embeddings[[0, 63, 1023]].floatValue == 0)

            var expectedBias = [Float16](repeating: -10_000, count: 64 * 64)
            for row in 0..<64 { expectedBias[row * 64 + row] = 0 }
            expectedBias[1] = 0
            expectedBias[64] = 0
            bias.withUnsafeBufferPointer(ofType: Float16.self) { values in
                #expect(Array(values) == expectedBias)
            }
            var expectedPool = [Float16](repeating: 0, count: 8 * 64)
            expectedPool[0] = 0.5
            expectedPool[1] = 0.5
            expectedPool[64 + 2] = 1
            pool.withUnsafeBufferPointer(ofType: Float16.self) { values in
                #expect(Array(values) == expectedPool)
            }

            // The predictor can return its own padded, column-strided output.
            let pooled = MLMultiArray(shape: [8, 1024], dataType: .float32, strides: [1, 9])
            pooled.withUnsafeMutableBufferPointer(ofType: Float.self) { values, strides in
                values.initialize(repeating: .nan)
                for slot in 0..<8 {
                    for column in 0..<1024 {
                        values[slot * strides[0] + column * strides[1]] = Float(slot + column)
                    }
                }
            }
            return try features("pooled", pooled)
        }

        let values = try await PPLXFused.embed([[0, 1], [1]], length: 64, runner: runner, table: tokenTable())
        let firstRow = (0..<1024).map { Float($0) }
        let secondRow = (1...1024).map { Float($0) }
        let expected = [firstRow, secondRow]
        #expect(values == expected)
    }

    @Test func qwenPackingIsCausalWithinEachSequenceAndPoolsOnlyItsFinalRow() async throws {
        let runner = FakePredictor { _, inputs, outputs in
            let bias = try tensor(inputs, "bias", [1, 1, 64, 64], .float16)
            let pool = try tensor(inputs, "pool", [8, 64], .float16)
            bias.withUnsafeBufferPointer(ofType: Float16.self) { values in
                for row in 0..<64 {
                    for column in 0..<64 {
                        let visible = row == column || (row < 4 && row / 2 == column / 2 && column < row)
                        #expect(values[row * 64 + column] == (visible ? 0 : -10000))
                    }
                }
            }
            pool.withUnsafeBufferPointer(ofType: Float16.self) { values in
                #expect(values.indices.filter { values[$0] != 0 } == [1, 67])
                #expect(values[1] == 1 && values[67] == 1)
            }
            let pooled = try tensor(outputs, "pooled", [8, 1024], .float32)
            fill(pooled, with: 1)
            return try features("pooled", pooled)
        }
        _ = try await PPLXFused.embed([[0, 1], [1, 0]], length: 64, runner: runner, table: tokenTable(), causal: true)
    }

    @Test func packedAcceptsLastSlotAndTokenButRejectsOverflow() async throws {
        let calls = Mutex(0)
        let runner = FakePredictor { function, inputs, outputs in
            calls.withLock { $0 += 1 }
            #expect(function == "embed_64")
            let pool = try tensor(inputs, "pool", [8, 64], .float16)
            #expect(pool[[6, 63]].floatValue == 0)
            #expect(pool[[7, 62]].floatValue == 0)
            #expect(pool[[7, 63]].floatValue == 1)
            let pooled = try tensor(outputs, "pooled", [8, 1024], .float32)
            fill(pooled, with: 0)
            pooled[[7, 1023]] = 8
            return try features("pooled", pooled)
        }
        let table = try tokenTable()
        let sequences = [Array(repeating: Int32(0), count: 57)] + Array(repeating: [Int32(1)], count: 7)
        let values = try await PPLXFused.embed(sequences, length: 64, runner: runner, table: table)
        #expect(values.count == 8)
        #expect(values.last?.last == 8)

        for invalid in [Array(repeating: [Int32(0)], count: 9), [[Int32](repeating: 0, count: 65)], [[]]] {
            await #expect(throws: RuntimeError.self) {
                try await PPLXFused.embed(invalid, length: 64, runner: runner, table: table)
            }
        }
        #expect(calls.withLock { $0 } == 1)
    }

    @Test(arguments: [2048, 4096]) func fusedUsesKeyBiasAndUndoesPoolScale(length: Int) async throws {
        let tokenCount = length - 5
        let runner = FakePredictor { function, inputs, outputs in
            #expect(function == "fused_\(length)")
            #expect(Set(inputs.keys) == ["inputs_embeds", "bias", "pool"])
            let embeddings = try tensor(inputs, "inputs_embeds", [1, length, 1024], .float16)
            #expect(embeddings[[0, NSNumber(value: tokenCount - 1), 1023]].floatValue == 6)
            #expect(embeddings[[0, NSNumber(value: tokenCount), 0]].floatValue == 0)
            try checkLongMask(inputs, length: length, tokens: tokenCount)
            let pooled = try tensor(outputs, "pooled", [1, 1024], .float32)
            fill(pooled, with: 0.5 * Float(tokenCount) / 1024)
            return try features("pooled", pooled)
        }
        let ids = [Int32](repeating: 1, count: tokenCount)
        let values = try await PPLXFused.embed([ids], length: length, runner: runner, table: tokenTable())
        #expect(values == [[Float](repeating: 0.5, count: 1024)])
    }

    @Test(arguments: [false, true]) func splitChainsActualOutputAndAlternatesBackings(fallback: Bool) async throws {
        let length = 8192
        let tokenCount = 4097
        let calls = Mutex([String]())
        let backings = Mutex([ObjectIdentifier]())
        let priorOutput = Mutex<ObjectIdentifier?>(nil)
        let runner = FakePredictor { function, inputs, outputs in
            let part = calls.withLock { calls in
                calls.append(function)
                return calls.count - 1
            }
            #expect(function == "fused_8192_part\(part)")
            let key = part == 0 ? "inputs_embeds" : "hidden"
            let hidden = try tensor(inputs, key, [1, length, 1024], .float16)
            let bias = try tensor(inputs, "bias", [1, 1, 1, length], .float16)
            #expect(bias[[0, 0, 0, 4096]].floatValue == 0)
            #expect(bias[[0, 0, 0, 4097]].floatValue == -10_000)

            #expect(hidden[[0, 0, 0]].floatValue == Float(2 + part))
            if part > 0 { #expect(ObjectIdentifier(hidden) == priorOutput.withLock { $0 }) }
            if part == 3 {
                #expect(Set(inputs.keys) == [key, "bias", "pool"])
                #expect(Set(outputs.keys) == ["pooled"])
                try checkLongMask(inputs, length: length, tokens: tokenCount)
                let pooled = try tensor(outputs, "pooled", [1, 1024], .float32)
                fill(pooled, with: 2 * Float(tokenCount) / 1024)
                return try features("pooled", pooled)
            }

            #expect(Set(inputs.keys) == [key, "bias"])
            #expect(Set(outputs.keys) == ["hidden_out"])
            let backing = try tensor(outputs, "hidden_out", [1, length, 1024], .float16)
            #expect(backing !== hidden)
            backings.withLock { $0.append(ObjectIdentifier(backing)) }

            let output = fallback && part == 1
                ? MLMultiArray(shape: [1, length, 1024], dataType: .float16, strides: [length * 1025, 1025, 1])
                : backing
            output.withUnsafeMutableBufferPointer(ofType: Float16.self) { values, _ in
                values.initialize(repeating: Float16(3 + part))
            }
            priorOutput.withLock { $0 = ObjectIdentifier(output) }
            return try features("hidden_out", output)
        }

        let ids = [Int32](repeating: 0, count: tokenCount)
        let values = try await PPLXFused.embed([ids], length: length, runner: runner, table: tokenTable())
        #expect(values == [[Float](repeating: 2, count: 1024)])
        #expect(calls.withLock { $0 } == (0..<4).map { "fused_8192_part\($0)" })
        let used = backings.withLock { $0 }
        try #require(used.count == 3)
        #expect(used[0] != used[1])
        #expect(used[0] == used[2])
    }

    @Test func rejectsNonfiniteOrWrongDTypeOutputs() async throws {
        let table = try tokenTable()
        let nonfinite = FakePredictor { _, _, outputs in
            let pooled = try tensor(outputs, "pooled", [8, 1024], .float32)
            fill(pooled, with: 0)
            pooled[[0, 1023]] = .init(value: Float.infinity)
            return try features("pooled", pooled)
        }
        let wrongType = FakePredictor { _, _, _ in
            try features("pooled", PPLXTensor.half([8, 1024], zero: true))
        }
        for runner in [nonfinite, wrongType] {
            await #expect(throws: RuntimeError.self) {
                try await PPLXFused.embed([[0]], length: 64, runner: runner, table: table)
            }
        }
    }

    @Test func rejectsNonfiniteHiddenBeforeNextPrediction() async throws {
        let calls = Mutex([String]())
        let runner = FakePredictor { function, _, outputs in
            calls.withLock { $0.append(function) }
            let hidden = try tensor(outputs, "hidden_out", [1, 8192, 1024], .float16)
            hidden.withUnsafeMutableBufferPointer(ofType: Float16.self) { values, _ in values.initialize(repeating: 0) }
            hidden[[0, 4096, 1023]] = .init(value: Float.nan)
            return try features("hidden_out", hidden)
        }
        let table = try tokenTable()
        await #expect(throws: RuntimeError.self) {
            try await PPLXFused.embed([[Int32](repeating: 0, count: 4097)], length: 8192, runner: runner, table: table)
        }
        #expect(calls.withLock { $0 } == ["fused_8192_part0"])
    }

    @Test func cancellationBeforePredictionSkipsRunner() async throws {
        let runner = FakePredictor { _, _, _ in
            Issue.record("prediction after cancellation")
            throw CancellationError()
        }
        let table = try tokenTable()
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await PPLXFused.embed([[0]], length: 64, runner: runner, table: table)
        }
        await #expect(throws: CancellationError.self) { try await task.value }
    }

    @Test func selfTestChecksMeanMagnitudeAsWellAsDirection() {
        let mean = [Float](repeating: 0.25, count: 1024)
        let doubled = mean.map { 2 * $0 }
        #expect(PPLXSelfTest.compare([mean], with: [mean], name: "pool").passed)
        let invalid = PPLXSelfTest.compare([doubled], with: [mean], name: "pool")
        #expect(!invalid.passed)
        #expect(invalid.cosine == 1 && invalid.error?.contains("magnitude") == true)
    }

    @Test func predictionConvenienceSuppliesNoBackings() async throws {
        let runner: any PPLXPredicting = FakePredictor { function, inputs, outputs in
            #expect(function == "function")
            #expect(Set(inputs.keys) == ["xs"])
            #expect(outputs.isEmpty)
            return try MLDictionaryFeatureProvider(dictionary: [:])
        }
        _ = try await runner.predict("function", inputs: ["xs": PPLXTensor.half([1], zero: true)])
    }

    @Test(arguments: [64, 8192]) func cancellationDuringPredictionDiscardsOutput(length: Int) async throws {
        let calls = Mutex([String]())
        let runner = FakePredictor { function, _, _ in
            calls.withLock { $0.append(function) }
            withUnsafeCurrentTask { $0?.cancel() }
            return try MLDictionaryFeatureProvider(dictionary: [:])
        }
        let table = try tokenTable()
        let sequences = [[Int32](repeating: 0, count: length == 64 ? 1 : 4097)]
        let task = Task { try await PPLXFused.embed(sequences, length: length, runner: runner, table: table) }
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(calls.withLock { $0 } == [length == 64 ? "embed_64" : "fused_8192_part0"])
    }
}

private struct FakePredictor: PPLXPredicting {
    let body: @Sendable (String, [String: MLMultiArray], [String: MLMultiArray]) throws -> MLFeatureProvider

    init(_ body: @escaping @Sendable (String, [String: MLMultiArray], [String: MLMultiArray]) throws -> MLFeatureProvider) {
        self.body = body
    }

    func predict(_ function: String, inputs: [String: MLMultiArray], outputs: [String: MLMultiArray])
        async throws -> MLFeatureProvider
    {
        try body(function, inputs, outputs)
    }
}

private func tokenTable() throws -> PPLXTokenTable {
    let data = ([Float](repeating: 2, count: 1024) + [Float](repeating: 6, count: 1024)).withUnsafeBytes { Data($0) }
    return try PPLXTokenTable(data: data, rows: 2)
}

private func tensor(_ tensors: [String: MLMultiArray], _ name: String, _ shape: [Int], _ dtype: MLMultiArrayDataType)
    throws -> MLMultiArray
{
    let tensor = try #require(tensors[name])
    #expect(tensor.shape.map(\.intValue) == shape)
    #expect(tensor.dataType == dtype)
    return tensor
}

private func features(_ name: String, _ tensor: MLMultiArray) throws -> MLFeatureProvider {
    try MLDictionaryFeatureProvider(dictionary: [name: MLFeatureValue(multiArray: tensor)])
}

private func fill(_ tensor: MLMultiArray, with value: Float) {
    tensor.withUnsafeMutableBufferPointer(ofType: Float.self) { values, _ in values.initialize(repeating: value) }
}

private func checkLongMask(_ inputs: [String: MLMultiArray], length: Int, tokens: Int) throws {
    let bias = try tensor(inputs, "bias", [1, 1, 1, length], .float16)
    let pool = try tensor(inputs, "pool", [1, length], .float16)
    bias.withUnsafeBufferPointer(ofType: Float16.self) { values in
        #expect(values.prefix(tokens).allSatisfy { $0 == 0 })
        #expect(values.dropFirst(tokens).allSatisfy { $0 == -10_000 })
    }
    pool.withUnsafeBufferPointer(ofType: Float16.self) { values in
        #expect(values.prefix(tokens).allSatisfy { $0 == 1 / 1024 })
        #expect(values.dropFirst(tokens).allSatisfy { $0 == 0 })
    }
}
