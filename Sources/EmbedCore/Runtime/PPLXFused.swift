@preconcurrency import CoreML
import Foundation

enum PPLXFused {
    private static let width = PPLXTokenTable.width
    private static let masked: Float16 = -10_000
    private static let poolScale: Float16 = 1 / 1024

    static func embed(_ sequences: [[Int32]], length: Int, runner: any PPLXPredicting, table: PPLXTokenTable, causal: Bool = false)
        async throws -> [[Float]]
    {
        try Task.checkCancellation()
        let packed: Bool
        switch length {
        case 64, 512, 1024: packed = true
        case 2048, 4096, 8192: packed = false
        default: throw RuntimeError.model("unsupported pplx fused length \(length)")
        }
        let slots = packed ? max(8, length / 32) : 1
        guard !sequences.isEmpty, sequences.count <= slots else {
            throw RuntimeError.model("pplx length \(length) requires between 1 and \(slots) sequences")
        }
        var used = 0
        var ranges: [Range<Int>] = []
        ranges.reserveCapacity(sequences.count)
        for sequence in sequences {
            guard !sequence.isEmpty, sequence.count <= length - used else {
                throw RuntimeError.model("pplx sequences must be nonempty and fit length \(length)")
            }
            ranges.append(used..<(used + sequence.count))
            used += sequence.count
        }

        let embeddings = try PPLXTensor.half([1, length, width])
        try embeddings.withUnsafeMutableBufferPointer(ofType: Float16.self) { buffer, _ in
            let start = buffer.baseAddress!
            for (sequence, range) in zip(sequences, ranges) {
                try table.gatherHalf(sequence, into: start + range.lowerBound * width)
            }
            (start + used * width).initialize(repeating: 0, count: (length - used) * width)
        }
        let bias = try PPLXTensor.half([1, 1, packed ? length : 1, length])
        let pool = try PPLXTensor.half([slots, length], zero: true)

        bias.withUnsafeMutableBufferPointer(ofType: Float16.self) { buffer, _ in
            buffer.initialize(repeating: masked)
            let start = buffer.baseAddress!
            if packed {
                for range in ranges {
                    for row in range {
                        (start + row * length + range.lowerBound).update(repeating: 0, count: causal ? row - range.lowerBound + 1 : range.count)
                    }
                }
                for row in used..<length { start[row * length + row] = 0 }
            } else {
                start.update(repeating: 0, count: used)
            }
        }
        pool.withUnsafeMutableBufferPointer(ofType: Float16.self) { buffer, _ in
            let start = buffer.baseAddress!
            for (slot, range) in ranges.enumerated() {
                if causal {
                    start[slot * length + range.upperBound - 1] = 1
                    continue
                }
                let weight = packed ? 1 / Float16(range.count) : poolScale
                (start + slot * length + range.lowerBound).update(repeating: weight, count: range.count)
            }
        }

        let pooled = try PPLXTensor.contiguous([slots, width], dtype: .float32)
        let output: MLFeatureProvider
        if length == 8192 {
            output = try await split(embeddings, bias: bias, pool: pool, pooled: pooled, runner: runner)
        } else {
            let function = (packed ? "embed_" : "fused_") + String(length)
            try Task.checkCancellation()
            output = try await runner.predict(function, inputs: ["inputs_embeds": embeddings, "bias": bias, "pool": pool],
                                              outputs: ["pooled": pooled])
            try Task.checkCancellation()
        }
        return try readPooled(output, slots: slots, count: sequences.count,
                              divisor: packed ? 1 : Float(used) * Float(poolScale))
    }

    private static func split(_ embeddings: MLMultiArray, bias: MLMultiArray, pool: MLMultiArray,
                              pooled: MLMultiArray, runner: any PPLXPredicting) async throws -> MLFeatureProvider
    {
        let shape = [1, 8192, width]
        try Task.checkCancellation()
        let second = try PPLXTensor.half(shape)
        var hidden = embeddings
        for part in 0..<3 {
            // Part 0 consumes the embeddings. Its storage can back part 1 once that prediction completes.
            let backing = part.isMultiple(of: 2) ? second : embeddings
            guard hidden !== backing else { throw RuntimeError.model("pplx hidden output aliases the next backing") }
            let key = part == 0 ? "inputs_embeds" : "hidden"
            try Task.checkCancellation()
            let output = try await runner.predict("fused_8192_part\(part)", inputs: [key: hidden, "bias": bias],
                                                  outputs: ["hidden_out": backing])
            try Task.checkCancellation()
            hidden = try PPLXTensor.output(output, "hidden_out")
            try checkHidden(hidden, shape: shape)
        }
        try Task.checkCancellation()
        let output = try await runner.predict("fused_8192_part3", inputs: ["hidden": hidden, "bias": bias, "pool": pool],
                                              outputs: ["pooled": pooled])
        try Task.checkCancellation()
        return output
    }

    private static func checkHidden(_ hidden: MLMultiArray, shape: [Int]) throws(RuntimeError) {
        let layout = try matrixLayout(hidden, shape: shape, dtype: .float16, name: "hidden_out")
        let length = shape[1]
        let valid = hidden.withUnsafeBufferPointer(ofType: Float16.self) { buffer in
            guard layout.lastOffset < buffer.count else { return false }
            if layout.columnStride == 1 {
                let start = buffer.baseAddress!
                if layout.rowStride == width { return finiteHalfBuffer(start, count: length * width) }
                return (0..<length).allSatisfy { finiteHalfBuffer(start + $0 * layout.rowStride, count: width) }
            }
            for row in 0..<length {
                for column in 0..<width {
                    if !buffer[row * layout.rowStride + column * layout.columnStride].isFinite { return false }
                }
            }
            return true
        }
        guard valid else { throw .model("pplx hidden_out has invalid storage or nonfinite values") }
    }

    private static func finiteHalfBuffer(_ start: UnsafePointer<Float16>, count: Int) -> Bool {
        // Each row has 1,024 lanes. An all-ones Float16 exponent marks infinity or NaN.
        let exponent = SIMD8<UInt16>(repeating: 0x7c00)
        let raw = UnsafeRawPointer(start)
        for index in stride(from: 0, to: count, by: SIMD8<UInt16>.scalarCount) {
            let block = raw.loadUnaligned(fromByteOffset: index * MemoryLayout<Float16>.stride, as: SIMD8<UInt16>.self)
            if any((block & exponent) .== exponent) { return false }
        }
        return true
    }

    private static func readPooled(_ features: MLFeatureProvider, slots: Int, count: Int, divisor: Float)
        throws(RuntimeError) -> [[Float]]
    {
        let pooled = try PPLXTensor.output(features, "pooled")
        let layout = try matrixLayout(pooled, shape: [slots, width], dtype: .float32, name: "pooled")
        return try pooled.withUnsafeBufferPointer(ofType: Float.self) { buffer -> Result<[[Float]], RuntimeError> in
            guard layout.lastOffset < buffer.count else { return .failure(.model("pplx pooled has invalid storage")) }
            var result: [[Float]] = []
            result.reserveCapacity(count)
            for slot in 0..<slots {
                var values: [Float] = []
                if slot < count { values.reserveCapacity(width) }
                let offset = slot * layout.rowStride
                for column in 0..<width {
                    let value = buffer[offset + column * layout.columnStride] / divisor
                    guard value.isFinite else { return .failure(.model("pplx pooled has nonfinite values")) }
                    if slot < count { values.append(value) }
                }
                if slot < count { result.append(values) }
            }
            return .success(result)
        }.get()
    }

    private static func matrixLayout(_ array: MLMultiArray, shape: [Int], dtype: MLMultiArrayDataType, name: String)
        throws(RuntimeError) -> (rowStride: Int, columnStride: Int, lastOffset: Int)
    {
        let strides = array.strides.map(\.intValue)
        guard array.dataType == dtype, array.shape.map(\.intValue) == shape, strides.count == shape.count else {
            throw .model("pplx \(name) must be \(dtype == .float16 ? "fp16" : "fp32") \(shape)")
        }
        let rows = shape[shape.count - 2]
        let rowStride = strides[strides.count - 2]
        let columnStride = strides[strides.count - 1]
        let row = rowStride.multipliedReportingOverflow(by: rows - 1)
        let column = columnStride.multipliedReportingOverflow(by: width - 1)
        let lastOffset = row.partialValue.addingReportingOverflow(column.partialValue)
        guard rowStride >= (rows == 1 ? 0 : 1), columnStride > 0,
              !row.overflow, !column.overflow, !lastOffset.overflow
        else { throw .model("pplx \(name) has invalid strides \(strides)") }
        return (rowStride, columnStride, lastOffset.partialValue)
    }
}
