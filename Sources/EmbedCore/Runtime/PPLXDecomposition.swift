import Accelerate
@preconcurrency import CoreML
import Foundation

/// The mmap stays shared across requests. A valid row is passed straight to the
/// Float16 conversion or copied to the owned Float32 residual stream.
struct PPLXTokenTable: Sendable {
    static let width = 1024
    static let rowCount = 151_936
    private let data: Data
    private let rows: Int

    init(contentsOf url: URL, rows: Int = Self.rowCount) throws {
        try self.init(data: Data(contentsOf: url, options: .alwaysMapped), rows: rows)
    }

    init(data: Data, rows: Int) throws(RuntimeError) {
        guard rows > 0, rows <= Self.rowCount, data.count == rows * Self.width * MemoryLayout<Float>.stride else {
            throw .model("invalid pplx Float32 embedding table")
        }
        self.data = data
        self.rows = rows
    }

    private func withRows(_ ids: [Int32], _ visit: (Int, UnsafeRawPointer) throws -> Void) throws {
        try data.withUnsafeBytes { bytes in
            let stride = Self.width * MemoryLayout<Float>.stride
            for (offset, id) in ids.enumerated() {
                guard id >= 0, id < rows else { throw RuntimeError.model("pplx token id \(id) is outside the table") }
                try visit(offset, bytes.baseAddress! + Int(id) * stride)
            }
        }
    }

    func gatherFloat(_ ids: [Int32], into target: UnsafeMutablePointer<Float>) throws {
        try withRows(ids) { offset, row in
            UnsafeMutableRawPointer(target + offset * Self.width)
                .copyMemory(from: row, byteCount: Self.width * MemoryLayout<Float>.stride)
        }
    }

    func gatherHalf(_ ids: [Int32], into target: UnsafeMutablePointer<Float16>) throws {
        try withRows(ids) { offset, row in
            try PPLXHostMath.toHalf(row.assumingMemoryBound(to: Float.self), target + offset * Self.width, count: Self.width)
        }
    }
}

enum PPLXHostMath {
    private static let flags = vImage_Flags(kvImageDoNotTile)

    static func toHalf(_ input: UnsafePointer<Float>, _ output: UnsafeMutablePointer<Float16>, count: Int) throws(RuntimeError) {
        var from = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: input), height: 1, width: vImagePixelCount(count),
                                 rowBytes: count * MemoryLayout<Float>.stride)
        var to = vImage_Buffer(data: output, height: 1, width: vImagePixelCount(count),
                               rowBytes: count * MemoryLayout<Float16>.stride)
        let error = vImageConvert_PlanarFtoPlanar16F(&from, &to, flags)
        guard error == kvImageNoError else { throw .model("pplx Float16 conversion failed (vImage \(error))") }
    }

    static func toFloat(_ input: UnsafePointer<Float16>, _ output: UnsafeMutablePointer<Float>,
                        rows: Int, columns: Int, inputStride: Int) throws(RuntimeError) {
        var from = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: input), height: vImagePixelCount(rows),
                                 width: vImagePixelCount(columns), rowBytes: inputStride * MemoryLayout<Float16>.stride)
        var to = vImage_Buffer(data: output, height: vImagePixelCount(rows), width: vImagePixelCount(columns),
                               rowBytes: columns * MemoryLayout<Float>.stride)
        let error = vImageConvert_Planar16FtoPlanarF(&from, &to, flags)
        guard error == kvImageNoError else { throw .model("pplx Float32 conversion failed (vImage \(error))") }
    }

    static func copyHalf(_ input: UnsafePointer<Float16>, _ output: UnsafeMutablePointer<Float16>,
                         rows: Int, columns: Int, inputStride: Int, outputStride: Int) throws(RuntimeError) {
        var from = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: input), height: vImagePixelCount(rows),
                                 width: vImagePixelCount(columns), rowBytes: inputStride * MemoryLayout<Float16>.stride)
        var to = vImage_Buffer(data: output, height: vImagePixelCount(rows), width: vImagePixelCount(columns),
                               rowBytes: outputStride * MemoryLayout<Float16>.stride)
        let error = vImageCopyBuffer(&from, &to, MemoryLayout<Float16>.stride, flags)
        guard error == kvImageNoError else { throw .model("pplx head copy failed (vImage \(error))") }
    }

    static func halfPointer(_ array: MLMultiArray) -> UnsafeMutablePointer<Float16> {
        array.dataPointer.assumingMemoryBound(to: Float16.self)
    }

    static func withHalfOutput<T>(_ array: MLMultiArray, shape: [Int],
                                  _ read: (UnsafePointer<Float16>, [Int]) throws -> T) throws -> T
    {
        let strides = array.strides.map(\.intValue)
        guard array.dataType == .float16, array.shape.map(\.intValue) == shape,
              strides.count == shape.count, strides.last == 1
        else { throw RuntimeError.model("pplx output must be fp16 \(shape) with contiguous columns") }
        // Each axis must leave room for the inner axes. One-element axes
        // contribute no offset, including the batch axis.
        var required = 1
        for (size, stride) in zip(shape.reversed(), strides.reversed()) {
            guard size > 0, stride >= 0, size == 1 || stride >= required else {
                throw RuntimeError.model("invalid pplx output strides \(strides) for \(shape)")
            }
            let offset = stride.multipliedReportingOverflow(by: size - 1)
            let extent = required.addingReportingOverflow(offset.partialValue)
            guard !offset.overflow, !extent.overflow else {
                throw RuntimeError.model("pplx output strides overflow for \(shape)")
            }
            required = extent.partialValue
        }
        return try array.withUnsafeBufferPointer(ofType: Float16.self) { buffer in
            guard buffer.count >= required, let base = buffer.baseAddress else {
                throw RuntimeError.model("pplx output storage is too short for \(shape) and \(strides)")
            }
            return try read(base, strides)
        }
    }
}

/// One pair per row/frequency. Float64 argument generation matches the source
/// model's rotary convention. Multiplication and rotation use Float32.
struct PPLXRotary: Sendable {
    static let frequencies = 64
    let cosine: [Float]
    let sine: [Float]

    init(rows: Int) {
        let width = Self.frequencies
        let inverse = (0..<width).map { pow(1_000_000.0, -Double($0) / Double(width)) }
        var phase = [Double](repeating: 0, count: rows * width)
        inverse.withUnsafeBufferPointer { inv in
            phase.withUnsafeMutableBufferPointer { out in
                for row in 1..<max(rows, 1) {
                    var position = Double(row)
                    vDSP_vsmulD(inv.baseAddress!, 1, &position, out.baseAddress! + row * width, 1, vDSP_Length(width))
                }
            }
        }
        var count = Int32(phase.count)
        var sin = [Double](repeating: 0, count: phase.count)
        var cos = sin
        vvsincos(&sin, &cos, phase, &count)
        sine = sin.map(Float.init)
        cosine = cos.map(Float.init)
    }

    func rotate(first: Float, second: Float, row: Int, frequency: Int, scale: Float) -> (Float16, Float16) {
        let i = row * Self.frequencies + frequency
        return (Float16((first * cosine[i] - second * sine[i]) * scale),
                Float16((second * cosine[i] + first * sine[i]) * scale))
    }
}

struct PPLXLayerTiming: Sendable {
    var pre: Double = 0
    var attention: Double = 0
    var post: Double = 0
    var host: Double = 0

    static func seconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) + Double(parts.attoseconds) * 1e-18
    }
}

enum PPLXDecomposition {
    static let tileRows = 2048
    static let headWidth = 128
    static let queryHeads = 16
    static let keyHeads = 8
    static let hidden = PPLXTokenTable.width
    static let layers = 28
    private static let queryWidth = queryHeads * headWidth
    private static let keyWidth = keyHeads * headWidth
    private static let queryScale = 1 / sqrt(Float(headWidth))
    private static let epsilon: Float = 1e-6

    static func alpha(_ layer: Int) -> Float {
        switch layer {
        case 0..<10: return 1
        case 10..<19: return 1 / 8
        default: return 1 / 64
        }
    }

    struct Profile: Sendable {
        let vector: [Float]
        let layers: [PPLXLayerTiming]
        let preparationSeconds: Double
        let poolingSeconds: Double
    }

    struct Tile: Sendable {
        let start: Int
        let rows: Int
    }

    /// The residual has one owner. Pre-stage tasks only read it, and each
    /// post-stage task joins before its rows are overwritten. Keeping stable
    /// storage avoids copying the entire residual across task boundaries.
    private final class Residual: @unchecked Sendable {
        let values: UnsafeMutablePointer<Float>

        init(_ ids: [Int32], table: PPLXTokenTable) throws {
            let storage = UnsafeMutablePointer<Float>.allocate(capacity: ids.count * hidden)
            do { try table.gatherFloat(ids, into: storage) }
            catch { storage.deallocate(); throw error }
            values = storage
        }

        deinit { values.deallocate() }
    }

    /// Each slot has one writer and is borrowed by one prediction after its
    /// preparation task joins. The producer and the current prediction use
    /// opposite slots. Core ML can borrow the tensor storage directly.
    private final class ScaledInput: @unchecked Sendable {
        let tensor: MLMultiArray
        private var floatScratch = [Float](repeating: 0, count: tileRows * hidden)

        init() throws { tensor = try PPLXTensor.half([1, tileRows, hidden]) }

        func prepare(_ tile: Tile, residual: Residual, alpha: Float) throws {
            try Task.checkCancellation()
            let count = tile.rows * hidden
            let half = PPLXHostMath.halfPointer(tensor)
            let from = residual.values + tile.start * hidden
            if alpha == 1 {
                try PPLXHostMath.toHalf(from, half, count: count)
            } else {
                var scale = alpha
                try floatScratch.withUnsafeMutableBufferPointer { scratch in
                    vDSP_vsmul(from, 1, &scale, scratch.baseAddress!, 1, vDSP_Length(count))
                    try PPLXHostMath.toHalf(scratch.baseAddress!, half, count: count)
                }
            }
            if count < tileRows * hidden {
                (half + count).update(repeating: 0, count: tileRows * hidden - count)
            }
        }
    }

    /// Storage is local to one admitted long request and reused across all
    /// layers. Only valid tiles get query storage and predictions. The key
    /// tensors retain the model's full fixed shape with a zero/masked tail.
    private final class Workspace {
        let rotary: PPLXRotary
        let tiles: [Tile]
        let scaled: [ScaledInput]
        let queries: [MLMultiArray]
        let key: MLMultiArray
        let value: MLMultiArray
        let bias: MLMultiArray
        let preQ: MLMultiArray
        let preK: MLMultiArray
        let preV: MLMultiArray
        let attention: MLMultiArray
        let postInput: MLMultiArray
        let postA: MLMultiArray
        let postD: MLMultiArray
        // Float32 scratch is reused for Q, K and the two post outputs.
        var first = [Float](repeating: 0, count: tileRows * queryWidth)
        var second = [Float](repeating: 0, count: tileRows * hidden)
        var rotatedHead = [Float](repeating: 0, count: tileRows * headWidth)
        // Contiguous half-head work vectors for vDSP. Results are assembled
        // in head-major order with strided copies, then narrowed once.
        var rotaryScratch = [Float](repeating: 0, count: 6 * tileRows * PPLXRotary.frequencies)

        init(rows: Int, length: Int) throws {
            rotary = PPLXRotary(rows: rows)
            tiles = stride(from: 0, to: rows, by: tileRows).map { Tile(start: $0, rows: min(tileRows, rows - $0)) }
            scaled = try [ScaledInput(), ScaledInput()]
            queries = try tiles.map { _ in try PPLXTensor.half([1, queryHeads, tileRows, headWidth], zero: true) }
            key = try PPLXTensor.half([1, keyHeads, length, headWidth], zero: true)
            value = try PPLXTensor.half([1, keyHeads, length, headWidth], zero: true)
            bias = try PPLXTensor.half([1, 1, 1, length])
            let mask = PPLXHostMath.halfPointer(bias)
            mask.update(repeating: 0, count: rows)
            (mask + rows).update(repeating: -10_000, count: length - rows)
            preQ = try PPLXTensor.half([1, tileRows, queryWidth])
            preK = try PPLXTensor.half([1, tileRows, keyWidth])
            preV = try PPLXTensor.half([1, tileRows, keyWidth])
            attention = try PPLXTensor.half([1, queryHeads, tileRows, headWidth])
            postInput = try PPLXTensor.half([1, tileRows, queryWidth])
            postA = try PPLXTensor.half([1, tileRows, hidden])
            postD = try PPLXTensor.half([1, tileRows, hidden])
        }

        func collect(_ output: MLFeatureProvider, tile index: Int, length: Int) throws {
            let tile = tiles[index]
            let qOutput = try PPLXTensor.output(output, "q")
            let kOutput = try PPLXTensor.output(output, "k")
            let vOutput = try PPLXTensor.output(output, "v")
            let qHeads = PPLXHostMath.halfPointer(queries[index])
            let kHeads = PPLXHostMath.halfPointer(key)
            let vHeads = PPLXHostMath.halfPointer(value)
            try PPLXHostMath.withHalfOutput(qOutput, shape: [1, tileRows, queryWidth]) { q, strides in
                try first.withUnsafeMutableBufferPointer {
                    try PPLXHostMath.toFloat(q, $0.baseAddress!, rows: tile.rows, columns: queryWidth, inputStride: strides[1])
                }
            }
            try rotateHeads(count: queryHeads, destination: qHeads, length: tileRows, start: 0, tile: tile, scale: queryScale)
            try PPLXHostMath.withHalfOutput(kOutput, shape: [1, tileRows, keyWidth]) { k, strides in
                try first.withUnsafeMutableBufferPointer {
                    try PPLXHostMath.toFloat(k, $0.baseAddress!, rows: tile.rows, columns: keyWidth, inputStride: strides[1])
                }
            }
            try rotateHeads(count: keyHeads, destination: kHeads, length: length, start: tile.start, tile: tile, scale: 1)
            try PPLXHostMath.withHalfOutput(vOutput, shape: [1, tileRows, keyWidth]) { v, strides in
                for head in 0..<keyHeads {
                    try PPLXHostMath.copyHalf(v + head * headWidth,
                                              vHeads + (head * length + tile.start) * headWidth,
                                              rows: tile.rows, columns: headWidth, inputStride: strides[1], outputStride: headWidth)
                }
            }
        }

        private func rotateHeads(count: Int, destination: UnsafeMutablePointer<Float16>, length: Int, start: Int,
                                 tile: Tile, scale: Float) throws
        {
            let halfHead = PPLXRotary.frequencies
            let elements = tile.rows * halfHead
            let n = vDSP_Length(elements)
            let columns = vDSP_Length(halfHead)
            let rows = vDSP_Length(tile.rows)
            let inputStride = vDSP_Length(count * headWidth)
            let outputStride = vDSP_Length(headWidth)
            var multiplier = scale
            try rotary.cosine.withUnsafeBufferPointer { cosine in
                try rotary.sine.withUnsafeBufferPointer { sine in
                    let c = cosine.baseAddress! + tile.start * halfHead
                    let s = sine.baseAddress! + tile.start * halfHead
                    try first.withUnsafeBufferPointer { input in
                        try rotaryScratch.withUnsafeMutableBufferPointer { scratch in
                            let a = scratch.baseAddress!
                            let b = a + elements
                            let ac = b + elements
                            let bc = ac + elements
                            let bs = bc + elements
                            let sa = bs + elements
                            try rotatedHead.withUnsafeMutableBufferPointer { head in
                                for index in 0..<count {
                                    let from = input.baseAddress! + index * headWidth
                                    vDSP_mmov(from, a, columns, rows, inputStride, columns)
                                    vDSP_mmov(from + halfHead, b, columns, rows, inputStride, columns)
                                    vDSP_vmul(a, 1, c, 1, ac, 1, n)
                                    vDSP_vmul(b, 1, c, 1, bc, 1, n)
                                    vDSP_vmul(b, 1, s, 1, bs, 1, n)
                                    vDSP_vmul(a, 1, s, 1, sa, 1, n)
                                    vDSP_vsub(bs, 1, ac, 1, a, 1, n)
                                    vDSP_vadd(sa, 1, bc, 1, b, 1, n)
                                    if scale != 1 {
                                        vDSP_vsmul(a, 1, &multiplier, a, 1, n)
                                        vDSP_vsmul(b, 1, &multiplier, b, 1, n)
                                    }
                                    let rotated = head.baseAddress!
                                    vDSP_mmov(a, rotated, columns, rows, columns, outputStride)
                                    vDSP_mmov(b, rotated + halfHead, columns, rows, columns, outputStride)
                                    try PPLXHostMath.toHalf(rotated, destination + (index * length + start) * headWidth,
                                                            count: tile.rows * headWidth)
                                }
                            }
                        }
                    }
                }
            }
        }

        func preparePost(_ output: MLFeatureProvider, tile: Tile) throws {
            let heads = try PPLXTensor.output(output, "o")
            let target = PPLXHostMath.halfPointer(postInput)
            try PPLXHostMath.withHalfOutput(heads, shape: [1, queryHeads, tileRows, headWidth]) { input, strides in
                for head in 0..<queryHeads {
                    try PPLXHostMath.copyHalf(input + head * strides[1], target + head * headWidth,
                                              rows: tile.rows, columns: headWidth,
                                              inputStride: strides[2], outputStride: queryWidth)
                }
            }
            let count = tile.rows * queryWidth
            if count < tileRows * queryWidth {
                (target + count).update(repeating: 0, count: tileRows * queryWidth - count)
            }
        }

        func update(_ output: MLFeatureProvider, tile: Tile, inverseAlpha: Float, residual: Residual) throws {
            let aOutput = try PPLXTensor.output(output, "a")
            let dOutput = try PPLXTensor.output(output, "d")
            let count = vDSP_Length(tile.rows * hidden)
            var scale = inverseAlpha
            try first.withUnsafeMutableBufferPointer { aValues in
                try second.withUnsafeMutableBufferPointer { dValues in
                    let aa = aValues.baseAddress!
                    let dd = dValues.baseAddress!
                    try PPLXHostMath.withHalfOutput(aOutput, shape: [1, tileRows, hidden]) { a, strides in
                        try PPLXHostMath.toFloat(a, aa, rows: tile.rows, columns: hidden, inputStride: strides[1])
                    }
                    try PPLXHostMath.withHalfOutput(dOutput, shape: [1, tileRows, hidden]) { d, strides in
                        try PPLXHostMath.toFloat(d, dd, rows: tile.rows, columns: hidden, inputStride: strides[1])
                    }
                    vDSP_vadd(aa, 1, dd, 1, aa, 1, count)
                    let out = residual.values + tile.start * hidden
                    vDSP_vsma(aa, 1, &scale, out, 1, out, 1, count)
                }
            }
        }
    }

    static func run(_ ids: [Int32], length: Int, runner: any PPLXPredicting,
                    table: PPLXTokenTable, norm: [Float], causal: Bool = false) async throws -> Profile
    {
        guard !ids.isEmpty, ids.count <= length, norm.count == hidden else {
            throw RuntimeError.model("invalid pplx decomposition request for \(length) positions")
        }
        try Task.checkCancellation()
        let start = ContinuousClock.now
        let residual = try Residual(ids, table: table)
        let workspace = try Workspace(rows: ids.count, length: length)
        let preparation = PPLXLayerTiming.seconds(start.duration(to: .now))
        let tiledAttention = "attn_S\(length)_Q\(tileRows)"
        var timings: [PPLXLayerTiming] = []
        timings.reserveCapacity(layers)

        for layer in 0..<layers {
            try Task.checkCancellation()
            let layerStart = ContinuousClock.now
            let scale: Float = causal ? 1 : alpha(layer)
            // Pre projections must all finish before the post sweep updates the residual.
            var timing = PPLXLayerTiming()
            try await withThrowingTaskGroup(of: Void.self) { preparation in
                try workspace.scaled[0].prepare(workspace.tiles[0], residual: residual, alpha: scale)
                for index in workspace.tiles.indices {
                    if index > 0 { _ = try await preparation.next() }
                    let next = index + 1
                    if next < workspace.tiles.count {
                        let nextBuffer = workspace.scaled[next % 2]
                        let tile = workspace.tiles[next]
                        preparation.addTask { try nextBuffer.prepare(tile, residual: residual, alpha: scale) }
                    }
                    try Task.checkCancellation()
                    let before = ContinuousClock.now
                    let projections = try await runner.predict("pre\(layer)",
                                                                inputs: ["xs": workspace.scaled[index % 2].tensor],
                                                                outputs: ["q": workspace.preQ, "k": workspace.preK, "v": workspace.preV])
                    timing.pre += PPLXLayerTiming.seconds(before.duration(to: .now))
                    try workspace.collect(projections, tile: index, length: length)
                }
            }

            // All Q/K/V projections for the layer are now fixed. A query tile
            // can flow straight from attention through post into the residual:
            // neither later query tiles nor keys read the updated rows.
            let input = workspace.scaled[0]
            for (index, tile) in workspace.tiles.enumerated() {
                try Task.checkCancellation()
                async let prepare: Void = input.prepare(tile, residual: residual, alpha: scale)
                let attentionStart = ContinuousClock.now
                if causal {
                    // Store each key's row relative to this query tile. The clamped integers
                    // fit fp16 exactly, including the padding sentinel one past the last row.
                    let mask = PPLXHostMath.halfPointer(workspace.bias)
                    mask.update(repeating: -1, count: tile.start)
                    for row in 0..<tile.rows { mask[tile.start + row] = Float16(row) }
                    let end = tile.start + tile.rows
                    (mask + end).update(repeating: Float16(tileRows), count: length - end)
                }
                let heads = try await runner.predict(tiledAttention,
                                                     inputs: ["q": workspace.queries[index], "k": workspace.key,
                                                              "v": workspace.value, "bias": workspace.bias],
                                                     outputs: ["o": workspace.attention])
                timing.attention += PPLXLayerTiming.seconds(attentionStart.duration(to: .now))
                try workspace.preparePost(heads, tile: tile)
                try await prepare
                let postStart = ContinuousClock.now
                let deltas = try await runner.predict("post\(layer)",
                                                      inputs: ["xs": input.tensor, "o": workspace.postInput],
                                                      outputs: ["a": workspace.postA, "d": workspace.postD])
                timing.post += PPLXLayerTiming.seconds(postStart.duration(to: .now))
                try workspace.update(deltas, tile: tile, inverseAlpha: 1 / scale, residual: residual)
            }
            let wall = PPLXLayerTiming.seconds(layerStart.duration(to: .now))
            timing.host = max(0, wall - timing.pre - timing.attention - timing.post)
            timings.append(timing)
        }

        let poolStart = ContinuousClock.now
        let first = causal ? ids.count - 1 : 0
        let vector = pool(residual.values + first * hidden, validRows: ids.count - first, norm: norm)
        guard vector.allSatisfy(\.isFinite) else { throw RuntimeError.model("nonfinite pplx mean at \(length) positions") }
        return Profile(vector: vector, layers: timings, preparationSeconds: preparation,
                       poolingSeconds: PPLXLayerTiming.seconds(poolStart.duration(to: .now)))
    }

    static func pool(_ residual: UnsafePointer<Float>, validRows: Int, norm: [Float]) -> [Float] {
        var output = [Float](repeating: 0, count: hidden)
        var scratch = output
        let width = vDSP_Length(hidden)
        var mean = 1 / Float(validRows)
        norm.withUnsafeBufferPointer { w in
            output.withUnsafeMutableBufferPointer { result in
                scratch.withUnsafeMutableBufferPointer { row in
                    let out = result.baseAddress!
                    let normalized = row.baseAddress!
                    for i in 0..<validRows {
                        let input = residual + i * hidden
                        var sum: Float = 0
                        vDSP_svesq(input, 1, &sum, width)
                        var inverseRMS = 1 / sqrt(sum / Float(hidden) + epsilon)
                        vDSP_vsmul(input, 1, &inverseRMS, normalized, 1, width)
                        vDSP_vmul(normalized, 1, w.baseAddress!, 1, normalized, 1, width)
                        vDSP_vadd(normalized, 1, out, 1, out, 1, width)
                    }
                    vDSP_vsmul(out, 1, &mean, out, 1, width)
                }
            }
        }
        return output
    }
}
