@preconcurrency import CoreML
import Foundation
import Testing

@testable import EmbedCore

@Suite("Vision buffers and soft token extraction")
struct VisionBufferTests {
    private static func syntheticPositionTable() -> Data {
        let count = 2 * CoreMLVisionEncoder.tableRows * CoreMLVisionEncoder.patchWidth
        var values = [Float16](repeating: 0, count: count)
        for i in 0..<count {
            values[i] = Float16(Float((i % 251) - 125) * 0.01)
        }
        return values.withUnsafeBytes { Data($0) }
    }

    private static func syntheticImage(width: Int, height: Int, seed: Int) -> PreparedImage {
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0..<rgb.count {
            rgb[i] = UInt8((i &* 31 &+ seed &* 17) & 0xff)
        }
        return PreparedImage(rgb: rgb, width: width, height: height)
    }

    private static func expectBitExact(_ a: UnsafeMutablePointer<Float16>, _ b: UnsafeMutablePointer<Float16>, count: Int) {
        var mismatches = 0
        for i in 0..<count where a[i].bitPattern != b[i].bitPattern {
            mismatches += 1
        }
        #expect(mismatches == 0)
    }

    @Test func pixelLUTMatchesScalarFormula() {
        for value in 0..<256 {
            let expected = Float16(Float(value) / 255)
            #expect(VisionBuffers.pixelLUT[value].bitPattern == expected.bitPattern)
        }
    }

    @Test func pooledBuffersMatchFreshAcrossGeometries() throws {
        let table = Self.syntheticPositionTable()
        let budget = 70
        let total = budget * 9
        let patchWidth = CoreMLVisionEncoder.patchWidth
        let reused = try VisionBuffers(budget: budget)

        let cases: [(Int, Int, Int)] = [
            (480, 336, 1),  // 70 tokens (full budget)
            (240, 192, 2),  // 20 tokens (shrink: must clear stale patches/pool)
            (240, 192, 3),  // 20 tokens (same geometry, different pixels: cached geometry path)
            (192, 240, 4),  // 20 tokens (transposed geometry, same patch count: must refresh geometry/pool)
            (480, 144, 5),  // 30 tokens (grow again)
        ]

        for (width, height, seed) in cases {
            let image = Self.syntheticImage(width: width, height: height, seed: seed)
            _ = reused.fill(image, positionTable: table)
            let fresh = try VisionBuffers(budget: budget)
            _ = fresh.fill(image, positionTable: table)
            Self.expectBitExact(reused.patches, fresh.patches, count: total * patchWidth)
            Self.expectBitExact(reused.positions, fresh.positions, count: total * 2)
            Self.expectBitExact(reused.embeddings, fresh.embeddings, count: total * patchWidth)
            Self.expectBitExact(reused.valid, fresh.valid, count: total)
            Self.expectBitExact(reused.pool, fresh.pool, count: budget * total)
        }
    }

    @Test func poolReusesBuffers() throws {
        let pool = VisionBufferPool(maxIdlePerBudget: 1)
        let first = try pool.checkout(budget: 70)
        pool.checkin(first)
        let second = try pool.checkout(budget: 70)
        #expect(first === second)
    }

    @Test func softTokensDirectCopyMatchesReferenceForFloat16AndFloat32() throws {
        let width = BucketContract.hidden
        let rowsCount = 5
        let f16 = try MLMultiArray(shape: [1, NSNumber(value: rowsCount + 2), NSNumber(value: width)], dataType: .float16)
        let f16Ptr = f16.dataPointer.assumingMemoryBound(to: Float16.self)
        let f32 = try MLMultiArray(shape: [1, NSNumber(value: rowsCount + 2), NSNumber(value: width)], dataType: .float32)
        let f32Ptr = f32.dataPointer.assumingMemoryBound(to: Float.self)
        var expected = [Float16](repeating: 0, count: rowsCount * width)
        for r in 0..<(rowsCount + 2) {
            for c in 0..<width {
                let val = Float16(Float((r * 7 + c * 3) % 100 - 50) * 0.125)
                f16Ptr[r * width + c] = val
                f32Ptr[r * width + c] = Float(val)
                if r < rowsCount { expected[r * width + c] = val }
            }
        }
        let got16 = try CoreMLVisionEncoder.extractSoftTokens(f16, count: rowsCount)
        let got32 = try CoreMLVisionEncoder.extractSoftTokens(f32, count: rowsCount)
        #expect(got16.map(\.bitPattern) == expected.map(\.bitPattern))
        #expect(got32.map(\.bitPattern) == expected.map(\.bitPattern))
    }
}
