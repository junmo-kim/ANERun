import Foundation
import Testing

@testable import EmbedCore

/// Straightforward reference: the full matrices for a layout (local is [S, S] here, also for windowed lengths).
private func reference(length: Int, segments: [PackedSegment]) -> (global: [Float16], local: [Float16], pool: [Float16]) {
    let blocked = BucketContract.blocked
    var global = [Float16](repeating: blocked, count: length * length)
    var local = global
    var owner = [Int](repeating: -1, count: length)
    for (slot, segment) in segments.enumerated() {
        for row in segment.offset..<(segment.offset + segment.length) { owner[row] = slot }
    }
    for i in 0..<length {
        for j in 0..<length {
            let same = owner[i] >= 0 && owner[i] == owner[j]
            let padding = owner[i] < 0 && i == j
            if same || padding { global[i * length + j] = 0 }
            if (same && abs(i - j) <= BucketContract.window) || padding { local[i * length + j] = 0 }
        }
    }
    let slots = BucketContract.slots(forLength: length)
    var pool = [Float16](repeating: 0, count: slots * length)
    for (slot, segment) in segments.enumerated() {
        for column in segment.offset..<(segment.offset + segment.length) {
            pool[slot * length + column] = Float16(1 / Float(segment.length))
        }
    }
    return (global, local, pool)
}

private func layout(_ lengths: [Int]) -> [PackedSegment] {
    var offset = 0
    return lengths.enumerated().map { index, length in
        defer { offset += length }
        return PackedSegment(item: index, offset: offset, length: length)
    }
}

@Suite("Bias and pool builders")
struct BucketBufferTests {
    func expectMatches(_ buffers: BucketBuffers, _ segments: [PackedSegment]) {
        let length = buffers.length
        let expected = reference(length: length, segments: segments)
        let global = Array(UnsafeBufferPointer(start: buffers.globalBias, count: length * length))
        let pool = Array(UnsafeBufferPointer(start: buffers.pool, count: buffers.slots * length))
        #expect(global == expected.global)
        #expect(pool == expected.pool)
        if buffers.windowed {
            // band column c of row r is key (r / 512) * 512 - 512 + c; keys outside [0, S) must stay blocked
            var mismatches = 0
            let columns = WindowContract.bandColumns
            for row in 0..<length {
                for column in 0..<columns {
                    let key = WindowContract.key(row: row, column: column)
                    let want = key >= 0 && key < length ? expected.local[row * length + key] : BucketContract.blocked
                    if buffers.slidingBias[row * columns + column] != want { mismatches += 1 }
                }
            }
            #expect(mismatches == 0)
            // the band holds every open key of the local mask
            let open = expected.local.reduce(0) { $0 + ($1 == 0 ? 1 : 0) }
            let bandOpen = (0..<(length * columns)).reduce(0) { $0 + (buffers.slidingBias[$1] == 0 ? 1 : 0) }
            #expect(open == bandOpen)
        } else {
            let local = Array(UnsafeBufferPointer(start: buffers.slidingBias, count: length * length))
            #expect(local == expected.local)
        }
        // no fully masked row
        for row in 0..<length {
            #expect((0..<length).contains { global[row * length + $0] == 0 })
        }
    }

    @Test func smallBucketsShareOneBiasBuffer() {
        #expect(BucketBuffers(length: 512).localSharesGlobal)
        #expect(!BucketBuffers(length: 1024).localSharesGlobal)
    }

    @Test func incrementalFillsMatchTheReference() {
        let buffers = BucketBuffers(length: 128)
        for lengths in [[10], [3, 40, 5], [128], [1], [8, 8, 8, 8, 8, 8, 8, 8], []] {
            let segments = layout(lengths)
            buffers.setLayout(segments)
            expectMatches(buffers, segments)
        }
    }

    @Test func slidingWindowBandMatchesTheReference() {
        let buffers = BucketBuffers(length: 1024)
        for lengths in [[1000], [600, 300], [1024], [2, 700]] {
            let segments = layout(lengths)
            buffers.setLayout(segments)
            expectMatches(buffers, segments)
        }
    }

    @Test func windowedBandMatchesTheReference() {
        let buffers = BucketBuffers(length: 1536)
        #expect(buffers.windowed && buffers.slidingColumns == 1536 && !buffers.localSharesGlobal)
        for lengths in [[1300, 37, 150], [1536], [2, 700, 513, 1], [10], [600, 600, 300], []] {
            let segments = layout(lengths)
            buffers.setLayout(segments)
            expectMatches(buffers, segments)
        }
    }

    @Test func preparesTokenRowsAndZeroesStalePadding() throws {
        let (data, _) = TokenTableTests.table()
        let table = try TokenTable(data: data)
        let buffers = BucketBuffers(length: 64)
        buffers.prepare(PackedCall(length: 64, segments: layout([3, 2])), ids: [[1, 2, 3], [0, 1]], rows: table)
        let row4 = Array(UnsafeBufferPointer(start: buffers.embedsRow(4), count: TokenTable.width))
        #expect(row4.map(Float.init) == table.row(1))
        buffers.prepare(PackedCall(length: 64, segments: layout([2])), ids: [[3, 3]], rows: table)
        let stale = UnsafeBufferPointer(start: buffers.embedsRow(2), count: 3 * TokenTable.width)
        #expect(stale.allSatisfy { $0 == 0 })
    }

    @Test func poolReusesBuffers() {
        let pool = BucketBufferPool(maxIdlePerLength: 1)
        let first = pool.checkout(length: 64)
        pool.checkin(first)
        #expect(pool.checkout(length: 64) === first)
        pool.checkin(first)
        pool.checkin(BucketBuffers(length: 64))
        #expect(pool.idleCount(length: 64) == 1)
    }
}
