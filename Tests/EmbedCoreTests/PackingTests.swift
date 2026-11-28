import Foundation
import Testing

@testable import EmbedCore

@Suite("Packing planner")
struct PackingPlannerTests {
    let planner = PackingPlanner()

    func check(_ counts: [Int], _ calls: [PackedCall]) {
        var seen = [Int](repeating: 0, count: counts.count)
        for call in calls {
            #expect(BucketContract.lengths.contains(call.length))
            #expect(call.usedTokens <= call.length)
            #expect(call.segments.count <= call.slots)
            var offset = 0
            for segment in call.segments {
                #expect(segment.offset == offset)
                #expect(segment.length == counts[segment.item])
                offset += segment.length
                seen[segment.item] += 1
            }
            // the bucket is the smallest that holds the call
            #expect(planner.bucket(tokens: call.usedTokens, texts: call.segments.count) == call.length)
        }
        #expect(seen.allSatisfy { $0 == 1 })
    }

    @Test func singleShortTextTakesTheSmallestBucket() {
        let calls = planner.plan([7])
        #expect(calls == [PackedCall(length: 64, segments: [PackedSegment(item: 0, offset: 0, length: 7)])])
    }

    @Test func longTextTakesTheBucketThatFits() {
        #expect(planner.plan([2048]).map(\.length) == [2048])
        #expect(planner.plan([1281]).map(\.length) == [1536])
        #expect(planner.plan([1280]).map(\.length) == [1280])
        #expect(planner.plan([769]).map(\.length) == [1024])
        #expect(planner.plan([513]).map(\.length) == [768])
        #expect(planner.plan([65]).map(\.length) == [128])
        #expect(BucketContract.functionName(forLength: 1536) == "winp1536_b512")
        #expect(BucketContract.functionName(forLength: 1280) == "embed_1280")
    }

    @Test func packsShortQueriesAndRespectsSlots() {
        let counts = [Int](repeating: 5, count: 20)  // 8 slots per 64 bucket
        let calls = planner.plan(counts)
        check(counts, calls)
        #expect(calls.count == 3)
    }

    @Test(arguments: [1, 2, 3, 17, 100, 500])
    func coversRandomBatchesExactlyOnce(_ size: Int) {
        var state: UInt64 = UInt64(size) &* 0x9E37_79B9_7F4A_7C15
        var counts: [Int] = []
        for _ in 0..<size {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let r = Int(state >> 33)
            counts.append(r % 10 == 0 ? 1 + r % 2048 : 3 + r % 60)
        }
        let calls = planner.plan(counts)
        check(counts, calls)
        // never worse than one call per text
        let naive = counts.reduce(0.0) { $0 + planner.cost(ofLength: planner.bucket(tokens: $1)!) }
        let total = calls.reduce(0.0) { $0 + planner.cost(ofLength: $1.length) }
        #expect(total <= naive)
    }

    @Test func followsTheCostModel() {
        // When big calls are cheap, short texts go into one large call.
        let flat = PackingPlanner(costs: Dictionary(uniqueKeysWithValues: BucketContract.lengths.map { ($0, 10.0) }))
        let counts = [Int](repeating: 30, count: 40)
        #expect(flat.plan(counts).count == 1)
        #expect(planner.plan(counts).count > 1)
    }
}
