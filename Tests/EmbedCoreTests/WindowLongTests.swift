import Foundation
import Testing

@testable import EmbedCore

/// Token rows whose first value identifies the id; the rest are zero.
private struct TinyRows: TokenRowSource {
    func gather(_ ids: [Int32], into destination: UnsafeMutablePointer<Float16>) {
        for (position, id) in ids.enumerated() {
            let row = destination + position * BucketContract.hidden
            row.update(repeating: 0, count: BucketContract.hidden)
            row[0] = Float16(Float(id % 97) / 97)
        }
    }
}

/// Band, key bias and pool of a single text of n rows, straight from the contract.
private func expectLayout(_ buffers: LongBuffers, tokens n: Int) {
    let length = buffers.length, columns = WindowContract.bandColumns
    var bad = 0
    for row in 0..<length {
        for column in 0..<columns {
            let key = WindowContract.key(row: row, column: column)
            let open = (row < n && key >= 0 && key < n && abs(row - key) <= BucketContract.window) || key == row
            if buffers[band: row, column] != (open ? 0 : BucketContract.blocked) { bad += 1 }
        }
        if buffers.keyBias[row] != (row < n ? 0 : BucketContract.blocked) { bad += 1 }
        if buffers.pool[row] != (row < n ? WindowContract.poolWeight : 0) { bad += 1 }
    }
    #expect(bad == 0, "n = \(n)")
}

/// Records the calls; the "embedding" is the pool-weighted sum of the first value of each row, so it
/// depends on the token rows and on the pool.
private final class RecordingLong: LongFunctionRunner, @unchecked Sendable {
    let lengths: [Int]
    let lock = NSLock()
    var calls: [Int] = []

    init(lengths: [Int]) { self.lengths = lengths }

    func run(_ buffers: LongBuffers) async throws -> [Float] {
        lock.withLock { calls.append(buffers.length) }
        var sum: Float = 0
        for row in 0..<buffers.length {
            sum += Float(buffers.pool[row]) * Float(buffers.inputsEmbeds[row * BucketContract.hidden])
        }
        return [sum, Float(buffers.tokens)]
    }
}

@Suite("Windowed long texts")
struct WindowLongTests {
    @Test func contract() {
        #expect(WindowContract.bandColumns == 1536)
        #expect(WindowContract.key(row: 0, column: 512) == 0 && WindowContract.key(row: 1023, column: 0) == 0)
        #expect(WindowContract.column(row: 700, key: 700) == 700 - 512 + 512)
        #expect(WindowContract.longLengths(forMaxTokens: 2048).isEmpty)
        #expect(WindowContract.longLengths(forMaxTokens: 2049) == [3072])
        #expect(WindowContract.longLengths(forMaxTokens: 4096) == [3072, 4096])
        #expect(WindowContract.longLengths(forMaxTokens: 4097) == [3072, 4096, 6144])
        #expect(WindowContract.longLengths(forMaxTokens: 8192) == [3072, 4096, 6144, 8192])
        #expect(WindowContract.longName(8192) == "winl8192_b512")
        #expect(CoreMLBucketRunner.lengths(forMaxTokens: 2048) == BucketContract.lengths)
        #expect(CoreMLBucketRunner.lengths(forMaxTokens: 8192) == BucketContract.lengths)
        #expect(CoreMLBucketRunner.lengths(forMaxTokens: 700) == [64, 128, 256, 512, 768])
    }

    @Test func layoutChangesRewriteOnlyWhatChanges() {
        let buffers = LongBuffers(length: 3072)
        expectLayout(buffers, tokens: 0)
        for n in [2049, 2049, 3072, 2500, 1, 3000, 2048] {
            let before = buffers.tokens
            buffers.setTokens(n)
            expectLayout(buffers, tokens: n)
            #expect(buffers.lastRewrittenRows <= (before == n ? 0 : abs(before - n) + WindowContract.halo))
        }
    }

    @Test func eightThousandRowsBoundary() {
        let buffers = LongBuffers(length: 8192)
        for n in [8192, 6145, 8191] {
            buffers.setTokens(n)
            expectLayout(buffers, tokens: n)
        }
    }

    @Test func picksTheSmallestLengthAndZeroesStaleRows() async throws {
        let runner = RecordingLong(lengths: [3072, 4096, 6144, 8192])
        let engine = LongTextEngine(runner: runner, rows: TinyRows())
        for n in [2049, 3072, 3073, 4096, 4097, 6144, 6145, 8192] {
            _ = try await engine.embed(ids: [Int32](repeating: 5, count: n))
        }
        #expect(runner.calls == [3072, 3072, 4096, 4096, 6144, 6144, 8192, 8192])
        await #expect(throws: RuntimeError.self) { _ = try await engine.embed(ids: [Int32](repeating: 5, count: 8193)) }
        // A shorter text in a reused buffer gives the same result as in a fresh one.
        let long = (0..<4000).map { Int32($0 % 500 + 3) }, short = (0..<3100).map { Int32($0 % 300 + 7) }
        _ = try await engine.embed(ids: long)
        let reused = try await engine.embed(ids: short)
        let fresh = try await LongTextEngine(runner: RecordingLong(lengths: [4096]), rows: TinyRows()).embed(ids: short)
        #expect(reused == fresh)
    }

    /// No token ids is an error, not a precondition trap in the buffers.
    @Test func anEmptyTextIsAnError() async {
        let engine = LongTextEngine(runner: RecordingLong(lengths: [3072]), rows: TinyRows())
        await #expect(throws: RuntimeError.self) { _ = try await engine.embed(ids: []) }
    }
}
