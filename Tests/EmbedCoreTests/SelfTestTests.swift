import Foundation
import Testing

@testable import EmbedCore

/// A bucket runner that returns fixed rows (lengths: embed_64 only by default).
struct StubBucketRunner: BucketFunctionRunner {
    var lengths: [Int] = [64]
    let rows: @Sendable () -> [[Float]]

    init(lengths: [Int] = [64], rows: @escaping @Sendable () -> [[Float]]) {
        self.lengths = lengths
        self.rows = rows
    }

    func run(_ buffers: BucketBuffers) async throws -> [[Float]] { rows() }
    func warm() async throws {}
}

@Suite("Self-test data")
struct SelfTestTests {
    /// A two-case suite: little-endian float32 blocks addressed by offset, as the package's selftest.bin stores them.
    func suite() throws -> SelfTestSuite {
        var blob = Data()
        func put(_ values: [Float]) -> JSONValue {
            let offset = blob.count
            for value in values { withUnsafeBytes(of: value.bitPattern.littleEndian) { blob.append(contentsOf: $0) } }
            return ["offset": .int(offset), "shape": values.count == 768 ? [768] : .array([.int(values.count / 8), 8])]
        }
        let digest = put((0..<(512 * 8)).map { Float($0 % 8 == 0 ? 1 : 0) })
        let expected = put([Float](repeating: 0.5, count: 768))
        let json: JSONValue = [
            "format": "aneembed-selftest/1", "digest": digest, "doc_ids": [2, 10, 11, 1],
            "tolerance": ["vision": ["min_cos": 0.999, "max_rel_err": 0.05], "bucket": ["min_cos": 0.999, "zero_rows_max_abs": 0.001]],
            "tests": [["function": "winl3072_b512", "kind": "winl", "doc_rows": 4, "expected": expected]],
        ]
        return try SelfTestSuite(json: json.jsonData(), blob: blob)
    }

    @Test func parsesCasesTolerancesAndBlocks() throws {
        let suite = try suite()
        #expect(suite.cases.count == 1 && suite.cases[0].function == "winl3072_b512" && suite.cases[0].documentRows == 4)
        #expect(suite.documentIDs == [2, 10, 11, 1])
        #expect(suite.tolerance["vision"]?.maxRelativeError == 0.05 && suite.tolerance["bucket"]?.unusedRowsMaxAbs == 0.001)
        #expect(suite.expected(suite.cases[0]) == [Float](repeating: 0.5, count: 768))
        // digest: column 0 of R is all ones, so D[:, 0] is the row sum
        let rows = [Float](repeating: 2, count: 512) + [Float](repeating: 1, count: 512)
        let digest = suite.digest(of: rows, count: 2)
        #expect(digest[0] == 1024 && digest[8] == 512 && digest[1] == 0)
    }

    @Test func rejectsBlocksOutsideTheData() throws {
        let json: JSONValue = ["format": "aneembed-selftest/1", "digest": ["offset": 0, "shape": [512, 8]], "tests": []]
        #expect(throws: (any Error).self) { try SelfTestSuite(json: json.jsonData(), blob: Data(count: 16)) }
    }

    @Test func metrics() {
        let (cosine, error) = SelfTestSuite.metrics([1, 0, 0], [1, 0, 0])
        #expect(cosine == 1 && error == 0)
        let (opposite, _) = SelfTestSuite.metrics([-1, 0], [1, 0])
        #expect(opposite == -1)
    }

    /// The procedural image is exact integer arithmetic: spot values from the formula.
    @Test func patternImage() {
        let image = SelfTestSuite.pattern(columns: 27, rows: 18)
        #expect(image.width == 432 && image.height == 288 && image.tokens == 54)
        func value(_ x: Int, _ y: Int, _ c: Int) -> UInt8 { image.rgb[(y * image.width + x) * 3 + c] }
        #expect(value(0, 0, 0) == 0 && value(0, 0, 2) == 140)
        #expect(value(24, 0, 0) == UInt8((5 * 24 + 90) % 256))
        let corner: Int = (5 * 431 + 3 * 287 + 70 + 90 * ((17 + 11) % 2)) % 256
        #expect(value(431, 287, 1) == UInt8(corner))
    }

    /// A suite with a zero digest and the given tests; `put` stores float32 values in the blob and returns the
    /// `{"offset", "shape"}` block (`shape` defaults to the value count as one dimension).
    func makeSuite(tolerance: JSONValue? = nil, docIDs: [Int] = [],
                   _ tests: (_ put: ([Float], [Int]?) -> JSONValue) -> [JSONValue]) throws -> SelfTestSuite
    {
        var blob = Data()
        func put(_ values: [Float], _ shape: [Int]?) -> JSONValue {
            let offset = blob.count
            for value in values { withUnsafeBytes(of: value.bitPattern.littleEndian) { blob.append(contentsOf: $0) } }
            return ["offset": .int(offset), "shape": .array((shape ?? [values.count]).map { .int($0) })]
        }
        let digest = put([Float](repeating: 0, count: 512 * 8), [512, 8])
        var json: JSONObject = ["format": "aneembed-selftest/1", "digest": digest, "doc_ids": .array(docIDs.map { .int($0) })]
        if let tolerance { json["tolerance"] = tolerance }
        json["tests"] = .array(tests(put))
        return try SelfTestSuite(json: JSONValue.object(json).jsonData(), blob: blob)
    }

    static func ids(_ sequences: [[Int32]]) -> JSONValue { .array(sequences.map { .array($0.map { .int(Int($0)) }) }) }

    /// One embed_64 case: `sequences` with expectation rows `reference` (768 values each).
    func bucketSuite(sequences: [[Int32]], reference: [[Float]],
                     tolerance: JSONValue? = ["bucket": ["min_cos": 0.999, "zero_rows_max_abs": 0.001]]) throws -> SelfTestSuite
    {
        try makeSuite(tolerance: tolerance) { put in
            [["function": "embed_64", "kind": "bucket", "sequences": Self.ids(sequences),
              "expected": put(reference.flatMap { $0 }, [reference.count, 768])]]
        }
    }

    static let unit: [Float] = [1] + [Float](repeating: 0, count: 767)

    /// A function whose rows are NaN (used rows, then unused rows) must fail, never pass on a NaN comparison.
    @Test func nonFiniteOutputFails() async throws {
        let suite = try bucketSuite(sequences: [[2, 10, 1]], reference: [Self.unit])
        let nan = [Float](repeating: .nan, count: 768)
        let zero = [Float](repeating: 0, count: 768)
        let good = await suite.run(buckets: StubBucketRunner { [Self.unit] + [[Float]](repeating: zero, count: 7) }, long: nil,
                                   vision: nil, rows: HashedTokenRows())
        #expect(good.map(\.passed) == [true])
        let allNaN = await suite.run(buckets: StubBucketRunner { [[Float]](repeating: nan, count: 8) }, long: nil, vision: nil,
                                     rows: HashedTokenRows())
        #expect(allNaN.map(\.passed) == [false])
        var unusedNaN = [Self.unit] + [[Float]](repeating: zero, count: 7)
        unusedNaN[3][1] = .nan
        let unused = await suite.run(buckets: StubBucketRunner { [unusedNaN] in unusedNaN }, long: nil, vision: nil, rows: HashedTokenRows())
        #expect(unused.map(\.passed) == [false])
        var unusedInfinity = [Self.unit] + [[Float]](repeating: zero, count: 7)
        unusedInfinity[7][0] = -.infinity
        let infinite = await suite.run(buckets: StubBucketRunner { [unusedInfinity] in unusedInfinity }, long: nil, vision: nil, rows: HashedTokenRows())
        #expect(infinite.map(\.passed) == [false])
    }

    /// Serving is an allowlist: a loaded function the suite has no case for is reported as failed, not skipped.
    @Test func loadedFunctionsWithoutACaseAreNotServable() async throws {
        let suite = try bucketSuite(sequences: [[2, 10, 1]], reference: [Self.unit])
        let zero = [Float](repeating: 0, count: 768)
        let runner = StubBucketRunner(lengths: [64, 128]) { [Self.unit] + [[Float]](repeating: zero, count: 7) }
        let results = await suite.run(buckets: runner, long: nil, vision: nil, rows: HashedTokenRows())
        #expect(results.map(\.function) == ["embed_64", "embed_128"])
        #expect(results.map(\.passed) == [true, false])
        #expect(results[1].error?.contains("no self-test case") == true)
        let report = SelfTestReport(osBuild: "b", ranAt: Date(), seconds: 0, results: results)
        #expect(report.servable == ["embed_64"])
        #expect(report.summary == "1/2 passed; not served: embed_128")
        // A function with two cases is servable only when both pass.
        let twice = SelfTestReport(osBuild: "b", ranAt: Date(), seconds: 0, results: [
            SelfTestResult(function: "embed_64", kind: "bucket", passed: true, cosine: 1),
            SelfTestResult(function: "embed_64", kind: "bucket", passed: false, cosine: 0.5),
        ])
        #expect(twice.servable.isEmpty)
    }

    /// No self-test data, or data that cannot be read, refuses to serve (unless --selftest skip).
    @Test func missingOrUnreadableSelfTestDataIsAnError() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("aneembed-selftest-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let manifest = try Manifest.parse(JSONValue.object(["files": [
            ["path": .string(ModelLayout.selfTestData), "size": 16, "sha256": .string(SHA256Hasher.hex(Data(count: 16)))],
        ]]).jsonData())
        #expect(throws: ServiceError.self) { try SelfTestSuite.require(from: directory, manifest: manifest) }
        do { _ = try SelfTestSuite.require(from: directory, manifest: manifest) } catch { #expect(error.reason == "selftest_missing") }
        try Data("{bad".utf8).write(to: directory.appendingPathComponent(ModelLayout.selfTest))
        try Data(count: 16).write(to: directory.appendingPathComponent(ModelLayout.selfTestData))
        do { _ = try SelfTestSuite.require(from: directory, manifest: manifest) } catch { #expect(error.reason == "invalid_selftest") }
        #expect(try Config(environment: [:]).skipSelfTest == false)
        #expect(try Config(flags: ["selftest": "skip"], environment: [:]).skipSelfTest)
        #expect(throws: ServiceError.self) { try Config(flags: ["selftest": "maybe"], environment: [:]) }
    }

    /// Without a "tolerance" entry a bucket case uses the bucket defaults (no relative error), not the image ones.
    @Test func defaultTolerancesArePerKind() async throws {
        let suite = try bucketSuite(sequences: [[2, 10, 1]], reference: [Self.unit], tolerance: nil)
        let zero = [Float](repeating: 0, count: 768)
        let results = await suite.run(buckets: StubBucketRunner { [Self.unit] + [[Float]](repeating: zero, count: 7) }, long: nil,
                                      vision: nil, rows: HashedTokenRows())
        #expect(results.map(\.passed) == [true])
        // A package can tighten a tolerance, not loosen it, and a limit the kind does not measure is ignored.
        let loose = try bucketSuite(sequences: [[2, 10, 1]], reference: [Self.unit],
                                    tolerance: ["bucket": ["min_cos": 0.5, "zero_rows_max_abs": 10, "max_rel_err": 0.01],
                                                "vision": ["min_cos": 0.9999, "max_rel_err": 0.5]])
        #expect(loose.tolerance(for: "bucket").minCosine == 0.999 && loose.tolerance(for: "bucket").unusedRowsMaxAbs == 0.001)
        #expect(loose.tolerance(for: "bucket").maxRelativeError == nil)
        #expect(loose.tolerance(for: "vision").minCosine == 0.9999 && loose.tolerance(for: "vision").maxRelativeError == 0.05)
        #expect(loose.tolerance(for: "winl").maxRelativeError == nil && loose.tolerance(for: "winl").unusedRowsMaxAbs == nil)
    }

    /// Malformed selftest.json is an invalid_selftest error at load, never an overflow or index trap.
    @Test func rejectsMalformedCases() throws {
        let row = [Float](repeating: 0, count: 768)
        func bucket(_ sequences: [[Int32]], shape: [Int]?, values: Int = 768) -> () throws -> SelfTestSuite {
            { try self.makeSuite { put in
                [["function": "embed_64", "kind": "bucket", "sequences": Self.ids(sequences),
                  "expected": put([Float](repeating: 0, count: values), shape)]]
            } }
        }
        #expect(throws: Never.self, performing: bucket([[2, 1]], shape: [1, 768]))
        #expect(throws: SelfTestError.self, performing: bucket([[2, 1]], shape: [], values: 1))  // empty shape
        #expect(throws: SelfTestError.self, performing: bucket([[2, 1], [2, 1]], shape: [1, 768]))  // fewer rows than sequences
        #expect(throws: SelfTestError.self, performing: bucket([[2, 1]], shape: [1, 512], values: 512))  // wrong width
        #expect(throws: SelfTestError.self, performing: bucket([], shape: [0, 768], values: 0))  // no sequences
        #expect(throws: SelfTestError.self, performing: bucket([[]], shape: [1, 768]))  // an empty sequence
        #expect(throws: SelfTestError.self, performing: bucket([[2, -5, 1]], shape: [1, 768]))  // a negative id
        #expect(throws: SelfTestError.self, performing: bucket([[2, 1]], shape: [1, 768, 1]))
        #expect(throws: SelfTestError.self) {  // more sequences than embed_64 has slots
            try self.makeSuite { put in
                [["function": "embed_64", "kind": "bucket", "sequences": Self.ids([[Int32]](repeating: [2, 1], count: 9)),
                  "expected": put([Float](repeating: 0, count: 9 * 768), [9, 768])]]
            }
        }
        #expect(throws: SelfTestError.self) {  // more tokens than embed_64 holds
            try self.makeSuite { put in
                [["function": "embed_64", "kind": "bucket", "sequences": Self.ids([[Int32](repeating: 7, count: 65)]),
                  "expected": put(row, [1, 768])]]
            }
        }
        // Shapes whose byte size overflows Int, or offsets past the data.
        for block: JSONValue in [["offset": 0, "shape": [.int(1 << 40), .int(1 << 40)]], ["offset": .int(Int.max - 2), "shape": [1, 768]],
                                 ["offset": -4, "shape": [1, 768]], ["offset": 0, "shape": [-1, -768]]]
        {
            #expect(throws: SelfTestError.self) {
                try self.makeSuite { put in
                    _ = put(row, [1, 768])
                    return [["function": "embed_64", "kind": "bucket", "sequences": Self.ids([[2, 1]]), "expected": block]]
                }
            }
        }
        #expect(throws: SelfTestError.self) {  // winl rows beyond the document ids or the function's length
            try self.makeSuite(docIDs: [2, 3, 4]) { put in
                [["function": "winl3072_b512", "kind": "winl", "doc_rows": 4, "expected": put(row, [768])]]
            }
        }
        #expect(throws: SelfTestError.self) {  // a grid whose image does not fit the vision function
            try self.makeSuite { put in
                [["function": "vision_70", "kind": "vision", "grid": [30, 30], "expected": put([Float](repeating: 0, count: 100 * 8), [100, 8])]]
            }
        }
        #expect(throws: SelfTestError.self) {  // a grid that is not whole 48-pixel cells
            try self.makeSuite { put in
                [["function": "vision_70", "kind": "vision", "grid": [10, 9], "expected": put([Float](repeating: 0, count: 9 * 8), [9, 8])]]
            }
        }
        #expect(throws: SelfTestError.self) {  // expectations that do not match the grid's tokens
            try self.makeSuite { put in
                [["function": "vision_70", "kind": "vision", "grid": [27, 18], "expected": put([Float](repeating: 0, count: 8), [1, 8])]]
            }
        }
        #expect(throws: SelfTestError.self) {  // a non-finite tolerance
            try self.makeSuite(tolerance: ["bucket": ["min_cos": 2]]) { _ in [] }
        }
    }

    /// A runner that returns fewer or narrower rows than the case needs fails the function instead of trapping.
    @Test func shortOutputFailsTheFunction() async throws {
        let suite = try bucketSuite(sequences: [[2, 10, 1], [2, 1]], reference: [Self.unit, Self.unit])
        let one = await suite.run(buckets: StubBucketRunner { [Self.unit] }, long: nil, vision: nil, rows: HashedTokenRows())
        #expect(one.map(\.passed) == [false] && one[0].error != nil)
        let narrow = await suite.run(buckets: StubBucketRunner { [[Float]](repeating: [1], count: 8) }, long: nil, vision: nil,
                                     rows: HashedTokenRows())
        #expect(narrow.map(\.passed) == [false] && narrow[0].error != nil)
    }

    /// A load with a smaller --max-tokens tests fewer functions; the records of the others stay (same OS build and
    /// binary), so doctor still shows an earlier failure.
    @Test func reportsMergeWithTheLastRunOfTheSameBuild() {
        let earlier = SelfTestReport(osBuild: "26A", ranAt: Date(timeIntervalSince1970: 1), seconds: 1, results: [
            SelfTestResult(function: "embed_64", kind: "bucket", passed: true, cosine: 1),
            SelfTestResult(function: "winp2048_b512", kind: "winp", passed: false, cosine: 0.5),
        ], binary: "x")
        let now = SelfTestReport(osBuild: "26A", ranAt: Date(timeIntervalSince1970: 2), seconds: 1, results: [
            SelfTestResult(function: "embed_64", kind: "bucket", passed: false, cosine: 0.9),
        ], binary: "x")
        let merged = now.merged(with: earlier)
        #expect(merged.results.map(\.function) == ["embed_64", "winp2048_b512"])
        #expect(merged.results.map(\.passed) == [false, false])
        #expect(now.merged(with: SelfTestReport(osBuild: "26B", ranAt: .distantPast, seconds: 0, results: earlier.results,
                                                binary: "x")).results.count == 1)
        #expect(now.merged(with: SelfTestReport(osBuild: "26A", ranAt: .distantPast, seconds: 0, results: earlier.results,
                                                binary: "y")).results.count == 1)
        #expect(now.merged(with: nil) == now)
    }

    @Test func reportRoundTripsAndSummarizes() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("aneembed-selftest-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let report = SelfTestReport(osBuild: "26A434", ranAt: Date(timeIntervalSince1970: 1_800_000_000), seconds: 1.25, results: [
            SelfTestResult(function: "embed_64", kind: "bucket", passed: true, cosine: 0.99997),
            SelfTestResult(function: "winl8192_b512", kind: "winl", passed: false, cosine: 0.55),
        ])
        try report.save(to: directory)
        let loaded = try #require(SelfTestReport.load(from: directory))
        #expect(loaded.results.map(\.function) == ["embed_64", "winl8192_b512"] && loaded.failed.map(\.function) == ["winl8192_b512"])
        #expect(loaded.summary == "1/2 passed; not served: winl8192_b512")
    }
}
