// The load-time self-test: every loaded function runs once on golden inputs from the model package
// (selftest.json + selftest.bin, fp32 expectations from the original model) before it may serve.
//
// Why: the Neural Engine compiler can silently miscompile a graph (a constant attention band once came
// out wrong with no error), and the compile happens on the user's Mac, per macOS build. A function whose
// output does not match is not used; doctor shows which one and why.
//
// Inputs are rebuilt from the package's own files as selftest.json's "procedure" field describes:
//   bucket embed_S     the listed sequences packed into one call: cosine per row, unused rows ~0
//   winp winp{S}_b512  the same through the band buffers
//   winl winl{S}_b512  the first doc_rows document ids as one text: cosine of the embedding
//   vision_{B}         a procedural image (integer formula): digest of the image tokens
// A digest is D = rows @ R with R [512, 8] from selftest.bin; metrics are cosine and |Da - Dr| / |Dr|.
import Accelerate
import Foundation

public struct SelfTestResult: Sendable, Equatable {
    public var function: String
    public var kind: String
    public var passed: Bool
    public var cosine: Double
    public var relativeError: Double?
    public var unusedRowsMaxAbs: Double?
    public var error: String?

    public var json: JSONValue {
        var object: JSONObject = ["function": .string(function), "kind": .string(kind), "pass": .bool(passed),
                                  "cos": .number((cosine * 1e6).rounded() / 1e6)]
        if let relativeError { object["rel_err"] = .number((relativeError * 1e5).rounded() / 1e5) }
        if let unusedRowsMaxAbs { object["unused_max_abs"] = .number(unusedRowsMaxAbs) }
        if let error { object["error"] = .string(error) }
        return .object(object)
    }
}

/// What the last load found, kept next to the model for `doctor`.
public struct SelfTestReport: Sendable, Equatable {
    public var osBuild: String
    public var ranAt: Date
    public var seconds: Double
    public var results: [SelfTestResult]
    /// Whether the package carried self-test data at all.
    public var available: Bool
    /// SHA-256 of the executable that ran it (the Neural Engine compile is per executable).
    public var binary: String

    public var failed: [SelfTestResult] { results.filter { !$0.passed } }

    /// The functions that may serve: every one of their results passed (an allowlist; a function without a
    /// result is not in it).
    public var servable: Set<String> {
        Set(results.map(\.function)).subtracting(failed.map(\.function))
    }

    public init(osBuild: String, ranAt: Date, seconds: Double, results: [SelfTestResult], available: Bool = true,
                binary: String = Executable.identity)
    {
        self.osBuild = osBuild
        self.ranAt = ranAt
        self.seconds = seconds
        self.results = results
        self.available = available
        self.binary = binary
    }

    /// This run plus the records of `previous` for functions this run did not test, when both ran on the same
    /// macOS build with the same executable (a load with a smaller --max-tokens tests fewer functions, and an
    /// earlier failure must stay visible to doctor).
    public func merged(with previous: SelfTestReport?) -> SelfTestReport {
        guard let previous, previous.osBuild == osBuild, previous.binary == binary, previous.available else { return self }
        let tested = Set(results.map(\.function))
        var merged = self
        merged.results += previous.results.filter { !tested.contains($0.function) }
        return merged
    }

    public var summary: String {
        guard available else { return "unavailable: the package has no self-test data" }
        let failed = failed
        return failed.isEmpty
            ? "\(results.count)/\(results.count) passed"
            : "\(results.count - failed.count)/\(results.count) passed; not served: \(failed.map(\.function).joined(separator: ", "))"
    }

    public var json: JSONValue {
        [
            "os_build": .string(osBuild), "ran_at": .string(ISO8601DateFormatter().string(from: ranAt)),
            "seconds": .number((seconds * 100).rounded() / 100), "summary": .string(summary),
            "available": .bool(available), "binary_sha256": .string(binary), "results": .array(results.map(\.json)),
        ]
    }

    public func save(to directory: URL) throws {
        try (json.render(as: .json) + "\n").write(to: directory.appendingPathComponent(ModelLayout.selfTestState),
                                                  atomically: true, encoding: .utf8)
    }

    public static func load(from directory: URL) -> SelfTestReport? {
        guard let data = try? SmallFile.read(directory.appendingPathComponent(ModelLayout.selfTestState), limit: ModelLayout.stateLimit),
            let root = try? JSONValue.parse(data)
        else { return nil }
        return parse(root)
    }

    static func parse(_ root: JSONValue) -> SelfTestReport? {
        guard let build = root["os_build"]?.stringValue else { return nil }
        let results = (root["results"]?.arrayValue ?? []).compactMap { item -> SelfTestResult? in
            guard let function = item["function"]?.stringValue, let passed = item["pass"]?.boolValue else { return nil }
            return SelfTestResult(function: function, kind: item["kind"]?.stringValue ?? "", passed: passed, cosine: item["cos"]?.doubleValue ?? 0,
                                  relativeError: item["rel_err"]?.doubleValue, unusedRowsMaxAbs: item["unused_max_abs"]?.doubleValue,
                                  error: item["error"]?.stringValue)
        }
        return SelfTestReport(osBuild: build,
                              ranAt: root["ran_at"]?.stringValue.flatMap { ISO8601DateFormatter().date(from: $0) } ?? .distantPast,
                              seconds: root["seconds"]?.doubleValue ?? 0, results: results,
                              available: root["available"]?.boolValue ?? true, binary: root["binary_sha256"]?.stringValue ?? "")
    }
}

public enum SelfTestError: Error, CustomStringConvertible {
    case invalid(String)
    public var description: String {
        switch self { case .invalid(let message): return "selftest.json: \(message)" }
    }
}

/// selftest.json + selftest.bin of a model package.
public struct SelfTestSuite: Sendable {
    public struct Case: Sendable {
        public var function: String
        public var kind: String
        public var sequences: [[Int32]]
        public var documentRows: Int
        public var grid: (columns: Int, rows: Int)
        public var expectedOffset: Int
        public var expectedShape: [Int]
    }

    public struct Tolerance: Sendable {
        public var minCosine: Double
        public var maxRelativeError: Double?
        public var unusedRowsMaxAbs: Double?
    }

    public let cases: [Case]
    public let documentIDs: [Int32]
    public let tolerance: [String: Tolerance]
    let blob: Data
    let digest: [Float]  // [512, 8]
    static let digestWidth = 8

    /// The package's suite. Without one nothing can be checked, so loading stops (`--selftest skip` is the
    /// explicit, unsafe way around it). Both files must be regular files: selftest.json of at most
    /// ModelLayout.selfTestLimit bytes, selftest.bin of at most the size `manifest` declares for it.
    public static func require(from directory: URL, manifest: Manifest) throws(ServiceError) -> SelfTestSuite {
        let jsonURL = directory.appendingPathComponent(ModelLayout.selfTest)
        let binURL = directory.appendingPathComponent(ModelLayout.selfTestData)
        guard FileManager.default.fileExists(atPath: jsonURL.path), FileManager.default.fileExists(atPath: binURL.path) else {
            throw ServiceError(.unavailable, "selftest_missing",
                               "the package in \(directory.path) has no self-test data (\(ModelLayout.selfTest), "
                               + "\(ModelLayout.selfTestData)): its functions cannot be checked on this Mac",
                               hint: "reinstall with `anerun models install`; --selftest skip serves unchecked (unsafe)")
        }
        func invalid(_ message: String) -> ServiceError {
            ServiceError(.unavailable, "invalid_selftest", "cannot read the package's self-test: \(message)",
                         hint: "check the package with `anerun models verify`, or reinstall it")
        }
        guard let declared = manifest.files.first(where: { $0.path == ModelLayout.selfTestData })?.size else {
            throw invalid("\(ModelLayout.selfTestData) is not listed in \(ModelLayout.manifest)")
        }
        let json: Data, blob: Data
        do {
            json = try SmallFile.read(jsonURL, limit: ModelLayout.selfTestLimit)
        } catch {
            throw invalid(error.describe(ModelLayout.selfTest))
        }
        do {
            blob = try SmallFile.read(binURL, limit: declared)
        } catch {
            throw invalid(error.describe(ModelLayout.selfTestData))
        }
        do {
            return try SelfTestSuite(json: json, blob: blob)
        } catch {
            throw invalid("\(error)")
        }
    }

    /// Built-in tolerance of each kind. A package may tighten it, never loosen it, and a limit a kind does not
    /// measure (relative error outside vision, unused rows outside packing) does not apply.
    static let defaultTolerance: [String: Tolerance] = [
        "bucket": Tolerance(minCosine: 0.999, maxRelativeError: nil, unusedRowsMaxAbs: 1e-3),
        "winp": Tolerance(minCosine: 0.999, maxRelativeError: nil, unusedRowsMaxAbs: 1e-3),
        "winl": Tolerance(minCosine: 0.999, maxRelativeError: nil, unusedRowsMaxAbs: nil),
        "vision": Tolerance(minCosine: 0.999, maxRelativeError: 0.05, unusedRowsMaxAbs: nil),
    ]

    /// The tolerance a case of `kind` is held to: the stricter of the package's value and the built-in one.
    public func tolerance(for kind: String) -> Tolerance {
        let base = Self.defaultTolerance[kind] ?? Tolerance(minCosine: 0.999, maxRelativeError: nil, unusedRowsMaxAbs: nil)
        let declared = tolerance[kind]
        func stricter(_ limit: Double?, _ declared: Double?) -> Double? { limit.map { min($0, declared ?? $0) } }
        return Tolerance(minCosine: max(base.minCosine, declared?.minCosine ?? 0),
                         maxRelativeError: stricter(base.maxRelativeError, declared?.maxRelativeError),
                         unusedRowsMaxAbs: stricter(base.unusedRowsMaxAbs, declared?.unusedRowsMaxAbs))
    }

    /// Largest dimension of a data block and of a document; keeps every product far from Int overflow.
    static let maxDimension = 1 << 20

    /// Parses and checks every field the run indexes or multiplies with, so a malformed file is an error here
    /// and never an overflow or an out-of-bounds trap later.
    public init(json: Data, blob: Data) throws {
        let root = try JSONValue.parse(json)
        guard root["format"]?.stringValue == "aneembed-selftest/1" else { throw SelfTestError.invalid("unknown format") }
        self.blob = blob
        func block(_ value: JSONValue?, _ name: String) throws -> (Int, [Int]) {
            guard let offset = value?["offset"]?.intValue, let items = value?["shape"]?.arrayValue, offset >= 0,
                !items.isEmpty, items.count <= 3
            else { throw SelfTestError.invalid("\(name): bad data block") }
            var count = 4
            var shape: [Int] = []
            for item in items {
                guard let dimension = item.intValue, (1...Self.maxDimension).contains(dimension) else {
                    throw SelfTestError.invalid("\(name): bad shape")
                }
                let (product, overflow) = count.multipliedReportingOverflow(by: dimension)
                guard !overflow else { throw SelfTestError.invalid("\(name): shape too large") }
                count = product
                shape.append(dimension)
            }
            let (end, overflow) = offset.addingReportingOverflow(count)
            guard !overflow, end <= blob.count else { throw SelfTestError.invalid("\(name): block outside \(ModelLayout.selfTestData)") }
            return (offset, shape)
        }
        func ids(_ value: JSONValue?, _ name: String) throws -> [Int32] {
            guard let items = value?.arrayValue, items.count <= Self.maxDimension else { throw SelfTestError.invalid("\(name): not a list") }
            return try items.map { item in
                guard let id = item.intValue, let id32 = Int32(exactly: id), id32 >= 0 else {
                    throw SelfTestError.invalid("\(name): bad token id")
                }
                return id32
            }
        }
        let (digestOffset, digestShape) = try block(root["digest"], "digest")
        guard digestShape == [BucketContract.hidden, Self.digestWidth] else { throw SelfTestError.invalid("digest shape \(digestShape)") }
        digest = Self.floats(blob, offset: digestOffset, count: BucketContract.hidden * Self.digestWidth)
        documentIDs = root["doc_ids"] == nil ? [] : try ids(root["doc_ids"], "doc_ids")
        var tolerance: [String: Tolerance] = [:]
        for (kind, value) in root["tolerance"]?.objectValue?.pairs ?? [] {
            func limit(_ key: String) throws -> Double? {
                guard let item = value[key] else { return nil }
                guard let number = item.doubleValue, number.isFinite, number >= 0 else {
                    throw SelfTestError.invalid("tolerance \(kind).\(key)")
                }
                return number
            }
            guard let minCosine = try limit("min_cos") else { continue }
            guard minCosine <= 1 else { throw SelfTestError.invalid("tolerance \(kind).min_cos") }
            tolerance[kind] = Tolerance(minCosine: minCosine, maxRelativeError: try limit("max_rel_err"),
                                        unusedRowsMaxAbs: try limit("zero_rows_max_abs"))
        }
        self.tolerance = tolerance
        var cases: [Case] = []
        for test in root["tests"]?.arrayValue ?? [] {
            guard let function = test["function"]?.stringValue, let kind = test["kind"]?.stringValue else {
                throw SelfTestError.invalid("test without function or kind")
            }
            let (offset, shape) = try block(test["expected"], function)
            var item = Case(function: function, kind: kind, sequences: [], documentRows: 0, grid: (0, 0),
                            expectedOffset: offset, expectedShape: shape)
            switch kind {
            case "bucket", "winp":
                guard let length = BucketContract.lengths.first(where: { BucketContract.functionName(forLength: $0) == function }),
                    BucketContract.isWindowed(length) == (kind == "winp"), let list = test["sequences"]?.arrayValue
                else { throw SelfTestError.invalid("\(function): not a \(kind) function or no sequences") }
                item.sequences = try list.map { try ids($0, function) }
                let total = item.sequences.reduce(0) { $0 + $1.count }
                guard !item.sequences.isEmpty, item.sequences.count <= BucketContract.slots(forLength: length),
                    item.sequences.allSatisfy({ !$0.isEmpty }), total <= length
                else { throw SelfTestError.invalid("\(function): sequences do not fit \(length) rows") }
                guard shape == [item.sequences.count, BucketContract.dimension] else {
                    throw SelfTestError.invalid("\(function): expected shape \(shape)")
                }
            case "winl":
                guard let length = WindowContract.longLengths.first(where: { WindowContract.longName($0) == function }),
                    let rows = test["doc_rows"]?.intValue, rows >= 1, rows <= min(length, documentIDs.count)
                else { throw SelfTestError.invalid("\(function): bad doc_rows") }
                guard shape == [BucketContract.dimension] else { throw SelfTestError.invalid("\(function): expected shape \(shape)") }
                item.documentRows = rows
            case "vision":
                let grid = test["grid"]?.arrayValue?.compactMap(\.intValue) ?? []
                guard let budget = Int(function.dropFirst("vision_".count)), function.hasPrefix("vision_"),
                    CoreMLVisionEncoder.allBudgets.contains(budget), grid.count == 2,
                    grid.allSatisfy({ $0 >= 3 && $0 <= 3 * budget && $0 % 3 == 0 })
                else { throw SelfTestError.invalid("\(function): bad grid") }
                let tokens = (grid[0] / 3) * (grid[1] / 3)
                guard tokens <= budget else { throw SelfTestError.invalid("\(function): grid of \(tokens) tokens") }
                guard shape == [tokens, Self.digestWidth] else { throw SelfTestError.invalid("\(function): expected shape \(shape)") }
                item.grid = (grid[0], grid[1])
            default:
                break  // a kind this build does not know: its function has no case and is not served
            }
            cases.append(item)
        }
        self.cases = cases
    }

    static func floats(_ data: Data, offset: Int, count: Int) -> [Float] {
        data.withUnsafeBytes { raw in
            (0..<count).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset + $0 * 4, as: UInt32.self))) }
        }
    }

    func expected(_ test: Case) -> [Float] {
        Self.floats(blob, offset: test.expectedOffset, count: test.expectedShape.reduce(1, *))
    }

    /// rows [count, 512] @ digest [512, 8]
    func digest(of rows: [Float], count: Int) -> [Float] {
        var output = [Float](repeating: 0, count: count * Self.digestWidth)
        vDSP_mmul(rows, 1, digest, 1, &output, 1, vDSP_Length(count), vDSP_Length(Self.digestWidth), vDSP_Length(BucketContract.hidden))
        return output
    }

    static func metrics(_ actual: [Float], _ reference: [Float]) -> (cosine: Double, relativeError: Double) {
        var dot = 0.0, aa = 0.0, bb = 0.0, diff = 0.0
        for (a, b) in zip(actual, reference) {
            let x = Double(a), y = Double(b)
            dot += x * y
            aa += x * x
            bb += y * y
            diff += (x - y) * (x - y)
        }
        let cosine = aa > 0 && bb > 0 ? dot / (aa.squareRoot() * bb.squareRoot()) : 0
        return (cosine.isFinite ? cosine : 0, bb > 0 ? (diff / bb).squareRoot() : .infinity)
    }

    /// The procedural test image: value(X, Y, c) = (5X + 3Y + 70c + 90 * ((X / 24 + Y / 24) % 2)) % 256.
    public static func pattern(columns: Int, rows: Int) -> PreparedImage {
        let width = columns * 16, height = rows * 16
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0..<height {
            for x in 0..<width {
                for c in 0..<3 {
                    rgb[(y * width + x) * 3 + c] = UInt8((5 * x + 3 * y + 70 * c + 90 * ((x / 24 + y / 24) % 2)) % 256)
                }
            }
        }
        return PreparedImage(rgb: rgb, width: width, height: height)
    }

    /// Runs the case of every function that is loaded. A loaded function without a case gets a failed result
    /// (it is not served), so the report covers every loaded function.
    public func run(buckets: (any BucketFunctionRunner)?, long: (any LongFunctionRunner)?, vision: (any VisionEncoder)?,
                    visionBudgets: [Int] = [], rows: any TokenRowSource) async -> [SelfTestResult]
    {
        var results: [SelfTestResult] = []
        for test in cases {
            let tolerance = self.tolerance(for: test.kind)
            var result = SelfTestResult(function: test.function, kind: test.kind, passed: false, cosine: 0)
            do {
                switch test.kind {
                case "bucket", "winp":
                    guard let buckets, let length = buckets.lengths.first(where: {
                        BucketContract.functionName(forLength: $0) == test.function
                    }) else { continue }
                    try await runBucket(test, length: length, runner: buckets, rows: rows, into: &result)
                case "winl":
                    guard let long, let length = long.lengths.first(where: { WindowContract.longName($0) == test.function })
                    else { continue }
                    try await runLong(test, length: length, runner: long, rows: rows, into: &result)
                case "vision":
                    guard let vision, let budget = Int(test.function.dropFirst("vision_".count)),
                        visionBudgets.contains(budget)
                    else { continue }
                    let image = Self.pattern(columns: test.grid.columns, rows: test.grid.rows)
                    let tokens = try await vision.encode(image)
                    let valid = test.expectedShape[0]
                    guard tokens.count >= valid * BucketContract.hidden else { throw RuntimeError.model("too few image tokens") }
                    let actual = digest(of: tokens.prefix(valid * BucketContract.hidden).map(Float.init), count: valid)
                    (result.cosine, result.relativeError) = Self.metrics(actual, expected(test))
                default:
                    continue
                }
                result.passed = Self.passes(result, tolerance)
            } catch {
                result.error = "\(error)"
            }
            results.append(result)
        }
        let loaded = (buckets?.lengths.map { (BucketContract.functionName(forLength: $0), BucketContract.isWindowed($0) ? "winp" : "bucket") } ?? [])
            + (long?.lengths.map { (WindowContract.longName($0), "winl") } ?? [])
            + (vision == nil ? [] : visionBudgets.map { ("vision_\($0)", "vision") })
        let tested = Set(results.map(\.function))
        for (function, kind) in loaded where !tested.contains(function) {
            results.append(SelfTestResult(function: function, kind: kind, passed: false, cosine: 0,
                                          error: "no self-test case in the package"))
        }
        return results
    }

    /// Every metric must be finite and inside the tolerance: a NaN or an infinity anywhere fails, never passes
    /// through a comparison that is false for NaN.
    static func passes(_ result: SelfTestResult, _ tolerance: Tolerance) -> Bool {
        guard result.cosine.isFinite, result.cosine >= tolerance.minCosine else { return false }
        if let error = result.relativeError, !error.isFinite { return false }
        if let maxAbs = result.unusedRowsMaxAbs, !maxAbs.isFinite { return false }
        if let limit = tolerance.maxRelativeError {
            guard let error = result.relativeError, error <= limit else { return false }
        }
        if let limit = tolerance.unusedRowsMaxAbs {
            guard let maxAbs = result.unusedRowsMaxAbs, maxAbs <= limit else { return false }
        }
        return true
    }

    /// The largest |value|, or infinity when any value is NaN or infinite (`max()` skips NaN).
    static func maxAbs(_ rows: some Sequence<[Float]>) -> Double {
        var largest: Float = 0
        for row in rows {
            for value in row {
                guard value.isFinite else { return .infinity }
                largest = Swift.max(largest, abs(value))
            }
        }
        return Double(largest)
    }

    private func runBucket(_ test: Case, length: Int, runner: any BucketFunctionRunner, rows: any TokenRowSource,
                           into result: inout SelfTestResult) async throws
    {
        let total = test.sequences.reduce(0) { $0 + $1.count }
        guard !test.sequences.isEmpty, test.sequences.count <= BucketContract.slots(forLength: length), total <= length,
            test.sequences.allSatisfy({ !$0.isEmpty })
        else { throw SelfTestError.invalid("\(test.function): sequences do not fit \(length) rows") }
        var offset = 0
        let segments = test.sequences.enumerated().map { index, ids -> PackedSegment in
            defer { offset += ids.count }
            return PackedSegment(item: index, offset: offset, length: ids.count)
        }
        let buffers = BucketBuffers(length: length)
        buffers.prepare(PackedCall(length: length, segments: segments), ids: test.sequences, rows: rows)
        let output = try await runner.run(buffers)
        let width = BucketContract.dimension
        guard output.count >= test.sequences.count, output.allSatisfy({ $0.count == width }) else {
            throw RuntimeError.model("\(test.function) returned \(output.count) rows of \(output.first?.count ?? 0) values")
        }
        let reference = expected(test)
        guard reference.count == test.sequences.count * width else { throw SelfTestError.invalid("\(test.function): expected shape") }
        var worst = 1.0
        for row in test.sequences.indices {
            let cosine = Self.metrics(output[row], Array(reference[(row * width)..<((row + 1) * width)])).cosine
            if !(cosine >= worst) { worst = cosine }  // keeps a NaN (Swift's min(1.0, .nan) is 1.0)
        }
        result.cosine = worst
        result.unusedRowsMaxAbs = Self.maxAbs(output.dropFirst(test.sequences.count))
    }

    private func runLong(_ test: Case, length: Int, runner: any LongFunctionRunner, rows: any TokenRowSource,
                         into result: inout SelfTestResult) async throws
    {
        let n = test.documentRows
        guard n > 0, n <= documentIDs.count, n <= length else { throw SelfTestError.invalid("doc_rows \(n)") }
        let buffers = LongBuffers(length: length)
        buffers.prepare(ids: Array(documentIDs.prefix(n)), rows: rows)
        let embedding = try await runner.run(buffers)
        guard embedding.count == BucketContract.dimension else {
            throw RuntimeError.model("\(test.function) returned \(embedding.count) values")
        }
        result.cosine = Self.metrics(embedding, expected(test)).cosine
    }
}
