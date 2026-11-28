@preconcurrency import CoreML
import Foundation

/// A shape set published only after its entire prefix of routes has passed the package tests.
final class PPLXTextRuntime: EmbeddingRuntime {
    let info: RuntimeInfo
    private let runner: any PPLXPredicting
    private let table: PPLXTokenTable
    private let norm: [Float]
    private let longWorkspace = AsyncSemaphore(1)
    private let causal: Bool
    private let lengths: [Int]

    init(runner: any PPLXPredicting, table: PPLXTokenTable, norm: [Float], lengths: [Int], maxTokens: Int,
         functions: [String], notServed: [String], skipped: Bool, causal: Bool = false)
    {
        self.causal = causal
        self.runner = runner
        self.table = table
        self.norm = norm
        self.lengths = lengths
        info = RuntimeInfo(maxTokens: min(maxTokens, lengths.last ?? 0), maxImageTokens: 0,
                           bucketLengths: lengths, backend: "coreml-ane", functions: functions,
                           notServed: notServed, selfTestSkipped: skipped)
    }

    func warm() async throws {}

    func embed(images: [PreparedImage]) async throws -> [[Float]] {
        throw RuntimeError.unsupported("pplx-embed-v1-0.6b supports text inputs")
    }

    func embed(tokenized texts: [[Int32]]) async throws -> [[Float]] {
        var start = 0
        var vectors: [[Float]] = []
        vectors.reserveCapacity(texts.count)
        for (index, ids) in texts.enumerated() {
            guard !ids.isEmpty else { throw RuntimeError.unsupported("input[\(index)] has no tokens") }
            guard ids.count <= info.maxTokens else { throw RuntimeError.tooLong(tokens: ids.count, max: info.maxTokens) }
        }
        let packing = lengths.filter { $0 <= 1024 }
        guard let packedLimit = packing.last else { throw RuntimeError.model("no validated pplx packing function") }
        while start < texts.count {
            try Task.checkCancellation()
            let ids = texts[start]
            if ids.count > min(512, packedLimit) {
                let length = try bucket(for: ids.count)
                let output = try await longWorkspace.withPermit {
                    try Task.checkCancellation()
                    return try await Self.run([ids], length: length, runner: self.runner, table: self.table, norm: self.norm, causal: self.causal)
                }
                vectors.append(contentsOf: output)
                start += 1
                continue
            }

            var rows = ids.count
            var end = start + 1
            let slots = max(8, packedLimit / 32)
            while end < texts.count, end - start < slots, texts[end].count <= 512,
                  rows + texts[end].count <= packedLimit
            {
                rows += texts[end].count
                end += 1
            }
            let count = end - start
            guard let length = packing.first(where: { $0 >= rows && max(8, $0 / 32) >= count }) else {
                throw RuntimeError.model("cannot pack \(count) pplx texts with \(rows) rows")
            }
            vectors += try await PPLXFused.embed(Array(texts[start..<end]), length: length, runner: runner, table: table, causal: causal)
            start = end
        }
        return vectors
    }

    func bucket(for count: Int) throws(RuntimeError) -> Int {
        guard count <= info.maxTokens, let bucket = lengths.first(where: { $0 >= count }) else {
            throw .tooLong(tokens: count, max: info.maxTokens)
        }
        return bucket
    }

    /// Same work and permit as the production long path, with its per-layer timings.
    func profile(_ ids: [Int32], length: Int) async throws -> PPLXDecomposition.Profile {
        try await longWorkspace.withPermit {
            try Task.checkCancellation()
            return try await PPLXDecomposition.run(ids, length: length, runner: self.runner, table: self.table, norm: self.norm, causal: self.causal)
        }
    }

    static func run(_ sequences: [[Int32]], length: Int, runner: any PPLXPredicting, table: PPLXTokenTable, norm: [Float], causal: Bool = false)
        async throws -> [[Float]]
    {
        if length <= (causal ? 64 : 8192) {
            return try await PPLXFused.embed(sequences, length: length, runner: runner, table: table, causal: causal)
        }
        guard let ids = sequences.first, sequences.count == 1 else {
            throw RuntimeError.model("pplx decomposed route requires one text")
        }
        return [try await PPLXDecomposition.run(ids, length: length, runner: runner, table: table, norm: norm, causal: causal).vector]
    }
}

extension ModelLoader {
    static func pplx(_ config: Config, manifest: Manifest, unavailablePackages: [String: String],
                     started: Date, log: @Sendable (String) -> Void)
        async throws(ServiceError) -> (service: EmbeddingService, report: LoadReport)
    {
        guard let contract = manifest.pplxContract else {
            throw ServiceError(.unavailable, "invalid_manifest", "no text package contract in \(config.modelDirectory.path)")
        }
        let causal = contract.model == .qwen3Embedding
        let fusedLimit = causal ? 64 : 8192
        let directory = config.modelDirectory
        var functions: [String] = []
        var effectiveMaxTokens = 0
        var report: SelfTestReport?
        defer {
            if !Task.isCancelled {
                let now = Date()
                let warm = WarmState(osBuild: operatingSystemBuild, binaryVersion: aneembedVersion,
                                     manifestSHA256: manifest.sha256, maxTokens: effectiveMaxTokens,
                                     functions: functions, warmedAt: now,
                                     seconds: (now.timeIntervalSince(started) * 10).rounded() / 10, selfTest: report)
                do { try warm.recordCapability(requestedMaxTokens: config.maxTokens, in: directory) }
                catch { log("could not record warm state: \(error)") }
                if let report { log("\(contract.model.rawValue) self-test: \(report.summary)") }
            }
        }
        let suite = config.skipSelfTest ? nil : try PPLXSelfTest.load(from: directory, manifest: manifest)
        if config.skipSelfTest { log("warning: --selftest skip: \(contract.model.rawValue) predictions are served unchecked") }

        let table: PPLXTokenTable
        let norm: [Float]
        let tokenizer: any TextTokenizer
        do {
            table = try PPLXTokenTable(contentsOf: directory.appendingPathComponent("embeddings.f32"),
                                       rows: causal ? 151669 : PPLXTokenTable.rowCount)
            norm = try PPLXSelfTest.readNorm(directory.appendingPathComponent("final_norm.f32"))
            tokenizer = try makeTokenizer(contract.model, from: directory.appendingPathComponent(ModelLayout.tokenizer))
        } catch {
            throw ServiceError(.unavailable, "model_load_failed", "\(error)",
                               hint: "run anerun models verify --model \(contract.model.rawValue)")
        }

        let required = contract.requiredFunctions(for: config.maxTokens)
        let lengths = Set(required.compactMap(\.length)).sorted()
        let (loaded, failures) = await PPLXCoreMLRunner.load(required, from: directory, unavailablePackages: unavailablePackages)
        let testStart = Date()
        var published: [Int] = []
        var serving: Set<String> = []
        var checked: [String: SelfTestResult] = [:]

        for (index, length) in lengths.enumerated() {
            let route = required.filter { candidate in
                candidate.length == length || (length > fusedLimit && (candidate.role == .pre || candidate.role == .post))
            }
            let names = route.map(\.function)
            guard let first = names.first else { continue }
            let key = length > fusedLimit ? "attn_S\(length)_Q2048" : first
            let failuresHere = names.compactMap { name in failures[name].map { "\(name): \($0)" } }
            let check: PPLXSelfTest.Metrics
            if published.count < index {
                check = .failed("a shorter pplx route is unavailable")
            } else if !failuresHere.isEmpty {
                check = .failed(failuresHere.joined(separator: ", "))
            } else if let suite {
                check = await suite.run(key, length: length, runner: loaded, table: table, norm: norm, causal: causal)
            } else {
                // A diagnostic load still exercises each route before publishing it.
                // Only the comparison against the package's reference is skipped.
                do {
                    _ = try await PPLXTextRuntime.run([[0]], length: length, runner: loaded, table: table, norm: norm, causal: causal)
                    check = .pass
                } catch {
                    check = .failed(String(describing: error))
                }
            }

            if check.passed {
                published.append(length)
                serving.formUnion(names)
            } else {
                log("\(contract.model.rawValue) \(length) route unavailable: \(check.error ?? "self-test cosine \(check.cosine)")")
            }
            if suite != nil {
                for name in names where !serving.contains(name) || checked[name] == nil {
                    checked[name] = SelfTestResult(function: name, kind: "\(causal ? "qwen" : "pplx")_\(length)", passed: check.passed,
                                                   cosine: check.cosine, error: check.error)
                }
            }
        }

        report = suite.map { _ in
            SelfTestReport(osBuild: operatingSystemBuild, ranAt: Date(), seconds: Date().timeIntervalSince(testStart),
                           results: required.compactMap { checked[$0.function] })
        }
        functions = required.map(\.function).filter(serving.contains)
        effectiveMaxTokens = min(config.maxTokens, published.last ?? 0)
        guard !published.isEmpty else {
            throw ServiceError(.unavailable, suite == nil ? "model_load_failed" : "selftest_failed",
                               "no usable text route for \(contract.model.rawValue)",
                               hint: "run anerun doctor --model \(contract.model.rawValue) and inspect the load log")
        }
        let runtime = PPLXTextRuntime(runner: loaded.keeping(serving), table: table, norm: norm, lengths: published,
                                      maxTokens: config.maxTokens, functions: functions,
                                      notServed: required.map(\.function).filter { !serving.contains($0) },
                                      skipped: config.skipSelfTest, causal: causal)
        if runtime.info.maxTokens < config.maxTokens {
            log("\(contract.model.rawValue) max context reduced to \(runtime.info.maxTokens) after validating the loaded functions")
        }
        let service = EmbeddingService(tokenizer: tokenizer, runtime: runtime, limits: config.limits,
                                       modelName: config.modelName, modelID: contract.model)
        var loadReport = LoadReport()
        loadReport.functions = functions
        loadReport.manifestSHA256 = manifest.sha256
        loadReport.selfTest = report
        loadReport.seconds = Date().timeIntervalSince(started)
        return (service, loadReport)
    }
}
