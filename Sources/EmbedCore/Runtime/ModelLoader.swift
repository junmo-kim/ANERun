// Builds an EmbeddingService from the configuration: the Core ML packages in the model directory, or
// the fake runtime (--runtime fake) for testing integrations without a model.
//
// Loading = compile the packages if needed (once, normally at install) -> load the functions the
// configured --max-tokens needs -> run the package's self-test on each of them -> serve only those that
// pass. The self-test also pages every function in, so it doubles as the warm-up.
import Foundation

public struct LoadReport: Sendable {
    public var functions: [String] = []
    /// SHA-256 of the loaded package's manifest.json (empty for the fake runtime).
    public var manifestSHA256 = ""
    public var selfTest: SelfTestReport?
    public var seconds = 0.0
}

public enum ModelLoader {
    public static func makeService(_ config: Config, log: @Sendable (String) -> Void = { _ in }) async throws(ServiceError)
        -> EmbeddingService
    {
        try await load(config, log: log).service
    }

    public static func load(_ config: Config, log: @Sendable (String) -> Void = { _ in }) async throws(ServiceError)
        -> (service: EmbeddingService, report: LoadReport)
    {
        let config = try config.selectingModel(for: "load model")
        let started = Date()
        let directory = config.modelDirectory
        let tokenizerURL = directory.appendingPathComponent(ModelLayout.tokenizer)
        guard let model = config.modelID else {
            throw .badRequest("missing_model", "select a model with --model or --model-dir", param: "model")
        }
        guard !model.isDecision else { throw model.unsupportedEmbedding }
        if config.useFakeRuntime {
            let tokenizer = fakeTokenizer(model, tokenizerURL: tokenizerURL)
            log("fake runtime (no model): vectors are deterministic stand-ins, not embeddings")
            let cap = model != .embeddingGemma2 ? config.maxTokens : min(config.maxTokens, 2048)
            let service = EmbeddingService(tokenizer: tokenizer, runtime: FakeRuntime.make(maxTokens: cap, modelID: model),
                                           limits: config.limits, modelName: config.modelName, modelID: model)
            return (service, LoadReport())
        }

        guard NeuralEngine.device.available else {
            throw ServiceError(.unavailable, "no_neural_engine", "Core ML reports no Neural Engine on this Mac",
                               hint: "ANEEmbed runs only on Apple silicon with a Neural Engine")
        }
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw ServiceError(.unavailable, "model_missing", "no model directory at \(directory.path)",
                               hint: "run `anerun models install --model \(config.modelID?.rawValue ?? "embeddinggemma-2")` or pass --model-dir")
        }
        let (manifest, unavailablePackages) = try checkPackage(config, allowDegradedPPLX: true, log: log)
        if let note = config.unpinnedRevisionNote { log("note: " + note) }
        if manifest.pplxContract != nil {
            return try await pplx(config, manifest: manifest, unavailablePackages: unavailablePackages,
                                  started: started, log: log)
        }
        guard let textModel = ModelLayout.model(ModelLayout.textModel, in: directory) else {
            throw ServiceError(.unavailable, "model_missing", "no \(ModelLayout.textModel) model in \(directory.path)")
        }
        // Before any function is loaded: without the package's self-test nothing is checked, so nothing is served.
        let suite: SelfTestSuite?
        if config.skipSelfTest {
            log("warning: --selftest skip: every function is served without the self-test (unsafe: a miscompiled "
                + "function returns wrong vectors without an error)")
            suite = nil
        } else {
            suite = try SelfTestSuite.require(from: directory, manifest: manifest)
        }
        do {
            let tokenizer = try makeTokenizer(model, from: tokenizerURL)
            let table = try TokenTable(contentsOf: directory.appendingPathComponent(ModelLayout.tokenTable))
            let compiledText = try await compiledModelURL(textModel)
            var buckets = try await CoreMLBucketRunner.load(from: compiledText, maxTokens: config.maxTokens)
            var long: CoreMLLongRunner?
            let longLengths = WindowContract.longLengths(forMaxTokens: config.maxTokens)
            if !longLengths.isEmpty {
                log("loading the long-text functions for up to \(longLengths.last!) tokens")
                long = try await CoreMLLongRunner.load(from: compiledText, lengths: longLengths)
            }
            var vision: CoreMLVisionEncoder?
            if let visionModel = ModelLayout.model(ModelLayout.visionModel, in: directory) {
                vision = try await CoreMLVisionEncoder.load(
                    model: visionModel, positionTable: directory.appendingPathComponent(ModelLayout.positionTable))
            } else {
                log("no vision model: image inputs are rejected")
            }

            var report = LoadReport()
            report.manifestSHA256 = manifest.sha256
            let loaded = buckets.functionNames + (long?.functionNames ?? []) + (vision?.functionNames ?? [])
            let loadedImageTokens = vision?.maxTokens ?? 0
            if let suite {
                let testStarted = Date()
                let results = await suite.run(buckets: buckets, long: long, vision: vision,
                                              visionBudgets: vision?.budgets ?? [], rows: table)
                let selfTest = SelfTestReport(osBuild: operatingSystemBuild, ranAt: Date(),
                                              seconds: Date().timeIntervalSince(testStarted), results: results)
                for result in selfTest.failed {
                    log("self-test failed, not serving \(result.function): cos \(result.cosine)"
                        + (result.relativeError.map { ", rel err \($0)" } ?? "") + (result.error.map { ", \($0)" } ?? ""))
                }
                log(String(format: "self-test: %@ in %.1fs", selfTest.summary, selfTest.seconds))
                // Merged with the last run's records of functions this --max-tokens did not load, so doctor stays truthful.
                try? selfTest.merged(with: SelfTestReport.load(from: directory)).save(to: directory)
                report.selfTest = selfTest
                // Allowlist: only functions with a passing result in this run serve.
                let servable = selfTest.servable
                buckets = buckets.keeping(servable)
                long = long?.keeping(servable)
                vision = vision?.keeping(servable)
                guard !buckets.lengths.isEmpty else {
                    throw ServiceError(.unavailable, "selftest_failed", "every text function failed the self-test on this Mac",
                                       hint: "see `anerun doctor`; please report the macOS build")
                }
            }
            let longRunner = long.flatMap {
                $0.lengths.isEmpty ? nil : LongTextEngine(runner: $0, rows: table, maxIdlePerLength: max(1, config.limits.longConcurrency))
            }
            let visionEncoder = vision.flatMap { $0.budgets.isEmpty ? nil : $0 }
            report.functions = buckets.functionNames + (long?.functionNames ?? []) + (visionEncoder?.functionNames ?? [])
            let notServed = loaded.filter { !report.functions.contains($0) }
            let runtime = BucketTextRuntime(runner: buckets, rows: table, long: longRunner, vision: visionEncoder,
                                            maxIdleBuffers: max(2, config.limits.queryConcurrency), backend: "coreml-ane",
                                            functions: report.functions, notServed: notServed,
                                            selfTestSkipped: suite == nil, warmed: report.selfTest != nil)
            // Whatever was dropped, say what the limits are now.
            if runtime.info.maxTokens < config.maxTokens {
                log("texts are limited to \(runtime.info.maxTokens) tokens (--max-tokens \(config.maxTokens)): "
                    + "\(notServed.joined(separator: ", ")) not served")
            }
            if runtime.info.maxImageTokens < loadedImageTokens {
                log("images are limited to \(runtime.info.maxImageTokens) soft tokens (0: no images): "
                    + "\(notServed.filter { $0.hasPrefix("vision_") }.joined(separator: ", ")) not served")
            }
            report.seconds = Date().timeIntervalSince(started)
            let service = EmbeddingService(tokenizer: tokenizer, runtime: runtime, limits: config.limits,
                                           modelName: config.modelName, modelID: model)
            return (service, report)
        } catch let error as ServiceError {
            throw error
        } catch {
            throw ServiceError(.unavailable, "model_load_failed", "\(error)")
        }
    }

    /// The tokenizer-only CLI path needs no Core ML device or compiled package.
    public static func loadTokenizer(_ config: Config) throws(ServiceError) -> any TextTokenizer {
        let config = try config.selectingModel(for: "tokenize")
        guard let model = config.modelID else {
            throw .badRequest("missing_model", "select a model with --model or --model-dir", param: "model")
        }
        let directory = config.modelDirectory
        let file = directory.appendingPathComponent(ModelLayout.tokenizer)
        guard !model.isDecision else { throw model.unsupportedEmbedding }
        if config.useFakeRuntime { return fakeTokenizer(model, tokenizerURL: file) }
        let manifest = try Manifest.load(from: directory)
        try manifest.checkModel(model, directory: directory)
        try manifest.checkPinned(config.pinnedManifestSHA256)
        try manifest.checkContract()
        do {
            let data = try SmallFile.read(file, limit: 64 << 20)
            guard let entry = manifest.sourceFiles.first(where: { $0.path == ModelLayout.tokenizer }),
                  data.count == entry.size, SHA256Hasher.hex(data) == entry.sha256
            else {
                throw ServiceError(.unavailable, "model_files_changed", "\(file.path) doesn't match its manifest",
                                   hint: "reinstall with anerun models install --model \(model.rawValue)")
            }
            switch model {
            case .decision: throw model.unsupportedEmbedding
            case .embeddingGemma2: return try GemmaTokenizer(data: data)
            case .pplx, .qwen3Embedding: return try QwenTokenizer(data: data).checked(for: model)
            }
        } catch let error as ServiceError {
            throw error
        } catch {
            throw ServiceError(.unavailable, "tokenizer_missing", "\(file.path): \(error)",
                               hint: "reinstall with anerun models install --model \(model.rawValue)")
        }
    }

    private static func fakeTokenizer(_ model: ModelID, tokenizerURL: URL) -> any TextTokenizer {
        if FileManager.default.fileExists(atPath: tokenizerURL.path),
           let tokenizer = try? makeTokenizer(model, from: tokenizerURL) {
            return tokenizer
        }
        return ByteTokenizer(modelID: model)
    }

    static func makeTokenizer(_ model: ModelID, from url: URL) throws -> any TextTokenizer {
        switch model {
        case .decision: throw model.unsupportedEmbedding
        case .embeddingGemma2: return try GemmaTokenizer(contentsOf: url)
        case .pplx, .qwen3Embedding: return try QwenTokenizer(contentsOf: url).checked(for: model)
        }
    }

    /// The checks before anything of the package is loaded; returns its manifest. The manifest is required: it is
    /// what the contract and the file checks are read from. For the data directory's pinned revision it must be the
    /// manifest pinned in this build.
    static func checkPackage(_ config: Config, allowDegradedPPLX: Bool = false, log: (String) -> Void = { _ in })
        throws(ServiceError) -> (Manifest, unavailablePackages: [String: String])
    {
        let directory = config.modelDirectory
        let manifest = try Manifest.load(from: directory)
        if let model = config.modelID { try manifest.checkModel(model, directory: directory) }
        try manifest.checkPinned(config.pinnedManifestSHA256)
        try manifest.checkContract()
        // Files may have changed since install or `models verify`: sizes of all, hashes of what the self-test cannot see.
        let checkStarted = Date()
        let groups = Dictionary(grouping: manifest.checkForLoad(in: directory)) { problem in
            allowDegradedPPLX && manifest.pplxContract != nil ? problem.sourcePackage : nil
        }
        if let requiredAssets = groups[nil] {
            throw ServiceError(.unavailable, "model_files_changed",
                               "the model files do not match manifest.json: "
                               + requiredAssets.map { "\($0.path) \($0.issue)" }.joined(separator: ", "),
                               hint: "run `anerun models verify`; reinstall with `anerun models install`")
        }
        var unavailable: [String: String] = [:]
        for (key, problems) in groups {
            if let key { unavailable[key] = problems.map { "\($0.path) \($0.issue)" }.joined(separator: ", ") }
        }
        log(String(format: "checked model files against the manifest in %.2fs (%d invalid packages)",
                   Date().timeIntervalSince(checkStarted), unavailable.count))
        return (manifest, unavailable)
    }
}

extension Manifest.Problem {
    var sourcePackage: String? {
        guard let slash = path.firstIndex(of: "/") else { return nil }
        let directory = path[..<slash]
        guard directory.hasSuffix(".mlpackage") else { return nil }
        return String(directory.dropLast(".mlpackage".count))
    }
}
