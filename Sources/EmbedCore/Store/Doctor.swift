// `anerun doctor` runs the startup asset checks. --verify hashes every source file against the manifest.
import Foundation

public enum Doctor {
    public static func report(_ config: Config, verify: Bool = false) -> JSONValue {
        let os = ProcessInfo.processInfo.operatingSystemVersion
        let engine = NeuralEngine.device
        let system: JSONObject = [
            "aneembed": .string(aneembedVersion),
            "macos": .string("\(os.majorVersion).\(os.minorVersion).\(os.patchVersion)"),
            "macos_build": .string(operatingSystemBuild),
            "neural_engine": engine.available ? .string("available (\(engine.cores) cores)") : "unavailable",
            "runtime": .string(config.string("runtime")),
        ]
        func result(_ fields: JSONObject) -> JSONValue { .object(JSONObject(system.pairs + fields.pairs)) }
        if config.modelID == nil, config.string("model-dir").isEmpty, config.string("model").isEmpty {
            return result([
                "model": [:],
                "compile_cache": "no model selected",
                "ready": false,
                "problems": ["select a model with `anerun doctor --model ID`", .string(ModelID.catalog)],
            ])
        }
        let directory = config.modelDirectory
        let exists = FileManager.default.fileExists(atPath: directory.path)
        var manifestForPackage: Manifest?
        var problems: [JSONValue] = []

        var model: JSONObject = ["dir": .string(directory.path), "present": .bool(exists)]
        if let modelID = config.modelID { model["name"] = .string(modelID.rawValue) }
        if let error = config.modelResolutionError {
            model["present"] = false
            return result(["model": .object(model), "compile_cache": "no active model", "ready": false,
                           "problems": [.string(error.message)]])
        }
        let hasDecisionDescriptor = !FileManager.default.fileExists(atPath: directory.appendingPathComponent(ModelLayout.manifest).path)
            && FileManager.default.fileExists(atPath: directory.appendingPathComponent("model.json").path)
        if config.modelID?.isDecision == true || hasDecisionDescriptor {
            do {
                let identified = try ModelID.identify(in: directory, expected: config.modelID)
                model["name"] = .string(identified.rawValue)
                model["bundle_identity"] = "recognized"
                problems.append(.string("check decision runtime readiness with `anerun models verify` and the same model selector"))
            } catch {
                model["bundle_identity"] = "unverified"
                problems.append(.string(error.message))
            }
            return result(["model": .object(model), "compile_cache": "unchecked (CoreAI)", "ready": false,
                           "problems": .array(problems)])
        }
        let isTextOnly = config.modelID?.supportsImages == false
        let isQwen = config.modelID == .qwen3Embedding
        if exists {
            var files: [(String, Bool)] = [
                ("tokenizer", FileManager.default.fileExists(atPath: directory.appendingPathComponent(ModelLayout.tokenizer).path)),
                ("token_table", FileManager.default.fileExists(atPath: directory.appendingPathComponent(isTextOnly ? "embeddings.f32" : ModelLayout.tokenTable).path)),
            ]
            if !isQwen {
                files.insert(("text_model", ModelLayout.model(isTextOnly ? "PPLXText" : ModelLayout.textModel, in: directory) != nil),
                             at: 0)
            }
            do {
                let manifest = try Manifest.load(from: directory)
                manifestForPackage = manifest
                if isQwen, let contract = manifest.pplxContract {
                    let packages = contract.requiredPackages(for: config.maxTokens)
                    files.insert(("text_model", !packages.isEmpty && packages.allSatisfy {
                        ModelLayout.model($0, in: directory) != nil
                    }), at: 0)
                }
                let (issues, compiled) = verify
                    ? manifest.verifyAll(in: directory, hash: true)
                    : (manifest.checkForLoad(in: directory), [])
                model["name"] = .string(manifest.model)
                model["manifest_sha256"] = .string(manifest.sha256)
                model["manifest"] = [
                    "files": .int(manifest.sourceFiles.count), "checked": .string(verify ? "sha256" : "size and small-asset sha256"),
                    "problems": .array(issues.map { .string("\($0.path): \($0.issue)") }),
                ]
                if verify {
                    // Informational: another Core ML compiler stamps its version into model.mil.
                    model["compiled_differs_from_packager"] = .array(compiled.map { .string("\($0.path): \($0.issue)") })
                }
                do {
                    if let selected = config.modelID { try manifest.checkModel(selected, directory: directory) }
                    try manifest.checkPinned(config.pinnedManifestSHA256)
                } catch {
                    problems.append(.string(error.message + (error.hint.map { "; \($0)" } ?? "")))
                }
                let blocking: [Manifest.Problem]
                if !verify, let contract = manifest.pplxContract {
                    let required = Set(contract.requiredPackages(for: config.maxTokens))
                    blocking = issues.filter { $0.sourcePackage.map(required.contains) ?? true }
                } else {
                    blocking = issues
                }
                if !blocking.isEmpty { problems.append(.string("\(blocking.count) model files fail the manifest check")) }
                for item in manifest.compiledStatus(in: directory, maxTokens: config.maxTokens) where item.status.hasPrefix("incomplete") {
                    problems.append(.string("\(item.package).mlmodelc is \(item.status)"))
                }
                do { try manifest.checkContract() } catch { problems.append(.string(error.message)) }
            } catch {
                model["manifest"] = .string(error.reason == "manifest_missing" ? "missing" : "invalid")
                problems.append(.string(error.message))
            }
            for (name, present) in files where !present { problems.append(.string("model file missing: \(name)")) }
            if !isQwen { files.append(("vision_model", ModelLayout.model(ModelLayout.visionModel, in: directory) != nil)) }
            model["files"] = .object(JSONObject(files.map { (key: $0.0, value: .string($0.1 ? "present" : "missing")) }))
            if config.skipSelfTest {
                model["selftest_data"] = "not used (--selftest skip: every function is served unchecked)"
            } else if isTextOnly, let loadedManifest = manifestForPackage {
                model["selftest_data"] = .bool(loadedManifest.sourceFiles.contains { $0.path == ModelLayout.selfTestData })
            } else if !config.useFakeRuntime, let loadedManifest = manifestForPackage {
                do {
                    let suite = try SelfTestSuite.require(from: directory, manifest: loadedManifest)
                    model["selftest_data"] = .string("\(suite.cases.count) tests")
                } catch {
                    model["selftest_data"] = .string(error.reason)
                    problems.append(.string(error.message))
                }
            } else if !config.useFakeRuntime {
                model["selftest_data"] = "not checked: no valid manifest.json to check selftest.bin against"
            }
        } else if !config.useFakeRuntime {
            let name = config.modelID?.rawValue ?? ModelID.embeddingGemma2.rawValue
            let source = isQwen ? " --from DIR" : ""
            problems.append(.string("no model: run `anerun models install --model \(name)\(source)` or pass --model-dir"))
        }

        let cache: String
        let requiredFunctions = manifestForPackage?.pplxContract?.requiredFunctions(for: config.maxTokens).map(\.function) ?? []
        let warm = WarmState.load(from: directory)
        if let warm {
            if warm.osBuild != operatingSystemBuild {
                cache = "stale: warmed on macOS build \(warm.osBuild), now \(operatingSystemBuild)"
            } else if let manifest = manifestForPackage, warm.manifestSHA256 != manifest.sha256 {
                cache = "stale: warmed for another model package (manifest sha256 \(warm.manifestSHA256.prefix(12))); the next start compiles again"
            } else if warm.binary != Executable.identity {
                cache = "stale: warmed by another anerun binary (\(warm.binaryVersion)); the first start compiles again"
            } else if warm.maxTokens < config.maxTokens {
                cache = "partial: warmed up to \(warm.maxTokens) tokens, configured \(config.maxTokens)"
            } else if isTextOnly && (!Set(requiredFunctions).isSubset(of: warm.functions)
                        || (manifestForPackage?.compiledStatus(in: directory, maxTokens: config.maxTokens).contains(where: \.problem) ?? true))
            {
                cache = "partial: compiled package or validated function coverage is missing"
            } else {
                cache = "warm"
            }
        } else {
            cache = "cold: never warmed (the first start compiles for the Neural Engine)"
        }
        if cache != "warm" && exists { problems.append(.string("run `anerun models warm` (compile cache \(cache.prefix { $0 != ":" }))")) }

        // The self-test of the last load (serve, mcp, models warm/install): functions that failed are not served.
        var selfTest: JSONValue = "not run yet"
        let report = isTextOnly ? warm?.selfTest : SelfTestReport.load(from: directory)
        if var report {
            if isTextOnly && !verify {
                let required = Set(requiredFunctions)
                report.results = report.results.filter { required.contains($0.function) }
            }
            var object: JSONObject = ["summary": .string(report.summary), "os_build": .string(report.osBuild),
                                      "ran_at": .string(ISO8601DateFormatter().string(from: report.ranAt))]
            if !report.failed.isEmpty {
                object["failed"] = .array(report.failed.map(\.json))
                problems.append(.string("self-test failed for \(report.failed.map(\.function).joined(separator: ", ")); "
                    + "those functions are not served (please report the macOS build)"))
            }
            if report.osBuild != operatingSystemBuild { object["note"] = "ran on another macOS build; the next load runs it again" }
            selfTest = .object(object)
        }
        if !engine.available && !config.useFakeRuntime { problems.append(.string("no Neural Engine")) }

        return result([
            "max_tokens": .int(config.maxTokens),
            // What a load with this --max-tokens compiles and self-tests (long functions only above 2048).
            "text_functions": .array((isTextOnly ? requiredFunctions : CoreMLBucketRunner.lengths(forMaxTokens: config.maxTokens).map(BucketContract.functionName)
                + WindowContract.longLengths(forMaxTokens: config.maxTokens).map(WindowContract.longName)).map(JSONValue.string)),
            "model": .object(model),
            "compile_cache": .string(cache),
            "selftest": selfTest,
            "ready": .bool(problems.isEmpty),
            "problems": .array(problems),
        ])
    }
}
