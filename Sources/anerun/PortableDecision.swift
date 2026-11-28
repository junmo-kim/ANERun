import Darwin
import DecisionRuntime
import EmbedCore
import Foundation

enum PortableDecision {
    typealias Compiler = (URL, URL, String) throws -> Void
    typealias NativeAudit = (URL) async throws -> Void
    static let receiptName = ".decision-aot.json"
    static let inventoryName = "decision-source-inventory.json"

    struct Asset: Decodable {
        let asset: String
        let functions: [String]
        let source_hash: String
    }

    struct Evidence: Codable, Equatable {
        let asset: String
        let sourceHash: String
        let key: String
        let functions: [String]
        let symbols: [String]
        let resource: String
        let sha256: String
    }

    struct Receipt: Codable {
        let inventory: String
        let architecture: String
        let elephant: Bool
        let fingerprint: String
        let assets: [Evidence]
    }

    static func invalid(_ message: String) -> ServiceError {
        ServiceError(.invalidInput, "invalid_portable_bundle", message)
    }

    static func compile(_ input: URL, _ output: URL, _ architecture: String) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        process.arguments = ["coreai-build", "compile", input.path, "--output", output.path,
                             "--architecture", architecture, "--preferred-compute", "neural-engine",
                             "--platform", "macOS", "--min-deployment-version", "27.0"]
        process.standardOutput = FileHandle.standardError
        try process.run()
        while process.isRunning {
            if Task.isCancelled {
                process.terminate()
                process.waitUntilExit()
                throw CancellationError()
            }
            Thread.sleep(forTimeInterval: 0.1)
        }
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw ServiceError(.unavailable, "aot_failed", "coreai-build failed for \(input.lastPathComponent)",
                               hint: "select Xcode 27 with the Metal Toolchain and check xcrun --find coreai-build")
        }
    }

    static func matches(_ directory: URL, package: DecisionPackage) -> Bool {
        guard let bytes = try? SmallFile.read(directory.appendingPathComponent(receiptName), limit: DownloadLimit.manifestBytes),
              let receipt = try? JSONDecoder().decode(Receipt.self, from: bytes),
              receipt.inventory == package.manifest.sha256, receipt.architecture == package.architecture,
              (try? SmallFile.read(directory.appendingPathComponent(inventoryName), limit: DownloadLimit.manifestBytes)) == package.bytes,
              (try? ModelID.identify(in: directory, expected: package.model)) == package.model else { return false }
        return (try? DecisionInstall.fingerprint(directory, excludingRootFile: receiptName)) == receipt.fingerprint
    }

    // Transform only runtime asset records. Preserve the publisher's schedule, I/O ABI and sidecar fields.
    static func derive(in stage: URL, package: DecisionPackage, compile: Compiler) throws {
        let fm = FileManager.default
        let descriptor = stage.appendingPathComponent("model.json")
        let bytes = try SmallFile.read(descriptor, limit: DownloadLimit.manifestBytes)
        guard var model = try JSONSerialization.jsonObject(with: bytes) as? [String: Any],
              let backbone = model["assets"] as? [[String: Any]], !backbone.isEmpty else { throw invalid("missing decision assets") }
        let files = try fm.contentsOfDirectory(atPath: stage.path)
        guard !files.contains(where: { ["checkpoint", inventoryName].contains($0.lowercased()) || $0.lowercased().hasSuffix(".aimodelc") }),
              files.contains("tokens.json"), files.contains("bpe.bin") else { throw invalid("expected portable assets and prepared tokenizer sidecars") }
        var pending = Set(files.filter { $0.hasSuffix(".aimodel") })
        var assets: [Asset] = []
        func transform(_ row: [String: Any]) throws -> [String: Any] {
            let asset = try JSONDecoder().decode(Asset.self, from: JSONSerialization.data(withJSONObject: row))
            guard pending.remove(asset.asset) != nil, !asset.functions.isEmpty, asset.functions.allSatisfy({ !$0.isEmpty }),
                  Set(asset.functions).count == asset.functions.count else { throw invalid("missing, duplicate or invalid asset: \(asset.asset)") }
            assets.append(asset)
            var compiled = row
            compiled["asset"] = outputName(asset.asset, package.architecture)
            return compiled
        }
        model["assets"] = try backbone.map(transform)
        if let rounding = model["residual_rounding"] as? [[String: Any]] { model["residual_rounding"] = try rounding.map(transform) }
        if var lookup = model["embedding_lookup"] as? [String: Any], let banks = lookup["assets"] as? [[String: Any]] {
            lookup["assets"] = try banks.map(transform)
            model["embedding_lookup"] = lookup
        }
        for key in ["decision_head", "residual_blend", "result_stats", "embedding_unscale", "vision_position", "feature_scale", "role_pool", "role_mixer", "vision_256"] {
            if let row = model[key] as? [String: Any] { model[key] = try transform(row) }
        }
        if let name = model["vision_asset"] as? String {
            let row = try transform(["asset": name, "functions": model["vision_functions"] as Any,
                                     "source_hash": model["vision_source_hash"] as Any])
            model["vision_asset"] = row["asset"]
        }
        guard pending.isEmpty else { throw invalid("undeclared source assets: \(pending.sorted())") }
        model["architecture"] = package.architecture
        let elephant = package.model.rawValue == "elephant-1"
        var evidence: [Evidence] = []
        var keys = Set<String>()
        for asset in assets {
            try Task.checkCancellation()
            let input = stage.appendingPathComponent(asset.asset)
            let output = stage.appendingPathComponent(outputName(asset.asset, package.architecture))
            guard try hexKey(input) == asset.source_hash,
                  try SHA256Hasher.hex(fileAt: input.appendingPathComponent("main.mlirb")) == asset.source_hash else {
                throw invalid("source IR does not match \(asset.asset)")
            }
            let row = try compileVerified(input, into: output, architecture: package.architecture,
                                          asset: asset, elephant: elephant, compile: compile)
            guard keys.insert(row.key).inserted else { throw invalid("duplicate native key for \(asset.asset)") }
            evidence.append(row)
            try fm.removeItem(at: input)
        }
        try JSONSerialization.data(withJSONObject: model, options: [.sortedKeys]).write(to: descriptor, options: .atomic)
        try fm.moveItem(at: stage.appendingPathComponent(DecisionDownload.inventory), to: stage.appendingPathComponent(inventoryName))
        let receipt = Receipt(inventory: package.manifest.sha256, architecture: package.architecture, elephant: elephant,
                              fingerprint: try DecisionInstall.fingerprint(stage), assets: evidence)
        try JSONEncoder().encode(receipt).write(to: stage.appendingPathComponent(receiptName), options: .atomic)
    }

    // Same per-asset budget as prepare.py. Only compiler output with partial placement is retryable.
    static let placementAttempts = 8

    private static func compileVerified(_ input: URL, into output: URL, architecture: String,
                                        asset: Asset, elephant: Bool, compile: Compiler) throws -> Evidence {
        let fm = FileManager.default
        for attempt in 1...placementAttempts {
            try Task.checkCancellation()
            let scratch = output.deletingLastPathComponent().appendingPathComponent(UUID().uuidString)
            try fm.createDirectory(at: scratch, withIntermediateDirectories: false)
            defer { try? fm.removeItem(at: scratch) }
            let candidate = scratch.appendingPathComponent(output.lastPathComponent)
            let start = ContinuousClock.now
            Log.info("AOT \(asset.asset) for \(architecture), attempt \(attempt)/\(placementAttempts)")
            try compile(input, candidate, architecture)
            try Task.checkCancellation()
            do {
                let row = try inspect(candidate, asset: asset, elephant: elephant)
                try fm.moveItem(at: candidate, to: output)
                Log.info("Verified AOT \(asset.asset) in \(start.duration(to: .now))")
                return row
            } catch let error as ServiceError where error.reason == "partial_ane_placement" {
                Log.info("Rejected AOT \(asset.asset) attempt \(attempt)/\(placementAttempts): \(error)")
                if attempt == placementAttempts { throw error }
            }
        }
        throw invalid("exhausted placement attempts for \(asset.asset)")
    }

    static func outputName(_ source: String, _ architecture: String) -> String {
        String(source.dropLast(".aimodel".count)) + "." + architecture + ".aimodelc"
    }

    static func hexKey(_ url: URL) throws -> String {
        let bytes = try SmallFile.read(url.appendingPathComponent("main.hash"), limit: 32)
        guard bytes.count == 32 else { throw invalid("invalid main.hash in \(url.lastPathComponent)") }
        return bytes.map { String(format: "%02x", $0) }.joined()
    }

    static func payload(_ directory: URL, functions: [String], elephant: Bool) throws -> (symbols: [String], resource: URL) {
        let paths = try FileManager.default.subpathsOfDirectory(atPath: directory.path)
        func files(_ name: String) -> [URL] {
            paths.filter { ($0 as NSString).lastPathComponent == name }.map { directory.appendingPathComponent($0) }
        }
        let plists = files("manifest.plist")
        guard plists.count == 1,
              let plist = try PropertyListSerialization.propertyList(from: Data(contentsOf: plists[0]), format: nil) as? [String: Any],
              let versions = plist["Package Version"] as? [String: [String: Any]], !versions.isEmpty else {
            throw invalid("missing placement evidence in \(directory.lastPathComponent)")
        }
        var symbols = Set<String>()
        for version in versions.values {
            guard let modules = version["Optimized Modules"] as? [String: [String: Any]], !modules.isEmpty else { throw invalid("missing optimized modules") }
            var covered = Set<String>()
            var versionSymbols = Set<String>()
            for module in modules.values {
                guard let entries = module["Entry Function Attributes"] as? [String: [Any]], !entries.isEmpty else { throw invalid("missing entry attributes") }
                for (symbol, attributes) in entries {
                    let candidates = functions.filter { symbol == $0 || symbol.hasPrefix($0 + "_") }
                    let flags = attributes.compactMap { $0 as? String }
                    guard candidates.count == 1 else { throw invalid("unexpected compiled entry \(symbol)") }
                    guard flags.contains("mps.fullyPlacedOnANE"),
                          !elephant || flags.contains("mps.noGPUActivity") else {
                        throw ServiceError(.invalidInput, "partial_ane_placement", "incomplete placement for \(symbol)")
                    }
                    covered.insert(candidates[0])
                    versionSymbols.insert(symbol)
                }
            }
            symbols.formUnion(versionSymbols)
            guard versionSymbols.count == functions.count, covered == Set(functions) else { throw invalid("missing placement for declared function") }
        }
        var resources = files("resources.bin")
        if resources.isEmpty { resources = files("original_model_0.mpsgraph") }
        guard resources.count == 1 else { throw invalid("missing or ambiguous compiled resource") }
        return (symbols.sorted(), resources[0])
    }

    static func inspect(_ output: URL, asset: Asset, elephant: Bool) throws -> Evidence {
        struct Metadata: Decodable { let sourceHash: String }
        let data = try SmallFile.read(output.appendingPathComponent("metadata.json"), limit: DownloadLimit.manifestBytes)
        guard try JSONDecoder().decode(Metadata.self, from: data).sourceHash.lowercased() == asset.source_hash else {
            throw invalid("compiled source identity mismatch for \(asset.asset)")
        }
        let checked = try payload(output, functions: asset.functions, elephant: elephant)
        return Evidence(asset: output.lastPathComponent, sourceHash: asset.source_hash, key: try hexKey(output),
                        functions: asset.functions, symbols: checked.symbols, resource: checked.resource.lastPathComponent,
                        sha256: try SHA256Hasher.hex(fileAt: checked.resource))
    }

    static func verifyForActivation(_ directory: URL) async throws {
        try await verifyForActivation(directory, cache: nativeCacheRoot(), executable: Bundle.main.executableURL)
    }

    static func verifyForActivation(_ directory: URL, cache: URL, executable: URL?) async throws {
        try auditPrepared(directory, cache: cache)
        #if canImport(CoreAI)
        if #available(macOS 27.0, *), try ModelID.identify(in: directory) == .decision(.valen4B) {
            try await PortableRecovery.child(["_portable-check-valen", directory.path], executable: executable)
        }
        #endif
    }

    #if canImport(CoreAI)
    @available(macOS 27.0, *)
    static func checkValenPrediction(_ directory: URL) async throws {
        // Exercise the audited backbone at its permanent path after streaming the native resources.
        // This session is released before activation and the serving process starts from unloaded.
        // The one-pixel neutral PNG exercises the encoder and the text backbone together.
        guard let pixel = Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNoaGgAAAMEAYFL09IQAAAAAElFTkSuQmCC") else {
            throw invalid("invalid Valen preparation image")
        }
        let request = DecisionInput(state: "The first item is selected.", questions: [
            .init(name: "selected", kind: .choice, instructions: "Which item is selected?",
                  candidates: [.init(key: "first", description: "First item"),
                               .init(key: "second", description: "Second item")])
        ], image: pixel)
        do {
            let result = try await DecisionEngine(bundleURL: directory).decide(request)
            guard result.answers["selected"]?.probabilities?.count == 2 else {
                throw invalid("incomplete Valen preparation response")
            }
        } catch {
            throw ServiceError(.unavailable, "prepare_failed", "audited Valen prediction failed: \(error)")
        }
    }
    #endif

    static func auditPrepared(_ directory: URL) throws {
        try auditPrepared(directory, cache: nativeCacheRoot())
    }

    static func nativeCacheRoot() throws -> URL {
        var length = 0
        guard sysctlbyname("kern.osversion", nil, &length, nil, 0) == 0, length > 1 else { throw invalid("missing OS build") }
        var build = [CChar](repeating: 0, count: length)
        guard sysctlbyname("kern.osversion", &build, &length, nil, 0) == 0 else { throw invalid("unreadable OS build") }
        let support = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                                  appropriateFor: nil, create: false)
        let osBuild = String(decoding: build.prefix { $0 != 0 }.map(UInt8.init(bitPattern:)), as: UTF8.self)
        return support.appendingPathComponent("coreai-cache").appendingPathComponent(osBuild)
            .appendingPathComponent(ProcessInfo.processInfo.processName)
    }

    static func auditPrepared(_ directory: URL, cache: URL) throws {
        let data = try SmallFile.read(directory.appendingPathComponent(receiptName), limit: DownloadLimit.manifestBytes)
        let receipt = try JSONDecoder().decode(Receipt.self, from: data)
        let installed = try FileManager.default.contentsOfDirectory(atPath: directory.path).filter { $0.hasSuffix(".aimodelc") }
        guard !installed.isEmpty, installed.count == receipt.assets.count, Set(installed) == Set(receipt.assets.map(\.asset)),
              Set(receipt.assets.map(\.key)).count == installed.count else { throw invalid("incomplete AOT receipt") }
        for row in receipt.assets {
            try auditAsset(row, in: directory, cache: cache, elephant: receipt.elephant)
        }
    }

    static func auditAsset(_ row: Evidence, in directory: URL, cache: URL, elephant: Bool) throws {
        try Task.checkCancellation()
        let local = directory.appendingPathComponent(row.asset)
        let declared = Asset(asset: row.asset, functions: row.functions, source_hash: row.sourceHash)
        guard try inspect(local, asset: declared, elephant: elephant) == row else {
            throw invalid("installed AOT resources differ for \(row.asset)")
        }
        let key = cache.appendingPathComponent(row.key)
        let variants = try FileManager.default.contentsOfDirectory(at: key, includingPropertiesForKeys: nil)
            .map { $0.appendingPathComponent("model.aimodelx") }.filter { FileManager.default.fileExists(atPath: $0.path) }
        guard !variants.isEmpty else { throw invalid("missing native specialization for \(row.asset)") }
        for variant in variants {
            let checked = try payload(variant, functions: row.functions, elephant: elephant)
            guard checked.symbols == row.symbols, checked.resource.lastPathComponent == row.resource,
                  try SHA256Hasher.hex(fileAt: checked.resource) == row.sha256 else { throw invalid("native resources differ for \(row.asset)") }
        }
    }
}
