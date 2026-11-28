import DecisionRuntime
import DecisionProfiles
import Foundation
import Testing
@testable import EmbedCore
@testable import anerun

struct PortableDecisionTests {
    @Test(arguments: ["local", "remote", "retry", "sourceHash", "partial", "missingEntry", "nativeResource", "wrongChip", "fallback", "fallbackRemote", "sourceLoadFails", "compiledRetryFails", "fallbackMismatch", "secondEntry", "sameEntry", "badReport"])
    func verifiedActivation(_ scenario: String) async throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? fm.removeItem(at: root) }
        let source = root.appendingPathComponent("input")
        let cache = root.appendingPathComponent("native")
        let model = ModelID.decision(.valen4B)
        let store = ModelStore(dataDirectory: root, model: model)
        let old = try store.directory(for: "previous")
        try fm.createDirectory(at: old, withIntermediateDirectories: true)
        try store.activate(revision: "previous")
        let ir = Data("miniature".utf8)
        let digest = SHA256Hasher.hex(ir)
        let bytes = Data(stride(from: 0, to: 64, by: 2).map { offset in
            UInt8(digest.dropFirst(offset).prefix(2), radix: 16)!
        })
        let srcHash = scenario == "sourceHash" ? String(repeating: "0", count: 64) : digest
        let header: JSONValue = ["source": ["repo": .string(model.source.repository),
                                            "revision": .string(DecisionProfile.valen4B.revision)],
                                 "assets": [["asset": "stage.aimodel", "functions": ["entry_a", "entry_b"],
                                             "source_hash": .string(srcHash)]]]
        let files = ["model.json": header.jsonData(), "stage.aimodel/main.hash": bytes,
                     "stage.aimodel/main.mlirb": ir, "bpe.bin": Data([1]), "tokens.json": Data("{}".utf8)]
        let inventory: JSONValue = ["format": "portable-v1", "architectures": ["h16g"], "model": .string(model.rawValue),
                                    "files": .array(files.map { path, data in
                                        ["path": .string(path), "sha256": .string(SHA256Hasher.hex(data)), "size": .int(data.count)]
                                    })]
        for (path, data) in files {
            let target = source.appendingPathComponent(path)
            try fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: target)
        }
        let index = inventory.jsonData()
        try index.write(to: source.appendingPathComponent(DecisionDownload.inventory))
        var prepared: URL?
        var requests = [String]()
        var calls = 0
        let compiler: PortableDecision.Compiler = { _, output, arch in
            calls += 1
            #expect(!fm.fileExists(atPath: output.path))
            #expect(arch == "h16g")
            try fm.createDirectory(at: output, withIntermediateDirectories: true)
            try bytes.write(to: output.appendingPathComponent("main.hash"))
            try JSONValue.object(["sourceHash": .string(digest)]).jsonData().write(to: output.appendingPathComponent("metadata.json"))
            var entries = ["entry_a_HASH_0": ["mps.fullyPlacedOnANE"]]
            if scenario != "missingEntry" { entries["entry_b_HASH_1"] = (scenario == "partial" || (scenario == "retry" && calls == 1)) ? [] : ["mps.fullyPlacedOnANE"] }
            let plist = ["Package Version": ["1": ["Optimized Modules": ["m": ["Entry Function Attributes": entries]]]]]
            try PropertyListSerialization.data(fromPropertyList: plist, format: .binary, options: 0).write(to: output.appendingPathComponent("manifest.plist"))
            try ir.write(to: output.appendingPathComponent("resources.bin"))
        }
        let needsRecovery = ["fallback", "fallbackRemote", "sourceLoadFails", "compiledRetryFails", "fallbackMismatch", "secondEntry", "sameEntry", "badReport"].contains(scenario)
        var sourceLoads = 0
        var aotRetries = 0
        let recovery = PortableRecovery(cacheRoot: { cache }, warmSource: { url, name in
            sourceLoads += 1
            #expect(url.lastPathComponent == "stage.aimodel")
            #expect((try? Data(contentsOf: url.appendingPathComponent("main.mlirb"))) == ir)
            #expect(name == (sourceLoads == 2 ? "entry_a" : "entry_b"))
            if scenario == "sourceLoadFails" { throw PortableDecision.invalid("source load failed") }
        }, retryCompiled: { directory in
            aotRetries += 1
            #expect(try store.resolveDirectory() == old)
            #expect(directory == prepared)
            if (scenario == "secondEntry" && aotRetries == 1) || scenario == "badReport" {
                // A real child exercises exit status, report transport and native stdout noise together.
                let helper = root.appendingPathComponent("fixture-child")
                let report = scenario == "badReport" ? "truncated" : #"{"asset":"stage.h16g.aimodelc","function":"entry_a"}"#
                try ("#!/bin/sh\nprintf 'native diagnostic\\n'\nprintf '%s' '" + report + "' > \"$3\"\nexit 75\n")
                    .write(to: helper, atomically: true, encoding: .utf8)
                try fm.setAttributes([.posixPermissions: 0o700], ofItemAtPath: helper.path)
                try await PortableRecovery.child(["_portable-prepare-compiled", directory.path], executable: helper)
            }
            if scenario == "sameEntry" {
                throw CompiledFunctionLoadFailure(asset: "stage.h16g.aimodelc", function: "entry_b")
            }
            if scenario == "compiledRetryFails" { throw PortableDecision.invalid("compiled retry failed") }
        })
        let prepare: (URL) async throws -> Void = { directory in
            prepared = directory
            #expect(!directory.lastPathComponent.contains(".partial-"))
            #expect((try? store.resolveDirectory()) == old)
            let native = cache.appendingPathComponent(digest + "/options/model.aimodelx")
            do {
                try fm.createDirectory(at: native.deletingLastPathComponent(), withIntermediateDirectories: true)
                try fm.copyItem(at: directory.appendingPathComponent("stage.h16g.aimodelc"), to: native)
                if scenario == "nativeResource" || scenario == "fallbackMismatch" { try Data([2]).write(to: native.appendingPathComponent("resources.bin")) }
            } catch { throw PortableDecision.invalid(error.localizedDescription) }
            if needsRecovery { throw CompiledFunctionLoadFailure(asset: "stage.h16g.aimodelc", function: "entry_b") }
        }
        let audit: PortableDecision.NativeAudit = { directory in
            #if canImport(CoreAI)
            if #available(macOS 27.0, *), scenario == "secondEntry" {
                #expect(aotRetries == 2)
                #expect(try store.resolveDirectory() == old)
                let verifier = root.appendingPathComponent("activation-child")
                try """
                #!/bin/sh
                test "$1" = _portable-check-valen && test -f "$2/.decision-aot.json" && echo "$$" > "$2/../probe-pid"
                """.write(to: verifier, atomically: true, encoding: .utf8)
                try fm.setAttributes([.posixPermissions: 0o700], ofItemAtPath: verifier.path)
                try await PortableDecision.verifyForActivation(directory, cache: cache, executable: verifier)
                let pid = try String(contentsOf: directory.deletingLastPathComponent().appendingPathComponent("probe-pid"),
                                     encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
                #expect(Int32(pid) != nil && Int32(pid) != ProcessInfo.processInfo.processIdentifier)
                return
            }
            #endif
            try PortableDecision.auditPrepared(directory, cache: cache)
        }
        do {
            let result: URL
            if scenario == "remote" || scenario == "fallbackRemote" {
                result = try await DecisionDownload.installInventory(revision: String(repeating: "a", count: 40),
                    pin: SHA256Hasher.hex(index), model: model, architecture: "h16g", store: store,
                    fetch: { path, stage, size throws(ServiceError) in
                        requests.append(path)
                        guard size != nil else { return index }
                        let file = stage.appendingPathComponent(path)
                        do {
                            try fm.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
                            try files[path]!.write(to: file)
                        } catch { throw PortableDecision.invalid(error.localizedDescription) }
                        return Data()
                    }, compile: compiler, nativeAudit: audit, recovery: recovery, prepare: prepare)
            } else {
                result = try await DecisionInstall.install(source, model: model, store: store,
                    architecture: scenario == "wrongChip" ? "other" : "h16g", compile: compiler, nativeAudit: audit, recovery: recovery, prepare: prepare)
            }
            #expect(["remote", "local", "retry", "fallback", "fallbackRemote", "secondEntry"].contains(scenario))
            #expect(try store.resolveDirectory() == result)
            #expect(prepared == result)
            #expect(!fm.fileExists(atPath: result.appendingPathComponent("stage.aimodel").path))
            #expect(fm.fileExists(atPath: source.appendingPathComponent("stage.aimodel/main.mlirb").path))
            if scenario == "local" {
                let reused = try await DecisionInstall.install(source, model: model, store: store, architecture: "h16g",
                    compile: compiler, nativeAudit: audit, prepare: { _ in })
                #expect(reused == result)
            }
            if scenario == "local" {
                // Existing receipts keep no permanent source assets. Recover using only
                // the targeted files from the original --from input.
                let reused = try await DecisionInstall.install(source, model: model, store: store, architecture: "h16g",
                    compile: compiler, nativeAudit: audit, recovery: PortableRecovery(cacheRoot: { cache },
                        warmSource: { url, function in
                            #expect(url.lastPathComponent == "stage.aimodel")
                            #expect(function == "entry_b")
                            #expect((try? Data(contentsOf: url.appendingPathComponent("main.mlirb"))) == ir)
                        }, retryCompiled: { dir in #expect(dir == result) }),
                    prepare: { _ in throw CompiledFunctionLoadFailure(asset: "stage.h16g.aimodelc", function: "entry_b") })
                #expect(reused == result)
            }
            #expect(sourceLoads == (scenario == "secondEntry" ? 2 : needsRecovery ? 1 : 0))
            #expect(aotRetries == sourceLoads)
            if scenario == "fallbackRemote" || scenario == "remote" {
                for path in files.keys {
                    let count = requests.filter { $0 == path }.count
                    #expect(count == (scenario == "fallbackRemote" && path.hasPrefix("stage.aimodel/") ? 2 : 1))
                }
            }
            #expect(calls == (scenario == "retry" ? 2 : 1))
        } catch {
            let expected = ["sameEntry": "prepare_failed", "badReport": "prepare_failed", "sourceHash": "invalid_portable_bundle", "missingEntry": "invalid_portable_bundle",
                            "sourceLoadFails": "invalid_portable_bundle", "compiledRetryFails": "invalid_portable_bundle",
                            "fallbackMismatch": "invalid_portable_bundle", "partial": "partial_ane_placement", "nativeResource": "native_identity_failed", "wrongChip": "incompatible_architecture"]
            if scenario == "partial" { #expect(calls == 8) }
            if scenario == "missingEntry" { #expect(calls == 1) }
            #expect(error.reason == expected[scenario])
            #expect(try store.resolveDirectory() == old)
            #expect((prepared != nil) == (scenario == "nativeResource" || needsRecovery))
            #expect(sourceLoads == (["sourceLoadFails", "compiledRetryFails", "sameEntry", "badReport"].contains(scenario) ? 1 : 0))
            #expect(aotRetries == (["compiledRetryFails", "sameEntry", "badReport"].contains(scenario) ? 1 : 0))
        }
    }
}
