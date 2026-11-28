import DecisionProfiles
import Foundation
import Testing
@testable import EmbedCore
@testable import anerun

struct DecisionDownloadTests {
    @Test func valenDefaultDownloadIsQualifiedOnlyOnH16G() throws {
        let model = ModelID.decision(.valen4B)
        let source = try DecisionDownload.source(model: model, architecture: "h16g", repo: nil, revision: nil)
        #expect(source.repository == "junmo-k/valen-4b-ane")
        #expect(source.manifestSHA256 == "d54187559829df64b509b96028f91464b56b196d29e2adf7864d2244ac328708")
        let error = #expect(throws: ServiceError.self) {
            try DecisionDownload.source(model: model, architecture: "h17d", repo: nil, revision: nil)
        }
        #expect(error?.reason == "unpublished_model")
    }

    @Test func sourceRequiresImmutableCommitOrQualifiedArchitecture() throws {
        let model = ModelID.decision(.valen4B)
        let error = #expect(throws: ServiceError.self) {
            try DecisionDownload.source(model: model, architecture: "fixture-only", repo: nil, revision: nil)
        }
        #expect(error?.hint?.contains("--from DIR") == true)
        #expect(throws: ServiceError.self) {
            try DecisionDownload.source(model: model, architecture: "fixture-only", repo: "test/fixture", revision: "main")
        }
    }

    @Test(arguments: ["success", "corrupt", "architecture", "pin", "prepare"])
    func install(outcome: String) async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = ModelID.decision(.valen4B)
        let store = ModelStore(dataDirectory: root, model: model)
        let old = try store.directory(for: "previous")
        try FileManager.default.createDirectory(at: old, withIntermediateDirectories: true)
        try store.activate(revision: "previous")
        let descriptor: JSONValue = ["source": ["repo": .string(model.source.repository),
                                                "revision": .string(DecisionProfile.valen4B.revision)], "architecture": "test-chip"]
        let files = ["model.json": descriptor.jsonData(), "asset/weights": Data([1, 2])]
        let inventory: JSONValue = ["model": .string(model.rawValue), "architecture": "test-chip",
                                    "directories": ["asset/empty"], "files": .array(files.map { path, data in
                                        ["path": .string(path), "size": .int(data.count), "sha256": .string(SHA256Hasher.hex(data))]
                                    })]
        let bytes = inventory.jsonData()
        var source = try DecisionDownload.source(model: model, architecture: "test-chip", repo: "test/fixture",
                                                 revision: String(repeating: "a", count: 40))
        source.manifestSHA256 = outcome == "pin" ? String(repeating: "0", count: 64) : SHA256Hasher.hex(bytes)
        var requests: [String] = []
        var prepared: URL?
        do {
            let installed = try await DecisionDownload.install(source, model: model, architecture: outcome == "architecture" ? "other" : "test-chip",
                                                               store: store, fetch: { path, stage, size throws(ServiceError) in
                requests.append(path)
                if size == nil {
                    #expect(path == DecisionDownload.inventory)
                    return bytes
                }
                guard var data = files[path] else { throw .badRequest("path", path) }
                #expect(data.count == size)
                if outcome == "corrupt", path == "asset/weights" { data = Data([9, 9]) }
                let destination = stage.appendingPathComponent(path)
                do {
                    try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
                    try data.write(to: destination)
                } catch { throw .badRequest("fixture", error.localizedDescription) }
                return Data()
            }, prepare: { path throws(ServiceError) in
                prepared = path
                #expect(path.lastPathComponent == source.revision)
                #expect((try? store.resolveDirectory()) == old)
                if outcome == "prepare" { throw .badRequest("prepare_failed", "fixture") }
            })
            #expect(outcome == "success")
            #expect(installed == prepared)
            #expect(try FileManager.default.contentsOfDirectory(atPath: installed.appendingPathComponent("asset/empty").path).isEmpty)
            #expect(try store.resolveDirectory() == installed)
        } catch let error as ServiceError {
            let reasons = ["pin": "pinned_manifest_mismatch", "architecture": "incompatible_architecture",
                           "corrupt": "invalid_bundle", "prepare": "prepare_failed"]
            #expect(error.reason == reasons[outcome])
            #expect(try store.resolveDirectory() == old)
            #expect((prepared != nil) == (outcome == "prepare"))
        }
        #expect(requests.count == (["pin", "architecture"].contains(outcome) ? 1 : 3))
        #expect(try FileManager.default.contentsOfDirectory(atPath: store.modelsDirectory.path).allSatisfy { !$0.contains(".partial-") })
    }
}
