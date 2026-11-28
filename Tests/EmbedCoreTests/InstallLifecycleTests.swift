import DecisionProfiles
import Foundation
import Testing

@testable import EmbedCore
@testable import anerun

@Suite("Model installation lifecycle")
struct InstallLifecycleTests {
    private func descriptor(for model: ModelID, path: String = "payload", data: Data) -> Data {
        JSONValue.object([
            "name": .string(model.rawValue),
            "files": [["path": .string(path), "size": .int(data.count), "sha256": .string(SHA256Hasher.hex(data))]],
        ]).jsonData()
    }

    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("aneembed-install-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test func activeRevisionIsNotReplacedBeforeVerification() throws {
        let base = try directory()
        defer { try? FileManager.default.removeItem(at: base) }
        let store = ModelStore(dataDirectory: base, model: .embeddingGemma2)
        let revision = String(repeating: "1", count: 40)
        let target = try store.directory(for: revision)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        let good = Data("working".utf8)
        let bytes = descriptor(for: .embeddingGemma2, data: good)
        let incoming = try Manifest.parse(bytes)
        let file = target.appendingPathComponent("payload")
        try good.write(to: file)
        try bytes.write(to: target.appendingPathComponent(ModelLayout.manifest))
        try store.activate(revision: revision)
        #expect(try ModelInstaller.installTarget(target, in: store, manifest: incoming) == target)

        let differing = try Manifest.parse(descriptor(for: .embeddingGemma2, data: Data("replacement".utf8)))
        #expect(throws: ServiceError.self) { try ModelInstaller.installTarget(target, in: store, manifest: differing) }
        #expect(try store.resolveDirectory() == target)
        #expect(try Data(contentsOf: file) == good)

        // Repair a known revision without making its current pointer refer to a candidate.
        try Data("damaged".utf8).write(to: file)
        let repair = try ModelInstaller.installTarget(target, in: store, manifest: incoming)
        #expect(repair != target && repair.lastPathComponent.hasPrefix(revision + "-r-"))
        #expect(try store.resolveDirectory() == target)
        #expect(!FileManager.default.fileExists(atPath: repair.path))
    }

    @Test func pinnedDownloadRepairsAParseableManifestWithoutReplacingIt() async throws {
        let base = try directory()
        defer { try? FileManager.default.removeItem(at: base) }
        let store = ModelStore(dataDirectory: base, model: .embeddingGemma2)
        let lock = try InstallLock.acquire(in: store)
        defer { lock.release() }
        let good = Data("working".utf8)
        let entry: JSONValue = ["path": "payload", "size": .int(good.count), "sha256": .string(SHA256Hasher.hex(good))]
        let bytes = JSONValue.object(DoctorTests.contract(files: [entry])).jsonData()
        let source = ModelSource(repository: "owner/model", revision: String(repeating: "1", count: 40),
                                 manifestSHA256: SHA256Hasher.hex(bytes))
        let target = try store.directory(for: source.revision)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: false)
        let descriptorURL = target.appendingPathComponent(ModelLayout.manifest)
        let changed = bytes + Data([10])
        try changed.write(to: descriptorURL)
        try good.write(to: target.appendingPathComponent("payload"))
        try store.activate(revision: source.revision)
        #expect(try Manifest.load(from: target).sha256 != source.manifestSHA256)

        let installed = try await ModelInstaller.download(source, into: target, store: store) {
            path, staging, _ throws(ServiceError) in
            let data = path == ModelLayout.manifest ? bytes : good
            do { try data.write(to: staging.appendingPathComponent(path)) }
            catch { throw ServiceError(.internal, "fixture_write_failed", "\(error)") }
            return data
        }
        #expect(installed.lastPathComponent.hasPrefix(source.revision + "-r-"))
        #expect(try Data(contentsOf: descriptorURL) == changed)
        #expect(try store.resolveDirectory() == target)
        let resumed = try await ModelInstaller.download(source, into: target, store: store) { _, _, _ throws(ServiceError) in
            throw ServiceError(.internal, "unexpected_fetch", "the verified repair should be reused")
        }
        #expect(resumed == installed)
    }

    @Test func decisionRetryReusesPublishedRepairAfterPrepareFails() async throws {
        let base = try directory()
        defer { try? FileManager.default.removeItem(at: base) }
        let source = base.appendingPathComponent("bundle")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: false)
        let profile = DecisionProfile.valen4B
        let model = ModelID.decision(profile)
        let header: JSONValue = ["source": ["repo": .string(profile.repository), "revision": .string(profile.revision)]]
        try header.jsonData().write(to: source.appendingPathComponent("model.json"))
        let payload = Data("working".utf8)
        try payload.write(to: source.appendingPathComponent("payload"))
        let store = ModelStore(dataDirectory: base, model: model)
        let original = try await DecisionInstall.install(source, model: model, store: store, prepare: { _ in })
        try Data("damaged".utf8).write(to: original.appendingPathComponent("payload"))

        var pending: URL?
        let error = await #expect(throws: ServiceError.self) {
            try await DecisionInstall.install(source, model: model, store: store) { directory throws(ServiceError) in
                pending = directory
                throw ServiceError(.unavailable, "prepare_failed", "simulated interruption")
            }
        }
        #expect(error?.reason == "prepare_failed")
        let repair = try #require(pending)
        #expect(repair != original)
        #expect(try store.resolveDirectory() == original)
        let installed = try await DecisionInstall.install(source, model: model, store: store) { directory in
            #expect(directory == repair)
        }
        #expect(installed == repair)
        #expect(try store.resolveDirectory() == repair)
        #expect(try Data(contentsOf: repair.appendingPathComponent("payload")) == payload)
    }

    @Test func retriesSelectOnlyTheVerifiedActiveRepairOfTheRequestedRevision() throws {
        let base = try directory()
        defer { try? FileManager.default.removeItem(at: base) }
        let store = ModelStore(dataDirectory: base, model: .embeddingGemma2)
        let lock = try InstallLock.acquire(in: store)
        defer { lock.release() }
        let revision = String(repeating: "1", count: 40)
        let otherRevision = String(repeating: "2", count: 40)
        let target = try store.directory(for: revision)
        let other = try store.directory(for: otherRevision)
        let payload = Data("working".utf8)
        let bytes = descriptor(for: .embeddingGemma2, data: payload)
        let incoming = try Manifest.parse(bytes)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: false)
        try payload.write(to: target.appendingPathComponent("payload"))
        try bytes.write(to: target.appendingPathComponent(ModelLayout.manifest))
        let repair = try ModelInstaller.repairTarget(for: target, in: store)
        _ = try ModelInstaller.copy(from: target, into: repair, store: store)
        _ = try ModelInstaller.copy(from: target, into: other, store: store)
        try store.activate(revision: repair.lastPathComponent)

        #expect(try ModelInstaller.installTarget(target, in: store, manifest: incoming) == repair)
        #expect(try ModelInstaller.installTarget(other, in: store, manifest: incoming) == other)
        try Data("damaged".utf8).write(to: repair.appendingPathComponent("payload"))
        #expect(try ModelInstaller.installTarget(target, in: store, manifest: incoming) == target)
        #expect(try store.resolveDirectory() == repair)
    }

    @Test func retryResumesAPublishedRepairAndLeavesPublishedSourcesIntact() throws {
        let base = try directory()
        defer { try? FileManager.default.removeItem(at: base) }
        let store = ModelStore(dataDirectory: base, model: .embeddingGemma2)
        let lock = try InstallLock.acquire(in: store)
        defer { lock.release() }
        let target = try store.directory(for: "revision")
        let payload = Data("working".utf8)
        let damaged = Data("damaged".utf8)
        let bytes = descriptor(for: .embeddingGemma2, data: payload)
        let incoming = try Manifest.parse(bytes)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: false)
        let targetPayload = target.appendingPathComponent("payload")
        try payload.write(to: targetPayload)
        try bytes.write(to: target.appendingPathComponent(ModelLayout.manifest))
        try store.activate(revision: target.lastPathComponent)
        let pending = try ModelInstaller.repairTarget(for: target, in: store)
        _ = try ModelInstaller.copy(from: target, into: pending, store: store)

        // Interrupt after publishing the verified repair and before switching the pointer.
        ModelInstaller.removeStaleStaging(for: target)
        let fallback = try ModelInstaller.repairCandidate(for: target, in: store, manifestSHA256: incoming.sha256)
        #expect(fallback.directory == pending && fallback.verified)
        try damaged.write(to: targetPayload)
        let retried = try ModelInstaller.installCandidate(target, in: store, manifestSHA256: incoming.sha256)
        #expect(retried.directory == pending && retried.verified)
        #expect(try store.resolveDirectory() == target)

        try store.activate(revision: pending.lastPathComponent)
        // The original revision could still be in use by a reader that selected it before the switch.
        #expect(throws: ServiceError.self) { try ModelInstaller.copy(from: pending, into: target, store: store) }
        #expect(try Data(contentsOf: targetPayload) == damaged)
        #expect(try ModelInstaller.installTarget(target, in: store, manifest: incoming) == pending)
    }

    @Test(arguments: ["manifest.json", "Manifest.json", "MANIFEST.JSON/asset", ".aneembed-warm.json",
                      ".ANEEMBED-SELFTEST.JSON", ".aneembed-warm.lock", ".PPLXText.compile.lock",
                      ".PPLXText.COMPILE.PARTIAL/asset"])
    func manifestRootIsReservedButNestedPackageManifestsAreAllowed(path: String) throws {
        let payload = Data("payload".utf8)
        let error = #expect(throws: ServiceError.self) {
            try Manifest.parse(descriptor(for: .embeddingGemma2, path: path, data: payload))
        }
        #expect(error?.reason == "invalid_manifest")
        let nested = "EmbeddingGemma2Text.mlpackage/Manifest.json"
        let accepted = try Manifest.parse(descriptor(for: .embeddingGemma2, path: nested, data: payload))
        #expect(accepted.sourceFiles.map(\.path) == [nested])
        let nestedRuntimeName = "EmbeddingGemma2Text.mlpackage/.aneembed-warm.json"
        let acceptedNested = try Manifest.parse(descriptor(for: .embeddingGemma2, path: nestedRuntimeName, data: payload))
        #expect(acceptedNested.sourceFiles.map(\.path) == [nestedRuntimeName])
    }

    @Test func stagedManifestMustMatchVerifiedBytesBeforeReplacingARevision() throws {
        let base = try directory()
        defer { try? FileManager.default.removeItem(at: base) }
        let store = ModelStore(dataDirectory: base, model: .embeddingGemma2)
        let lock = try InstallLock.acquire(in: store)
        defer { lock.release() }
        let target = try store.directory(for: "prior")
        let stage = ModelInstaller.stagingDirectory(for: target)
        let old = Data("old".utf8)
        let new = Data("new".utf8)
        let priorDescriptor = descriptor(for: .embeddingGemma2, data: old)
        let bytes = descriptor(for: .embeddingGemma2, data: new)
        let incoming = try Manifest.parse(bytes)
        for url in [target, stage] {
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        }
        try old.write(to: target.appendingPathComponent("payload"))
        try priorDescriptor.write(to: target.appendingPathComponent(ModelLayout.manifest))
        try store.activate(revision: "prior")
        try new.write(to: stage.appendingPathComponent("payload"))
        try (bytes + Data("\n".utf8)).write(to: stage.appendingPathComponent(ModelLayout.manifest))
        #expect(try Manifest.load(from: stage).verify(in: stage, hash: true).isEmpty)

        let error = #expect(throws: ServiceError.self) {
            try ModelInstaller.finish(staging: stage, into: target, store: store, manifest: incoming, log: { _ in })
        }
        #expect(error?.reason == "checksum_mismatch")
        #expect(try store.resolveDirectory() == target)
        #expect(try Data(contentsOf: target.appendingPathComponent(ModelLayout.manifest)) == priorDescriptor)
        #expect(try Data(contentsOf: target.appendingPathComponent("payload")) == old)
        #expect(!FileManager.default.fileExists(atPath: stage.path))
    }

    @Test func interruptedRepairStagesAreCleanedFromAnActiveRepair() throws {
        let base = try directory()
        let files = FileManager.default
        defer { try? files.removeItem(at: base) }
        let store = ModelStore(dataDirectory: base, model: .embeddingGemma2)
        let lock = try InstallLock.acquire(in: store)
        defer { lock.release() }
        let original = try store.directory(for: "revision")
        let activeRepair = try store.directory(for: "revision-r-0123abcd")
        try files.createDirectory(at: activeRepair, withIntermediateDirectories: false)
        try store.activate(revision: activeRepair.lastPathComponent)
        let nextRepair = try ModelInstaller.repairTarget(for: activeRepair, in: store)
        #expect(nextRepair != activeRepair)
        #expect(nextRepair.lastPathComponent.hasPrefix("revision-r-"))
        #expect(nextRepair.lastPathComponent.count == "revision-r-0123abcd".count)
        let stages = [original, activeRepair, nextRepair].map { ModelInstaller.stagingDirectory(for: $0) }
        let sentinels = [
            ".revision.partial-not-a-stage",
            ".revision-r-0123abcg.partial-12-deadbeef",
            ".revision-r-0123abcd.partial-0-deadbeef",
            ".revision-r-0123abcd.partial-12-deadbeef.backup",
            ".revision-other.partial-12-deadbeef",
        ].map { store.modelsDirectory.appendingPathComponent($0) }
        for url in stages + sentinels {
            try files.createDirectory(at: url, withIntermediateDirectories: false)
        }
        let symlink = store.modelsDirectory.appendingPathComponent(".revision-r-0123abcd.partial-12-deadbeef")
        try files.createSymbolicLink(at: symlink, withDestinationURL: activeRepair)

        ModelInstaller.removeStaleStaging(for: activeRepair)

        for url in stages { #expect(!files.fileExists(atPath: url.path), "\(url.path)") }
        for url in sentinels + [symlink] { #expect(files.fileExists(atPath: url.path), "\(url.path)") }
        #expect(try store.resolveDirectory() == activeRepair)
    }

    @Test func fileVerificationRejectsSymlinkParentsBelowThePackageRoot() throws {
        let base = try directory()
        let files = FileManager.default
        defer { try? files.removeItem(at: base) }
        let store = ModelStore(dataDirectory: base, model: .embeddingGemma2)
        let lock = try InstallLock.acquire(in: store)
        defer { lock.release() }
        let source = base.appendingPathComponent("source")
        let alias = base.appendingPathComponent("package-link")
        let assetParent = "EmbeddingGemma2Text.mlpackage/Data"
        let asset = assetParent + "/weight.bin"
        let payload = Data("weights".utf8)
        let bytes = descriptor(for: .embeddingGemma2, path: asset, data: payload)
        try files.createDirectory(at: source.appendingPathComponent(assetParent), withIntermediateDirectories: true)
        try payload.write(to: source.appendingPathComponent(asset))
        try bytes.write(to: source.appendingPathComponent(ModelLayout.manifest))
        try files.createSymbolicLink(at: alias, withDestinationURL: source)
        let manifest = try Manifest.load(from: alias)
        #expect(manifest.verify(in: alias, hash: true).isEmpty)
        let target = try store.directory(for: "candidate")
        _ = try ModelInstaller.copy(from: alias, into: target, store: store)
        try store.activate(revision: "candidate")

        let stage = ModelInstaller.stagingDirectory(for: target)
        try files.copyItem(at: target, to: stage)
        let external = source.appendingPathComponent(assetParent)
        for url in [target, stage] {
            let parent = url.appendingPathComponent(assetParent)
            try files.removeItem(at: parent)
            try files.createSymbolicLink(at: parent, withDestinationURL: external)
        }
        let problem = Manifest.Problem(path: asset, issue: "not_regular_file")
        let selected = try store.resolveDirectory()
        #expect(manifest.verify(in: selected, hash: true) == [problem])
        #expect(manifest.checkForLoad(in: selected).contains(problem))
        #expect(!ModelInstaller.isInstalled(at: selected, manifestSHA256: manifest.sha256))
        let error = #expect(throws: ServiceError.self) {
            try ModelInstaller.finish(staging: stage, into: target, store: store, manifest: manifest, log: { _ in })
        }
        #expect(error?.reason == "checksum_mismatch")
        #expect(!files.fileExists(atPath: stage.path))
        #expect(try Data(contentsOf: external.appendingPathComponent("weight.bin")) == payload)
    }

    @Test func selectedModelMismatchStopsAfterManifest() async throws {
        let base = try directory()
        defer { try? FileManager.default.removeItem(at: base) }
        let store = ModelStore(dataDirectory: base, model: .pplx)
        let target = try store.directory(for: "candidate")
        let lock = try InstallLock.acquire(in: store)
        defer { lock.release() }
        var requests: [String] = []
        let descriptor: JSONValue = ["name": "embeddinggemma-2-ane", "files": []]
        let source = ModelSource(repository: "owner/model", revision: "candidate")
        do {
            _ = try await ModelInstaller.download(source, into: target, store: store) {
                path, _, _ throws(ServiceError) in
                requests.append(path)
                return descriptor.jsonData()
            }
            Issue.record("accepted a different model")
        } catch {
            #expect(error.reason == "model_mismatch")
        }
        #expect(requests == [ModelLayout.manifest])
        #expect(!FileManager.default.fileExists(atPath: target.path))
        #expect(!FileManager.default.fileExists(atPath: store.currentPointer.path))
        #expect(try FileManager.default.contentsOfDirectory(atPath: store.modelsDirectory.path) == [".install.lock"])
    }

    @Test func lockIsPerModelAndCancelledWaiterDoesNotTakeIt() async throws {
        let base = try directory()
        defer { try? FileManager.default.removeItem(at: base) }
        let eg2 = ModelStore(dataDirectory: base, model: .embeddingGemma2)
        let pplx = ModelStore(dataDirectory: base, model: .pplx)
        let first = try InstallLock.acquire(in: eg2)
        let second = try InstallLock.acquire(in: pplx, wait: false)
        defer { first.release(); second.release() }
        #expect(throws: ServiceError.self) { try InstallLock.acquire(in: eg2, wait: false) }
        let waiting = Task.detached {
            do {
                let lock = try InstallLock.acquire(in: eg2)
                lock.release()
                return "unexpected success"
            } catch let error as ServiceError { return error.reason }
            catch { return String(describing: error) }
        }
        waiting.cancel()
        #expect(await waiting.value == "install_cancelled")
        first.release()
        let retried = try InstallLock.acquire(in: eg2, wait: false)
        retried.release()
    }
}
