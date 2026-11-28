import DecisionRuntime
import EmbedCore
import Foundation

enum DecisionDownload {
    static let inventory = "decision-manifest.json"
    // Add a source only after the exact published revision has passed qualification on this architecture.
    static let pins: [ModelID: [String: ModelSource]] = [
        .decision(.valen4B): ["h16g": ModelSource(
            repository: "junmo-k/valen-4b-ane", revision: "70c656b1824e1940bb1d530e8c0ee664512b10b5",
            manifestSHA256: "d54187559829df64b509b96028f91464b56b196d29e2adf7864d2244ac328708")],
        .decision(.glide9B): ["h17d": ModelSource(
            repository: "junmo-k/glide-9b-ane", revision: "6654b0d1789384f4a48ebded47d5e7cdb41ff457",
            manifestSHA256: "2c26646f4fc32da82254d4a32b1db86392e8bcbcec8ac71e096daa9947a62a99")],
        .decision(.elephant1): ["h17d": ModelSource(
            repository: "junmo-k/elephant-1-ane", revision: "24addcc73080d278cbe1f9833eea58db4487da64",
            manifestSHA256: "298bd3b5411b87a3fee7d6d30384978667d8ca7b984fc4f0a44b5fbe82c90e1f")],
    ]

    static func source(model: ModelID, architecture: String, repo: String?, revision: String?) throws(ServiceError) -> ModelSource {
        let source: ModelSource
        if repo == nil && revision == nil {
            guard let pin = pins[model]?[architecture], pin.manifestSHA256?.count == 64 else {
                throw ServiceError(.unavailable, "unpublished_model", "no qualified package for \(model.rawValue) on \(architecture)",
                                   hint: "run anerun models install --model \(model.rawValue) --from DIR with a matching packed bundle")
            }
            source = pin
        } else {
            guard let repo, let revision else {
                throw .badRequest("invalid_model_source", "supply both --repo OWNER/NAME and --revision FULL_COMMIT")
            }
            if let pin = pins[model]?[architecture], pin.repository == repo, pin.revision == revision.lowercased() {
                source = pin
            } else {
                source = ModelSource(repository: repo, revision: revision.lowercased())
            }
        }
        try source.validate()
        guard source.revision.utf8.count == 40, source.revision.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw .badRequest("invalid_revision", "supply a full 40-character hexadecimal HF commit")
        }
        return source
    }

    static func install(_ source: ModelSource, model: ModelID, architecture: String, store: ModelStore,
                        fetch: (String, URL, Int?) async throws(ServiceError) -> Data,
                        prepare: (URL) async throws -> Void) async throws(ServiceError) -> URL {
        try await installInventory(revision: source.revision, pin: source.manifestSHA256, model: model,
                                   architecture: architecture, store: store, fetch: fetch, prepare: prepare)
    }

    static func installInventory(revision: String, pin: String?, model: ModelID, architecture: String, store: ModelStore,
                                 fetch: (String, URL, Int?) async throws(ServiceError) -> Data,
                                 compile: PortableDecision.Compiler = PortableDecision.compile,
                                 nativeAudit: PortableDecision.NativeAudit = PortableDecision.verifyForActivation,
                        recovery: PortableRecovery = .live,
                                 prepare: (URL) async throws -> Void) async throws(ServiceError) -> URL {
        let target = try store.directory(for: revision)
        let lock = try ModelsCommand.lock(store)
        defer { lock.release() }
        let scratch = ModelInstaller.stagingDirectory(for: target)
        let fm = FileManager.default
        do {
            ModelInstaller.removeStaleStaging(for: target)
            try fm.createDirectory(at: scratch, withIntermediateDirectories: false)
            defer { try? fm.removeItem(at: scratch) }
            let bytes: Data
            if let digest = pin,
               let cached = [inventory, PortableDecision.inventoryName].lazy.compactMap({
                   try? SmallFile.read(target.appendingPathComponent($0), limit: DownloadLimit.manifestBytes)
               }).first,
               SHA256Hasher.hex(cached) == digest {
                bytes = cached
            } else {
                bytes = try await fetch(inventory, scratch, nil)
            }
            let package = try DecisionPackage(bytes, pin: pin, model: model, architecture: architecture)
            return try await DecisionInstall.publish(target, store: store, matches: package.installed, stage: { stage in
                try fm.createDirectory(at: stage, withIntermediateDirectories: false)
                try bytes.write(to: stage.appendingPathComponent(inventory))
                for path in package.directories {
                    try fm.createDirectory(at: stage.appendingPathComponent(path), withIntermediateDirectories: true)
                }
                for file in package.manifest.files {
                    try Task.checkCancellation()
                    _ = try await fetch(file.path, stage, file.size)
                }
                guard package.matches(stage) else {
                    throw ServiceError(.invalidInput, "invalid_bundle", "decision bundle identity, architecture or file checksums differ from inventory")
                }
                if package.portable {
                    try PortableDecision.derive(in: stage, package: package, compile: compile)
                }
            }, prepare: { directory in
                do { try await prepare(directory) }
                catch let load as CompiledFunctionLoadFailure where package.portable {
                    try await recovery.recover(load, directory: directory, scratch: scratch, package: package, fetch: fetch)
                }
                catch let error as CompiledFunctionLoadFailure {
                    throw ServiceError(.unavailable, "prepare_failed", "cannot prepare packed model: \(error)")
                }
                if package.portable {
                    do { try await nativeAudit(directory) }
                    catch let error as ServiceError where error.reason == "prepare_failed" { throw error }
                    catch { throw ServiceError(.invalidInput, "native_identity_failed", "native cache audit: \(error)") }
                }
            })
        } catch let error as ServiceError {
            throw error
        } catch {
            throw ServiceError(.unavailable, "install_failed", "decision download failed: \(error)")
        }
    }
}

struct DecisionPackage {
    private struct Header: Decodable {
        let architecture: String?
        let format: String?
        let architectures: [String]?
        let directories: [String]?
    }

    let portable: Bool
    let bytes: Data
    let model: ModelID
    let architecture: String
    let manifest: Manifest
    let directories: [String]
    private var expectedEntries: Set<String>

    init(_ bytes: Data, pin: String?, model: ModelID, architecture: String) throws {
        self.bytes = bytes
        self.model = model
        self.architecture = architecture
        if let pin, SHA256Hasher.hex(bytes) != pin {
            throw ServiceError(.invalidInput, "pinned_manifest_mismatch", "decision-manifest.json differs from the pinned SHA-256")
        }
        manifest = try Manifest.parse(bytes)
        let header = try JSONDecoder().decode(Header.self, from: bytes)
        guard header.format == nil || header.format == "portable-v1" else {
            throw ServiceError(.invalidInput, "invalid_manifest", "unsupported decision inventory format")
        }
        portable = header.format == "portable-v1"
        let supported = portable ? (header.architectures ?? []) : [header.architecture ?? ""]
        guard supported.contains(architecture) else {
            throw ServiceError(.invalidInput, "incompatible_architecture", "package supports \(supported.joined(separator: ", ")), device is \(architecture)",
                               hint: "use anerun models install --model \(model.rawValue) --from DIR with a matching bundle")
        }
        guard manifest.model == model.rawValue, manifest.files.contains(where: { $0.path == "model.json" }) else {
            throw ServiceError(.invalidInput, "model_mismatch", "expected a decision package for \(model.rawValue) with model.json")
        }
        directories = header.directories ?? []
        expectedEntries = [DecisionDownload.inventory]
        var occupied = Set<String>()
        for path in manifest.files.map(\.path) + directories {
            let parts = path.split(separator: "/")
            guard Manifest.isValidPath(path), parts.first?.lowercased() != DecisionDownload.inventory,
                  !parts.contains(where: { $0.hasPrefix(".") }), occupied.insert(path.lowercased()).inserted else {
                throw ServiceError(.invalidInput, "invalid_manifest", "reserved or duplicate path: \(path)")
            }
            var prefix = ""
            for part in parts {
                prefix += (prefix.isEmpty ? "" : "/") + part
                expectedEntries.insert(prefix)
            }
        }
    }

    func installed(_ directory: URL) -> Bool {
        if portable { return PortableDecision.matches(directory, package: self) }
        return matches(directory)
    }

    func matches(_ directory: URL) -> Bool {
        guard let entries = try? FileManager.default.subpathsOfDirectory(atPath: directory.path), Set(entries) == expectedEntries,
              (try? SmallFile.read(directory.appendingPathComponent(DecisionDownload.inventory), limit: DownloadLimit.manifestBytes)) == bytes,
              (try? ModelID.identify(in: directory, expected: model)) == model,
              let descriptor = try? SmallFile.read(directory.appendingPathComponent("model.json"), limit: DownloadLimit.manifestBytes),
              let header = try? JSONDecoder().decode(Header.self, from: descriptor),
              header.architecture == (portable ? nil : architecture) else { return false }
        for path in directories {
            let url = directory.appendingPathComponent(path)
            guard (try? url.resourceValues(forKeys: [.fileResourceTypeKey]).fileResourceType) == .directory else { return false }
        }
        return manifest.files.allSatisfy { Manifest.fileProblem($0, in: directory, hash: true) == nil }
    }
}
