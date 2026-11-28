import DecisionRuntime
import EmbedCore
import Foundation

enum DecisionInstall {
    static func install(_ source: URL, model: ModelID, store: ModelStore, architecture: String? = nil,
                        compile: PortableDecision.Compiler = PortableDecision.compile,
                        nativeAudit: PortableDecision.NativeAudit = PortableDecision.verifyForActivation,
                        recovery: PortableRecovery = .live,
                        prepare: (URL) async throws -> Void) async throws(ServiceError) -> URL {
        _ = try ModelID.identify(in: source, expected: model)
        let fm = FileManager.default
        let namespace = store.modelsDirectory.standardizedFileURL.resolvingSymlinksInPath().path
        guard namespace != source.path, !namespace.hasPrefix(source.path + "/") else {
            throw ServiceError.badRequest("invalid_model_source", "the source cannot contain the model store")
        }
        if let architecture, fm.fileExists(atPath: source.appendingPathComponent(DecisionDownload.inventory).path) {
            let bytes: Data
            do { bytes = try SmallFile.read(source.appendingPathComponent(DecisionDownload.inventory), limit: DownloadLimit.manifestBytes) }
            catch { throw ServiceError(.invalidInput, "invalid_manifest", error.describe(DecisionDownload.inventory)) }
            let package: DecisionPackage
            do { package = try DecisionPackage(bytes, pin: nil, model: model, architecture: architecture) }
            catch let error as ServiceError { throw error }
            catch { throw ServiceError(.invalidInput, "invalid_manifest", error.localizedDescription) }
            if package.portable {
                return try await DecisionDownload.installInventory(revision: "local-" + package.manifest.sha256.prefix(32),
                    pin: package.manifest.sha256, model: model, architecture: architecture, store: store,
                    fetch: { path, stage, size throws(ServiceError) in
                        if size == nil { return bytes }
                        guard let file = package.manifest.files.first(where: { $0.path == path }),
                              Manifest.fileProblem(file, in: source, hash: false) == nil else {
                            throw ServiceError(.invalidInput, "invalid_bundle", "missing or invalid source file: \(path)")
                        }
                        do {
                            let target = stage.appendingPathComponent(path)
                            try fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                            try fm.copyItem(at: source.appendingPathComponent(path), to: target)
                        } catch { throw ServiceError(.unavailable, "copy_failed", error.localizedDescription) }
                        return Data()
                    }, compile: compile, nativeAudit: nativeAudit, recovery: recovery, prepare: prepare)
            }
        }
        let lock = try ModelsCommand.lock(store)
        defer { lock.release() }
        do {
            let digest = try fingerprint(source)
            let preferred = try store.directory(for: "local-" + digest.prefix(32))
            ModelInstaller.removeStaleStaging(for: preferred)
            return try await publish(preferred, store: store, matches: { (try? fingerprint($0)) == digest }, stage: { stage in
                // Foundation uses copy-on-write clones on APFS where available.
                try fm.copyItem(at: source, to: stage)
                guard try fingerprint(stage) == digest else {
                    throw ServiceError(.invalidInput, "model_files_changed", "bundle changed during installation")
                }
            }, prepare: prepare)
        } catch let error as CompiledFunctionLoadFailure {
            throw ServiceError(.unavailable, "prepare_failed", "cannot prepare packed model: \(error)")
        } catch let error as ServiceError {
            throw error
        } catch {
            throw ServiceError(.unavailable, "install_failed", "decision install failed: \(error)")
        }
    }

    // Caller holds the install lock. Prepare at the permanent path before switching current.
    static func publish(_ preferred: URL, store: ModelStore, matches: (URL) -> Bool,
                        stage materialize: (URL) async throws -> Void,
                        prepare: (URL) async throws -> Void) async throws -> URL {
        let fm = FileManager.default
        var target = preferred
        let active = try? store.resolveDirectory()
        let revision = preferred.lastPathComponent
        if let active, ModelInstaller.isRepair(active.lastPathComponent, of: revision), matches(active) {
            target = active
        } else if !matches(preferred) {
            if let pending = ModelInstaller.verifiedPendingRepair(of: revision, in: store, excluding: active, matches: matches) {
                target = pending
            } else if fm.fileExists(atPath: preferred.path) {
                target = try ModelInstaller.repairTarget(for: preferred, in: store)
            }
        }
        if !fm.fileExists(atPath: target.path) {
            let stage = ModelInstaller.stagingDirectory(for: target)
            defer { try? fm.removeItem(at: stage) }
            try await materialize(stage)
            try fm.moveItem(at: stage, to: target)
        }
        try await prepare(target)
        try Task.checkCancellation()
        try store.activate(revision: target.lastPathComponent)
        return target
    }

    /// Include filenames and contents so different exports of the same checkpoint keep separate
    /// final specialization paths. Refuse links, devices and other non-package entries.
    static func fingerprint(_ root: URL, excludingRootFile: String? = nil) throws -> String {
        var digest = SHA256Hasher()
        func visit(_ directory: URL, prefix: String) throws {
            for name in try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted() {
                if prefix.isEmpty && name == excludingRootFile { continue }
                let url = directory.appendingPathComponent(name)
                let path = prefix + name
                let values = try url.resourceValues(forKeys: [.fileResourceTypeKey])
                switch values.fileResourceType {
                case .directory:
                    digest.update(Data(("D" + path + "\0").utf8))
                    try visit(url, prefix: path + "/")
                case .regular:
                    digest.update(Data(("F" + path + "\0").utf8))
                    digest.update(Data(try SHA256Hasher.hex(fileAt: url).utf8))
                default:
                    throw ServiceError(.invalidInput, "invalid_bundle", "expected a regular file or directory at \(path)")
                }
            }
        }
        try visit(root, prefix: "")
        return digest.finalize()
    }
}
