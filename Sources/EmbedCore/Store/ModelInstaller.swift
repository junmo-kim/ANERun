// Installing a model package: copy (from a local directory) or download (ModelDownloader) into a staging
// directory, check every file against the manifest's SHA-256, move it into place, then compile the
// .mlpackage files on this Mac. Warming and the self-test follow in `models install`.
@preconcurrency import CoreML
import Foundation

/// How far a download may go: what the manifest declares, and at most `manifestBytes` for manifest.json itself.
public enum DownloadLimit {
    public static let manifestBytes = 4 << 20

    /// Why the download must stop, or nil. `expectedLength` is the response's Content-Length (-1: unknown);
    /// `finished` checks the final size too.
    public static func problem(expectedLength: Int64, received: Int64, declaredSize: Int?, finished: Bool) -> String? {
        let cap = Int64(declaredSize ?? manifestBytes)
        if let declaredSize, expectedLength >= 0, expectedLength != Int64(declaredSize) {
            return "the server sends \(expectedLength) bytes; the manifest says \(declaredSize)"
        }
        if expectedLength > cap { return "the server sends \(expectedLength) bytes, more than \(cap)" }
        if received > cap { return "more than \(cap) bytes" }
        if finished, let declaredSize, received != Int64(declaredSize) {
            return "\(received) bytes; the manifest says \(declaredSize)"
        }
        return nil
    }
}

/// Serializes installs: an exclusive flock on `models/.install.lock`, held until `release()` (or deinit).
public final class InstallLock: @unchecked Sendable {
    private var descriptor: Int32

    private init(descriptor: Int32) { self.descriptor = descriptor }

    /// Takes the lock; when another install holds it, waits (`wait`, after calling `onWait`) or fails.
    public static func acquire(in store: ModelStore, wait: Bool = true, onWait: () -> Void = {}) throws(ServiceError)
        -> InstallLock
    {
        try store.checkNamespace()
        let path = store.modelsDirectory.appendingPathComponent(".install.lock").path
        do { try FileManager.default.createDirectory(at: store.modelsDirectory, withIntermediateDirectories: true) } catch {
            throw ServiceError(.internal, "install_failed", "cannot create \(store.modelsDirectory.path): \(error.localizedDescription)")
        }
        let descriptor = open(path, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK, 0o644)
        guard descriptor >= 0 else {
            throw ServiceError(.internal, "install_failed", "cannot open \(path): \(String(cString: strerror(errno)))")
        }
        var info = stat()
        guard fstat(descriptor, &info) == 0, SmallFile.isRegular(info) else {
            close(descriptor)
            throw ServiceError(.internal, "install_failed", "\(path) is not a regular lock file")
        }
        var reported = false
        while !Task.isCancelled {
            if flock(descriptor, LOCK_EX | LOCK_NB) == 0 { return InstallLock(descriptor: descriptor) }
            let failure = errno
            if failure == EINTR { continue }
            guard failure == EWOULDBLOCK else {
                close(descriptor)
                throw ServiceError(.internal, "install_failed", "cannot lock \(path): \(String(cString: strerror(failure)))")
            }
            guard wait else {
                close(descriptor)
                throw ServiceError(.unavailable, "install_in_progress", "another installation for this model is running",
                                   hint: "wait for it to finish, then run the command again")
            }
            if !reported { onWait(); reported = true }
            usleep(20_000)
        }
        close(descriptor)
        throw ServiceError(.loading, "install_cancelled", "installation cancelled while waiting for \(path)",
                           hint: "retry anerun models install")
    }

    public func release() {
        guard descriptor >= 0 else { return }
        flock(descriptor, LOCK_UN)
        close(descriptor)
        descriptor = -1
    }

    deinit { release() }
}

/// Puts a model package in place: copy or download, check, compile.
public enum ModelInstaller {
    /// A staging directory next to `target`, unique to this call (process id and a random suffix).
    public static func stagingDirectory(for target: URL) -> URL {
        let suffix = UUID().uuidString.prefix(8).lowercased()
        return target.deletingLastPathComponent()
            .appendingPathComponent(".\(target.lastPathComponent).partial-\(ProcessInfo.processInfo.processIdentifier)-\(suffix)")
    }

    /// Removes staging directories left by an interrupted install or repair of this revision.
    /// Call with the model's install lock held.
    public static func removeStaleStaging(for target: URL) {
        let parent = target.deletingLastPathComponent()
        let prefix = ".\(originalRevision(target.lastPathComponent))"
        for name in (try? FileManager.default.contentsOfDirectory(atPath: parent.path)) ?? [] where name.hasPrefix(prefix) {
            let suffix = name.dropFirst(prefix.count)
            guard let marker = suffix.range(of: ".partial-") else { continue }
            let repair = suffix[..<marker.lowerBound]
            guard repair.isEmpty || (repair.hasPrefix("-r-") && isShortUUID(repair.dropFirst(3))) else { continue }
            let processAndSuffix = suffix[marker.upperBound...].split(separator: "-", omittingEmptySubsequences: false)
            guard processAndSuffix.count == 2, let pid = Int32(processAndSuffix[0]), pid > 0,
                  String(pid) == processAndSuffix[0], isShortUUID(processAndSuffix[1])
            else { continue }
            let stage = parent.appendingPathComponent(name)
            var info = stat()
            guard lstat(stage.path, &info) == 0, (info.st_mode & S_IFMT) == S_IFDIR else { continue }
            try? FileManager.default.removeItem(at: stage)
        }
    }

    private static func isShortUUID(_ suffix: Substring) -> Bool {
        suffix.utf8.count == 8 && suffix.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }

    private static func originalRevision(_ name: String) -> Substring {
        var revision = name[...]
        while revision.count > 11, revision.dropLast(8).hasSuffix("-r-"), isShortUUID(revision.suffix(8)) {
            revision = revision.dropLast(11)
        }
        return revision
    }

    /// The revision directory name for a package installed from a local directory.
    public static func localRevision(_ manifest: Manifest) -> String { "local-" + manifest.sha256.prefix(12) }

    /// Whether `target` holds exactly the package whose manifest.json hashes to `manifestSHA256`, every source file
    /// matching it. A package that is only consistent with its own manifest is not enough.
    public static func isInstalled(at target: URL, manifestSHA256: String) -> Bool {
        guard let manifest = try? Manifest.load(from: target), manifest.sha256 == manifestSHA256.lowercased() else { return false }
        return manifest.verify(in: target, hash: false).isEmpty && manifest.verify(in: target, hash: true).isEmpty
    }

    /// An active revision stays readable while a repair is compiled and warmed at a new path.
    public static func installTarget(_ preferred: URL, in store: ModelStore, manifest: Manifest)
        throws(ServiceError) -> URL
    {
        try installCandidate(preferred, in: store, manifestSHA256: manifest.sha256).directory
    }

    // Call with the store's install lock held. A trusted download pin can recover from a
    // changed manifest by publishing at a new path. A verified candidate needs no second hash pass.
    package static func installCandidate(_ preferred: URL, in store: ModelStore, manifestSHA256: String,
                                         trustedPin: Bool = false)
        throws(ServiceError) -> (directory: URL, verified: Bool)
    {
        try store.checkInside(preferred)
        guard store.model != nil else {
            return (preferred, isInstalled(at: preferred, manifestSHA256: manifestSHA256))
        }
        if !trustedPin, let current = try? Manifest.load(from: preferred), current.sha256 != manifestSHA256 {
            throw ServiceError(.invalidInput, "revision_conflict",
                               "\(preferred.path) already has manifest \(current.sha256), received \(manifestSHA256)",
                               hint: "install from a new revision or use anerun models install --from DIR with the selected model")
        }
        let revision = preferred.lastPathComponent
        let active = try? store.activeDirectory()
        if let active, isRepair(active.lastPathComponent, of: revision), isInstalled(at: active, manifestSHA256: manifestSHA256) {
            return (active, true)
        }
        if isInstalled(at: preferred, manifestSHA256: manifestSHA256) { return (preferred, true) }
        if let pending = verifiedPendingRepair(of: revision, in: store, excluding: active, matches: {
            isInstalled(at: $0, manifestSHA256: manifestSHA256)
        }) {
            return (pending, true)
        }
        guard FileManager.default.fileExists(atPath: preferred.path) else { return (preferred, false) }
        return (try newRepairTarget(for: revision, in: store), false)
    }

    // Call with the store's install lock held. Resume an inactive repair only when
    // the caller verifies that it contains the requested package.
    package static func verifiedPendingRepair(of revision: String, in store: ModelStore, excluding active: URL?,
                                              matches: (URL) -> Bool) -> URL?
    {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: store.modelsDirectory.path)) ?? []
        for name in names.sorted() where name != active?.lastPathComponent && isRepair(name, of: revision) {
            if let directory = try? store.directory(for: name), matches(directory) {
                return directory
            }
        }
        return nil
    }

    package static func isRepair(_ name: String, of revision: String) -> Bool {
        let prefix = revision + "-r-"
        return name.hasPrefix(prefix) && isShortUUID(name.dropFirst(prefix.count))
    }

    package static func repairCandidate(for preferred: URL, in store: ModelStore, manifestSHA256: String)
        throws(ServiceError) -> (directory: URL, verified: Bool)
    {
        try store.checkInside(preferred)
        let revision = preferred.lastPathComponent
        if let pending = verifiedPendingRepair(of: revision, in: store, excluding: try? store.activeDirectory(),
                                               matches: { isInstalled(at: $0, manifestSHA256: manifestSHA256) })
        {
            return (pending, true)
        }
        return (try newRepairTarget(for: revision, in: store), false)
    }

    /// Gives a repair its own path without changing the original revision's identity.
    public static func repairTarget(for preferred: URL, in store: ModelStore) throws(ServiceError) -> URL {
        try store.checkInside(preferred)
        return try newRepairTarget(for: String(originalRevision(preferred.lastPathComponent)), in: store)
    }

    private static func newRepairTarget(for revision: String, in store: ModelStore) throws(ServiceError) -> URL {
        while true {
            let suffix = UUID().uuidString.prefix(8).lowercased()
            let candidate = try store.directory(for: "\(revision)-r-\(suffix)")
            if !FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        }
    }

    static func checkCancellation() throws(ServiceError) {
        guard !Task.isCancelled else {
            throw ServiceError(.loading, "install_cancelled", "installation cancelled", hint: "retry anerun models install")
        }
    }

    /// Copies every source file the manifest of `source` lists into `target` through a staging directory
    /// and checks each against its SHA-256. `target` is replaced only when everything matches.
    public static func copy(from source: URL, into target: URL, store: ModelStore, manifest sourceManifest: Manifest? = nil,
                            log: (String) -> Void = { _ in }) throws(ServiceError) -> Manifest
    {
        let manifest: Manifest
        if let sourceManifest { manifest = sourceManifest } else { manifest = try Manifest.load(from: source) }
        let fileManager = FileManager.default
        let staging = stagingDirectory(for: target)
        try store.checkInside(target)
        try store.checkInside(staging)
        defer { try? fileManager.removeItem(at: staging) }
        do {
            try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
            for (index, file) in manifest.sourceFiles.enumerated() {
                try checkCancellation()
                log("copying \(index + 1)/\(manifest.sourceFiles.count) \(file.path)")
                let origin = source.appendingPathComponent(file.path)
                guard case .success = Manifest.regularFileSize(file.path, in: source) else {
                    throw ServiceError(.invalidInput, "checksum_mismatch", "\(file.path) is not a regular file")
                }
                let destination = staging.appendingPathComponent(file.path)
                try fileManager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
                try fileManager.copyItem(at: origin, to: destination)
            }
            let current = try SmallFile.read(source.appendingPathComponent(ModelLayout.manifest), limit: ModelLayout.manifestLimit)
            guard SHA256Hasher.hex(current) == manifest.sha256 else {
                throw ServiceError(.invalidInput, "checksum_mismatch", "manifest.json changed while copying \(source.path)")
            }
            try current.write(to: staging.appendingPathComponent(ModelLayout.manifest))
        } catch let error as ServiceError {
            throw error
        } catch {
            throw ServiceError(.unavailable, "copy_failed", "cannot copy the package: \(error.localizedDescription)")
        }
        try finish(staging: staging, into: target, store: store, manifest: manifest, log: log)
        return manifest
    }

    /// Downloads `source` into `target` through a staging directory: manifest.json first (checked against the
    /// source's pinned hash), then every source file it lists; `finish` checks each against its SHA-256.
    /// `fetch(path, directory, size)` stores one file of the revision in `directory` and returns the bytes of
    /// manifest.json (`size` nil); ModelDownloader passes the network. Every failure once the staging directory
    /// exists removes it, so a failed install leaves no partial download behind.
    @discardableResult
    public static func download(_ source: ModelSource, into target: URL, store: ModelStore, log: (String) -> Void = { _ in },
                                fetch: (_ path: String, _ directory: URL, _ size: Int?) async throws(ServiceError) -> Data)
        async throws(ServiceError) -> URL
    {
        try source.validate()
        try checkCancellation()
        try store.checkInside(target)
        let fileManager = FileManager.default
        // Skip only when the installed package is the pinned one (by the hash compiled into this build).
        if let pinned = source.manifestSHA256 {
            let candidate = try installCandidate(target, in: store, manifestSHA256: pinned, trustedPin: true)
            if candidate.verified {
                if let model = store.model {
                    let manifest = try Manifest.load(from: candidate.directory)
                    try manifest.checkModel(model, directory: candidate.directory)
                    try manifest.checkContract()
                }
                log("\(candidate.directory.lastPathComponent) is already installed and matches the pinned manifest")
                return candidate.directory
            }
        }
        let staging = stagingDirectory(for: target)
        try store.checkInside(staging)
        do {
            try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
        } catch {
            throw ServiceError(.internal, "staging_failed", "cannot create \(staging.path): \(error.localizedDescription)")
        }
        do throws(ServiceError) {
            let manifestData = try await fetch(ModelLayout.manifest, staging, nil)
            if let pinned = source.manifestSHA256, SHA256Hasher.hex(manifestData) != pinned {
                throw ServiceError(.invalidInput, "checksum_mismatch", "manifest.json does not match the hash pinned in this build")
            }
            let manifest = try Manifest.parse(manifestData)
            if let model = store.model {
                try manifest.checkModel(model, directory: target)
                try manifest.checkContract()
            }
            let candidate = try installCandidate(target, in: store, manifestSHA256: manifest.sha256,
                                                 trustedPin: source.manifestSHA256 != nil)
            let installed = candidate.directory
            if candidate.verified {
                try? fileManager.removeItem(at: staging)
                log("\(installed.lastPathComponent) is already installed and matches \(source.repository)'s manifest")
                return installed
            }
            for (index, file) in manifest.sourceFiles.enumerated() {
                try checkCancellation()
                log("downloading \(index + 1)/\(manifest.sourceFiles.count) \(file.path) (\(file.size / (1 << 20)) MiB)")
                _ = try await fetch(file.path, staging, file.size)
            }
            try finish(staging: staging, into: installed, store: store, manifest: manifest, log: log)
            return installed
        } catch {
            try? fileManager.removeItem(at: staging)
            throw error
        }
    }

    /// Verifies a staged package against `manifest` and moves it to `target`. Both must be inside the store's
    /// models directory. Named revisions are published only to new paths.
    public static func finish(staging: URL, into target: URL, store: ModelStore, manifest: Manifest, log: (String) -> Void)
        throws(ServiceError)
    {
        try store.checkInside(target)
        try store.checkInside(staging)
        try checkCancellation()
        let problems = withoutActuallyEscaping(log) { log in manifest.verify(in: staging, hash: true) { log("verifying \($0)") } }
        guard problems.isEmpty else {
            try? FileManager.default.removeItem(at: staging)
            throw ServiceError(.invalidInput, "checksum_mismatch",
                               "the package does not match its manifest: \(problems.map { "\($0.path) \($0.issue)" }.joined(separator: ", "))")
        }
        let descriptor = staging.appendingPathComponent(ModelLayout.manifest)
        guard let bytes = try? SmallFile.read(descriptor, limit: ModelLayout.manifestLimit),
              SHA256Hasher.hex(bytes) == manifest.sha256
        else {
            try? FileManager.default.removeItem(at: staging)
            throw ServiceError(.invalidInput, "checksum_mismatch", "staged manifest.json does not match the verified manifest")
        }
        try checkCancellation()
        // Preserve named revision paths for readers that already selected them.
        let swapped = store.model == nil && FileManager.default.fileExists(atPath: target.path)
        let flags = swapped ? UInt32(RENAME_SWAP) : UInt32(RENAME_EXCL)
        let result = renamex_np(staging.path, target.path, flags)
        guard result == 0 else {
            let reason = String(cString: strerror(errno))
            try? FileManager.default.removeItem(at: staging)
            throw ServiceError(.internal, "install_failed", "cannot move the package into place: \(reason)")
        }
        if swapped { try? FileManager.default.removeItem(at: staging) }
    }

    /// Compiles packages for this cap, optionally rebuilding their compiled caches. Returns the compiled files that
    /// differ from the manifest's reference compile (expected only with another Core ML compiler; the
    /// self-test decides whether they compute correctly).
    public static func compile(in directory: URL, maxTokens: Int = 2048, rebuild: Bool = false, log: (String) -> Void = { _ in })
        async throws(ServiceError) -> [Manifest.Problem]
    {
        let manifest = try Manifest.load(from: directory)
        let packages = manifest.pplxContract?.requiredPackages(for: maxTokens) ?? ModelLayout.packages
        for name in packages {
            try checkCancellation()
            let package = directory.appendingPathComponent(name + ".mlpackage")
            guard FileManager.default.fileExists(atPath: package.path) else { continue }
            log("compiling \(name).mlpackage")
            do { _ = try await compiledModelURL(package, rebuild: rebuild) } catch {
                throw ServiceError(.unavailable, "compile_failed", "cannot compile \(name): \(error)")
            }
        }
        return manifest.verifyAll(in: directory, hash: true).compiled
    }
}
