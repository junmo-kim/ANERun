// Model files on disk.
//
//   <data-dir>/models/<id>/<revision>/          one directory per installed revision, never moved: the OS keys its
//                                          ANE compile cache on the model path, so a moved model recompiles
//   <data-dir>/models/<id>/current              text file naming the active revision directory; switched
//                                          atomically after a new revision is downloaded, verified and warmed
//   <model-dir>/manifest.json              files with sizes and SHA-256 (written by the packaging tools)
//   <model-dir>/.aneembed-warm.json        what was last warmed, on which OS build (doctor, serve)
import Foundation

/// Where the published model package comes from.
public struct ModelSource: Sendable, Equatable {
    public var repository: String
    /// A commit hash, so the files behind it cannot change.
    public var revision: String
    /// SHA-256 of manifest.json at that revision. With it, everything downloaded is checked against a
    /// hash compiled into the binary rather than one fetched alongside the files.
    public var manifestSHA256: String?

    public init(repository: String, revision: String, manifestSHA256: String? = nil) {
        self.repository = repository
        self.revision = revision
        self.manifestSHA256 = manifestSHA256
    }

    /// The package this build installs: junmo-k/embeddinggemma-2-ane at the commit of its 0.3.2 upload, checked
    /// against the manifest hash below.
    public static let pinned = ModelSource(
        repository: "junmo-k/embeddinggemma-2-ane", revision: "97558c0a567bd56e04f1a9945d69a16e1628587b",
        manifestSHA256: "91ff64fc39cec9a55ea51eeab8440425093728df57b20770f04fe55989231d6e")

    public var isPublished: Bool { revision != "UNPUBLISHED" && !revision.isEmpty }

    /// Whether `revision` can name a directory under models/: one plain path component of letters, digits, `.`, `_`
    /// and `-`, starting with a letter or digit (so never `.`, `..` or a hidden name), at most 64 characters.
    public static func isValidRevision(_ revision: String) -> Bool {
        isName(revision, maxLength: 64)
    }

    /// Whether `repository` is a Hugging Face repository id, `owner/name`, each part a plain name.
    public static func isValidRepository(_ repository: String) -> Bool {
        let parts = repository.split(separator: "/", omittingEmptySubsequences: false)
        return parts.count == 2 && parts.allSatisfy { isName(String($0), maxLength: 96) && !$0.contains("..") }
    }

    static func isName(_ name: String, maxLength: Int) -> Bool {
        guard let first = name.unicodeScalars.first, name.unicodeScalars.count <= maxLength,
            first.isASCII, first.properties.isAlphabetic || ("0"..."9").contains(first)
        else { return false }
        return name.unicodeScalars.allSatisfy {
            $0.isASCII && ($0.properties.isAlphabetic || ("0"..."9").contains($0) || ".-_".unicodeScalars.contains($0))
        }
    }

    /// Fails for a repository or revision that `--repo` / `--revision` must not take.
    public func validate() throws(ServiceError) {
        guard Self.isValidRepository(repository) else {
            throw .badRequest("invalid_repo", "--repo \(repository): expected a Hugging Face repository id, owner/name", param: "repo")
        }
        guard Self.isValidRevision(revision) else {
            throw .badRequest("invalid_revision",
                              "--revision \(revision): expected a commit hash or a name of letters, digits, '.', '_' and '-'",
                              param: "revision")
        }
    }

    /// Hugging Face "resolve" URL of a file (a manifest path) at the revision; every part is percent-encoded.
    public func url(for path: String) throws(ServiceError) -> URL {
        try validate()
        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.path = "/\(repository)/resolve/\(revision)/\(path)"
        guard Manifest.isValidPath(path), let url = components.url else {
            throw ServiceError(.invalidInput, "invalid_manifest", "cannot build a download URL for \(path)")
        }
        return url
    }
}

/// File names inside a model directory (the layout of the published package, see its manifest.json).
public enum ModelLayout {
    public static let manifest = "manifest.json"
    public static let tokenizer = "tokenizer.json"
    public static let tokenTable = "embeddings.bf16"
    public static let textModel = "EmbeddingGemma2Text"
    public static let visionModel = "EmbeddingGemma2Vision"
    public static let positionTable = "position_embeddings.f16"
    public static let selfTest = "selftest.json"
    public static let selfTestData = "selftest.bin"
    public static let warmState = ".aneembed-warm.json"
    public static let selfTestState = ".aneembed-selftest.json"
    static let warmLock = ".aneembed-warm.lock"
    static let compileLockSuffix = ".compile.lock"
    static let compileStageSuffix = ".compile.partial"
    public static let packages = [textModel, visionModel]

    static func isReservedManifestRoot(_ root: Substring) -> Bool {
        if [manifest, warmState, selfTestState, warmLock].contains(where: { root.compare($0, options: .caseInsensitive) == .orderedSame }) {
            return true
        }
        let name = root.folding(options: .caseInsensitive, locale: nil)
        return name.hasPrefix(".") && (name.hasSuffix(compileLockSuffix) || name.hasSuffix(compileStageSuffix))
    }

    /// Caps on the files read whole, far above their real size (the published package's in brackets), so that only
    /// a file that is not what it should be reaches them: manifest.json the cap of its download, 4 MiB
    /// [24,581 B]; selftest.json 16 MiB [146,366 B]; the state files and models/current 1 MiB [1,251, 451 and 41 B].
    /// selftest.bin is capped at the size manifest.json declares for it.
    public static let manifestLimit = DownloadLimit.manifestBytes
    public static let selfTestLimit = 16 << 20
    public static let stateLimit = 1 << 20

    /// Compiled model when present, else the package (compiled at install/warm time).
    public static func model(_ name: String, in directory: URL) -> URL? {
        for suffix in [".mlmodelc", ".mlpackage"] {
            let url = directory.appendingPathComponent(name + suffix)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return nil
    }

    /// Whether `path` (relative to the model directory) lies inside a compiled model (`*.mlmodelc/...`).
    /// Those files are produced on this Mac from the `.mlpackage`, not downloaded.
    public static func isCompiled(_ path: String) -> Bool {
        path.split(separator: "/").first?.hasSuffix(".mlmodelc") ?? false
    }
}

public struct ManifestEntry: Sendable, Equatable {
    public var path: String
    public var size: Int
    public var sha256: String
}

public struct Manifest: Sendable, Equatable {
    /// Package name (`name`; older drafts used `model`).
    public var model: String
    public var modelID: ModelID? {
        if model == "embeddinggemma-2-ane" { return .embeddingGemma2 }
        guard let id = ModelID(rawValue: model), !id.isDecision else { return nil }
        return id
    }
    public private(set) var pplxContract: PPLXPackageContract?
    /// Revision recorded in the manifest, if any (the published one is pinned by commit, not by this field).
    public var revision: String
    public var created: String
    public var files: [ManifestEntry]
    /// SHA-256 of the manifest bytes: identifies a package exactly.
    public var sha256: String
    /// The embed_S functions the package declares (`buckets`): length -> pool slots.
    public var buckets: [Int: Int]
    /// The windowed functions (`window`), when declared.
    public var window: WindowFunctions?
    /// The package has the per-layer long path of the first layout (`long_path`), which this build no longer runs.
    public var legacyLongPath: Bool

    public struct WindowFunctions: Sendable, Equatable {
        public var block: Int
        public var halo: Int
        public var bandColumns: Int
        /// winp: length -> pool slots.
        public var packing: [Int: Int]
        /// winl lengths.
        public var long: [Int]
    }

    /// Files to download or copy; compiled models are built on this Mac instead.
    public var sourceFiles: [ManifestEntry] { files.filter { !ModelLayout.isCompiled($0.path) } }

    public static func parse(_ data: Data) throws(ServiceError) -> Manifest {
        let root: JSONValue
        do { root = try JSONValue.parse(data) } catch {
            throw ServiceError(.unavailable, "invalid_manifest", "manifest.json is not valid JSON: \(error.description)")
        }
        guard let files = root["files"]?.arrayValue else {
            throw ServiceError(.unavailable, "invalid_manifest", "manifest.json has no files list")
        }
        var entries: [ManifestEntry] = []
        var paths: Set<String> = []
        for file in files {
            guard let path = file["path"]?.stringValue, let size = file["size"]?.intValue, size >= 0,
                let sha = file["sha256"]?.stringValue, sha.count == 64, sha.allSatisfy(\.isHexDigit), isValidPath(path)
            else { throw ServiceError(.unavailable, "invalid_manifest", "manifest.json has an invalid file entry") }
            let root = path.prefix { $0 != "/" }
            guard !ModelLayout.isReservedManifestRoot(root) else {
                throw ServiceError(.unavailable, "invalid_manifest", "\(path) uses the reserved root name \(root)")
            }
            guard paths.insert(path).inserted else {
                throw ServiceError(.unavailable, "invalid_manifest", "manifest.json lists \(path) more than once")
            }
            entries.append(ManifestEntry(path: path, size: size, sha256: sha.lowercased()))
        }
        var buckets: [Int: Int] = [:]
        for bucket in root["buckets"]?.arrayValue ?? [] {
            if let tokens = bucket["tokens"]?.intValue, let slots = bucket["slots"]?.intValue { buckets[tokens] = slots }
        }
        var window: WindowFunctions?
        if let declared = root["window"] {
            var packing: [Int: Int] = [:]
            for item in declared["packing"]?.arrayValue ?? [] {
                if let tokens = item["tokens"]?.intValue, let slots = item["slots"]?.intValue { packing[tokens] = slots }
            }
            window = WindowFunctions(block: declared["block"]?.intValue ?? 0, halo: declared["halo"]?.intValue ?? 0,
                                     bandColumns: declared["band_columns"]?.intValue ?? 0, packing: packing,
                                     long: (declared["long"]?.arrayValue ?? []).compactMap { $0["tokens"]?.intValue })
        }
        let name = root["name"]?.stringValue
        let declaredModel = root["model"]?.stringValue
        let model = name ?? declaredModel ?? ""
        var pplx: PPLXPackageContract?
        let textModels = [ModelID.pplx.rawValue, ModelID.qwen3Embedding.rawValue]
        if textModels.contains(model) || declaredModel.map(textModels.contains) == true {
            guard model == declaredModel else {
                throw ServiceError(.unavailable, "invalid_manifest", "manifest.json has inconsistent pplx model identifiers")
            }
            pplx = try PPLXPackageContract.parse(root, files: entries)
        }
        return Manifest(model: model, pplxContract: pplx,
                        revision: root["revision"]?.stringValue ?? "", created: root["created"]?.stringValue ?? "",
                        files: entries, sha256: SHA256Hasher.hex(data), buckets: buckets, window: window,
                        legacyLongPath: root["long_path"] != nil)
    }

    /// A relative path inside the package: non-empty components, none `.` or `..`, no control characters.
    public static func isValidPath(_ path: String) -> Bool {
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        return !path.hasPrefix("/") && components.allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
            && !path.contains("\\") && !path.unicodeScalars.contains { $0.properties.generalCategory == .control }
    }

    public static func load(from directory: URL) throws(ServiceError) -> Manifest {
        let data: Data
        do { data = try SmallFile.read(directory.appendingPathComponent(ModelLayout.manifest), limit: ModelLayout.manifestLimit) } catch {
            guard error != .missing else {
                throw ServiceError(.unavailable, "manifest_missing", "no manifest.json in \(directory.path)",
                                   hint: "install with `anerun models install --model ID` or point --model-dir at a model package")
            }
            throw ServiceError(.unavailable, "invalid_manifest", "\(directory.path): \(error.describe(ModelLayout.manifest))")
        }
        return try parse(data)
    }

    /// Fails unless this is the manifest `pinned` names (nil: any manifest). Cheap: only manifest.json (about 24 KB)
    /// is hashed, not the files it lists; those are checked against it by `checkForLoad`.
    public func checkPinned(_ pinned: String?) throws(ServiceError) {
        guard let pinned, sha256 != pinned.lowercased() else { return }
        throw ServiceError(.unavailable, "pinned_manifest_mismatch",
                           "the installed package of the pinned revision is not the one pinned in this build: its "
                           + "manifest.json has sha256 \(sha256), not \(pinned)",
                           hint: "reinstall it with `anerun models install`")
    }

    /// Checks the package identity before a named load or install uses any of its files.
    public func checkModel(_ expected: ModelID, directory: URL) throws(ServiceError) {
        guard let modelID else { throw unsupportedModel(in: directory) }
        guard modelID == expected else {
            throw ServiceError(.invalidInput, "model_mismatch", "expected \(expected.rawValue) but found \(modelID.rawValue) in \(directory.path)",
                               hint: "use the matching package with `anerun models install --model \(expected.rawValue) --from DIR` "
                               + "or point --model-dir at the package you want to load")
        }
    }

    func unsupportedModel(in directory: URL) -> ServiceError {
        let declared = model.isEmpty ? "(missing model identifier)" : model
        return ServiceError(.unavailable, "unsupported_model", "unsupported model \(declared) in \(directory.path)",
                            hint: "point --model-dir at a supported model package")
    }

    /// Fails unless the package declares exactly the contract this build implements: every packing bucket and
    /// windowed function the runtime may load, with their slots and window sizes.
    public func checkContract() throws(ServiceError) {
        if pplxContract != nil { return }
        guard modelID == .embeddingGemma2 else {
            throw ServiceError(.unavailable, "unsupported_model", "manifest.json does not declare a supported model (\(model))")
        }
        if legacyLongPath {
            throw ServiceError(.unavailable, "unsupported_model",
                               "the package has the per-layer long path of the first model layout; this anerun runs the "
                               + "windowed layout (winp/winl functions)",
                               hint: "install the model package this build pins (`anerun models install`)")
        }
        let full = BucketContract.fullAttentionLengths
        let expectedBuckets = Dictionary(uniqueKeysWithValues: full.map { ($0, BucketContract.slots(forLength: $0)) })
        guard buckets == expectedBuckets else {
            throw ServiceError(.unavailable, "unsupported_model",
                               "the package declares buckets \(buckets.keys.sorted()) (slots \(buckets.keys.sorted().map { buckets[$0]! })); "
                               + "this build loads \(full) with max(8, S/32) slots",
                               hint: "install the model package this build pins (`anerun models install`), or update anerun")
        }
        let expectedPacking = Dictionary(uniqueKeysWithValues: WindowContract.packingLengths.map {
            ($0, BucketContract.slots(forLength: $0))
        })
        guard let window, window.block == WindowContract.block, window.halo == WindowContract.halo,
            window.bandColumns == WindowContract.bandColumns, window.packing == expectedPacking,
            window.long.sorted() == WindowContract.longLengths
        else {
            let declared = window.map {
                "block \($0.block), halo \($0.halo), packing \($0.packing.keys.sorted()), long \($0.long)"
            } ?? "none"
            throw ServiceError(.unavailable, "unsupported_model",
                               "the package's windowed functions (\(declared)) differ from this build's (block "
                               + "\(WindowContract.block), halo \(WindowContract.halo), packing \(WindowContract.packingLengths), "
                               + "long \(WindowContract.longLengths))",
                               hint: "install the model package this build pins (`anerun models install`), or update anerun")
        }
    }

    /// Required files in each compiled model. Attention graphs also use weight.bin for constants.
    public static let compiledCoreFiles = ["model.mil", "coremldata.bin", "weights/weight.bin"]

    /// State of each compiled model whose package is present or listed: compiled (and whether it matches the
    /// packager's compile, informational), not compiled, or incomplete. `problem` is true for the last two.
    /// A compile is not byte-for-byte reproducible across Core ML compilers, so its SHA-256 is not a check;
    /// the load-time self-test is what decides whether it computes correctly.
    /// `hash` compares the compiled files with the packager's (informational; reads every compiled file).
    public func compiledStatus(in directory: URL, hash: Bool = false, maxTokens: Int? = nil)
        -> [(package: String, status: String, problem: Bool)]
    {
        let packages = pplxContract?.requiredPackages(for: maxTokens ?? ModelID.pplx.tokenRange.upperBound) ?? ModelLayout.packages
        let compiledNames = Set(packages.map { $0 + ".mlmodelc" })
        var compiledOnly = self
        compiledOnly.files = files.filter { file in
            guard let root = file.path.split(separator: "/").first else { return false }
            return compiledNames.contains(String(root))
        }
        let differing = Set(compiledOnly.verifyAll(in: directory, hash: hash).compiled.map { String($0.path.split(separator: "/")[0]) })
        return packages.compactMap { name in
            let compiledName = name + ".mlmodelc"
            let listed = pplxContract != nil
                || files.contains { $0.path.hasPrefix(name + ".mlpackage/") || $0.path.hasPrefix(compiledName + "/") }
            let compiled = directory.appendingPathComponent(compiledName)
            guard listed || FileManager.default.fileExists(atPath: compiled.path) else { return nil }
            guard FileManager.default.fileExists(atPath: compiled.path) else {
                return (name, "not compiled: run `anerun models warm`", true)
            }
            let damaged = Self.compiledCoreFiles.filter { file in
                guard case .success(let size) = Self.regularFileSize("\(compiledName)/\(file)", in: directory) else { return true }
                return size == 0
            }
            guard damaged.isEmpty else {
                return (name, "incomplete: \(damaged.joined(separator: ", ")) missing or empty; reinstall, or remove "
                    + "\(compiledName) and run `anerun models warm`", true)
            }
            guard hash else { return (name, "compiled", false) }
            return (name, differing.contains(compiledName)
                ? "compiled; differs from the packager's compile (expected with another Core ML compiler)"
                : "compiled; matches the packager's compile", false)
        }
    }

    /// Files whose bytes the load-time self-test does not exercise: hashed on every load.
    public static let hashedAtLoad = [ModelLayout.tokenizer, ModelLayout.selfTest, ModelLayout.selfTestData]

    /// The cheap check of every load: each source file present, regular and of its manifest size (the token table,
    /// the .mlpackage weights), and the files in `hashedAtLoad` hashed. A file the loader needs must be listed.
    /// `models verify` / `doctor --verify` still hash everything.
    public func checkForLoad(in directory: URL) -> [Problem] {
        var problems: [Problem] = []
        let listed = Set(files.map(\.path))
        let required: [String]
        var hashed = Self.hashedAtLoad
        if pplxContract != nil {
            hashed += ["tokenizer_config.json", "final_norm.f32"]
            required = PPLXPackageContract.requiredAssets
        } else {
            let selfTestFiles = [ModelLayout.selfTest, ModelLayout.selfTestData].filter {
                FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path)
            }
            required = [ModelLayout.tokenizer, ModelLayout.tokenTable] + selfTestFiles
        }
        for path in required where !listed.contains(path) {
            problems.append(Problem(path: path, issue: "not_in_manifest"))
        }
        for file in sourceFiles {
            if let problem = Self.fileProblem(file, in: directory, hash: hashed.contains(file.path)) {
                problems.append(problem)
            }
        }
        return problems
    }

    public struct Problem: Sendable, Equatable {
        public var path: String
        public var issue: String  // missing | not_regular_file | size_mismatch | sha256_mismatch
    }

    /// A regular file itself, not a symlink, FIFO, socket or device (those could hang the hash or point elsewhere).
    public static func isRegularFile(_ url: URL) -> Bool {
        var info = stat()
        return lstat(url.path, &info) == 0 && SmallFile.isRegular(info)
    }

    enum FileCheck: String, Error {
        case missing
        case notRegular = "not_regular_file"
    }

    /// The package root may be an explicitly selected symlink. Below it, each parent must be a real directory.
    static func regularFileSize(_ path: String, in directory: URL) -> Result<Int64, FileCheck> {
        guard isValidPath(path) else { return .failure(.notRegular) }
        var info = stat()
        var parent = directory
        for component in path.split(separator: "/").dropLast() {
            parent.appendPathComponent(String(component), isDirectory: false)
            guard lstat(parent.path, &info) == 0 else { return .failure(.missing) }
            guard (info.st_mode & S_IFMT) == S_IFDIR else { return .failure(.notRegular) }
        }
        let file = directory.appendingPathComponent(path)
        guard lstat(file.path, &info) == 0 else { return .failure(.missing) }
        guard SmallFile.isRegular(info) else { return .failure(.notRegular) }
        return .success(info.st_size)
    }

    package static func fileProblem(_ file: ManifestEntry, in directory: URL, hash: Bool, progress: ((String) -> Void)? = nil)
        -> Problem?
    {
        switch regularFileSize(file.path, in: directory) {
        case .failure(let error):
            return Problem(path: file.path, issue: error.rawValue)
        case .success(let size) where size != Int64(file.size):
            return Problem(path: file.path, issue: "size_mismatch")
        case .success:
            guard hash else { return nil }
            progress?(file.path)
            let url = directory.appendingPathComponent(file.path)
            return (try? SHA256Hasher.hex(fileAt: url)) == file.sha256 ? nil : Problem(path: file.path, issue: "sha256_mismatch")
        }
    }

    /// Checks source files; `hash: false` only checks presence, type and size.
    public func verify(in directory: URL, hash: Bool, progress: ((String) -> Void)? = nil) -> [Problem] {
        sourceFiles.compactMap { Self.fileProblem($0, in: directory, hash: hash, progress: progress) }
    }

    /// Checks source and compiled files.
    ///
    /// Files of a compiled model (`X.mlmodelc/...`) are checked only when that compiled model exists:
    /// they are built on this Mac from `X.mlpackage`, and the Core ML compiler of another macOS release
    /// may produce different bytes (the load-time self-test is what decides whether they compute correctly).
    /// Their problems come back separately in `compiled`.
    public func verifyAll(in directory: URL, hash: Bool, progress: ((String) -> Void)? = nil)
        -> (source: [Problem], compiled: [Problem])
    {
        var source: [Problem] = []
        var compiled: [Problem] = []
        for file in files {
            let isCompiled = ModelLayout.isCompiled(file.path)
            if isCompiled {
                let root = String(file.path.split(separator: "/")[0])
                var info = stat()
                guard lstat(directory.appendingPathComponent(root).path, &info) == 0 else { continue }
            }
            if let problem = Self.fileProblem(file, in: directory, hash: hash, progress: progress) {
                if isCompiled { compiled.append(problem) } else { source.append(problem) }
            }
        }
        return (source, compiled)
    }
}

/// What `models warm` compiled, so a later start knows whether the OS cache is likely still warm.
///
/// The OS keeps the Neural Engine compile per model path, macOS build and executable: the same binary
/// copied to another path compiles again (measured), so an upgraded anerun recompiles once too.
public struct WarmState: Sendable, Equatable {
    public var osBuild: String
    public var binaryVersion: String
    /// SHA-256 of the executable that warmed (empty in states written before it was recorded).
    public var binary: String
    /// SHA-256 of the manifest of the package that was warmed (empty in states written before it was recorded).
    public var manifestSHA256: String
    public var maxTokens: Int
    public var functions: [String]
    public var warmedAt: Date
    public var seconds: Double
    public var selfTest: SelfTestReport?

    public init(osBuild: String, binaryVersion: String, binary: String = Executable.identity, manifestSHA256: String,
                maxTokens: Int, functions: [String], warmedAt: Date, seconds: Double, selfTest: SelfTestReport? = nil)
    {
        self.osBuild = osBuild
        self.binaryVersion = binaryVersion
        self.binary = binary
        self.manifestSHA256 = manifestSHA256
        self.maxTokens = maxTokens
        self.functions = functions
        self.warmedAt = warmedAt
        self.seconds = seconds
        self.selfTest = selfTest
    }

    public var json: JSONValue {
        var object: JSONObject = [
            "os_build": .string(osBuild), "binary_version": .string(binaryVersion), "binary_sha256": .string(binary),
            "manifest_sha256": .string(manifestSHA256), "max_tokens": .int(maxTokens), "functions": .array(functions.map(JSONValue.string)),
            "warmed_at": .string(ISO8601DateFormatter().string(from: warmedAt)), "seconds": .number(seconds),
        ]
        if let selfTest { object["selftest"] = selfTest.json }
        return .object(object)
    }

    public static func load(from directory: URL) -> WarmState? {
        guard let data = try? SmallFile.read(directory.appendingPathComponent(ModelLayout.warmState), limit: ModelLayout.stateLimit),
            let root = try? JSONValue.parse(data), let build = root["os_build"]?.stringValue,
            let maxTokens = root["max_tokens"]?.intValue
        else { return nil }
        return WarmState(
            osBuild: build, binaryVersion: root["binary_version"]?.stringValue ?? "",
            binary: root["binary_sha256"]?.stringValue ?? "", manifestSHA256: root["manifest_sha256"]?.stringValue ?? "",
            maxTokens: maxTokens,
            functions: root["functions"]?.arrayValue?.compactMap(\.stringValue) ?? [],
            warmedAt: root["warmed_at"]?.stringValue.flatMap { ISO8601DateFormatter().date(from: $0) } ?? .distantPast,
            seconds: root["seconds"]?.doubleValue ?? 0, selfTest: root["selftest"].flatMap(SelfTestReport.parse))
    }

    public func save(to directory: URL) throws {
        try (json.render(as: .json) + "\n").write(to: directory.appendingPathComponent(ModelLayout.warmState),
                                                  atomically: true, encoding: .utf8)
    }

    /// Publish a pplx load's validated capability. A healthy shorter load preserves broader coverage.
    func recordCapability(requestedMaxTokens: Int, in directory: URL) throws {
        let path = directory.appendingPathComponent(ModelLayout.warmLock).path
        let descriptor = open(path, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK, 0o644)
        guard descriptor >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        defer { close(descriptor) }
        var info = stat()
        guard fstat(descriptor, &info) == 0, SmallFile.isRegular(info) else {
            throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: path])
        }
        while flock(descriptor, LOCK_EX) != 0 {
            if errno != EINTR { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        }
        var snapshot = self
        if maxTokens == requestedMaxTokens, let previous = Self.load(from: directory),
           previous.covers(osBuild: osBuild, maxTokens: maxTokens, binary: binary,
                           manifestSHA256: manifestSHA256, requiredFunctions: functions)
        {
            guard let selfTest else { return }
            snapshot = previous
            snapshot.selfTest = selfTest.merged(with: previous.selfTest)
        }
        try snapshot.save(to: directory)
    }

    /// Fresh when this executable warmed this package on this OS build for at least `maxTokens`.
    public func covers(osBuild: String, maxTokens: Int, binary: String = Executable.identity, manifestSHA256: String,
                       requiredFunctions: [String] = []) -> Bool
    {
        self.osBuild == osBuild && self.maxTokens >= maxTokens && self.binary == binary && self.manifestSHA256 == manifestSHA256
            && Set(requiredFunctions).isSubset(of: functions)
    }
}

/// Reads a small file whole, only if it is a regular file within a size cap. A FIFO, socket or device (or, unless
/// `followSymlink`, a symlink) is refused before a byte is read, so the read can neither block nor run without end.
/// The file is opened without blocking and checked through the open descriptor: what is checked is what is read.
public enum SmallFile {
    public enum Failure: Error, Equatable {
        case missing
        case notRegularFile
        case tooLarge(limit: Int)
        case unreadable(String)

        /// What went wrong with the file called `name`.
        public func describe(_ name: String) -> String {
            switch self {
            case .missing: return "\(name) is missing"
            case .notRegularFile: return "\(name) is not a regular file (a symlink, FIFO, socket or device)"
            case .tooLarge(let limit): return "\(name) is larger than \(limit) bytes"
            case .unreadable(let reason): return "cannot read \(name): \(reason)"
            }
        }
    }

    static func isRegular(_ info: stat) -> Bool { (info.st_mode & S_IFMT) == S_IFREG }

    public static func read(_ url: URL, limit: Int, followSymlink: Bool = false) throws(Failure) -> Data {
        let descriptor = open(url.path, O_RDONLY | O_NONBLOCK | O_NOCTTY | O_CLOEXEC | (followSymlink ? 0 : O_NOFOLLOW))
        guard descriptor >= 0 else {
            switch errno {
            case ENOENT, ENOTDIR: throw .missing
            case ELOOP: throw .notRegularFile  // a symlink, refused by O_NOFOLLOW
            default: throw .unreadable(String(cString: strerror(errno)))
            }
        }
        let file = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        var info = stat()
        guard fstat(descriptor, &info) == 0 else { throw .unreadable(String(cString: strerror(errno))) }
        guard isRegular(info) else { throw .notRegularFile }
        guard info.st_size <= limit else { throw .tooLarge(limit: limit) }
        let data: Data
        do { data = try file.read(upToCount: limit + 1) ?? Data() } catch { throw .unreadable(error.localizedDescription) }
        guard data.count <= limit else { throw .tooLarge(limit: limit) }  // it grew after the check
        return data
    }
}

/// The running executable, identified by content.
public enum Executable {
    public static let identity: String = {
        guard let path = Bundle.main.executablePath, let hash = try? SHA256Hasher.hex(fileAt: URL(fileURLWithPath: path))
        else { return "" }
        return hash
    }()
}

public struct ModelStore: Sendable {
    public let dataDirectory: URL
    let model: ModelID?

    public init(dataDirectory: URL, model: ModelID? = nil) {
        self.dataDirectory = dataDirectory
        self.model = model
    }

    private var namespace: URL { dataDirectory.appendingPathComponent("models", isDirectory: true) }

    public var modelsDirectory: URL {
        guard let model else { return namespace }
        return namespace.appendingPathComponent(model.rawValue, isDirectory: true)
    }

    var currentPointer: URL { modelsDirectory.appendingPathComponent("current") }

    /// For a named store, resolveDirectory() reports invalid or missing revisions.
    public var currentDirectory: URL {
        guard model == nil else { return (try? resolveDirectory()) ?? modelsDirectory }
        return legacyActiveDirectory() ?? modelsDirectory.appendingPathComponent(ModelSource.pinned.revision, isDirectory: true)
    }

    private func legacyActiveDirectory() -> URL? {
        guard let data = try? SmallFile.read(currentPointer, limit: ModelLayout.stateLimit, followSymlink: true) else { return nil }
        let revision = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        guard ModelSource.isValidRevision(revision) else { return nil }
        return modelsDirectory.appendingPathComponent(revision, isDirectory: true)
    }

    /// Resolves a single named revision. EG2 can read the old namespace when its own pointer is absent.
    public func resolveDirectory() throws(ServiceError) -> URL {
        guard let model else { return currentDirectory }
        if let directory = try activeDirectory() { return directory }
        guard model == .embeddingGemma2 else { throw missingModel("no active revision at \(currentPointer.path)") }

        let legacy = ModelStore(dataDirectory: dataDirectory).currentDirectory
        let revision = legacy.lastPathComponent.lowercased()
        guard revision != "current", ModelID(rawValue: revision) == nil, try isDirectory(legacy) else {
            throw missingModel("no active revision at \(currentPointer.path) or legacy package at \(legacy.path)")
        }
        return legacy
    }

    /// Reports whether the store's own pointer names this directory.
    public func isActive(_ directory: URL) -> Bool {
        let active = model == nil ? legacyActiveDirectory() : try? activeDirectory()
        return active?.standardizedFileURL.resolvingSymlinksInPath().path
            == directory.standardizedFileURL.resolvingSymlinksInPath().path
    }

    func activeDirectory() throws(ServiceError) -> URL? {
        try checkNamespace()
        let data: Data
        do { data = try SmallFile.read(currentPointer, limit: ModelLayout.stateLimit) } catch {
            if error == .missing { return nil }
            throw missingModel(error.describe(currentPointer.path))
        }
        let revision = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        guard ModelSource.isValidRevision(revision), revision.lowercased() != "current" else {
            throw missingModel("\(currentPointer.path) does not name a valid revision")
        }
        let directory = modelsDirectory.appendingPathComponent(revision, isDirectory: true)
        guard try isDirectory(directory) else {
            throw missingModel("\(currentPointer.path) points to missing revision \(directory.path)")
        }
        return directory
    }

    private func missingModel(_ detail: String) -> ServiceError {
        let id = model?.rawValue ?? ModelID.embeddingGemma2.rawValue
        return ServiceError(.unavailable, "model_missing", "\(id): \(detail)",
                            hint: "install with `anerun models install --model \(id)` or set --model-dir to an existing package")
    }

    /// Validates the directories that hold a named install lock, allowing a new store to be created.
    public func checkNamespace() throws(ServiceError) {
        guard model != nil else { return }
        if try isDirectory(namespace) { _ = try isDirectory(modelsDirectory) }
    }

    /// Checks an existing named store component without following symbolic links.
    private func isDirectory(_ url: URL) throws(ServiceError) -> Bool {
        var info = stat()
        if lstat(url.path, &info) != 0 {
            let error = errno
            if error == ENOENT { return false }
            throw ServiceError(.unavailable, "invalid_model_store", "cannot inspect \(url.path): \(String(cString: strerror(error)))")
        }
        guard (info.st_mode & S_IFMT) == S_IFDIR else {
            throw ServiceError(.unavailable, "invalid_model_store", "\(url.path) must be a directory and cannot be a symbolic link")
        }
        return true
    }

    /// The directory of a validated revision name.
    public func directory(for revision: String) throws(ServiceError) -> URL {
        guard ModelSource.isValidRevision(revision), model == nil || revision.lowercased() != "current" else {
            throw .badRequest("invalid_revision", "\(revision) is not a valid revision name", param: "revision")
        }
        let url = modelsDirectory.appendingPathComponent(revision, isDirectory: true)
        try checkInside(url)
        return url
    }

    /// Checks that the installer is operating on a direct child of the models directory.
    public func checkInside(_ url: URL) throws(ServiceError) {
        let parent = url.deletingLastPathComponent()
        let name = url.lastPathComponent
        let isInside: Bool
        if model == nil {
            isInside = parent.standardizedFileURL.resolvingSymlinksInPath().path
                == modelsDirectory.standardizedFileURL.resolvingSymlinksInPath().path
        } else {
            try checkNamespace()
            isInside = parent.path == modelsDirectory.path && name.lowercased() != "current"
        }
        guard isInside, ModelSource.isName(String(name.drop { $0 == "." }), maxLength: 160) else {
            throw ServiceError(.internal, "outside_models_directory", "refusing to replace \(url.path): not inside \(modelsDirectory.path)")
        }
        if model != nil { _ = try isDirectory(url) }
    }

    /// Points `current` at a revision by writing a temporary file and atomically renaming it.
    public func activate(revision: String) throws {
        let directory = try directory(for: revision)
        if model != nil, try !isDirectory(directory) { throw missingModel("revision \(directory.path) does not exist") }
        try FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
        let temporary = modelsDirectory.appendingPathComponent(".current.\(UUID().uuidString)")
        try Data((revision + "\n").utf8).write(to: temporary, options: [.withoutOverwriting])
        defer { _ = unlink(temporary.path) }
        if rename(temporary.path, currentPointer.path) != 0 {
            throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: currentPointer.path])
        }
    }
}

/// The running macOS build, e.g. "26A434".
public var operatingSystemBuild: String {
    var size = 0
    sysctlbyname("kern.osversion", nil, &size, nil, 0)
    var buffer = [UInt8](repeating: 0, count: max(size, 1))
    sysctlbyname("kern.osversion", &buffer, &size, nil, 0)
    return String(decoding: buffer.prefix { $0 != 0 }, as: UTF8.self)
}
