import Foundation
import Testing

@testable import EmbedCore

@Suite("Config")
struct ConfigTests {
    @Test func defaultsEnvironmentAndFlags() throws {
        let defaults = try Config(environment: [:])
        #expect(defaults.port == 18180 && defaults.host == "127.0.0.1" && defaults.maxTokens == 2048)
        #expect(defaults.sources["port"] == .default)

        let env = try Config(environment: ["ANEEMBED_PORT": "9000", "ANEEMBED_MAX_TOKENS": "8192"])
        #expect(env.port == 9000 && env.maxTokens == 8192 && env.sources["port"] == .environment)

        let both = ["ANERUN_PORT": "9002", "ANEEMBED_PORT": "9000", "ANERUN_MAX_TOKENS": "", "ANEEMBED_MAX_TOKENS": "8192"]
        let branded = try Config(environment: both)
        #expect(branded.port == 9002 && branded.maxTokens == 8192 && branded.sources["port"] == .environment)

        let flag = try Config(flags: ["port": "9001"], environment: both)
        #expect(flag.port == 9001 && flag.sources["port"] == .flag)
    }

    @Test func everyOptionHasTheSameNameAsFlagAndEnvironmentVariable() {
        for option in Config.options {
            #expect(option.environmentName == "ANERUN_" + option.name.uppercased().replacingOccurrences(of: "-", with: "_"))
        }
    }

    @Test func rejectsInvalidValues() {
        #expect(throws: ServiceError.self) { try Config(flags: ["max-tokens": "32769"], environment: [:]) }
        #expect(throws: ServiceError.self) { try Config(flags: ["port": "x"], environment: [:]) }
        #expect(throws: ServiceError.self) { try Config(flags: ["nope": "1"], environment: [:]) }
        #expect(throws: ServiceError.self) { try Config(environment: ["ANEEMBED_RUNTIME": "gpu"]) }
        #expect(throws: ServiceError.self) { try Config(flags: ["max-connections": "0"], environment: [:]) }
    }

    @Test func connectionLimit() throws {
        #expect(try Config(environment: [:]).maxConnections == 64)
        #expect(try Config(environment: ["ANEEMBED_MAX_CONNECTIONS": "8"]).maxConnections == 8)
    }

    @Test func modelDirectoryFollowsTheDataDirectory() throws {
        let config = try Config(flags: ["data-dir": "/tmp/x"], environment: [:])
        #expect(config.modelDirectory.path.hasPrefix("/tmp/x/models/"))
        let explicit = try Config(flags: ["model-dir": "/opt/m"], environment: [:])
        #expect(explicit.modelDirectory.path == "/opt/m")
        #expect(config.printable["port"]?["source"] == "default")
        #expect(config.printable["data-dir"]?["env"] == "ANERUN_DATA_DIR")
    }

    @Test func defaultStoreStillUsesTheLegacyPath() throws {
        let config = try Config(environment: [:])
        let expected = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/ANEEmbed", isDirectory: true)
        #expect(config.dataDirectory.path == expected.path)
    }

    @Test func modelSelectionResolvesPrecedenceAndCaps() throws {
        let dataDirectory = FileManager.default.temporaryDirectory.appendingPathComponent("aneembed-selection-\(UUID().uuidString)")
        let candidate = try Config(
            flags: ["model": "pplx-embed-v1-0.6b", "runtime": "fake", "max-tokens": "32768", "data-dir": dataDirectory.path],
            environment: ["ANEEMBED_MODEL": "embeddinggemma-2"])
        let selected = try candidate.selectingModel(for: "serve")
        #expect(selected.modelID == .pplx)
        #expect(selected.modelName == "pplx-embed-v1-0.6b")
        #expect(selected.sources["model"] == .flag)
        #expect(selected.maxTokens == 32768)
        #expect(selected.modelDirectory == ModelStore(dataDirectory: dataDirectory, model: .pplx).modelsDirectory)
        #expect(try selected.selectingModel(for: "load") == selected)

        let install = try Config(
            flags: ["model-name": "demo", "data-dir": dataDirectory.path],
            environment: ["ANEEMBED_MODEL": "pplx-embed-v1-0.6b", "ANEEMBED_MAX_TOKENS": "1"])
        let target = try install.selectingModel(for: "models install", requiresName: true)
        #expect(target.modelID == .pplx && target.modelName == "demo")
        let installed = target.withModelDirectory(URL(fileURLWithPath: "/tmp/aneembed-test/revision"))
        #expect(installed.modelID == .pplx)
        #expect(try installed.selectingModel(for: "models warm") == installed)
        #expect(!FileManager.default.fileExists(atPath: dataDirectory.path))

        for cap in ["1", "8193"] {
            let eg2 = try Config(flags: ["model": "embeddinggemma-2", "max-tokens": cap, "runtime": "fake"], environment: [:])
            let error = #expect(throws: ServiceError.self) { try eg2.selectingModel(for: "embed") }
            #expect(error?.reason == "invalid_option" && error?.param == "max-tokens")
        }
    }

    @Test func missingUnknownAndConflictingSelectionsAreStructuredErrors() throws {
        let unselected = try Config(flags: ["runtime": "fake"], environment: [:])
        let missing = #expect(throws: ServiceError.self) { try unselected.selectingModel(for: "serve") }
        #expect(missing?.exitCode == .usage && missing?.reason == "missing_model")
        #expect(missing?.message.contains("serve") == true && missing?.message.contains(ModelID.catalog) == true)
        let printed = try unselected.selectingModel(for: "--print-config", required: false)
        #expect(printed.modelID == nil && printed.printable["model-dir"]?["value"] == "")

        let unknown = try Config(environment: ["ANEEMBED_MODEL": "other"])
        let invalid = #expect(throws: ServiceError.self) { try unknown.selectingModel(for: "embed") }
        #expect(invalid?.reason == "unknown_model" && invalid?.message.contains("other") == true)
        let conflict = try Config(flags: ["model": "embeddinggemma-2"], environment: ["ANEEMBED_MODEL_DIR": "/tmp/package"])
        let conflicting = #expect(throws: ServiceError.self) { try conflict.selectingModel(for: "serve") }
        #expect(conflicting?.reason == "conflicting_options" && conflicting?.exitCode == .usage)

        let pathOnly = try Config(flags: ["model-dir": "/tmp/package"], environment: [:])
        let requiresName = #expect(throws: ServiceError.self) {
            try pathOnly.selectingModel(for: "models install", requiresName: true)
        }
        #expect(requiresName?.reason == "missing_model")
    }

    @Test func explicitDirectoryInfersTheLegacyPackageID() throws {
        let directory = try DoctorTests.modelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try Config(flags: ["model-dir": directory.path], environment: ["ANEEMBED_MODEL_DIR": "/tmp/other"])
        let selected = try config.selectingModel(for: "models verify")
        #expect(selected.modelID == .embeddingGemma2 && selected.modelName == "embeddinggemma-2")
        #expect(selected.modelDirectory.path == directory.resolvingSymlinksInPath().path)
        #expect(selected.pinnedManifestSHA256 == nil)
    }

}

@Suite("Model store")
struct ModelStoreTests {
    func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("aneembed-store-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test func decisionBundlesUseTheSameNamedPointerAndRequireThePinnedCheckpoint() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = ModelID.decision(.glide2B)
        let store = ModelStore(dataDirectory: root, model: model)
        let bundle = try store.directory(for: "fixture")
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        let manifest = bundle.appendingPathComponent("model.json")
        func writeRevision(_ revision: String) throws {
            let header: JSONValue = ["source": ["repo": "fastino/GLiDE-2B", "revision": .string(revision)]]
            try header.jsonData().write(to: manifest)
        }
        try writeRevision("00f122c120cdeed82284106cb77d9bab5a50ad16")
        try store.activate(revision: "fixture")
        let named = try Config(flags: ["data-dir": root.path, "model": "glide-2b"], environment: [:])
            .selectingModel(for: "serve")
        let direct = try Config(flags: ["model-dir": bundle.path], environment: [:]).selectingModel(for: "decide")
        #expect(named.modelID == model && direct.modelID == model)
        #expect(named.modelDirectory == direct.modelDirectory)
        #expect(throws: ServiceError.self) { try ModelID.identify(in: bundle, expected: .decision(.valen4B)) }
        try writeRevision("wrong")
        #expect(throws: ServiceError.self) { try ModelID.identify(in: bundle) }
    }

    @Test func namedStoresKeepIndependentPointersAndSnapshotTheResolvedRevision() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let legacy = ModelStore(dataDirectory: directory)
        let eg2 = ModelStore(dataDirectory: directory, model: .embeddingGemma2)
        let pplx = ModelStore(dataDirectory: directory, model: .pplx)
        let package = try DoctorTests.modelDirectory()
        let legacyPath = try legacy.directory(for: "legacy")
        try legacy.activate(revision: "legacy")
        try FileManager.default.moveItem(at: package, to: legacyPath)
        #expect(try eg2.resolveDirectory().path == legacyPath.path)
        #expect(!eg2.isActive(legacyPath))
        let missing = #expect(throws: ServiceError.self) { try pplx.resolveDirectory() }
        #expect(missing?.reason == "model_missing")

        let raw = try Config(flags: ["data-dir": directory.path, "model": "embeddinggemma-2"], environment: [:])
        let selected = try raw.selectingModel(for: "serve")
        let eg2Path = try eg2.directory(for: "v2")
        let pplxPath = try pplx.directory(for: "v2")
        for path in [eg2Path, pplxPath] {
            try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: legacyPath, to: path)
        }
        try eg2.activate(revision: "v2")
        try pplx.activate(revision: "v2")
        #expect(try eg2.resolveDirectory().path == eg2Path.path)
        #expect(try pplx.resolveDirectory().path == pplxPath.path)
        #expect(eg2.isActive(eg2Path) && pplx.isActive(pplxPath) && !pplx.isActive(eg2Path))
        #expect(legacy.currentDirectory.path == legacyPath.path)
        #expect(selected.modelDirectory.path == legacyPath.resolvingSymlinksInPath().path)
        #expect(try raw.selectingModel(for: "serve").modelDirectory.path == eg2Path.resolvingSymlinksInPath().path)

        let wrongModel = try Config(flags: ["data-dir": directory.path, "model": "pplx-embed-v1-0.6b"], environment: [:])
        let mismatch = #expect(throws: ServiceError.self) { try wrongModel.selectingModel(for: "serve") }
        #expect(mismatch?.reason == "model_mismatch" && mismatch?.exitCode == .dataError)
        #expect(mismatch?.message.contains(pplxPath.resolvingSymlinksInPath().path) == true)

        try "absent\n".write(to: eg2.currentPointer, atomically: true, encoding: .utf8)
        let broken = #expect(throws: ServiceError.self) { try eg2.resolveDirectory() }
        #expect(broken?.reason == "model_missing")
        try "../legacy\n".write(to: eg2.currentPointer, atomically: true, encoding: .utf8)
        #expect(throws: ServiceError.self) { try eg2.resolveDirectory() }
    }

    @Test func namedStoreRejectsSymlinksAtTheNamespacePointerAndRevision() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ModelStore(dataDirectory: directory, model: .embeddingGemma2)
        let models = store.modelsDirectory.deletingLastPathComponent()
        let external = directory.appendingPathComponent("elsewhere")
        try FileManager.default.createDirectory(at: external, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: models, withDestinationURL: external)
        #expect(throws: ServiceError.self) { try store.checkNamespace() }
        #expect(throws: ServiceError.self) { try store.directory(for: "v1") }

        try FileManager.default.removeItem(at: models)
        try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: store.modelsDirectory, withDestinationURL: external)
        #expect(throws: ServiceError.self) { try store.checkNamespace() }

        try FileManager.default.removeItem(at: store.modelsDirectory)
        try Data().write(to: store.modelsDirectory)
        #expect(throws: ServiceError.self) { try store.checkNamespace() }
        try FileManager.default.removeItem(at: store.modelsDirectory)
        try FileManager.default.createDirectory(at: store.modelsDirectory, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: store.currentPointer, withDestinationURL: external.appendingPathComponent("absent"))
        #expect(throws: ServiceError.self) { try store.resolveDirectory() }
        try FileManager.default.removeItem(at: store.currentPointer)

        let revision = store.modelsDirectory.appendingPathComponent("v1")
        try FileManager.default.createSymbolicLink(at: revision, withDestinationURL: external)
        try "v1\n".write(to: store.currentPointer, atomically: true, encoding: .utf8)
        #expect(throws: ServiceError.self) { try store.directory(for: "v1") }
        #expect(throws: ServiceError.self) { try store.resolveDirectory() }
        #expect(!store.isActive(revision))
    }

    @Test func duplicateManifestPathsAreRejected() {
        let entry: JSONValue = ["path": "tokenizer.json", "size": 1, "sha256": .string(String(repeating: "a", count: 64))]
        let manifest: JSONValue = ["files": .array([entry, entry])]
        let error = #expect(throws: ServiceError.self) { try Manifest.parse(manifest.jsonData()) }
        #expect(error?.reason == "invalid_manifest")
    }

    @Test func verifiesFilesAgainstTheManifest() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let payload = Data("weights".utf8)
        try payload.write(to: directory.appendingPathComponent("a.bin"))
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("m.mlmodelc"), withIntermediateDirectories: true)
        try Data("x".utf8).write(to: directory.appendingPathComponent("m.mlmodelc/model.mil"))
        let manifest: JSONValue = [
            "model": "embeddinggemma-2", "revision": "r1",
            "files": [
                ["path": "a.bin", "size": .int(payload.count), "sha256": .string(SHA256Hasher.hex(payload))],
                ["path": "m.mlmodelc/model.mil", "size": 1, "sha256": .string(String(repeating: "0", count: 64))],
                ["path": "missing.bin", "size": 1, "sha256": .string(String(repeating: "0", count: 64))],
            ],
        ]
        try manifest.jsonData().write(to: directory.appendingPathComponent("manifest.json"))
        let loaded = try Manifest.load(from: directory)
        #expect(loaded.revision == "r1" && loaded.files.count == 3)
        #expect(loaded.sourceFiles.map(\.path) == ["a.bin", "missing.bin"])
        #expect(loaded.verify(in: directory, hash: false) == [.init(path: "missing.bin", issue: "missing")])
        // Compiled files are built on this Mac: reported apart, and only when the compiled model exists.
        let checked = loaded.verifyAll(in: directory, hash: true)
        #expect(checked.source == [.init(path: "missing.bin", issue: "missing")])
        #expect(checked.compiled == [.init(path: "m.mlmodelc/model.mil", issue: "sha256_mismatch")])
        try FileManager.default.removeItem(at: directory.appendingPathComponent("m.mlmodelc"))
        #expect(loaded.verifyAll(in: directory, hash: true).compiled.isEmpty)
    }

    /// "Already installed" means this exact package: the manifest must hash to the expected value (pinned in the
    /// binary, or fetched for the requested source), not just match the files next to it.
    @Test func anInstalledPackageMustMatchTheExpectedManifest() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let payload = Data("weights".utf8)
        try payload.write(to: directory.appendingPathComponent("a.bin"))
        let manifest: JSONValue = ["files": [["path": "a.bin", "size": .int(payload.count), "sha256": .string(SHA256Hasher.hex(payload))]]]
        let data = manifest.jsonData()
        try data.write(to: directory.appendingPathComponent("manifest.json"))
        #expect(ModelInstaller.isInstalled(at: directory, manifestSHA256: SHA256Hasher.hex(data)))
        // A self-consistent package that is not the expected one is not "installed".
        #expect(!ModelInstaller.isInstalled(at: directory, manifestSHA256: ModelSource.pinned.manifestSHA256!))
        try Data("Weights".utf8).write(to: directory.appendingPathComponent("a.bin"))
        #expect(!ModelInstaller.isInstalled(at: directory, manifestSHA256: SHA256Hasher.hex(data)))
        #expect(!ModelInstaller.isInstalled(at: directory.appendingPathComponent("none"), manifestSHA256: SHA256Hasher.hex(data)))
    }

    /// Only regular files are hashed or copied: a symlink (to /dev/zero, say) or a FIFO would hang the hash.
    @Test func packageFilesMustBeRegularFiles() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("pkg")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        let payload = Data("weights".utf8)
        try payload.write(to: directory.appendingPathComponent("real.bin"))
        try FileManager.default.createSymbolicLink(atPath: source.appendingPathComponent("a.bin").path,
                                                   withDestinationPath: directory.appendingPathComponent("real.bin").path)
        #expect(mkfifo(source.appendingPathComponent("f.bin").path, 0o600) == 0)
        let manifest: JSONValue = ["files": [
            ["path": "a.bin", "size": .int(payload.count), "sha256": .string(SHA256Hasher.hex(payload))],
            ["path": "f.bin", "size": 0, "sha256": .string(SHA256Hasher.hex(Data()))],
        ]]
        try manifest.jsonData().write(to: source.appendingPathComponent("manifest.json"))
        let loaded = try Manifest.load(from: source)
        #expect(loaded.verify(in: source, hash: true) == [.init(path: "a.bin", issue: "not_regular_file"),
                                                         .init(path: "f.bin", issue: "not_regular_file")])
        let store = ModelStore(dataDirectory: directory)
        try FileManager.default.createDirectory(at: store.modelsDirectory, withIntermediateDirectories: true)
        #expect(throws: ServiceError.self) {
            try ModelInstaller.copy(from: source, into: store.directory(for: "local-x"), store: store)
        }
        #expect(!FileManager.default.fileExists(atPath: store.modelsDirectory.appendingPathComponent("local-x").path))
    }

    /// Replaces `name` in `directory` with a FIFO.
    func makeFIFO(_ name: String, in directory: URL) throws {
        try? FileManager.default.removeItem(at: directory.appendingPathComponent(name))
        #expect(mkfifo(directory.appendingPathComponent(name).path, 0o600) == 0)
    }

    /// Replaces `name` in `directory` with a symlink to `destination`.
    func makeSymlink(_ name: String, in directory: URL, to destination: String) throws {
        try? FileManager.default.removeItem(at: directory.appendingPathComponent(name))
        try FileManager.default.createSymbolicLink(atPath: directory.appendingPathComponent(name).path, withDestinationPath: destination)
    }

    /// The small files read whole are read only when they are regular files: a FIFO, or a symlink to a device, is an
    /// error at once instead of a read that blocks or never ends. A test that hangs here fails at the time limit.
    @Test(.timeLimit(.minutes(1))) func aFIFOOrDeviceInPlaceOfAStoreFileIsAnError() throws {
        let package = try DoctorTests.modelDirectory()
        defer { try? FileManager.default.removeItem(at: package) }
        let config = try Config(flags: ["model-dir": package.path], environment: [:])
        try WarmState(osBuild: "b", binaryVersion: "v", manifestSHA256: "m", maxTokens: 2048, functions: [], warmedAt: Date(),
                      seconds: 1).save(to: package)
        try SelfTestReport(osBuild: "b", ranAt: Date(), seconds: 0, results: []).save(to: package)
        #expect(WarmState.load(from: package) != nil && SelfTestReport.load(from: package) != nil)
        #expect(DoctorTests.problems(config).allSatisfy { !$0.contains("self-test data") && !$0.contains("manifest.json") })

        let specials: [(String) throws -> Void] = [
            { try makeFIFO($0, in: package) }, { try makeSymlink($0, in: package, to: "/dev/zero") },
        ]
        for special in specials {
            for name in [ModelLayout.selfTest, ModelLayout.selfTestData, ModelLayout.manifest] {
                let original = try Data(contentsOf: package.appendingPathComponent(name))
                try special(name)
                if name == ModelLayout.manifest {
                    #expect(throws: ServiceError.self) { try Manifest.load(from: package) }
                    #expect(Doctor.report(config)["model"]?["manifest"] == "invalid")
                } else {
                    #expect(Doctor.report(config)["model"]?["selftest_data"] == "invalid_selftest", "\(name)")
                }
                try FileManager.default.removeItem(at: package.appendingPathComponent(name))
                try original.write(to: package.appendingPathComponent(name))
            }
            for name in [ModelLayout.warmState, ModelLayout.selfTestState] { try special(name) }
            #expect(WarmState.load(from: package) == nil && SelfTestReport.load(from: package) == nil)
        }

        // models/current names the active revision: a FIFO or device there is ignored; a symlink to a regular file
        // still works.
        let store = ModelStore(dataDirectory: package)
        try FileManager.default.createDirectory(at: store.modelsDirectory, withIntermediateDirectories: true)
        try makeFIFO("current", in: store.modelsDirectory)
        #expect(store.currentDirectory.lastPathComponent == ModelSource.pinned.revision)
        try makeSymlink("current", in: store.modelsDirectory, to: "/dev/zero")
        #expect(store.currentDirectory.lastPathComponent == ModelSource.pinned.revision)
        try Data("abc123\n".utf8).write(to: package.appendingPathComponent("pointer"))
        try makeSymlink("current", in: store.modelsDirectory, to: package.appendingPathComponent("pointer").path)
        #expect(store.currentDirectory.lastPathComponent == "abc123")
    }

    @Test(.timeLimit(.minutes(1))) func smallFilesAreReadOnlyAsRegularFilesWithinTheCap() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("a.json")
        try Data("12345".utf8).write(to: file)
        #expect(try SmallFile.read(file, limit: 5) == Data("12345".utf8))
        #expect(throws: SmallFile.Failure.tooLarge(limit: 4)) { try SmallFile.read(file, limit: 4) }
        #expect(throws: SmallFile.Failure.missing) { try SmallFile.read(directory.appendingPathComponent("none"), limit: 5) }
        try makeFIFO("fifo", in: directory)
        try makeSymlink("zero", in: directory, to: "/dev/zero")
        try makeSymlink("link", in: directory, to: file.path)
        for name in ["fifo", "zero", "link"] {
            #expect(throws: SmallFile.Failure.notRegularFile, "\(name)") { try SmallFile.read(directory.appendingPathComponent(name), limit: 5) }
        }
        #expect(throws: SmallFile.Failure.notRegularFile) {
            try SmallFile.read(directory.appendingPathComponent("zero"), limit: 5, followSymlink: true)
        }
        #expect(try SmallFile.read(directory.appendingPathComponent("link"), limit: 5, followSymlink: true) == Data("12345".utf8))
    }

    /// Each small file read whole has a cap far above its real size, so a huge file is an error, not a huge read.
    @Test func anOversizedStoreFileIsAnError() throws {
        let package = try DoctorTests.modelDirectory()
        defer { try? FileManager.default.removeItem(at: package) }
        let config = try Config(flags: ["model-dir": package.path], environment: [:])
        /// Valid JSON (`value` padded with spaces) of `size` bytes.
        func padded(_ value: JSONValue, to size: Int) -> Data {
            let json = value.jsonData()
            return json + Data(repeating: 0x20, count: size - json.count)
        }

        // selftest.bin: the size manifest.json declares.
        let blob = try Data(contentsOf: package.appendingPathComponent(ModelLayout.selfTestData))
        try (blob + Data(count: 4)).write(to: package.appendingPathComponent(ModelLayout.selfTestData))
        #expect(Doctor.report(config)["model"]?["selftest_data"] == "invalid_selftest")
        try blob.write(to: package.appendingPathComponent(ModelLayout.selfTestData))
        #expect(Doctor.report(config)["model"]?["selftest_data"] == "0 tests")
        // selftest.json: 16 MiB.
        let selfTest = try Data(contentsOf: package.appendingPathComponent(ModelLayout.selfTest))
        try padded(try JSONValue.parse(selfTest), to: (16 << 20) + 1).write(to: package.appendingPathComponent(ModelLayout.selfTest))
        #expect(Doctor.report(config)["model"]?["selftest_data"] == "invalid_selftest")
        try selfTest.write(to: package.appendingPathComponent(ModelLayout.selfTest))

        // The state files and models/current: 1 MiB.
        let state = WarmState(osBuild: "b", binaryVersion: "v", manifestSHA256: "m", maxTokens: 2048, functions: [],
                              warmedAt: Date(timeIntervalSince1970: 0), seconds: 1)
        try padded(state.json, to: 1 << 20).write(to: package.appendingPathComponent(ModelLayout.warmState))
        #expect(WarmState.load(from: package) == state)
        try padded(state.json, to: (1 << 20) + 1).write(to: package.appendingPathComponent(ModelLayout.warmState))
        #expect(WarmState.load(from: package) == nil)
        let report = SelfTestReport(osBuild: "b", ranAt: Date(timeIntervalSince1970: 0), seconds: 0, results: [])
        try padded(report.json, to: (1 << 20) + 1).write(to: package.appendingPathComponent(ModelLayout.selfTestState))
        #expect(SelfTestReport.load(from: package) == nil)
        let store = ModelStore(dataDirectory: package)
        try FileManager.default.createDirectory(at: store.modelsDirectory, withIntermediateDirectories: true)
        try (Data("abc123\n".utf8) + Data(repeating: 0x20, count: 1 << 20)).write(to: store.modelsDirectory.appendingPathComponent("current"))
        #expect(store.currentDirectory.lastPathComponent == ModelSource.pinned.revision)

        // manifest.json: 4 MiB, as for its download.
        let manifest = try Data(contentsOf: package.appendingPathComponent(ModelLayout.manifest))
        try padded(try JSONValue.parse(manifest), to: DownloadLimit.manifestBytes).write(to: package.appendingPathComponent(ModelLayout.manifest))
        #expect(throws: Never.self) { try Manifest.load(from: package) }
        try padded(try JSONValue.parse(manifest), to: DownloadLimit.manifestBytes + 1)
            .write(to: package.appendingPathComponent(ModelLayout.manifest))
        #expect(throws: ServiceError.self) { try Manifest.load(from: package) }
    }

    /// At load: every source file present with its manifest size, and tokenizer.json and the self-test files
    /// (which the self-test cannot check itself) hashed. The full hash pass stays in `models verify`.
    @Test func loadChecksSizesAndTheFilesTheSelfTestCannotSee() throws {
        let directory = try DoctorTests.modelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let manifest = try Manifest.load(from: directory)
        #expect(manifest.checkForLoad(in: directory).isEmpty)
        try Data("[]".utf8).write(to: directory.appendingPathComponent(ModelLayout.tokenizer))  // same size, other bytes
        try Data(count: 4).write(to: directory.appendingPathComponent(ModelLayout.tokenTable))
        #expect(manifest.checkForLoad(in: directory) == [.init(path: ModelLayout.tokenTable, issue: "size_mismatch"),
                                                         .init(path: ModelLayout.tokenizer, issue: "sha256_mismatch")])
        var unlisted = manifest
        unlisted.files.removeAll { $0.path == ModelLayout.tokenizer }
        #expect(unlisted.checkForLoad(in: directory).contains(.init(path: ModelLayout.tokenizer, issue: "not_in_manifest")))
    }

    /// Two installers never share a staging directory or race on the revision directory: an exclusive lock on
    /// models/.install.lock, a staging directory per process, and an atomic swap into place.
    @Test func installsAreSerializedAndSwappedAtomically() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ModelStore(dataDirectory: directory)
        let first = try InstallLock.acquire(in: store)
        #expect(throws: ServiceError.self) { try InstallLock.acquire(in: store, wait: false) }
        first.release()
        let second = try InstallLock.acquire(in: store, wait: false)
        defer { second.release() }

        let target = try store.directory(for: "r1")
        let staging = ModelInstaller.stagingDirectory(for: target)
        #expect(staging != ModelInstaller.stagingDirectory(for: target))
        #expect(throws: Never.self) { try store.checkInside(staging) }
        // A target in use is replaced by one rename (swap), and the old contents go away.
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        try Data("old".utf8).write(to: target.appendingPathComponent("a.bin"))
        let payload = Data("new".utf8)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        try payload.write(to: staging.appendingPathComponent("a.bin"))
        let descriptor = JSONValue.object(["files": [
            ["path": "a.bin", "size": .int(payload.count), "sha256": .string(SHA256Hasher.hex(payload))],
        ]]).jsonData()
        try descriptor.write(to: staging.appendingPathComponent(ModelLayout.manifest))
        let manifest = try Manifest.parse(descriptor)
        try ModelInstaller.finish(staging: staging, into: target, store: store, manifest: manifest, log: { _ in })
        #expect(try Data(contentsOf: target.appendingPathComponent("a.bin")) == payload)
        #expect(!FileManager.default.fileExists(atPath: staging.path))
        // Staging left by an interrupted install is removed by the next one.
        let stale = ModelInstaller.stagingDirectory(for: target)
        try FileManager.default.createDirectory(at: stale, withIntermediateDirectories: true)
        ModelInstaller.removeStaleStaging(for: target)
        #expect(!FileManager.default.fileExists(atPath: stale.path))
        #expect(FileManager.default.fileExists(atPath: target.path))
    }

    /// Whatever fails once the staging directory exists (the manifest download, the manifest's pinned hash, a file it
    /// lists), the staging directory is removed and nothing is installed.
    @Test func aFailedDownloadLeavesNoStagingDirectory() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ModelStore(dataDirectory: directory)
        try FileManager.default.createDirectory(at: store.modelsDirectory, withIntermediateDirectories: true)
        let target = try store.directory(for: "r1")
        let payload = Data("weights".utf8)
        let manifest = JSONValue.object(["files": [
            ["path": "a.bin", "size": .int(payload.count), "sha256": .string(SHA256Hasher.hex(payload))],
        ]]).jsonData()
        let pinned = ModelSource(repository: "owner/name", revision: "r1", manifestSHA256: SHA256Hasher.hex(manifest))
        let otherPin = ModelSource(repository: "owner/name", revision: "r1", manifestSHA256: String(repeating: "0", count: 64))

        /// Downloads through a stand-in for the network that fails on `failing`; returns the error.
        func download(_ source: ModelSource, failing: String?) async -> ServiceError? {
            do {
                try await ModelInstaller.download(source, into: target, store: store) { path, staging, _ throws(ServiceError) in
                    guard path != failing else { throw ServiceError(.unavailable, "download_failed", "\(path): offline") }
                    let data = path == ModelLayout.manifest ? manifest : payload
                    do { try data.write(to: staging.appendingPathComponent(path)) } catch {
                        throw ServiceError(.internal, "write_failed", "\(error)")
                    }
                    return path == ModelLayout.manifest ? data : Data()
                }
                return nil
            } catch {
                return error
            }
        }
        func staged() throws -> [String] {
            try FileManager.default.contentsOfDirectory(atPath: store.modelsDirectory.path).filter { $0.contains(".partial") }
        }

        #expect(await download(pinned, failing: ModelLayout.manifest)?.reason == "download_failed")
        #expect(try staged().isEmpty)
        #expect(await download(otherPin, failing: nil)?.reason == "checksum_mismatch")
        #expect(try staged().isEmpty)
        #expect(await download(pinned, failing: "a.bin")?.reason == "download_failed")
        #expect(try staged().isEmpty && !FileManager.default.fileExists(atPath: target.path))
        #expect(await download(pinned, failing: nil) == nil)
        #expect(try staged().isEmpty && Data(contentsOf: target.appendingPathComponent("a.bin")) == payload)
    }

    /// A download stops as soon as it is larger than the manifest says, or announces another length.
    @Test func downloadsAreCappedAtTheDeclaredSize() {
        #expect(DownloadLimit.problem(expectedLength: -1, received: 10, declaredSize: 100, finished: false) == nil)
        #expect(DownloadLimit.problem(expectedLength: 100, received: 100, declaredSize: 100, finished: true) == nil)
        #expect(DownloadLimit.problem(expectedLength: 200, received: 0, declaredSize: 100, finished: false) != nil)
        #expect(DownloadLimit.problem(expectedLength: -1, received: 101, declaredSize: 100, finished: false) != nil)
        #expect(DownloadLimit.problem(expectedLength: -1, received: 99, declaredSize: 100, finished: true) != nil)
        // manifest.json has no declared size: capped at DownloadLimit.manifestBytes.
        #expect(DownloadLimit.problem(expectedLength: -1, received: 30_000, declaredSize: nil, finished: true) == nil)
        #expect(DownloadLimit.problem(expectedLength: -1, received: Int64(DownloadLimit.manifestBytes) + 1, declaredSize: nil,
                                      finished: false) != nil)
    }

    /// `models verify` covers the compiled models too: present and complete, or a problem. Their bytes differ between
    /// Core ML compilers, so a hash difference is reported, not failed.
    @Test func compiledModelsAreReportedByPresenceAndCompleteness() throws {
        let directory = try DoctorTests.modelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let manifest = try Manifest.load(from: directory)
        var status = manifest.compiledStatus(in: directory)
        #expect(status.map(\.package) == ["EmbeddingGemma2Text"] && status[0].problem && status[0].status.hasPrefix("not compiled"))
        let compiled = directory.appendingPathComponent("EmbeddingGemma2Text.mlmodelc")
        try FileManager.default.createDirectory(at: compiled.appendingPathComponent("weights"), withIntermediateDirectories: true)
        try Data("program".utf8).write(to: compiled.appendingPathComponent("model.mil"))
        status = manifest.compiledStatus(in: directory)
        #expect(status[0].problem && status[0].status.hasPrefix("incomplete"))
        try Data("x".utf8).write(to: compiled.appendingPathComponent("coremldata.bin"))
        try Data("w".utf8).write(to: compiled.appendingPathComponent("weights/weight.bin"))
        status = manifest.compiledStatus(in: directory)
        #expect(!status[0].problem && status[0].status.hasPrefix("compiled"))
    }

    @Test(arguments: [false, true])
    func completeCacheReclaimsAbandonedStagingWithoutCompiling(symlink: Bool) async throws {
        let directory = try temporaryDirectory()
        let files = FileManager.default
        defer { try? files.removeItem(at: directory) }
        let compiled = directory.appendingPathComponent("Fixture.mlmodelc")
        try files.createDirectory(at: compiled.appendingPathComponent("weights"), withIntermediateDirectories: true)
        for path in Manifest.compiledCoreFiles {
            try Data([1]).write(to: compiled.appendingPathComponent(path))
        }
        let lock = directory.appendingPathComponent(".Fixture" + ModelLayout.compileLockSuffix)
        #expect(try await compiledModelURL(compiled).path == compiled.path)
        #expect(!files.fileExists(atPath: lock.path))

        let stage = directory.appendingPathComponent(".Fixture" + ModelLayout.compileStageSuffix)
        if symlink {
            try files.createSymbolicLink(at: stage, withDestinationURL: compiled)
        } else {
            try files.createDirectory(at: stage, withIntermediateDirectories: false)
        }
        #expect(try await compiledModelURL(compiled).path == compiled.path)
        #expect(!files.fileExists(atPath: stage.path))
        #expect(try Data(contentsOf: compiled.appendingPathComponent("weights/weight.bin")) == Data([1]))
    }

    @Test func readsThePackageManifest() throws {
        let manifest: JSONValue = [
            "name": "embeddinggemma-2-ane", "manifest_version": 2, "created": "2026-11-15",
            "buckets": .array(BucketContract.fullAttentionLengths.map {
                ["function": .string("embed_\($0)"), "tokens": .int($0), "slots": .int(BucketContract.slots(forLength: $0))]
            }),
            "window": ["block": 512, "halo": 512, "band_columns": 1536,
                       "packing": [["function": "winp1536_b512", "tokens": 1536, "slots": 48],
                                   ["function": "winp2048_b512", "tokens": 2048, "slots": 64]],
                       "long": .array([3072, 4096, 6144, 8192].map { ["function": .string("winl\($0)_b512"), "tokens": .int($0)] })],
            "files": [["path": "tokenizer.json", "size": 1, "sha256": .string(String(repeating: "a", count: 64))]],
        ]
        let data = manifest.jsonData()
        let parsed = try Manifest.parse(data)
        #expect(parsed.model == "embeddinggemma-2-ane" && parsed.created == "2026-11-15")
        #expect(parsed.sha256 == SHA256Hasher.hex(data))
        #expect(parsed.buckets[1280] == 40 && parsed.window?.packing[2048] == 64 && parsed.window?.long == [3072, 4096, 6144, 8192])
        #expect(!parsed.legacyLongPath)
        #expect(throws: Never.self) { try parsed.checkContract() }
        var other = parsed
        other.window?.halo = 256
        #expect(throws: ServiceError.self) { try other.checkContract() }
        other = parsed
        other.window?.long.append(16384)
        #expect(throws: ServiceError.self) { try other.checkContract() }
        other = parsed
        other.buckets[512] = 8
        #expect(throws: ServiceError.self) { try other.checkContract() }
        other = parsed
        other.buckets[2048] = 64  // the first layout's largest bucket is a winp length now
        #expect(throws: ServiceError.self) { try other.checkContract() }
        // The first layout (per-layer long path) is rejected with a clear message.
        var legacy = manifest.objectValue!
        legacy["long_path"] = ["chunk": 1024, "halo": 512, "buckets": [4096, 8192]]
        let old = try Manifest.parse(JSONValue.object(legacy).jsonData())
        #expect(old.legacyLongPath)
        #expect(throws: ServiceError.self) { try old.checkContract() }
        #expect(ModelInstaller.localRevision(parsed) == "local-" + parsed.sha256.prefix(12))
    }

    /// The runtime loads every function of the contract, so the manifest must declare all of them.
    @Test func contractRequiresEveryDeclaredFunction() throws {
        let full = DoctorTests.contract(files: [])
        #expect(throws: Never.self) { try Manifest.parse(JSONValue.object(full).jsonData()).checkContract() }
        for key in ["buckets", "window"] {
            var missing = full
            missing[key] = nil
            #expect(throws: ServiceError.self) { try Manifest.parse(JSONValue.object(missing).jsonData()).checkContract() }
        }
        var partial = full
        partial["buckets"] = [["tokens": 64, "slots": 8]]
        #expect(throws: ServiceError.self) { try Manifest.parse(JSONValue.object(partial).jsonData()).checkContract() }
        var noLong = full
        noLong["window"] = ["block": 512, "halo": 512, "band_columns": 1536,
                            "packing": [["tokens": 1536, "slots": 48], ["tokens": 2048, "slots": 64]]]
        #expect(throws: ServiceError.self) { try Manifest.parse(JSONValue.object(noLong).jsonData()).checkContract() }
    }

    @Test func rejectsPathsOutsideTheModelDirectory() {
        let bad: JSONValue = ["files": [["path": "../x", "size": 1, "sha256": .string(String(repeating: "a", count: 64))]]]
        #expect(throws: ServiceError.self) { try Manifest.parse(bad.jsonData()) }
    }

    @Test func activatesRevisionsAtomically() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ModelStore(dataDirectory: directory)
        try store.activate(revision: "abc123")
        #expect(store.currentDirectory.lastPathComponent == "abc123")
        try store.activate(revision: "def456")
        #expect(try store.currentDirectory == store.directory(for: "def456"))
    }

    @Test func warmStateRoundTrips() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let state = WarmState(osBuild: "26A434", binaryVersion: "0.1.0", binary: "abc", manifestSHA256: "m1", maxTokens: 2048,
                              functions: ["embed_64"], warmedAt: Date(timeIntervalSince1970: 1_800_000_000), seconds: 31.5)
        try state.save(to: directory)
        #expect(WarmState.load(from: directory) == state)
        #expect(state.covers(osBuild: "26A434", maxTokens: 512, binary: "abc", manifestSHA256: "m1"))
        #expect(!state.covers(osBuild: "26B1", maxTokens: 512, binary: "abc", manifestSHA256: "m1"))
        #expect(!state.covers(osBuild: "26A434", maxTokens: 4096, binary: "abc", manifestSHA256: "m1"))
        #expect(!state.covers(osBuild: "26A434", maxTokens: 512, binary: "def", manifestSHA256: "m1"))  // another executable
        #expect(!state.covers(osBuild: "26A434", maxTokens: 512, binary: "abc", manifestSHA256: "m2"))  // another package
        #expect(Executable.identity.count == 64)
        #expect(!operatingSystemBuild.isEmpty)
    }

    @Test func warmCapabilityDemotesOnFailureAndPreservesHealthyBroaderCoverage() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let passed = SelfTestResult(function: "short", kind: "test", passed: true, cosine: 1)
        let failed = SelfTestResult(function: "wide", kind: "test", passed: false, cosine: 0)
        func record(_ effective: Int, requested: Int, functions: [String], results: [SelfTestResult]? = nil) throws {
            let report = results.map {
                SelfTestReport(osBuild: "macOS", ranAt: Date(), seconds: 0, results: $0, binary: "binary")
            }
            let warm = WarmState(osBuild: "macOS", binaryVersion: "test", binary: "binary", manifestSHA256: "manifest",
                                 maxTokens: effective, functions: functions, warmedAt: Date(), seconds: 1, selfTest: report)
            try warm.recordCapability(requestedMaxTokens: requested, in: directory)
        }
        try record(32768, requested: 32768, functions: ["short", "wide"])
        try record(2048, requested: 2048, functions: ["short"], results: [passed])
        #expect(WarmState.load(from: directory)?.maxTokens == 32768)
        #expect(WarmState.load(from: directory)?.selfTest?.results == [passed])
        try record(2048, requested: 32768, functions: ["short"], results: [passed, failed])
        try record(2048, requested: 2048, functions: ["short"], results: [passed])
        let snapshot = try #require(WarmState.load(from: directory))
        #expect(snapshot.functions == ["short"] && snapshot.maxTokens == 2048)
        #expect(snapshot.selfTest?.results == [passed, failed])
        #expect(SelfTestReport.load(from: directory) == nil)
        try record(0, requested: 32768, functions: [])
        #expect(WarmState.load(from: directory)?.maxTokens == 0)
        #expect(WarmState.load(from: directory)?.functions == [])
        #expect(WarmState.load(from: directory)?.selfTest == nil)
        try record(2048, requested: 2048, functions: ["short"])
        #expect(WarmState.load(from: directory)?.maxTokens == 2048)
    }

    /// The data directory's pinned revision loads only with the manifest pinned in this build; another installed
    /// revision and --model-dir are checked against their own manifest.
    @Test func thePinnedRevisionLoadsOnlyWithThePinnedManifest() throws {
        let (dataDirectory, revision) = try DoctorTests.replacedPinnedRevision()
        defer { try? FileManager.default.removeItem(at: dataDirectory) }
        let raw = try Config(flags: ["data-dir": dataDirectory.path, "model": "embeddinggemma-2"], environment: [:])
        let config = try raw.selectingModel(for: "serve")
        #expect(config.pinnedManifestSHA256 == ModelSource.pinned.manifestSHA256)
        let error = #expect(throws: ServiceError.self) { try ModelLoader.checkPackage(config) }
        #expect(error?.reason == "pinned_manifest_mismatch" && error?.hint?.contains("anerun models install") == true)

        let explicit = try Config(flags: ["model-dir": revision.path], environment: [:])
        #expect(explicit.pinnedManifestSHA256 == nil)
        #expect(throws: Never.self) { try ModelLoader.checkPackage(explicit) }
        let store = ModelStore(dataDirectory: dataDirectory)
        try FileManager.default.moveItem(at: revision, to: store.modelsDirectory.appendingPathComponent("local-0123456789ab"))
        try store.activate(revision: "local-0123456789ab")
        #expect(config.pinnedManifestSHA256 == ModelSource.pinned.manifestSHA256)
        let selected = try raw.selectingModel(for: "serve")
        #expect(selected.pinnedManifestSHA256 == nil)
        #expect(config.unpinnedActiveRevision == nil && explicit.unpinnedActiveRevision == nil)
        #expect(selected.unpinnedActiveRevision == "local-0123456789ab")
        #expect(selected.unpinnedRevisionNote?.contains("anerun models install --model embeddinggemma-2") == true)
        // Warming an install target is not the active selection: no reminder before it is activated.
        #expect(selected.withModelDirectory(store.modelsDirectory.appendingPathComponent("local-0123456789ab")).unpinnedActiveRevision == nil)
        #expect(throws: Never.self) { try ModelLoader.checkPackage(selected) }

        let named = ModelStore(dataDirectory: dataDirectory, model: .embeddingGemma2)
        let restored = try named.directory(for: ModelSource.pinned.revision + "-r-0123abcd")
        try FileManager.default.createDirectory(at: named.modelsDirectory, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: selected.modelDirectory, to: restored)
        try named.activate(revision: restored.lastPathComponent)
        let repaired = try raw.selectingModel(for: "serve")
        #expect(repaired.pinnedManifestSHA256 == ModelSource.pinned.manifestSHA256)
        #expect(repaired.withModelDirectory(restored).pinnedManifestSHA256 == ModelSource.pinned.manifestSHA256)
    }

    @Test func pinnedSourceIsAPublishedCommit() throws {
        #expect(ModelSource.pinned.isPublished)
        #expect(ModelSource.pinned.revision.count == 40 && ModelSource.pinned.revision.allSatisfy(\.isHexDigit))
        #expect(!ModelSource(repository: "a/b", revision: "UNPUBLISHED").isPublished)
        #expect(ModelSource.pinned.manifestSHA256?.count == 64)
        #expect(try ModelSource(repository: "a/b", revision: "0123abc").url(for: "manifest.json").absoluteString
                == "https://huggingface.co/a/b/resolve/0123abc/manifest.json")
    }

    /// A revision becomes a directory name under models/, so only one plain path component is accepted.
    @Test func revisionsAreOnePlainPathComponent() throws {
        for good in [ModelSource.pinned.revision, "local-0123456789ab", "v1.0", "main", "a_b-c.d"] {
            #expect(ModelSource.isValidRevision(good), "\(good)")
        }
        for bad in ["", ".", "..", "../x", "a/b", "/", "-x", ".hidden", " a", "a b", "a\u{0}", String(repeating: "a", count: 65)] {
            #expect(!ModelSource.isValidRevision(bad), "\(bad)")
        }
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ModelStore(dataDirectory: directory)
        #expect(throws: ServiceError.self) { try store.directory(for: "..") }
        #expect(throws: ServiceError.self) { try store.directory(for: "../../etc") }
        #expect(throws: ServiceError.self) { try store.activate(revision: "..") }
        #expect(try store.directory(for: "abc").deletingLastPathComponent().standardizedFileURL
                == store.modelsDirectory.standardizedFileURL)
        // A hand-edited `current` pointer is held to the same rule (else the pinned revision is used).
        try FileManager.default.createDirectory(at: store.modelsDirectory, withIntermediateDirectories: true)
        try "..\n".write(to: store.modelsDirectory.appendingPathComponent("current"), atomically: true, encoding: .utf8)
        #expect(store.currentDirectory.lastPathComponent == ModelSource.pinned.revision)
        try store.activate(revision: "abc")
        #expect(store.currentDirectory.lastPathComponent == "abc")
        // The installer replaces only direct children of models/, after resolving `..` and symlinks.
        let manifest = try Manifest.parse(JSONValue.object(["files": []]).jsonData())
        let staging = store.modelsDirectory.appendingPathComponent(".abc.partial")
        for target in [store.modelsDirectory.appendingPathComponent(".."), directory,
                       store.modelsDirectory.appendingPathComponent("link/abc")]
        {
            try? FileManager.default.createSymbolicLink(at: store.modelsDirectory.appendingPathComponent("link"),
                                                        withDestinationURL: FileManager.default.temporaryDirectory)
            #expect(throws: ServiceError.self, "\(target.path)") {
                try ModelInstaller.finish(staging: staging, into: target, store: store, manifest: manifest, log: { _ in })
            }
        }
        #expect(FileManager.default.fileExists(atPath: store.currentDirectory.deletingLastPathComponent().path))
    }

    /// Repository, revision and file paths end up in a URL: malformed ones are errors, not a crash.
    @Test func downloadURLsAreBuiltSafely() throws {
        #expect(try ModelSource(repository: "a/b", revision: "r1").url(for: "dir/tok enizer.json").absoluteString
                == "https://huggingface.co/a/b/resolve/r1/dir/tok%20enizer.json")
        for repository in ["a b/c", "a", "a/b/c", "../b", "a/..", "a/b?x", ""] {
            #expect(throws: ServiceError.self, "\(repository)") { try ModelSource(repository: repository, revision: "r1").url(for: "x") }
        }
        #expect(throws: ServiceError.self) { try ModelSource(repository: "a/b", revision: "r 1").url(for: "x") }
        #expect(throws: ServiceError.self) { try ModelSource(repository: "a/b", revision: "..").url(for: "x") }
        for path in ["a//b", "./a", "a/./b", "a\u{1}b"] {
            let bad: JSONValue = ["files": [["path": .string(path), "size": 1, "sha256": .string(String(repeating: "a", count: 64))]]]
            #expect(throws: ServiceError.self, "\(path)") { try Manifest.parse(bad.jsonData()) }
        }
    }
}
