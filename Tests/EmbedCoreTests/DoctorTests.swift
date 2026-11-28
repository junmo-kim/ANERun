import DecisionProfiles
import Foundation
import Testing

@testable import EmbedCore
@testable import anerun

@Suite("Doctor")
struct DoctorTests {
    @Test func reportsAMissingModel() throws {
        let config = try Config(flags: ["model-dir": "/nonexistent/aneembed"], environment: [:])
        let report = Doctor.report(config)
        #expect(report["model"]?["present"] == false)
        #expect(report["ready"] == false)
        #expect(report["compile_cache"]?.stringValue?.hasPrefix("cold") == true)
        #expect(report["neural_engine"] != nil && report["macos_build"] != nil)
    }

    /// A model directory with the files doctor looks at (no real model): manifest with the full contract,
    /// placeholder model, tokenizer and token table, and optionally self-test data.
    static func modelDirectory(selfTest: Bool = true, manifest: Bool = true) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("aneembed-doctor-\(UUID().uuidString)")
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: directory.appendingPathComponent("EmbeddingGemma2Text.mlpackage"),
                                        withIntermediateDirectories: true)
        var files: [String: Data] = [ModelLayout.tokenizer: Data("{}".utf8), ModelLayout.tokenTable: Data(count: 8),
                                     "EmbeddingGemma2Text.mlpackage/Manifest.json": Data("{}".utf8)]
        if selfTest {
            var blob = Data()
            for _ in 0..<(512 * 8) { withUnsafeBytes(of: Float(0).bitPattern.littleEndian) { blob.append(contentsOf: $0) } }
            let json: JSONValue = ["format": "aneembed-selftest/1", "digest": ["offset": 0, "shape": [512, 8]], "tests": []]
            files[ModelLayout.selfTest] = json.jsonData()
            files[ModelLayout.selfTestData] = blob
        }
        for (path, data) in files { try data.write(to: directory.appendingPathComponent(path)) }
        if manifest {
            let entries: [JSONValue] = files.keys.sorted().map {
                ["path": .string($0), "size": .int(files[$0]!.count), "sha256": .string(SHA256Hasher.hex(files[$0]!))]
            }
            try JSONValue.object(Self.contract(files: entries)).jsonData().write(to: directory.appendingPathComponent("manifest.json"))
        }
        return directory
    }

    /// The manifest fields of the package contract this build implements.
    static func contract(files: [JSONValue]) -> JSONObject {
        [
            "name": "embeddinggemma-2-ane", "manifest_version": 2,
            "buckets": .array(BucketContract.fullAttentionLengths.map {
                ["tokens": .int($0), "slots": .int(BucketContract.slots(forLength: $0))]
            }),
            "window": ["block": 512, "halo": 512, "band_columns": 1536,
                       "packing": [["tokens": 1536, "slots": 48], ["tokens": 2048, "slots": 64]],
                       "long": .array(WindowContract.longLengths.map { ["tokens": .int($0)] })],
            "files": .array(files),
        ]
    }

    static func problems(_ config: Config) -> [String] {
        Doctor.report(config)["problems"]?.arrayValue?.compactMap(\.stringValue) ?? []
    }

    @Test func aPackageWithoutSelfTestDataIsNotReady() throws {
        let directory = try Self.modelDirectory(selfTest: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let problems = Self.problems(try Config(flags: ["model-dir": directory.path], environment: [:]))
        #expect(problems.contains { $0.contains("self-test data") })
        let skipped = Self.problems(try Config(flags: ["model-dir": directory.path, "selftest": "skip"], environment: [:]))
        #expect(!skipped.contains { $0.contains("self-test data") })
        let complete = try Self.modelDirectory()
        defer { try? FileManager.default.removeItem(at: complete) }
        #expect(!Self.problems(try Config(flags: ["model-dir": complete.path], environment: [:])).contains { $0.contains("self-test") })
    }

    @Test func aPackageWithoutAManifestIsNotReady() throws {
        let directory = try Self.modelDirectory(manifest: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let report = Doctor.report(try Config(flags: ["model-dir": directory.path], environment: [:]))
        #expect(report["ready"] == false)
        #expect(Self.problems(try Config(flags: ["model-dir": directory.path], environment: [:])).contains { $0.contains("manifest") })
    }

    /// A warm state recorded for another package (same path, replaced files) is stale.
    @Test func warmStateOfAnotherPackageIsStale() throws {
        let directory = try Self.modelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try Config(flags: ["model-dir": directory.path], environment: [:])
        let manifest = try Manifest.load(from: directory)
        func warm(_ sha: String) throws {
            try WarmState(osBuild: operatingSystemBuild, binaryVersion: "x", manifestSHA256: sha, maxTokens: 2048, functions: [],
                          warmedAt: Date(), seconds: 1).save(to: directory)
        }
        try warm(manifest.sha256)
        #expect(Doctor.report(config)["compile_cache"] == "warm")
        try warm(String(repeating: "0", count: 64))
        #expect(Doctor.report(config)["compile_cache"]?.stringValue?.hasPrefix("stale") == true)
    }

    @Test func plainDoctorRejectsAnAssetWithTheRightSizeAndWrongHash() throws {
        let directory = try Self.modelDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try Config(flags: ["model-dir": directory.path, "runtime": "fake"], environment: [:])
        let manifest = try Manifest.load(from: directory)
        try WarmState(osBuild: operatingSystemBuild, binaryVersion: "test", manifestSHA256: manifest.sha256,
                      maxTokens: 2048, functions: [], warmedAt: Date(), seconds: 0).save(to: directory)
        #expect(Doctor.report(config)["ready"] == true)

        try Data("[]".utf8).write(to: directory.appendingPathComponent(ModelLayout.tokenizer))
        let report = Doctor.report(config)
        #expect(report["ready"] == false)
        #expect(report["model"]?["manifest"]?["problems"] == ["tokenizer.json: sha256_mismatch"])
    }

    /// A data directory whose pinned revision holds `modelDirectory()`: a package consistent with its own manifest,
    /// but not the one pinned in this build.
    static func replacedPinnedRevision() throws -> (dataDirectory: URL, revision: URL) {
        let dataDirectory = FileManager.default.temporaryDirectory.appendingPathComponent("aneembed-pinned-\(UUID().uuidString)")
        let store = ModelStore(dataDirectory: dataDirectory)
        try FileManager.default.createDirectory(at: store.modelsDirectory, withIntermediateDirectories: true)
        let revision = store.modelsDirectory.appendingPathComponent(ModelSource.pinned.revision, isDirectory: true)
        try FileManager.default.moveItem(at: try modelDirectory(), to: revision)
        return (dataDirectory, revision)
    }

    @Test func aReplacedPinnedRevisionIsNotReady() throws {
        let (dataDirectory, revision) = try Self.replacedPinnedRevision()
        defer { try? FileManager.default.removeItem(at: dataDirectory) }
        let config = try Config(flags: ["data-dir": dataDirectory.path, "model": "embeddinggemma-2"], environment: [:])
            .selectingModel(for: "doctor")
        #expect(config.modelDirectory.lastPathComponent == ModelSource.pinned.revision)
        #expect(Self.problems(config).contains { $0.contains("pinned in this build") })
        // The same files as --model-dir are checked against their own manifest only.
        #expect(!Self.problems(try Config(flags: ["model-dir": revision.path], environment: [:])).contains { $0.contains("pinned") })
    }

    @Test func cliReportsAnInvalidActiveManifest() async throws {
        let (base, revision) = try Self.replacedPinnedRevision()
        defer { try? FileManager.default.removeItem(at: base) }
        try Data("{".utf8).write(to: revision.appendingPathComponent(ModelLayout.manifest))
        let flags = ["data-dir": base.path, "model": ModelID.embeddingGemma2.rawValue, "runtime": "coreml"]
        let config = try Config(flags: flags, environment: [:]).selectingModel(for: "doctor")
        let report = Doctor.report(config, verify: true)
        #expect(report["model"]?["dir"] == .string(revision.resolvingSymlinksInPath().path))
        #expect(report["model"]?["manifest"] == "invalid")
        #expect(report["ready"] == false)
        let status = await CLI.run(["doctor", "--verify", "--format", "json", "--data-dir", base.path,
                                    "--model", ModelID.embeddingGemma2.rawValue, "--runtime", "coreml"])
        #expect(status == 0)
    }

    @Test func cliReportsDecisionBundleIdentityWithoutPreparing() async throws {
        let directory = try Self.modelDirectory(manifest: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let profile = DecisionProfile.valen4B
        let header: JSONValue = ["source": ["repo": .string(profile.repository), "revision": .string(profile.revision)]]
        let descriptor = directory.appendingPathComponent("model.json")
        try header.jsonData().write(to: descriptor)
        let flags = ["model-dir": directory.path, "runtime": "coreml"]
        let config = try Config(flags: flags, environment: [:]).selectingModel(for: "doctor")
        let report = Doctor.report(config, verify: true)
        #expect(report["model"]?["bundle_identity"] == "recognized")
        #expect(report["model"]?["name"] == .string(profile.rawValue))
        #expect(report["ready"] == false)
        let status = await CLI.run(["doctor", "--verify", "--format", "json", "--model-dir", directory.path,
                                    "--runtime", "coreml"])
        #expect(status == 0)
        try Data("{".utf8).write(to: descriptor)
        let damaged = try Config(flags: flags, environment: [:]).selectingModel(for: "doctor")
        #expect(Doctor.report(damaged)["model"]?["bundle_identity"] == "unverified")
    }

    @Test func fakeRuntimeNeedsNoModel() throws {
        let config = try Config(flags: ["model-dir": "/nonexistent/aneembed", "runtime": "fake"], environment: [:])
        #expect(Doctor.report(config)["problems"] == [])
    }
}
