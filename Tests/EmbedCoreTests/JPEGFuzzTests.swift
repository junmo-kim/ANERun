import Foundation
import Testing

@testable import EmbedCore

/// Runs the jpeg-fuzz tool (Sources/jpeg-fuzz) as a child process over the fixture JPEGs: a decoder trap kills
/// only the child, and its exit status fails the test. ANEEMBED_FUZZ_ITERATIONS overrides the iteration count.
@Suite("JPEG decoder fuzzing")
struct JPEGFuzzTests {
    static var tool: URL? {
        // The tool is built next to the test bundle (EmbedCoreTests depends on the jpeg-fuzz target).
        for bundle in Bundle.allBundles where bundle.bundlePath.hasSuffix(".xctest") {
            let candidate = bundle.bundleURL.deletingLastPathComponent().appendingPathComponent("jpeg-fuzz")
            if FileManager.default.isExecutableFile(atPath: candidate.path) { return candidate }
        }
        let arguments = CommandLine.arguments.compactMap { $0.hasSuffix(".xctest") || $0.contains(".xctest/") ? $0 : nil }
        for path in arguments {
            var url = URL(fileURLWithPath: path)
            while url.pathComponents.count > 1, !url.lastPathComponent.hasSuffix(".xctest") { url.deleteLastPathComponent() }
            let candidate = url.deletingLastPathComponent().appendingPathComponent("jpeg-fuzz")
            if FileManager.default.isExecutableFile(atPath: candidate.path) { return candidate }
        }
        return nil
    }

    func run(_ arguments: [String]) throws -> (Int32, Process.TerminationReason, String) {
        let tool = try #require(Self.tool, "jpeg-fuzz was not built next to the test bundle")
        let process = Process()
        process.executableURL = tool
        process.arguments = arguments
        let output = Pipe()
        process.standardOutput = output
        process.standardError = output
        try process.run()
        let data = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, process.terminationReason, String(decoding: data, as: UTF8.self))
    }

    @Test func mutatedFixturesNeverCrashTheDecoder() throws {
        let iterations = ProcessInfo.processInfo.environment["ANEEMBED_FUZZ_ITERATIONS"].flatMap(Int.init) ?? 3000
        let crash = FileManager.default.temporaryDirectory.appendingPathComponent("jpeg-fuzz-crash-\(getpid()).jpg")
        defer { try? FileManager.default.removeItem(at: crash) }
        // This checks crashes only. The per-input time limit (10 s by default) is calibrated for the release build that
        // CI fuzzes separately; a debug build on a shared runner can take longer than that on a legitimate input.
        let (status, reason, output) = try run(["--iterations", "\(iterations)", "--seed", "1", "--max-seconds", "600",
                                                "--crash-file", crash.path, Fixtures.url("images").path])
        #expect(reason == .exit && status == 0, "jpeg-fuzz failed (\(reason == .uncaughtSignal ? "signal" : "exit") \(status)): \(output)")
        #expect(output.contains("0 failures"))
    }

    /// The harness itself must notice a crash: a JPEG whose DC symbol is 255 traps an unfixed decoder, so here it
    /// only has to be rejected, and a missing corpus is a usage error rather than a silent pass.
    @Test func replaysAFileAndRefusesAnEmptyCorpus() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("jpeg-fuzz-replay-\(getpid()).jpg")
        try CraftedJPEG.make(dcSymbol: 255).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        let (status, _, output) = try run(["--replay", file.path])
        #expect(status == 0 && output.contains("rejected"), "\(output)")
        let empty = FileManager.default.temporaryDirectory.appendingPathComponent("jpeg-fuzz-empty-\(getpid())")
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: empty) }
        #expect(try run([empty.path]).0 == 64)
    }

    /// The memory measurement CI runs at 40 MP (README "Memory"), here on a small image: a flat JPEG of any declared
    /// size decodes, and --max-peak-mib fails a decode that adds more.
    @Test func measuresThePeakOfADecode() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("jpeg-fuzz-flat-\(getpid()).jpg")
        defer { try? FileManager.default.removeItem(at: file) }
        #expect(try run(["--flat-jpeg", "640x480", "--sampling", "420", "--out", file.path]).0 == 0)
        let data = try Data(contentsOf: file)
        #expect(data.count < 200)
        let (rgb, width, height) = try #require(JPEGDecoder.decode(data, maxPixels: 640 * 480))
        #expect(width == 640 && height == 480 && Set(rgb).count == 1)
        let (status, _, output) = try run(["--peak", file.path, "--max-peak-mib", "200"])
        #expect(status == 0 && output.contains("canvas"), "\(output)")
        #expect(try run(["--peak", file.path, "--max-peak-mib", "0"]).0 == 1)
    }

    /// --progressive writes the same flat image as a progressive JPEG with a single DC scan, which the decoder reads
    /// like any progressive file (CI measures both kinds at 40 MP).
    @Test func flatProgressiveJPEGsDecodeToTheSameImage() throws {
        var decoded: [[UInt8]] = []
        for options in [[], ["--progressive"]] {
            let file = FileManager.default.temporaryDirectory.appendingPathComponent("jpeg-fuzz-flat-\(getpid())-\(options.count).jpg")
            defer { try? FileManager.default.removeItem(at: file) }
            #expect(try run(["--flat-jpeg", "100x60", "--sampling", "420"] + options + ["--out", file.path]).0 == 0)
            let data = try Data(contentsOf: file)
            #expect(data[data.startIndex + 71 ... data.startIndex + 72] == (options.isEmpty ? Data([0xFF, 0xC0]) : Data([0xFF, 0xC2])))
            let (rgb, width, height) = try #require(JPEGDecoder.decode(data, maxPixels: 100 * 60))
            #expect(width == 100 && height == 60)
            decoded.append(rgb)
        }
        #expect(decoded[0] == decoded[1])
    }

    /// A decode that does not return ends the tool with a message once --max-seconds have passed, instead of
    /// hanging until the CI job times out (--hang stands in for such a decode).
    @Test(.timeLimit(.minutes(1))) func aDecodeThatNeverReturnsIsCutWithAMessage() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("jpeg-fuzz-hang-\(getpid()).jpg")
        defer { try? FileManager.default.removeItem(at: file) }
        #expect(try run(["--flat-jpeg", "64x64", "--out", file.path]).0 == 0)
        let (status, reason, output) = try run(["--peak", file.path, "--max-seconds", "1", "--hang"])
        #expect(reason == .exit && status == 1, "\(output)")
        #expect(output.contains("no result after 1.0 s"), "\(output)")
    }
}
