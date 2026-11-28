import Foundation
import Darwin
import Testing

@testable import EmbedCore

/// Inference on an isolated, verified package. The launcher holds the M4 ANE lock.
@Suite("PPLX on the Neural Engine", .serialized,
       .enabled(if: PPLXHardwareData.enabled, "needs ANEEMBED_TEST_MODEL_DIR and ANEEMBED_PPLX_REFERENCE, without CI"))
struct PPLXHardwareTests {
    private struct Reference: Decodable {
        let name: String
        let text: String
        let tokens: Int
        let vector: [Float]
    }

    private func record(_ line: String) {
        FileHandle.standardOutput.write(Data((line + "\n").utf8))
    }

    @Test func swiftMatchesThePinnedReferenceThrough32768Tokens() async throws {
        defer {
            var usage = rusage()
            if getrusage(RUSAGE_SELF, &usage) == 0 {
                record(String(format: "PPLX MEMORY max_rss_bytes=%lld peak_footprint_mib=%.1f",
                              usage.ru_maxrss, Memory.peakFootprintMB))
            }
        }
        let (path, reference) = try #require(PPLXHardwareData.paths)
        let selected = try Config(flags: ["model-dir": path, "max-tokens": "32768"], environment: [:])
            .selectingModel(for: "test pplx")
        let loadStart = ContinuousClock.now
        let (service, report) = try await ModelLoader.load(selected)
        record(String(format: "PPLX READY %.3fs", PPLXLayerTiming.seconds(loadStart.duration(to: .now))))
        let runtime = try #require(service.runtime as? PPLXTextRuntime)
        let suite = try #require(report.selfTest)
        #expect(suite.failed.isEmpty)
        #expect(service.maxTokens == 32768)
        #expect(service.maxImageTokens == 0)
        let cases = try JSONDecoder().decode([Reference].self, from: Data(contentsOf: reference))
        #expect(cases.filter { $0.tokens == 32768 }.count >= 3)
        let limits = PPLXHardwareData.lengths
        for sample in cases where limits.isEmpty || limits.contains(sample.tokens) {
            let ids = service.tokenize(sample.text)
            let length = try runtime.bucket(for: ids.count)
            #expect(ids.count == sample.tokens)
            let start = ContinuousClock.now
            let vector: [Float]
            if length > 8192 {
                let profile = try await runtime.profile(ids, length: length)
                vector = profile.vector
                let host = profile.layers.reduce(0) { $0 + $1.host } + profile.preparationSeconds + profile.poolingSeconds
                let attention = profile.layers.reduce(0) { $0 + $1.attention }
                let prePost = profile.layers.reduce(0) { $0 + $1.pre + $1.post }
                let layerHost = profile.layers.map { String(format: "%.4f", $0.host) }.joined(separator: ",")
                record(String(format: "PPLX HOST %@ %.4fs attention %.4fs pre/post %.4fs layers [%@]",
                             sample.name, host, attention, prePost, layerHost))
            } else {
                let result = try await runtime.embed(tokenized: [ids])
                vector = try #require(result.first)
            }
            let elapsed = PPLXLayerTiming.seconds(start.duration(to: .now))
            #expect(vector.count == 1024 && vector.allSatisfy(\.isFinite))
            try #require(vector.count == sample.vector.count)
            let raw = Vectors.cosine(vector, sample.vector)
            let actual = Vectors.output(vector, to: 1024, modelID: .pplx)
            let expected = Vectors.output(sample.vector, to: 1024, modelID: .pplx)
            let quantized = Vectors.cosine(actual, expected)
            let matches = zip(actual, expected).filter { $0.0 == $0.1 }.count
            let maxError = zip(actual, expected).map { abs($0.0 - $0.1) }.max() ?? 0
            let prefixes = [512, 256, 128].map { n in
                String(format: "%d=%.6f", n, Vectors.cosine(Array(actual.prefix(n)), Array(expected.prefix(n))))
            }.joined(separator: " ")
            record(String(format: "PPLX CASE %@ n=%d shape=%d %.3fs raw=%.7f int8=%.7f exact=%d/1024 maxError=%.0f %@",
                         sample.name, ids.count, length, elapsed, raw, quantized, matches, maxError, prefixes))
            #expect(raw >= 0.995 && quantized >= 0.995, "\(sample.name): raw \(raw), int8 \(quantized)")
        }
        if limits.isEmpty {
            let boundaryText = try #require(cases.first(where: { $0.tokens == 8192 })?.text)
            let boundaryIDs = service.tokenize(boundaryText)
            try #require(boundaryIDs.count == 8192)
            for count in [1, 2047, 2048, 2049, 4096, 8192] {
                let ids = Array(boundaryIDs.prefix(count))
                let fusedStart = ContinuousClock.now
                let fused = try await runtime.embed(tokenized: [ids])
                let fusedSeconds = PPLXLayerTiming.seconds(fusedStart.duration(to: .now))
                let tiledStart = ContinuousClock.now
                let tiled = try await runtime.profile(ids, length: 16384)
                let tiledSeconds = PPLXLayerTiming.seconds(tiledStart.duration(to: .now))
                let raw = Vectors.cosine(try #require(fused.first), tiled.vector)
                record(String(format: "PPLX FORCED n=%d fused %.3fs decomposed %.3fs raw=%.7f",
                              count, fusedSeconds, tiledSeconds, raw))
                #expect(raw >= 0.995)
            }
        }
        let over = String(repeating: " x", count: 32769)
        #expect(service.tokenize(over).count == 32769)
        do {
            _ = try await service.embedTexts([over], truncate: false)
            Issue.record("accepted 32,769 tokens")
        } catch {
            #expect(error.reason == "context_length_exceeded")
        }
    }
}

private enum PPLXHardwareData {
    static var paths: (String, URL)? {
        let env = ProcessInfo.processInfo.environment
        guard env["CI"] == nil, let dir = env["ANEEMBED_TEST_MODEL_DIR"],
              let ref = env["ANEEMBED_PPLX_REFERENCE"], FileManager.default.fileExists(atPath: ref)
        else { return nil }
        return (dir, URL(fileURLWithPath: ref))
    }

    static var enabled: Bool { paths != nil }

    static var lengths: Set<Int> {
        Set((ProcessInfo.processInfo.environment["ANEEMBED_PPLX_LENGTHS"] ?? "").split(separator: ",").compactMap { Int($0) })
    }
}
