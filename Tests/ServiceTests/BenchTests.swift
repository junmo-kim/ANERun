@preconcurrency import CoreML
import Foundation
import Testing

@testable import EmbedCore

@Suite("Benchmark request contract")
struct BenchTests {
    private func pplxConfig(cap: Int) throws -> Config {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        return try Config(flags: ["model": ModelID.pplx.rawValue, "runtime": "fake",
                                  "max-tokens": String(cap), "data-dir": directory.path], environment: [:])
    }

    @Test(arguments: [1, 8]) func smallPPLXCapProducesSuccessfulRequests(cap: Int) async throws {
        var options = Bench.Options()
        options.sizes = [cap]
        options.repeats = 1
        options.concurrency = 2

        let report = try await Bench.run(pplxConfig(cap: cap), options: options, log: { _ in })
        #expect(report["single"]?[String(cap)]?["tokens"] == .int(cap))
        let concurrent = try #require(report["concurrent"])
        #expect(concurrent["requests"] == 64)
        #expect((concurrent["requests_per_s"]?.doubleValue ?? 0) > 0)
    }

    @Test func rejectedResponsesFailTheConcurrentMeasurement() async throws {
        let service = PPLXServiceTests.service(maxTokens: 1)
        let router = Router(state: ServerState(), modelName: service.modelName, modelID: service.modelID)
        do {
            _ = try await Bench.concurrent(router: router, service: service, clients: 1, log: { _ in })
            Issue.record("reported benchmark throughput from rejected HTTP requests")
        } catch {
            #expect(error.kind == .internal && error.reason == "bench_failed")
            #expect(error.message.contains("HTTP 503") && error.message.contains("model_loading"))
        }
    }

    @Test func gemmaBenchmarkTextKeepsItsDocumentPrompt() throws {
        let service = TestSupport.service()
        let input = try Bench.text(tokens: 32, service: service)
        #expect(input.hasPrefix(InputType.document.prefix))
        #expect(service.tokenize(input).count == 32)
    }

    @Test func qwenBenchProfilesOnlyDecomposedSizes() async throws {
        let width = PPLXTokenTable.width
        let raw = [Float](repeating: 1, count: 259 * width).withUnsafeBytes { Data($0) }
        let table = try PPLXTokenTable(data: raw, rows: 259)
        let runtime = PPLXTextRuntime(runner: QwenProfileFixture(), table: table, norm: [Float](repeating: 1, count: width),
                                      lengths: [64, 2048], maxTokens: 128, functions: [], notServed: [],
                                      skipped: true, causal: true)
        let service = EmbeddingService(tokenizer: ByteTokenizer(modelID: .pplx), runtime: runtime, modelID: .qwen3Embedding)
        let report = try await Bench.textBreakdown(runtime, service: service, sizes: [64, 65, 129], fusedLimit: 64, log: { _ in })
        #expect(Set(report.keys) == ["65"])
        #expect(report["65"]?["tokens"] == 65 && report["65"]?["shape"] == 2048)
        #expect(report["65"]?["layers"]?.arrayValue?.count == 28)
    }

    private struct QwenProfileFixture: PPLXPredicting {
        func predict(_ function: String, inputs: [String: MLMultiArray], outputs: [String: MLMultiArray])
            async throws -> MLFeatureProvider
        {
            for tensor in outputs.values {
                tensor.withUnsafeMutableBufferPointer(ofType: Float16.self) { buffer, _ in
                    buffer.initialize(repeating: 0)
                }
            }
            return try MLDictionaryFeatureProvider(dictionary: outputs.mapValues { MLFeatureValue(multiArray: $0) })
        }
    }

    @Test func pplxRejectsImageBenchmark() async throws {
        let config = try pplxConfig(cap: 1)
        var options = Bench.Options()
        options.image = Data([0])
        do {
            _ = try await Bench.run(config, options: options, log: { _ in })
            Issue.record("accepted an image benchmark for the text-only model")
        } catch {
            #expect(error.kind == .invalidInput && error.reason == "images_unsupported")
        }
    }
}
