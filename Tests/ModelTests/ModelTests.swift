import Foundation
import Testing

@testable import EmbedCore

/// Real-model checks on the Neural Engine. They run only when a model package exists
/// (ANEEMBED_TEST_MODEL_DIR, or the installed current revision) and never on CI runners
/// (GitHub's virtualized macOS runners have no Neural Engine).
enum ModelGate {
    static var directory: URL? {
        let environment = ProcessInfo.processInfo.environment
        guard environment["CI"] == nil else { return nil }
        let url: URL
        if let path = environment["ANEEMBED_TEST_MODEL_DIR"] {
            url = URL(fileURLWithPath: path)
        } else {
            url = ModelStore(dataDirectory: Config.defaultDataDirectory).currentDirectory
        }
        return ModelLayout.model(ModelLayout.textModel, in: url) != nil ? url : nil
    }

    static var enabled: Bool { directory != nil }

    static func service(maxTokens: Int = 2048) async throws -> EmbeddingService {
        let config = try Config(flags: ["model-dir": directory!.path, "max-tokens": String(maxTokens)], environment: [:])
        return try await ModelLoader.makeService(config)
    }
}

@Suite("Model on the Neural Engine", .serialized, .enabled(if: ModelGate.enabled, "needs a model package and no CI"))
struct ModelTests {
    @Test func everyLoadedFunctionPassesTheSelfTest() async throws {
        let config = try Config(flags: ["model-dir": ModelGate.directory!.path], environment: [:])
        let (service, report) = try await ModelLoader.load(config)
        let selfTest = try #require(report.selfTest)
        #expect(selfTest.available && selfTest.failed.isEmpty, "\(selfTest.summary)")
        #expect(Set(selfTest.results.map(\.function)) == Set(service.runtime.info.functions))
        // seven embed_S, two winp, three vision budgets; no long functions at the default 2048 tokens
        #expect(service.runtime.info.functions.count == 12)
        #expect(!service.runtime.info.functions.contains { $0.hasPrefix("winl") })
    }

    /// A text of exactly `tokens` tokens (with <bos>/<eos>, no prompt).
    static func text(tokens: Int, service: EmbeddingService, seed: Int = 0) -> String {
        let words = ["neural", "engine", "임베딩", "window", "attention", "文書", "block", "fixed", "shape", "graph"]
        var text = ""
        var index = seed
        while service.tokenize(text + " " + words[index % words.count]).count <= tokens {
            text += " " + words[index % words.count]
            index += 1
        }
        while service.tokenize(text).count < tokens { text += "." }
        return text
    }

    @Test func longTextsRunInOneWindowedCall() async throws {
        let service = try await ModelGate.service(maxTokens: 8192)
        #expect(service.maxTokens == 8192 && service.runtime.info.functions.count == 16)
        let runtime = try #require(service.runtime as? BucketTextRuntime)
        let engine = try #require(runtime.long as? LongTextEngine)
        #expect(engine.lengths == [3072, 4096, 6144, 8192])
        for (tokens, length) in [(2049, 3072), (3073, 4096), (4097, 6144), (6145, 8192), (8192, 8192)] {
            let text = Self.text(tokens: tokens, service: service, seed: tokens)
            let ids = service.tokenize(text)
            #expect(abs(ids.count - tokens) <= 1)
            #expect(engine.length(forTokens: ids.count) == length)
            let vector = try await service.embedTexts([text]).vectors[0]
            #expect(abs(vector.reduce(0) { $0 + $1 * $1 }.squareRoot() - 1) < 1e-3)
        }
        // A text at the packing limit, through winp2048 and through winl3072: the same embedding.
        let text = Self.text(tokens: 2040, service: service, seed: 3)
        let ids = service.tokenize(text)
        let packed = try await runtime.embed(tokenized: [ids])[0]
        let long = try await engine.embed(ids: ids)
        #expect(Vectors.cosine(packed, long) > 0.999)
        // Without --max-tokens above 2048 the long functions are not loaded: such texts are rejected unless truncated.
        await #expect(throws: ServiceError.self) {
            _ = try await ModelGate.service(maxTokens: 2048).embedTexts([Self.text(tokens: 2100, service: service)], truncate: false)
        }
    }

    @Test func reusedBuffersGiveTheSameVectors() async throws {
        let service = try await ModelGate.service(maxTokens: 4096)
        // Same function (winl3072, then winp1536, then embed_1024), different texts in between.
        for tokens in [2500, 1400, 900] {
            let a = Self.text(tokens: tokens, service: service, seed: 1), b = Self.text(tokens: tokens - 37, service: service, seed: 5)
            let first = try await service.embedTexts([a]).vectors[0]
            let other = try await service.embedTexts([b]).vectors[0]
            let again = try await service.embedTexts([a]).vectors[0]
            #expect(Vectors.cosine(first, again) > 0.99999)
            #expect(Vectors.cosine(first, other) < 0.9999)
        }
    }

    @Test func embedsUnitVectors() async throws {
        let service = try await ModelGate.service()
        let result = try await service.embedTexts(["hello world", "task: search result | query: 서울 날씨"])
        for vector in result.vectors {
            #expect(vector.count == 768)
            #expect(abs(vector.reduce(0) { $0 + $1 * $1 }.squareRoot() - 1) < 1e-3)
        }
    }

    @Test func packingDoesNotChangeVectors() async throws {
        let service = try await ModelGate.service()
        let texts = (0..<12).map { "packed query number \($0) about topic \($0 * 7)" }
        let packed = try await service.embedTexts(texts)
        for (index, text) in texts.enumerated() {
            let alone = try await service.embedTexts([text])
            #expect(Vectors.cosine(alone.vectors[0], packed.vectors[index]) > 0.9999)
        }
    }

    @Test func windowedPackingKeepsVectors() async throws {
        let service = try await ModelGate.service()
        let runtime = try #require(service.runtime as? BucketTextRuntime)
        let texts = [Self.text(tokens: 1500, service: service, seed: 2)] + (0..<6).map { "query \($0) about windowed attention" }
        let ids = texts.map { service.tokenize($0) }
        // all seven in one winp2048 call
        var offset = 0
        let segments = ids.enumerated().map { index, item -> PackedSegment in
            defer { offset += item.count }
            return PackedSegment(item: index, offset: offset, length: item.count)
        }
        let buffers = BucketBuffers(length: 2048)
        buffers.prepare(PackedCall(length: 2048, segments: segments), ids: ids, rows: runtime.rows)
        let packed = try await runtime.runner.run(buffers)
        for (index, text) in texts.enumerated() {
            let alone = try await service.embedTexts([text]).vectors[0]
            #expect(Vectors.cosine(alone, packed[index]) > 0.9999)
        }
    }

    @Test(.enabled(if: ModelGate.directory.map { ModelLayout.model(ModelLayout.visionModel, in: $0) != nil } ?? false))
    func embedsImages() async throws {
        let service = try await ModelGate.service()
        let result = try await service.embedImages([try Data(contentsOf: URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Fixtures/images/up_320x240.png"))])
        #expect(result.tokenCounts == [266 + 4])
    }
}
