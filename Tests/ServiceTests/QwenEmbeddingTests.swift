import Foundation
import Testing

@testable import EmbedCore

@Suite("Qwen3 Embedding public contract")
struct QwenEmbeddingTests {
    private let model = ModelID.qwen3Embedding
    private let query = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:fox"

    private func config(cap: Int = 2048) throws -> Config {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        return try Config(flags: ["model": model.rawValue, "runtime": "fake",
                                  "max-tokens": String(cap), "data-dir": directory.path], environment: [:])
    }

    @Test func selectorUsesPublishedQwenPackage() throws {
        let selected = try config().selectingModel(for: "doctor")
        #expect(selected.modelID == model && selected.modelName == "qwen3-embedding-0.6b")
        #expect(selected.maxTokens == 2048 && selected.limits.shortTokens == 64)
        #expect(model.source.isPublished && model.source.repository == "junmo-k/qwen3-embedding-0.6b-ane")
        let report = Doctor.report(selected)
        #expect(report["model"]?["name"] == .string(model.rawValue))
        #expect(report["text_functions"] == [])
        #expect(report["ready"] == true)
    }

    @Test func promptsDimensionsAndTextOnlyRouting() async throws {
        let service = try await ModelLoader.makeService(config(), log: { _ in })
        let router = Router(state: ServerState(.ready(service)), modelName: service.modelName, modelID: model)
        func request(_ path: String, _ body: JSONValue? = nil) async throws -> (Int, JSONValue) {
            let response = await router.handle(HTTPRequest(method: body == nil ? "GET" : "POST", path: path,
                                                         body: body?.jsonData() ?? Data()))
            let body = try JSONValue.parse(response.body)
            if path == "/v1/embeddings", response.status == 200 {
                var object = try #require(body.objectValue)
                var usage = try #require(object["usage"]?.objectValue)
                #expect((usage["prompt_tokens_per_second"]?.doubleValue ?? 0) > 0)
                usage["prompt_tokens_per_second"] = nil
                object["usage"] = .object(usage)
                return (response.status, .object(object))
            }
            return (response.status, body)
        }
        func embedding(_ body: JSONValue) throws -> [Float] {
            let array = try #require(body["data"]?.arrayValue?.first?["embedding"]?.arrayValue)
            return try array.map { Float(try #require($0.doubleValue)) }
        }

        let plain = try await request("/v1/embeddings", ["input": "fox"])
        let document = try await request("/v1/embeddings", ["input": "fox", "input_type": "document"])
        #expect(plain.0 == 200 && document.0 == 200 && plain.1 == document.1)
        let full = try embedding(plain.1)
        #expect(full.count == 1024 && abs(full.reduce(0) { $0 + $1 * $1 } - 1) < 1e-5)

        let prompted = try await request("/v1/embeddings", ["input": "fox", "input_type": "query", "dimensions": 37])
        let manual = try await request("/v1/embeddings", ["input": .string(query), "dimensions": 37])
        let prefix = try embedding(prompted.1)
        #expect(prompted.0 == 200 && prompted.1 == manual.1 && prefix.count == 37)
        #expect(prompted.1["usage"]?["prompt_tokens"]?.intValue == query.utf8.count + 1)
        #expect(abs(prefix.reduce(0) { $0 + $1 * $1 } - 1) < 1e-5)

        for size in [31, 1025] {
            let invalid = try await request("/v1/embeddings", ["input": "fox", "dimensions": .int(size)])
            #expect(invalid.0 == 400 && invalid.1["error"]?["code"] == "invalid_dimensions")
        }
        let images = try await request("/v1/embeddings", ["input": [HTTPContractTests.imageItem("data:image/png;base64,!!!!")]])
        #expect(images.0 == 400 && images.1["error"]?["code"] == "images_unsupported")

        let models = try await request("/v1/models")
        let metadata = try #require(models.1["data"]?.arrayValue?.first)
        #expect(metadata["dimensions"] == .array((32...1024).reversed().map { .int($0) }))
        #expect(metadata["max_image_tokens"] == 0 && metadata["image_preprocessing"] == nil)
        let tools = Tools.definitions(for: model)
        #expect(tools.compactMap { $0["name"]?.stringValue } == ["embed_text", "similarity"])
        #expect(tools.first?["inputSchema"]?["properties"]?["dimensions"]?["minimum"] == 32)

        let context = ToolContext(service: service, allowPaths: true)
        let similarity = try await Tools.call("similarity", arguments: [
            "a": "fox", "a_input_type": "query", "b": .string(query), "b_input_type": "document", "dimensions": 37,
        ], context: context)
        #expect(abs((similarity["cosine"]?.doubleValue ?? 0) - 1) < 1e-6)
    }

    @Test func smallBenchmarkCountsEOSAndFallsBackToRawText() async throws {
        var options = Bench.Options()
        options.sizes = [1]
        options.repeats = 1
        options.concurrency = 1
        let report = try await Bench.run(config(cap: 1), options: options, log: { _ in })
        #expect(report["single"]?["1"]?["tokens"] == 1)
        #expect(report["concurrent"]?["requests"] == 32)
    }
}
