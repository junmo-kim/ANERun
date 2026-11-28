import Foundation
import Testing

@testable import EmbedCore

@Suite("PPLX HTTP and MCP routing")
struct PPLXRoutingTests {
    private let alias = "pplx-local"

    private func request(_ router: Router, _ method: String, _ path: String, _ body: JSONValue? = nil,
                         query: [(String, String)] = []) async throws -> (Int, JSONValue)
    {
        let response = await router.handle(HTTPRequest(method: method, path: path, query: query,
                                                       headers: [("Content-Type", "application/json")],
                                                       body: body?.jsonData() ?? Data()))
        return (response.status, try JSONValue.parse(response.body))
    }

    private func embedding(_ body: JSONValue) throws -> JSONValue {
        try #require(body["data"]?.arrayValue?.first?["embedding"])
    }

    @Test func httpAndMCPCatalogsUseTheSelectionBeforeAndAfterLoading() async throws {
        let state = ServerState()
        let router = Router(state: state, modelName: alias, modelID: .pplx, configuredMaxTokens: 16384)
        let handler = MCPHandler(state: state, allowPaths: true, waitForModel: true, modelID: .pplx,
                                 configuredMaxTokens: 16384)
        let list: JSONValue = ["jsonrpc": "2.0", "id": 1, "method": "tools/list"]
        let (status, loadingError) = try await request(router, "POST", "/v1/embeddings", ["input": "x"])
        #expect(status == 503 && loadingError["error"]?["code"] == "model_loading")
        let (mcpStatus, loadingMCP) = try await request(router, "POST", "/mcp", [
            "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": ["name": "embed_text", "arguments": ["text": "x"]],
        ])
        let loadingText = try #require(loadingMCP["result"]?["content"]?.arrayValue?.first?["text"]?.stringValue)
        #expect(mcpStatus == 200 && loadingMCP["result"]?["isError"] == true)
        #expect(try JSONValue.parse(loadingText)["reason"] == "model_loading")

        for ready in [false, true] {
            if ready { state.set(.ready(PPLXServiceTests.service(maxTokens: 8193))) }
            let expectedCap = JSONValue.int(ready ? 8193 : 16384)
            let (modelsStatus, models) = try await request(router, "GET", "/v1/models")
            let model = try #require(models["data"]?.arrayValue?.first)
            #expect(modelsStatus == 200 && models["data"]?.arrayValue?.count == 1)
            #expect(model["id"] == .string(alias) && model["max_image_tokens"] == 0 && model["image_preprocessing"] == nil)
            #expect(model["dimensions"] == .array((1...1024).reversed().map { .int($0) }))
            #expect(model["max_context_length"] == expectedCap)

            let (listStatus, http) = try await request(router, "POST", "/mcp", list)
            let stdio = await handler.handle(list)
            #expect(listStatus == 200 && http == stdio)
            let tools = try #require(http["result"]?["tools"]?.arrayValue)
            #expect(tools.compactMap { $0["name"]?.stringValue } == ["embed_text", "similarity"])
            for tool in tools {
                #expect(tool["_meta"]?["aneembed/max_context_length"] == expectedCap)
                let properties = try #require(tool["inputSchema"]?["properties"])
                let width = try #require(properties["dimensions"])
                #expect(width["type"] == "integer" && width["minimum"] == 1 && width["maximum"] == 1024 && width["enum"] == nil)
                for key in ["a_image", "b_image", "a_path", "b_path"] { #expect(properties[key] == nil) }
            }
        }
    }

    @Test func httpUsesQuantizedWidthsAndIgnoresValidatedPromptsAndBodyModel() async throws {
        let router = Router(state: ServerState(.ready(PPLXServiceTests.service())), modelName: alias, modelID: .pplx)
        let (status, plain) = try await request(router, "POST", "/v1/embeddings", ["input": "x", "model": "embeddinggemma-2"])
        #expect(status == 200 && plain["model"] == .string(alias))
        #expect(plain["usage"]?["prompt_tokens"] == 1 && plain["usage"]?["total_tokens"] == 1)
        #expect((plain["usage"]?["prompt_tokens_per_second"]?.doubleValue ?? 0) > 0)
        #expect(plain["usage"]?["generation_tokens_per_second"] == nil)
        let full = try #require(try embedding(plain).arrayValue)
        #expect(full.count == 1024 && full.allSatisfy { (-128...127).contains($0.intValue ?? 1000) })
        #expect(full.contains { ($0.intValue ?? 0).magnitude > 1 })

        for prompt in ["query", "document"] {
            let (status, body) = try await request(router, "POST", "/v1/embeddings", [
                "input": "x", "input_type": .string(prompt), "dimensions": 300, "encoding_format": "base64",
            ])
            let encoded = try #require(try embedding(body).stringValue)
            #expect(status == 200 && body["usage"]?["prompt_tokens"] == 1 && body["usage"]?["total_tokens"] == 1)
            #expect((body["usage"]?["prompt_tokens_per_second"]?.doubleValue ?? 0) > 0)
            #expect(Vectors.decodeBase64(encoded) == full.prefix(300).map { Float($0.intValue!) })
        }
        for (body, reason): (JSONValue, String) in [
            (["input": "x", "dimensions": 1025], "invalid_dimensions"),
            (["input": "x", "input_type": "passage"], "invalid_input_type"),
            (["input": ["x", ""]], "empty_sequence"),
            (["input": [HTTPContractTests.imageItem("data:image/png;base64,!!!!")]], "images_unsupported"),
        ] {
            let (status, error) = try await request(router, "POST", "/v1/embeddings", body)
            #expect(status == 400 && error["error"]?["code"] == .string(reason))
        }
        let (queryStatus, query) = try await request(router, "QUERY", "/embed_text", ["text": "x", "input_type": "query"],
                                                    query: [("format", "json"), ("dimensions", "1")])
        #expect(queryStatus == 200 && query["model"] == .string(alias))
        #expect(try embedding(query) == .array(Array(full.prefix(1))))
    }

    @Test func mcpUsesQuantizedOutputsAndRejectsImagesBeforeDecodingOrReading() async throws {
        let handler = MCPHandler(state: ServerState(.ready(PPLXServiceTests.service(maxInputs: 1))), allowPaths: true,
                                 waitForModel: true, modelID: .pplx)
        let mcp = MCPTests()
        let (failed, plain) = try await mcp.call(handler, "embed_text", ["text": "x", "dimensions": 17])
        let (promptFailed, prompted) = try await mcp.call(handler, "embed_text", [
            "text": "x", "dimensions": 17, "input_type": "query", "encoding_format": "base64",
        ])
        let head = try #require(try embedding(plain).arrayValue)
        let encoded = try #require(try embedding(prompted).stringValue)
        #expect(!failed && !promptFailed && head.count == 17 && plain["model"] == .string(alias))
        #expect(Vectors.decodeBase64(encoded) == head.map { Float($0.intValue!) })
        let (_, other) = try await mcp.call(handler, "embed_text", ["text": "y", "dimensions": 17])
        let otherHead = try #require(try embedding(other).arrayValue)
        let (similarityFailed, comparison) = try await mcp.call(handler, "similarity", [
            "a": "x", "b": "y", "dimensions": 17, "a_input_type": "query", "b_input_type": "document",
        ])
        let cosine = Vectors.cosine(head.map { Float($0.intValue!) }, otherHead.map { Float($0.intValue!) })
        #expect(!similarityFailed && comparison["tokens"] == [1, 1])
        #expect(abs((comparison["cosine"]?.doubleValue ?? 2) - Double(cosine)) < 1e-7)

        for (tool, args, reason): (String, JSONValue, String) in [
            ("embed_text", ["text": "x", "dimensions": 1025], "invalid_dimensions"),
            ("embed_image", ["data": "!!!!"], "images_unsupported"),
            ("embed_image", ["path": "/does/not/exist/aneembed.png"], "images_unsupported"),
            ("similarity", ["a": "x", "b_path": "/does/not/exist/aneembed.png"], "images_unsupported"),
        ] {
            let (failed, error) = try await mcp.call(handler, tool, args)
            #expect(failed && error["reason"] == .string(reason))
        }
    }
}
