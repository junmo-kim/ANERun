import Foundation
import Testing

@testable import EmbedCore

@Suite("MCP (stdio semantics, fake runtime)")
struct MCPTests {
    func call(_ handler: MCPHandler, _ name: String, _ arguments: JSONValue) async throws -> (Bool, JSONValue) {
        let response = await handler.handle(["jsonrpc": "2.0", "id": 9, "method": "tools/call",
                                             "params": ["name": .string(name), "arguments": arguments]])
        let result = try #require(response?["result"])
        let text = try #require(result["content"]?.arrayValue?.first?["text"]?.stringValue)
        return (result["isError"] == true, try JSONValue.parse(text))
    }

    @Test func protocolBasics() async {
        let handler = MCPHandler(state: ServerState(.loading), allowPaths: true, waitForModel: true)
        let initialize = await handler.handle(["jsonrpc": "2.0", "id": 1, "method": "initialize", "params": ["protocolVersion": "1999-01-01"]])
        #expect(initialize?["result"]?["protocolVersion"] == "2025-06-18")
        #expect(await handler.handle(["jsonrpc": "2.0", "id": 2, "method": "ping"])?["result"] == [:])
        #expect(await handler.handle(["jsonrpc": "2.0", "method": "notifications/initialized"]) == nil)
        #expect(await handler.handle(["jsonrpc": "2.0", "id": 3, "method": "nope"])?["error"]?["code"] == -32601)
        #expect(await handler.handle(["jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": ["name": "nope"]])?["error"]?["code"] == -32602)
        #expect(await handler.handle(raw: Data("{".utf8))?["error"]?["code"] == -32700)
    }

    @Test func gemmaCatalogPreservesToolInputsAndPublishesTheTextLimit() async throws {
        let state = ServerState()
        let handler = MCPHandler(state: state, allowPaths: true, waitForModel: true, configuredMaxTokens: 1024)
        let request: JSONValue = ["jsonrpc": "2.0", "id": 1, "method": "tools/list"]
        for cap in [1024, 64] {
            if cap == 64 { state.set(.ready(TestSupport.service(maxTokens: cap))) }
            let response = await handler.handle(request)
            let tools = try #require(response?["result"]?["tools"]?.arrayValue)
            #expect(tools.compactMap { $0["name"]?.stringValue } == ["embed_text", "embed_image", "similarity"])
            #expect(tools.map { $0["inputSchema"] } == Tools.definitions.map { $0["inputSchema"] })
            for index in [0, 2] {
                let truncate = tools[index]["inputSchema"]?["properties"]?["truncate"]
                #expect(truncate?["type"] == "boolean" && truncate?["default"] == true)
            }
            #expect(tools[0]["_meta"]?["aneembed/max_context_length"] == .int(cap))
            #expect(tools[1] == Tools.definitions[1])
            #expect(tools[2]["_meta"]?["aneembed/max_context_length"] == .int(cap))
        }
    }

    @Test func toolsEmbedAndCompare() async throws {
        let handler = MCPHandler(state: ServerState(.ready(TestSupport.service())), allowPaths: true, waitForModel: true)
        let (failed, output) = try await call(handler, "embed_text", ["texts": ["a", "b"], "input_type": "document", "dimensions": 512])
        #expect(!failed)
        #expect(output["data"]?.arrayValue?.count == 2)
        #expect(output["data"]?.arrayValue?[0]["embedding"]?.arrayValue?.count == 512)
        #expect(output["data"]?.arrayValue?.allSatisfy { $0["truncated"] == nil } == true)

        let (_, same) = try await call(handler, "similarity", ["a": "x", "b": "x"])
        #expect(abs((same["cosine"]?.doubleValue ?? 0) - 1) < 1e-5)
        #expect(same["truncated_indices"] == nil)

        let png = TestSupport.png(width: 64, height: 64)
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("aneembed-mcp-\(UUID().uuidString).png")
        try png.write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        let (_, image) = try await call(handler, "embed_image", ["path": .string(file.path)])
        let (_, inline) = try await call(handler, "embed_image", ["data": .string(png.base64EncodedString())])
        #expect(image["data"] == inline["data"])
        #expect(image["data"]?.arrayValue?.first?["truncated"] == nil)
        let (_, mixed) = try await call(handler, "similarity", ["a_path": .string(file.path), "b_image": .string(png.base64EncodedString())])
        #expect(abs((mixed["cosine"]?.doubleValue ?? 0) - 1) < 1e-5)
    }

    @Test func toolErrorsAreStructured() async throws {
        let handler = MCPHandler(state: ServerState(.ready(TestSupport.service(maxTokens: 64))), allowPaths: true, waitForModel: true)
        let (failed, error) = try await call(handler, "embed_text", [
            "text": .string(String(repeating: "z", count: 100)), "truncate": false,
        ])
        #expect(failed)
        #expect(error["error"] == "invalid_input" && error["reason"] == "context_length_exceeded")
        let (missing, missingError) = try await call(handler, "embed_text", [:])
        #expect(missing && missingError["reason"] == "missing_input")
    }

    @Test func embedTextReportsOnlyClippedRows() async throws {
        let handler = MCPHandler(state: ServerState(.ready(TestSupport.service(maxTokens: 64))), allowPaths: true, waitForModel: true)
        let overlong: JSONValue = .string(String(repeating: "z", count: 100))
        let exact: JSONValue = .string(String(repeating: "z", count: 62))
        let (failed, output) = try await call(handler, "embed_text", ["texts": ["x", overlong, exact]])
        let rows = try #require(output["data"]?.arrayValue)
        #expect(!failed)
        #expect(rows.map { $0["index"] } == [0, 1, 2])
        #expect(rows.map { $0["tokens"] } == [3, 64, 64])
        #expect(rows.map { $0["truncated"] } == [nil, true, nil])
        #expect(output["usage"]?["prompt_tokens"] == 131)
    }

    @Test(arguments: [ModelID.embeddingGemma2, .pplx, .qwen3Embedding])
    func similarityReportsClippedOperandsAndOffersStrictMode(_ model: ModelID) async throws {
        var limits = ServiceLimits()
        limits.maxTokens = 64
        let service = EmbeddingService(tokenizer: ByteTokenizer(modelID: model),
                                       runtime: FakeRuntime.make(maxTokens: 64, modelID: model),
                                       limits: limits, modelID: model)
        let handler = MCPHandler(state: ServerState(.ready(service)), allowPaths: true, waitForModel: true)
        let overlong: JSONValue = .string(String(repeating: "z", count: 100))
        let (failed, output) = try await call(handler, "similarity", ["a": "x", "b": overlong])
        #expect(!failed)
        #expect(output["tokens"]?.arrayValue?.last == 64)
        #expect(output["truncated_indices"] == [1])
        let (bothFailed, both) = try await call(handler, "similarity", ["a": overlong, "b": overlong])
        #expect(!bothFailed && both["truncated_indices"] == [0, 1])
        let (strictFailure, error) = try await call(handler, "similarity", ["a": "x", "b": overlong, "truncate": false])
        #expect(strictFailure && error["reason"] == "context_length_exceeded")
    }

    @Test func truncateRequiresABoolean() async throws {
        let handler = MCPHandler(state: ServerState(.ready(TestSupport.service())), allowPaths: true, waitForModel: true)
        for invalid: JSONValue in ["false", .null] {
            let (failed, error) = try await call(handler, "embed_text", ["text": "x", "truncate": invalid])
            #expect(failed && error["reason"] == "invalid_truncate")
        }
    }

    /// embed_image (MCP, and QUERY /embed_image with a JSON body) counts the images before decoding any, as
    /// /v1/embeddings does: one image too many is too_many_images even when it is not valid base64.
    @Test func imagesAreCountedBeforeAnyIsDecoded() async {
        let context = ToolContext(service: TestSupport.service(), allowPaths: false)
        let png = TestSupport.png(width: 2, height: 2).base64EncodedString()
        let list = [JSONValue](repeating: .string(png), count: 8) + [.string("!")]
        do {
            _ = try await Tools.call("embed_image", arguments: ["images": .array(list)], context: context)
            Issue.record("nine images were accepted")
        } catch {
            #expect(error.reason == "too_many_images" && error.httpStatus == 400 && error.param == "images")
        }
    }

    /// Each image's size is known from its base64 before it is decoded: one over --max-image-bytes is refused as
    /// image_too_large naming the argument, and line breaks or other characters the decoder skips do not count.
    @Test func anImageOverTheByteLimitIsRefusedBeforeDecoding() async throws {
        var limits = ServiceLimits()
        limits.maxImageBytes = 4096
        let context = ToolContext(service: TestSupport.service(limits: limits), allowPaths: false)
        let png = TestSupport.png(width: 2, height: 2)
        #expect(png.count < 4096)
        // 8 KB of line breaks around a small image: longer than the limit, but it decodes to the image alone.
        let padded = "data:image/png;base64," + png.base64EncodedString() + String(repeating: "\r\n", count: 4096)
        let accepted = try await Tools.call("embed_image", arguments: ["images": [.string(padded)]], context: context)
        #expect(accepted["data"]?.arrayValue?.count == 1)
        let big = Data(repeating: 0xAB, count: 4097).base64EncodedString()
        for arguments: JSONObject in [["data": .string(big)], ["images": [.string(png.base64EncodedString()), .string(big)]],
                                      ["a_image": .string(big), "b": "x"]]
        {
            let tool = arguments["b"] == nil ? "embed_image" : "similarity"
            do {
                _ = try await Tools.call(tool, arguments: arguments, context: context)
                Issue.record("\(arguments.keys.sorted()) over the limit was accepted")
            } catch {
                #expect(error.reason == "image_too_large" && error.httpStatus == 413)
                #expect(["data", "images[1]", "a_image"].contains(error.param ?? "") && error.message.hasPrefix("\(error.param!) is 4097 bytes"))
            }
        }
    }

    @Test func anOverlongStdioMessageIsAnInvalidRequest() {
        let error = MCPHandler.messageTooLarge(bytes: 2_000_000, limit: 1 << 20)
        #expect(error["id"] == .null && error["error"]?["code"] == -32600)
        #expect(error["error"]?["message"]?.stringValue?.contains("1048576") == true)
    }

    @Test func toolCallsWaitForTheModel() async throws {
        let state = ServerState(.loading)
        let handler = MCPHandler(state: state, allowPaths: true, waitForModel: true)
        Task {
            try? await Task.sleep(for: .milliseconds(50))
            state.set(.ready(TestSupport.service()))
        }
        let (failed, _) = try await call(handler, "embed_text", ["text": "hi"])
        #expect(!failed)
    }
}
