import Foundation
import Testing

@testable import EmbedCore

/// The HTTP/OpenAI contract of the prototype's blackbox.py, through the router and the fake runtime.
@Suite("HTTP contract (fake runtime)")
struct HTTPContractTests {
    let router: Router

    init() {
        router = Router(state: ServerState(.ready(TestSupport.service())), modelName: "embeddinggemma-2")
    }

    func call(_ method: String, _ path: String, _ body: JSONValue? = nil, router: Router? = nil) async -> (Int, JSONValue) {
        let request = HTTPRequest(method: method, path: path, headers: [("Content-Type", "application/json")],
                                  body: body?.jsonData() ?? Data())
        let response = await (router ?? self.router).handle(request)
        return (response.status, (try? JSONValue.parse(response.body)) ?? .null)
    }

    func embedding(_ body: JSONValue, at index: Int = 0) -> [Float] {
        body["data"]?.arrayValue?[index]["embedding"]?.arrayValue?.map { Float($0.doubleValue!) } ?? []
    }

    static func imageItem(_ url: String) -> JSONValue {
        ["content": [["type": "image_url", "image_url": ["url": .string(url)]]]]
    }

    @Test func healthAndModels() async {
        let (status, body) = await call("GET", "/health")
        #expect(status == 200 && body == ["status": "ok"])
        let (_, models) = await call("GET", "/v1/models")
        let model = models["data"]?.arrayValue
        #expect(model?.count == 1)
        #expect(model?[0]["id"] == "embeddinggemma-2")
        // The fields of OpenAI's model object
        #expect(model?[0]["object"] == "model" && model?[0]["owned_by"] == "local")
        #expect(model?[0]["created"]?.intValue == 1_794_700_800)
        #expect(model?[0]["max_context_length"]?.intValue == 2048)
        #expect(model?[0]["max_image_tokens"]?.intValue == 280)
        #expect(model?[0]["dimensions"] == [768, 512, 256, 128])
    }

    /// Functions that failed the self-test (or a skipped self-test) leave the server up but degraded: still 200,
    /// so a client keeps working, with what is not served and the context limit that remains.
    @Test func healthReportsADegradedModel() async {
        let lengths = [64, 128, 256, 512, 768, 1024, 1280, 1536]
        let runtime = BucketTextRuntime(runner: PoolingFakeRunner(lengths: lengths), rows: HashedTokenRows(), backend: "fake",
                                        notServed: ["winp2048_b512"])
        let degraded = Router(state: ServerState(.ready(EmbeddingService(tokenizer: ByteTokenizer(), runtime: runtime))), modelName: "m")
        let (status, body) = await call("GET", "/health", router: degraded)
        #expect(status == 200 && body["status"] == "degraded")
        #expect(body["not_served"] == ["winp2048_b512"] && body["max_context_length"]?.intValue == 1536)
        let unchecked = BucketTextRuntime(runner: PoolingFakeRunner(), rows: HashedTokenRows(), backend: "coreml-ane",
                                          selfTestSkipped: true)
        let skipped = Router(state: ServerState(.ready(EmbeddingService(tokenizer: ByteTokenizer(), runtime: unchecked))), modelName: "m")
        let (skippedStatus, skippedBody) = await call("GET", "/health", router: skipped)
        #expect(skippedStatus == 200 && skippedBody["status"] == "degraded" && skippedBody["selftest"] == "skipped")
    }

    @Test func loadingAndFailedStates() async {
        let loading = Router(state: ServerState(.loading), modelName: "m")
        #expect(await call("GET", "/health", router: loading) == (503, ["status": "loading"]))
        let (status, body) = await call("POST", "/v1/embeddings", ["input": "x"], router: loading)
        #expect(status == 503)
        #expect(body["error"]?["type"] == "server_error")
        #expect(body["error"]?["code"] == "model_loading")
        let failed = Router(state: ServerState(.failed(ServiceError(.unavailable, "model_missing", "no model"))), modelName: "m")
        let (failedStatus, failedBody) = await call("GET", "/health", router: failed)
        #expect(failedStatus == 503 && failedBody["status"] == "error")
    }

    @Test func singleBatchAndUsage() async {
        let (status, single) = await call("POST", "/v1/embeddings", ["input": "hello there"])
        #expect(status == 200)
        #expect(embedding(single).count == 768)
        #expect(single["usage"]?["prompt_tokens"]?.intValue == 13)
        #expect(single["model"] == "embeddinggemma-2")

        let texts: [JSONValue] = (0..<20).map { .string("query \($0) " + String(repeating: "w", count: $0)) }
        let (_, batch) = await call("POST", "/v1/embeddings", ["input": .array(texts), "model": "anything"])
        #expect(batch["data"]?.arrayValue?.map { $0["index"]?.intValue } == (0..<20).map { $0 })
        for index in [0, 7, 19] {
            let (_, alone) = await call("POST", "/v1/embeddings", ["input": texts[index]])
            #expect(Vectors.cosine(embedding(alone), embedding(batch, at: index)) > 0.999)
        }
    }

    @Test func base64AndDimensions() async throws {
        let (_, floats) = await call("POST", "/v1/embeddings", ["input": "x"])
        let (_, encoded) = await call("POST", "/v1/embeddings", ["input": "x", "encoding_format": "base64"])
        let decoded = try #require(Vectors.decodeBase64(encoded["data"]!.arrayValue![0]["embedding"]!.stringValue!))
        #expect(zip(decoded, embedding(floats)).allSatisfy { abs($0 - $1) < 1e-6 })
        let (_, short) = await call("POST", "/v1/embeddings", ["input": "x", "dimensions": 256])
        let truncated = embedding(short)
        #expect(truncated.count == 256 && abs(TestSupport.norm(truncated) - 1) < 1e-5)
        #expect(Vectors.cosine(truncated, Array(embedding(floats).prefix(256))) > 0.99999)
    }

    @Test func badRequests() async {
        for body: JSONValue in [["input": "x", "dimensions": 300], ["input": [1, 2, 3]], ["input": []]] {
            let (status, error) = await call("POST", "/v1/embeddings", body)
            #expect(status == 400, "\(body.jsonString())")
            #expect(error["error"]?["type"] == "invalid_request_error")
        }
        let raw = await router.handle(HTTPRequest(method: "POST", path: "/v1/embeddings", body: Data("{".utf8)))
        #expect(raw.status == 400)
    }

    @Test func defaultTruncationMarksOnlyClippedRowsAndCountsUsedTokens() async {
        let exact = String(repeating: "y", count: 2046)
        let (status, body) = await call("POST", "/v1/embeddings", ["input": [.string(exact), .string(exact + "z")]])
        #expect(status == 200)
        #expect(body["data"]?.arrayValue?[0]["truncated"] == nil)
        #expect(body["data"]?.arrayValue?[1]["truncated"] == true)
        #expect(body["usage"]?["prompt_tokens"]?.intValue == 4096)
        #expect(embedding(body) == embedding(body, at: 1))
        for value: JSONValue in [.null, "true", 1] {
            let (status, error) = await call("POST", "/v1/embeddings", ["input": "short", "truncate": value])
            #expect(status == 400 && error["error"]?["code"] == "invalid_truncate")
        }
    }

    @Test func overTheLimitIsRejectedInStrictMode() async {
        let tooLong = String(repeating: "y", count: 2100)
        let (status, body) = await call("POST", "/v1/embeddings", ["input": ["ok", .string(tooLong)], "truncate": false])
        #expect(status == 400)
        #expect(body["error"]?["code"] == "context_length_exceeded")
        #expect(body["error"]?["message"]?.stringValue?.contains("2048") == true)
    }

    @Test func longTextsAndMixedBatches() async {
        let long = String(repeating: "long text ", count: 150)  // 1502 tokens
        let (status, body) = await call("POST", "/v1/embeddings", ["input": .string(long)])
        #expect(status == 200 && body["usage"]?["prompt_tokens"]?.intValue == 1502)
        #expect(abs(TestSupport.norm(embedding(body)) - 1) < 1e-4)
        let mixed: [JSONValue] = ["a", .string(long), "b", "c"]
        let (_, batch) = await call("POST", "/v1/embeddings", ["input": .array(mixed)])
        #expect(batch["data"]?.arrayValue?.map { $0["index"]?.intValue } == [0, 1, 2, 3])
        #expect(Vectors.cosine(embedding(batch, at: 1), embedding(body)) > 0.999)
    }

    @Test func imageItems() async {
        let small = TestSupport.dataURL(TestSupport.png(width: 320, height: 240))
        let (status, one) = await call("POST", "/v1/embeddings", ["input": [Self.imageItem(small)]])
        #expect(status == 200 && one["usage"]?["prompt_tokens"]?.intValue == 270)
        #expect(abs(TestSupport.norm(embedding(one)) - 1) < 1e-4)
        let urls = [TestSupport.png(width: 1600, height: 1200), TestSupport.png(width: 1, height: 1),
                    TestSupport.png(width: 3000, height: 10)].map { Self.imageItem(TestSupport.dataURL($0)) }
        let (_, three) = await call("POST", "/v1/embeddings", ["input": .array(urls)])
        #expect(three["usage"]?["prompt_tokens"]?.intValue == 266 + 256 + 280 + 12)
        #expect(Vectors.cosine(embedding(three), embedding(one)) > 0.999)

        let nine = [JSONValue](repeating: Self.imageItem(small), count: 9)
        for body: JSONValue in [
            ["input": .array(nine)], ["input": [Self.imageItem(small), "text"]],
            ["input": [Self.imageItem("http://x/y.png")]], ["input": [Self.imageItem("data:image/png;base64,AAAA")]],
        ] {
            let (status, error) = await call("POST", "/v1/embeddings", body)
            #expect(status == 400)
            #expect(error["error"] != nil)
        }
    }

    /// A 141-byte JPEG whose DC Huffman symbol is 255 used to trap the server process. Now it is a 400.
    static let crashJPEG = "/9j/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAAL"
        + "CAAIAAgBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAA///EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AAP/Z"

    @Test func aHostileJPEGIsABadRequest() async throws {
        let (status, body) = await call("POST", "/v1/embeddings", ["input": [Self.imageItem("data:image/jpeg;base64," + Self.crashJPEG)]])
        #expect(status == 400 && body["error"]?["code"] == "image_decode_failed")
        let raw = await router.handle(HTTPRequest(method: "QUERY", path: "/embed_image", query: [("format", "json")],
                                                  headers: [("Content-Type", "image/jpeg")],
                                                  body: try #require(Data(base64Encoded: Self.crashJPEG))))
        #expect(raw.status == 400)
    }

    /// Nine image items are refused for their number before any is decoded.
    @Test func tooManyImagesAreCountedBeforeDecoding() async {
        let items = [JSONValue](repeating: Self.imageItem("data:image/png;base64,!!!!"), count: 9)
        let (status, body) = await call("POST", "/v1/embeddings", ["input": .array(items)])
        #expect(status == 400 && body["error"]?["code"] == "too_many_images")
    }

    /// A body that is not UTF-8 is refused like any other invalid JSON, not embedded with U+FFFD.
    @Test func invalidUTF8IsABadRequest() async throws {
        let body = Data(#"{"input":"a"#.utf8) + Data([0xFF]) + Data(#"b"}"#.utf8)
        let response = await router.handle(HTTPRequest(method: "POST", path: "/v1/embeddings", body: body))
        #expect(response.status == 400)
        #expect(try JSONValue.parse(response.body)["error"]?["code"] == "invalid_json")
        let raw = await router.handle(HTTPRequest(method: "QUERY", path: "/embed_text", query: [("format", "json")],
                                                  headers: [("Content-Type", "text/plain")], body: Data([0x61, 0xFF, 0x62])))
        #expect(raw.status == 400)
        #expect(try JSONValue.parse(raw.body)["reason"] == "invalid_utf8")
    }

    @Test func manyConcurrentRequests() async {
        let statuses = await withTaskGroup(of: Int.self) { group in
            for index in 0..<64 {
                group.addTask { await call("POST", "/v1/embeddings", ["input": .string("q\(index)")]).0 }
            }
            return await group.reduce(into: [Int]()) { $0.append($1) }
        }
        #expect(statuses.count == 64 && statuses.allSatisfy { $0 == 200 })
    }

    @Test func unknownRoutesAndMethods() async {
        #expect(await call("GET", "/nope").0 == 404)
        #expect(await call("GET", "/v1/embeddings").0 == 405)
    }

    /// llama-server's POST /tokenize and /detokenize (the fake runtime's tokenizer: one id per byte + 3).
    @Test func tokenizeAndDetokenizeLikeLlamaServer() async {
        // add_special defaults to false, as in llama-server
        #expect(await call("POST", "/tokenize", ["content": "hi"]) == (200, ["tokens": [107, 108]]))
        #expect(await call("POST", "/tokenize", ["content": "hi", "add_special": true]) == (200, ["tokens": [2, 107, 108, 1]]))
        #expect(await call("POST", "/tokenize", ["content": ""]) == (200, ["tokens": []]))
        let (status, pieces) = await call("POST", "/tokenize", ["content": "é", "add_special": true, "with_pieces": true])
        #expect(status == 200)
        #expect(pieces == ["tokens": [["id": 2, "piece": "<bos>"], ["id": 198, "piece": [195]], ["id": 172, "piece": [169]],
                                      ["id": 1, "piece": "<eos>"]]])
        // Not limited to the embedding context: a client measures long texts to split them.
        let long = String(repeating: "x", count: 5000)
        let (_, measured) = await call("POST", "/tokenize", ["content": .string(long)])
        #expect(measured["tokens"]?.arrayValue?.count == 5000)

        #expect(await call("POST", "/detokenize", ["tokens": [2, 107, 108, 1]]) == (200, ["content": "<bos>hi<eos>"]))
        #expect(await call("POST", "/detokenize", ["tokens": [198, 172]]) == (200, ["content": "é"]))
        #expect(await call("POST", "/detokenize", ["tokens": [198]]) == (200, ["content": "\u{FFFD}"]))
        #expect(await call("POST", "/detokenize", ["tokens": []]) == (200, ["content": ""]))
    }

    /// The simulated long call blocks synchronously. Give it its own thread so
    /// short requests can run even with a single cooperative executor thread.
    private final class BlockingTokenizerExecutor: TaskExecutor {
        private let queue = DispatchQueue(label: "test.blocking-tokenizer")

        func enqueue(_ job: consuming ExecutorJob) {
            let job = UnownedJob(job)
            queue.async { job.runSynchronously(on: self.asUnownedTaskExecutor()) }
        }
    }

    @Test func smallRawTokenizerRequestsUseReservedSlot() async throws {
        var limits = ServiceLimits()
        limits.longConcurrency = 2
        let service = EmbeddingService(tokenizer: ByteTokenizer(), runtime: FakeRuntime.make(), limits: limits)
        let router = Router(state: ServerState(.ready(service)), modelName: service.modelName)
        let (entered, ready) = AsyncStream<Void>.makeStream()
        let release = DispatchSemaphore(value: 0)
        let large = Task(executorPreference: BlockingTokenizerExecutor()) {
            try await service.withTokenizerLane {
                ready.yield(())
                ready.finish()
                return release.wait(timeout: .now() + 120) == .success
            }
        }
        for await _ in entered { break }

        async let tokens = call("POST", "/tokenize", ["content": "hi"], router: router)
        async let text = call("POST", "/detokenize", ["tokens": [107, 108]], router: router)
        let responses = await (tokens, text)
        release.signal()
        #expect(try await large.value)
        #expect(responses.0.0 == 200)
        #expect(responses.1.0 == 200)
    }

    @Test func tokenizerAPIErrors() async {
        func failure(_ path: String, _ body: JSONValue?) async -> (Int, JSONValue?, JSONValue?) {
            let (status, response) = await call("POST", path, body)
            return (status, response["error"]?["code"], response["error"]?["param"])
        }
        #expect(await failure("/tokenize", ["text": "hi"]) == (400, "invalid_content", "content"))
        #expect(await failure("/tokenize", ["content": 3]) == (400, "invalid_content", "content"))
        #expect(await failure("/tokenize", ["content": "hi", "add_special": "yes"]) == (400, "invalid_add_special", "add_special"))
        #expect(await failure("/tokenize", ["content": "hi", "with_pieces": 1]) == (400, "invalid_with_pieces", "with_pieces"))
        #expect(await failure("/tokenize", ["content": "hi", "parse_special": "no"]) == (400, "invalid_parse_special", "parse_special"))
        #expect(await failure("/detokenize", ["text": "hi"]) == (400, "invalid_tokens", "tokens"))
        // Bounded: at most 1 MiB of content, and of detokenized text (plus room for <bos>/<eos>)
        let limit = TokenizeRequest.maxContentBytes
        #expect(await call("POST", "/tokenize", ["content": .string(String(repeating: "x", count: limit))]).0 == 200)
        #expect(await failure("/tokenize", ["content": .string(String(repeating: "x", count: limit + 1))])
            == (413, "content_too_long", "content"))
        let atLimit = JSONValue.array([2] + Array(repeating: JSONValue.int(123), count: limit) + [1])
        #expect(await call("POST", "/detokenize", ["tokens": atLimit]).0 == 200)
        let overLimit = JSONValue.array(Array(repeating: JSONValue.int(123), count: DetokenizeRequest.maxContentBytes + 1))
        #expect(await failure("/detokenize", ["tokens": overLimit]) == (413, "content_too_long", "tokens"))
        let (notJSON, notJSONCode, _) = await failure("/tokenize", nil)
        #expect(notJSON == 400 && notJSONCode == "invalid_json")
        #expect(await failure("/detokenize", ["tokens": "1 2"]) == (400, "invalid_tokens", "tokens"))
        #expect(await failure("/detokenize", ["tokens": [107, 259]]) == (400, "invalid_token", "tokens[1]"))
        #expect(await failure("/detokenize", ["tokens": [-1]]) == (400, "invalid_token", "tokens[0]"))
        #expect(await failure("/detokenize", ["tokens": [1.5]]) == (400, "invalid_token", "tokens[0]"))
        #expect(await call("GET", "/tokenize").0 == 405)
        #expect(await call("GET", "/detokenize").0 == 405)
        let loading = Router(state: ServerState(.loading), modelName: "m")
        let (status, body) = await call("POST", "/tokenize", ["content": "x"], router: loading)
        #expect(status == 503 && body["error"]?["code"] == "model_loading")
    }
}

@Suite("QUERY and MCP over HTTP (fake runtime)")
struct ToolEndpointTests {
    let router = Router(state: ServerState(.ready(TestSupport.service())), modelName: "embeddinggemma-2")

    func query(_ target: String, body: Data, contentType: String? = "text/plain") async -> HTTPResponse {
        let (path, query) = HTTPParser.splitTarget(target)
        return await router.handle(HTTPRequest(method: "QUERY", path: path, query: query,
                                               headers: contentType.map { [("Content-Type", $0)] } ?? [], body: body))
    }

    @Test func embedTextWithRawBody() async throws {
        let response = await query("/embed_text?input_type=query&dimensions=128&format=json", body: Data("hello".utf8))
        #expect(response.status == 200)
        let body = try JSONValue.parse(response.body)
        #expect(body["dimensions"] == 128)
        #expect(body["data"]?.arrayValue?[0]["tokens"]?.intValue == "task: search result | query: hello".utf8.count + 2)
        let yaml = await query("/embed_text", body: Data("hello".utf8))
        #expect(String(decoding: yaml.body, as: UTF8.self).hasPrefix("model: embeddinggemma-2\n"))
        #expect(yaml.header("Content-Type")?.hasPrefix("application/yaml") == true)
    }

    @Test func rawTextTruncationFlag() async throws {
        let text = Data(repeating: 97, count: 2050)
        let strict = await query("/embed_text?format=json&truncate=false", body: text)
        #expect(strict.status == 400)
        #expect(try JSONValue.parse(strict.body)["reason"] == "context_length_exceeded")
        let clipped = await query("/embed_text?format=json&truncate=true", body: text)
        #expect(clipped.status == 200)
        #expect(try JSONValue.parse(clipped.body)["data"]?.arrayValue?[0]["truncated"] == true)
    }

    @Test func embedImageWithRawBytes() async throws {
        let response = await query("/embed_image?format=json", body: TestSupport.png(width: 320, height: 240), contentType: "image/png")
        #expect(response.status == 200)
        #expect(try JSONValue.parse(response.body)["usage"]?["prompt_tokens"] == 270)
    }

    @Test func similarityWithJSONArguments() async throws {
        let args: JSONValue = ["a": "same", "b": "same"]
        let response = await query("/similarity?format=json", body: args.jsonData(), contentType: "application/json")
        let cosine = try #require(try JSONValue.parse(response.body)["cosine"]?.doubleValue)
        #expect(abs(cosine - 1) < 1e-5)
    }

    @Test func queryErrors() async throws {
        #expect(await query("/embed_text", body: Data("x".utf8), contentType: nil).status == 400)
        #expect(await query("/nope", body: Data()).status == 404)
        #expect(await query("/similarity", body: Data("x".utf8)).status == 400)
        let bad = await query("/embed_text?dimensions=3&format=json", body: Data("x".utf8))
        #expect(bad.status == 400)
        #expect(try JSONValue.parse(bad.body)["reason"] == "invalid_dimensions")
    }

    @Test func mcpOverHTTP() async throws {
        func post(_ message: JSONValue) async -> HTTPResponse {
            await router.handle(HTTPRequest(method: "POST", path: "/mcp", body: message.jsonData()))
        }
        let initialize = try JSONValue.parse(await post(["jsonrpc": "2.0", "id": 1, "method": "initialize",
                                                         "params": ["protocolVersion": "2025-06-18"]]).body)
        #expect(initialize["result"]?["serverInfo"]?["name"] == "aneembed")
        #expect(await post(["jsonrpc": "2.0", "method": "notifications/initialized"]).status == 202)
        let list = try JSONValue.parse(await post(["jsonrpc": "2.0", "id": 2, "method": "tools/list"]).body)
        #expect(list["result"]?["tools"]?.arrayValue?.compactMap { $0["name"]?.stringValue } == Tools.names)
        // path is refused over HTTP
        let refused = try JSONValue.parse(await post(["jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                                      "params": ["name": "embed_image", "arguments": ["path": "/etc/hosts"]]]).body)
        #expect(refused["result"]?["isError"] == true)
        let text = try #require(refused["result"]?["content"]?.arrayValue?[0]["text"]?.stringValue)
        #expect(try JSONValue.parse(text)["reason"] == "path_not_allowed")
    }
}
