import Foundation
import Testing

@testable import EmbedCore

/// A runtime whose every call fails with an error that names an internal path.
struct FailingRuntime: EmbeddingRuntime {
    struct Failure: Error, CustomStringConvertible {
        var description: String { "Core ML failed at /Users/someone/Library/Application Support/ANEEmbed/models/x.mlmodelc" }
    }
    var info: RuntimeInfo { RuntimeInfo(maxTokens: 2048, maxImageTokens: 280, bucketLengths: [64], backend: "fake") }
    func embed(tokenized texts: [[Int32]]) async throws -> [[Float]] { throw Failure() }
    func embed(images: [PreparedImage]) async throws -> [[Float]] { throw Failure() }
    func warm() async throws {}
}

final class LogRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String] = []
    func record(_ line: String) { lock.withLock { lines.append(line) } }
    var all: [String] { lock.withLock { lines } }
}

/// 500 and 503 responses name a stable error code and a generic message; the internal detail (paths, Core ML
/// errors) goes to the server log only.
@Suite("Server errors over HTTP")
struct ServerErrorTests {
    static let secret = "/Users/someone"

    func body(_ response: HTTPResponse) -> String { String(decoding: response.body, as: UTF8.self) }

    @Test func internalErrorsAreGenericAndLogged() async throws {
        let log = LogRecorder()
        let service = EmbeddingService(tokenizer: ByteTokenizer(), runtime: FailingRuntime())
        let router = Router(state: ServerState(.ready(service)), modelName: "m", log: log.record)
        let response = await router.handle(HTTPRequest(method: "POST", path: "/v1/embeddings", body: Data(#"{"input":"x"}"#.utf8)))
        #expect(response.status == 500)
        let error = try JSONValue.parse(response.body)["error"]
        #expect(error?["code"] == "embedding_failed" && error?["type"] == "server_error")
        #expect(!body(response).contains(Self.secret) && !body(response).contains("Core ML"), "\(body(response))")
        #expect(log.all.contains { $0.contains(Self.secret) && $0.contains("embedding_failed") }, "\(log.all)")

        let query = await router.handle(HTTPRequest(method: "QUERY", path: "/embed_text", query: [("format", "json")],
                                                    headers: [("Content-Type", "text/plain")], body: Data("x".utf8)))
        #expect(query.status == 500 && !body(query).contains(Self.secret))
        #expect(try JSONValue.parse(query.body)["reason"] == "embedding_failed")

        let mcp = await router.handle(HTTPRequest(method: "POST", path: "/mcp", body: Data(
            #"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"embed_text","arguments":{"text":"x"}}}"#.utf8)))
        #expect(mcp.status == 200 && !body(mcp).contains(Self.secret) && body(mcp).contains("embedding_failed"))
    }

    @Test func aFailedLoadIsGeneric() async throws {
        let failure = ServiceError(.unavailable, "model_missing", "no model directory at \(Self.secret)/models/current",
                                   hint: "run `anerun models install`, or pass --model-dir")
        let router = Router(state: ServerState(.failed(failure)), modelName: "m")
        let health = await router.handle(HTTPRequest(method: "GET", path: "/health"))
        #expect(health.status == 503 && !body(health).contains(Self.secret))
        let status = try JSONValue.parse(health.body)
        #expect(status["status"] == "error" && status["reason"] == "model_missing")
        let embeddings = await router.handle(HTTPRequest(method: "POST", path: "/v1/embeddings", body: Data(#"{"input":"x"}"#.utf8)))
        #expect(embeddings.status == 503 && !body(embeddings).contains(Self.secret))
        #expect(try JSONValue.parse(embeddings.body)["error"]?["code"] == "model_missing")
    }

    /// Client errors keep their specific messages: they describe the request, not the server.
    @Test func clientErrorsKeepTheirMessages() async throws {
        let router = Router(state: ServerState(.ready(TestSupport.service())), modelName: "m")
        let response = await router.handle(HTTPRequest(method: "POST", path: "/v1/embeddings", body: Data(#"{"input":"x","dimensions":3}"#.utf8)))
        #expect(response.status == 400 && body(response).contains("dimensions must be one of"))
    }
}
