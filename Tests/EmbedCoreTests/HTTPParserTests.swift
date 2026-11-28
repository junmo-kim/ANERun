import Foundation
import Testing

@testable import EmbedCore

@Suite("HTTP parsing")
struct HTTPParserTests {
    func head(_ text: String, maxBody: Int = 1000) throws(HTTPParseError) -> HTTPHead? {
        try HTTPParser.parseHead(Data(text.utf8), maxBody: maxBody)
    }

    @Test func parsesRequestLineHeadersAndQuery() throws {
        let parsed = try #require(try head("QUERY /embed_text?input_type=query&x=a%20b+c HTTP/1.1\r\nHost: h\r\nContent-Type: text/plain\r\nContent-Length: 5\r\n\r\nhello"))
        #expect(parsed.request.method == "QUERY")
        #expect(parsed.request.path == "/embed_text")
        #expect(parsed.request.queryValue("input_type") == "query")
        #expect(parsed.request.queryValue("x") == "a b c")
        #expect(parsed.request.header("content-type") == "text/plain")
        #expect(parsed.contentLength == 5)
        #expect(parsed.headLength == 111)
        #expect(parsed.request.keepAlive)
    }

    @Test func waitsForTheWholeHead() throws {
        #expect(try head("GET /health HTTP/1.1\r\nHost: x\r\n") == nil)
    }

    @Test func refusesChunkedAndOversizedBodies() {
        #expect(throws: HTTPParseError.lengthRequired) {
            try head("POST /v1/embeddings HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n")
        }
        #expect(throws: HTTPParseError.bodyTooLarge(1000)) {
            try head("POST /v1/embeddings HTTP/1.1\r\nContent-Length: 1001\r\n\r\n")
        }
        #expect(throws: HTTPParseError.headerTooLarge) {
            try HTTPParser.parseHead(Data(repeating: UInt8(ascii: "a"), count: HTTPParser.maxHeadBytes + 1), maxBody: 10)
        }
    }

    @Test(arguments: ["GARBAGE\r\n\r\n", "GET / HTTP/2\r\n\r\n", "GET http://x/ HTTP/1.1\r\n\r\n", "GET / HTTP/1.1\r\nNoColon\r\n\r\n",
                      "POST / HTTP/1.1\r\nContent-Length: -1\r\n\r\n"])
    func rejectsMalformedHeads(_ text: String) {
        #expect(throws: HTTPParseError.self) { try head(text) }
    }

    /// Framing must not depend on which of two Content-Length (or Transfer-Encoding) headers a proxy in front
    /// of the server believes; anything ambiguous is a 400 (and the connection closes).
    @Test(arguments: [
        "Content-Length: 0\r\nContent-Length: 44", "Content-Length: 44\r\nContent-Length: 0", "Content-Length: 5, 6",
        "Content-Length: +5", "Content-Length: 0x5", "Content-Length: ", "Content-Length: 99999999999999999999",
        "Transfer-Encoding: chunked\r\nContent-Length: 5", "Content-Length: 5\r\nTransfer-Encoding: identity",
        "Transfer-Encoding: chunked\r\nTransfer-Encoding: identity", "Transfer-Encoding: identity",
        "Transfer-Encoding: gzip, chunked", "Transfer-Encoding: chunked, identity", "Transfer-Encoding: x",
    ])
    func rejectsAmbiguousFraming(_ headers: String) {
        do {
            let parsed = try head("POST /v1/embeddings HTTP/1.1\r\n\(headers)\r\n\r\n")
            Issue.record("accepted with Content-Length \(parsed?.contentLength ?? -1)")
        } catch {
            #expect(error.status == 400, "\(error)")
        }
    }

    @Test func acceptsRepeatedIdenticalContentLengths() throws {
        #expect(try head("POST / HTTP/1.1\r\nContent-Length: 5\r\ncontent-length: 5\r\n\r\n")?.contentLength == 5)
        #expect(try head("POST / HTTP/1.1\r\nContent-Length: 5, 5\r\n\r\n")?.contentLength == 5)
    }

    @Test func connectionHandling() throws {
        #expect(try head("GET / HTTP/1.1\r\nConnection: close\r\n\r\n")?.request.keepAlive == false)
        #expect(try head("GET / HTTP/1.0\r\n\r\n")?.request.keepAlive == false)
        #expect(try head("GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n")?.request.keepAlive == true)
    }

    @Test func serializesResponses() {
        let response = HTTPResponse.json(200, ["status": "ok"])
        let text = String(decoding: response.serialized(keepAlive: false), as: UTF8.self)
        #expect(text == "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\nConnection: close\r\n\r\n{\"status\":\"ok\"}")
        let head = String(decoding: response.serialized(keepAlive: true, includeBody: false), as: UTF8.self)
        #expect(head == "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\nConnection: keep-alive\r\n\r\n")
    }
}
