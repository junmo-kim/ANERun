import Foundation
import Testing

@testable import EmbedCore

/// The real EmbeddingGemma 2 tokenizer, loaded once.
enum RealTokenizer {
    static let shared: GemmaTokenizer? = try? GemmaTokenizer(data: Fixtures.data("tokenizer.json.xz"))
}

@Suite("Tokenizer golden (Hugging Face tokenizers)")
struct TokenizerGoldenTests {
    @Test func loadsTheRealTokenizer() throws {
        let tokenizer = try #require(RealTokenizer.shared)
        #expect(tokenizer.vocabularySize == 262_144)
        #expect(tokenizer.mergeCount == 514_906)
        #expect(tokenizer.prefixIDs == [2])
        #expect(tokenizer.suffixIDs == [1])
        #expect(tokenizer.encode("hello world") == [2, 23391, 1902, 1])
    }

    @Test(arguments: [
        "The quick brown fox jumps over the lazy dog. An ordinary document follows. ",
        "오늘 회의에서는 다음 주 일정을 정하고 고객 문의에 답변할 담당자를 확인했습니다 ",
    ])
    func longDocumentsClipAtSafeBoundaries(sentence: String) throws {
        let tokenizer = try #require(RealTokenizer.shared)
        let document = String(repeating: sentence, count: 1_048_576 / sentence.utf8.count + 1)
        let full = tokenizer.encode(document)
        for cap in [2, 64, 2048] {
            var scratch = BPEScratch()
            let expected = Array(full.prefix(cap - tokenizer.suffixIDs.count)) + tokenizer.suffixIDs
            #expect(tokenizer.encodeForEmbedding(document, limit: cap, truncate: true, scratch: &scratch) == .clipped(expected))
            #expect(scratch.ids.count < 4200)
        }
        #expect(tokenizer.encodeForEmbedding(document, limit: 64, truncate: false) == .tooLong(atLeast: 65))
    }

    @Test func boundedEncodingKeepsUnicodeFallbacksAndAddedTokensAcrossCuts() throws {
        let tokenizer = try #require(RealTokenizer.shared)
        let text = String(repeating: "한글𐐷e\u{301}▁문서 ", count: 650) + "<bos>"
            + String(repeating: "질문과 답변을 확인합니다 ", count: 300)
        let full = tokenizer.encode(text)
        #expect(tokenizer.encodeForEmbedding(text, limit: full.count, truncate: false) == .ids(full))
        let cap = full.count - 1
        #expect(tokenizer.encodeForEmbedding(text, limit: cap, truncate: true)
                == .clipped(Array(full.prefix(cap - 1)) + tokenizer.suffixIDs))
    }

    @Test func matchesEveryGoldenText() throws {
        let tokenizer = try #require(RealTokenizer.shared)
        let lines = try Fixtures.data("tokenizer-golden.jsonl.xz").split(separator: UInt8(ascii: "\n"))
        var checked = 0, ids = 0
        var mismatches: [String] = []
        var scratch = BPEScratch()
        var output: [Int32] = []
        for line in lines where !line.isEmpty {
            let record = try JSONValue.parse(Data(line))
            let text = try #require(record["text"]?.stringValue)
            let expected = try #require(record["ids"]?.arrayValue).map { Int32($0.intValue!) }
            output.removeAll(keepingCapacity: true)
            tokenizer.encode(text, into: &output, scratch: &scratch)
            if output != expected, mismatches.count < 10 {
                mismatches.append("\(text.debugDescription.prefix(80)): got \(output.prefix(12)) want \(expected.prefix(12))")
            }
            checked += 1
            ids += expected.count
        }
        #expect(mismatches.isEmpty, "\(mismatches.joined(separator: "\n"))")
        #expect(checked > 10_000)
        print("tokenizer golden: \(checked) texts, \(ids) ids, \(mismatches.count == 0 ? "all equal" : "MISMATCH")")
    }

    /// Decoding the ids of every golden text (without <bos>/<eos>) gives the text back, as Hugging Face's decoder does,
    /// except that a "▁" in the text comes back as " " there too. Compared as bytes: String equality would accept a
    /// canonically equivalent text, and String replacement skips a "▁" that a combining mark follows.
    @Test func decodesEveryGoldenTextBack() throws {
        let tokenizer = try #require(RealTokenizer.shared)
        let lines = try Fixtures.data("tokenizer-golden.jsonl.xz").split(separator: UInt8(ascii: "\n"))
        var checked = 0
        var mismatches: [String] = []
        for line in lines where !line.isEmpty {
            let text = try #require(try JSONValue.parse(Data(line))["text"]?.stringValue)
            let ids = tokenizer.encode(text, addSpecialTokens: false)
            let decoded = Array(try TokenizerResponse.content(ids, tokenizer: tokenizer).utf8)
            var expected: [UInt8] = []
            let utf8 = Array(text.utf8)
            var index = 0
            while index < utf8.count {
                if utf8[index...].starts(with: [0xE2, 0x96, 0x81]) {
                    expected.append(0x20)
                    index += 3
                } else {
                    expected.append(utf8[index])
                    index += 1
                }
            }
            if decoded != expected, mismatches.count < 10 { mismatches.append(text.debugDescription.prefix(80).description) }
            checked += 1
        }
        #expect(mismatches.isEmpty, "\(mismatches.joined(separator: "\n"))")
        #expect(checked > 10_000)
    }

    /// The pieces llama-server reports for EmbeddingGemma 2: text where a token is valid UTF-8 alone, else its bytes.
    @Test func reportsPiecesLikeLlamaServer() throws {
        let tokenizer = try #require(RealTokenizer.shared)
        let ids = tokenizer.encode("hello world", addSpecialTokens: true)
        let body = try JSONValue.parse(TokenizerResponse.tokens(ids, pieces: tokenizer))
        #expect(body == ["tokens": [["id": 2, "piece": "<bos>"], ["id": 23391, "piece": "hello"],
                                    ["id": 1902, "piece": " world"], ["id": 1, "piece": "<eos>"]]])
        #expect(try JSONValue.parse(TokenizerResponse.tokens(ids, pieces: nil)) == ["tokens": [2, 23391, 1902, 1]])
        // A byte-fallback token is part of a character: its piece is a list of bytes.
        let byteToken = try #require(tokenizer.id(of: "<0xC3>"))
        let pieces = try JSONValue.parse(TokenizerResponse.tokens([byteToken], pieces: tokenizer))
        #expect(pieces == ["tokens": [["id": .int(Int(byteToken)), "piece": [195]]]])
        #expect(try TokenizerResponse.content([byteToken], tokenizer: tokenizer) == "\u{FFFD}")
    }

    /// parse_special false, pinned to llama-server b12428's ids: control tokens such as <bos> become plain text, the six
    /// user-defined added tokens are still cut out.
    @Test func parseSpecialFalseMatchesLlamaServer() throws {
        let tokenizer = try #require(RealTokenizer.shared)
        func plain(_ text: String) -> [Int32] { tokenizer.encode(text, addSpecialTokens: false, parseSpecial: false) }
        #expect(tokenizer.encode("<bos>x", addSpecialTokens: false) == [2, 236781])
        #expect(plain("<bos>x") == [236820, 46757, 236813, 236781])
        #expect(plain("<|channel>x<channel|>") == [100, 236781, 101])
        #expect(plain("<|tool_call>call") == [48, 6639])
        #expect(plain(#"<|"|>q"#) == [52, 236809])
        #expect(plain("x<|tool_response>y") == [236781, 236820, 236909, 13205, 236779, 6275, 236813, 236762])
        #expect(tokenizer.encode("<|tool_response>", addSpecialTokens: false) == [50])
        for token in GemmaTokenizer.userDefinedTokens {
            #expect(plain(String(decoding: token, as: UTF8.self)).count == 1)
        }
    }

    /// The routes with the real tokenizer: the request options reach it.
    @Test func tokenizerRoutesWithTheRealTokenizer() async throws {
        let tokenizer = try #require(RealTokenizer.shared)
        let router = Router(state: ServerState(.ready(EmbeddingService(tokenizer: tokenizer, runtime: FakeRuntime.make()))),
                            modelName: "m")
        func post(_ path: String, _ body: JSONValue) async throws -> (Int, JSONValue) {
            let response = await router.handle(HTTPRequest(method: "POST", path: path, body: body.jsonData()))
            return (response.status, try JSONValue.parse(response.body))
        }
        #expect(try await post("/tokenize", ["content": "<bos>x"]) == (200, ["tokens": [2, 236781]]))
        #expect(try await post("/tokenize", ["content": "<bos>x", "parse_special": false])
            == (200, ["tokens": [236820, 46757, 236813, 236781]]))
        #expect(try await post("/tokenize", ["content": "hello world", "add_special": true])
            == (200, ["tokens": [2, 23391, 1902, 1]]))
        // U+0FF0 is not in the vocabulary: three byte tokens, each piece a list of one byte
        #expect(try await post("/tokenize", ["content": "\u{0FF0}", "with_pieces": true])
            == (200, ["tokens": [["id": 462, "piece": [224]], ["id": 429, "piece": [191]], ["id": 414, "piece": [176]]]]))
        #expect(try await post("/detokenize", ["tokens": [2, 23391, 1902, 1]]) == (200, ["content": "<bos>hello world<eos>"]))
        let (status, body) = try await post("/detokenize", ["tokens": [262_144]])
        #expect(status == 400 && body["error"]?["param"] == "tokens[0]")
    }

    /// No token of the real tokenizer stands for more bytes than its longest entry, so every golden text has at least
    /// bytes / maxBytesPerToken tokens (the bound the service checks before tokenizing).
    @Test func noTokenStandsForMoreThanTheLongestEntry() throws {
        let tokenizer = try #require(RealTokenizer.shared)
        let bound = try #require(tokenizer.maxBytesPerToken)
        #expect(bound == 93)
        let lines = try Fixtures.data("tokenizer-golden.jsonl.xz").split(separator: UInt8(ascii: "\n"))
        var violations = 0
        for line in lines where !line.isEmpty {
            let record = try JSONValue.parse(Data(line))
            let text = try #require(record["text"]?.stringValue)
            let ids = try #require(record["ids"]?.arrayValue).count - 2  // without <bos>/<eos>
            if ids * bound < text.utf8.count { violations += 1 }
        }
        #expect(violations == 0)
    }

    /// Optional large local corpus (not stored in the repository), e.g. from
    /// `make_tokenizer_golden.py --lines`: ANEEMBED_EXTRA_GOLDEN=/path/to/golden.jsonl[.xz]
    @Test(.enabled(if: ProcessInfo.processInfo.environment["ANEEMBED_EXTRA_GOLDEN"] != nil))
    func matchesExtraGoldenCorpus() throws {
        let tokenizer = try #require(RealTokenizer.shared)
        let path = ProcessInfo.processInfo.environment["ANEEMBED_EXTRA_GOLDEN"]!
        var data = try Data(contentsOf: URL(fileURLWithPath: path))
        if path.hasSuffix(".xz") { data = try (data as NSData).decompressed(using: .lzma) as Data }
        var checked = 0, failed = 0, ids = 0
        for line in data.split(separator: UInt8(ascii: "\n")) where !line.isEmpty {
            let record = try JSONValue.parse(Data(line))
            let expected = record["ids"]!.arrayValue!.map { Int32($0.intValue!) }
            if tokenizer.encode(record["text"]!.stringValue!) != expected {
                failed += 1
                if failed <= 5 { Issue.record("mismatch: \(record["text"]!.stringValue!.debugDescription.prefix(100))") }
            }
            checked += 1
            ids += expected.count
        }
        print("extra tokenizer golden: \(checked) texts, \(ids) ids, \(failed) mismatches")
        #expect(failed == 0)
    }
}
