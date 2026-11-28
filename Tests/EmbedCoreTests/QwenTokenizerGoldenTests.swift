import Foundation
import Testing

@testable import EmbedCore

@Suite("Qwen tokenizer golden (Hugging Face tokenizers)")
struct QwenTokenizerGoldenTests {
    private struct Golden: Decodable {
        var name: String
        var text: String
        var ids: [Int32]
        var pieces: [String]
    }

    private static let model = Result { try Fixtures.data("Qwen/tokenizer.json.xz") }
    private static let loaded = Result { try QwenTokenizer(data: model.get()) }

    @Test func matchesAllGoldenIDsAndSplitBoundaries() throws {
        let tokenizer = try Self.loaded.get()
        let lines = try Fixtures.data("Qwen/golden.jsonl.xz").split(separator: UInt8(ascii: "\n"))
        let decoder = JSONDecoder()
        var idCount = 0
        var idMismatches: [String] = []
        var splitMismatches: [String] = []
        for line in lines {
            let record = try decoder.decode(Golden.self, from: Data(line))
            if tokenizer.encode(record.text) != record.ids
                || tokenizer.encode(record.text, addSpecial: false, parseSpecial: true) != record.ids {
                idMismatches.append(record.name)
            }
            let cap = max(1, record.ids.count / 2)
            let expected: BoundedTokenization = record.ids.count > cap
                ? .clipped(Array(record.ids.prefix(cap))) : .ids(record.ids)
            if tokenizer.encodeForEmbedding(record.text, limit: cap, truncate: true) != expected {
                idMismatches.append(record.name + " (bounded)")
            }
            var pieces: [[UInt8]] = []
            tokenizer.forEachSplit(in: record.text) { pieces.append(Array($0.utf8)) }
            if pieces != record.pieces.map({ Array($0.utf8) }) {
                splitMismatches.append(record.name)
            }
            idCount += record.ids.count
        }
        #expect(lines.count == 6_220)
        #expect(idCount == 551_244)
        #expect(idMismatches.isEmpty, "\(idMismatches.count) ID mismatches: \(idMismatches.prefix(10))")
        #expect(splitMismatches.isEmpty, "\(splitMismatches.count) split mismatches: \(splitMismatches.prefix(10))")
    }

    @Test func embeddingTemplateAppendsEOSAndIncludesItInTheTokenCap() throws {
        var json = try #require(JSONValue.parse(Self.model.get()).objectValue)
        let byteLevel = try #require(json["post_processor"])
        json["post_processor"] = [
            "type": "Sequence", "processors": [byteLevel, [
                "type": "TemplateProcessing",
                "single": [
                    ["Sequence": ["type_id": 0, "id": "A"]],
                    ["SpecialToken": ["type_id": 0, "id": "<|endoftext|>"]],
                ],
                "special_tokens": ["<|endoftext|>": ["id": "<|endoftext|>", "ids": [151643], "tokens": ["<|endoftext|>"]]],
            ]],
        ]
        let tokenizer = try QwenTokenizer(data: JSONValue.object(json).jsonData())
        #expect(tokenizer.encode("hello world") == [14990, 1879, 151643])
        #expect(tokenizer.encode("hello world", addSpecial: false, parseSpecial: true) == [14990, 1879])
        #expect(tokenizer.encode("<|endoftext|>") == [151643, 151643])
        let query = InputType.query.apply("한글 e\u{0301} <|endoftext|> fox", modelID: .qwen3Embedding)
        let full = tokenizer.encode(query)
        for cap in 1..<full.count {
            #expect(tokenizer.encodeForEmbedding(query, limit: cap, truncate: true)
                    == .clipped(Array(full.prefix(cap - 1)) + [151643]))
        }
        #expect(tokenizer.encodeForEmbedding("", limit: 1) == .ids([151643]))
        #expect(tokenizer.encodeForEmbedding("", limit: 0) == .tooLong(atLeast: 1))
        #expect(tokenizer.encodeForEmbedding("<|endoftext|>", limit: 1) == .tooLong(atLeast: 2))
    }

    @Test func usesPinnedUnicodeClassesAndNFC() throws {
        let tokenizer = try Self.loaded.get()
        var pieces: [[UInt8]] = []
        tokenizer.forEachSplit(in: "\u{32B40}'s") { pieces.append(Array($0.utf8)) }
        #expect(pieces == [Array("\u{32B40}'".utf8), [115]])
        #expect(tokenizer.encode("\u{32B40}'s") == [172, 110, 255, 222, 6, 82])

        // Unicode 9 predates Todhri. The private-use scalar must also stay intact.
        #expect(tokenizer.encode("\u{105D2}\u{0307}") == [123934, 245, 240, 136, 229])
        #expect(tokenizer.encode("\u{F00D8}\u{0301}") == [175, 108, 225, 246, 53839])
        #expect(tokenizer.encode("e\u{0301}") == [963])
        #expect(tokenizer.encode("한글") == [23573, 83291])
    }

    @Test func specialFlagsOnlyAffectRawSpecialMatches() throws {
        let tokenizer = try Self.loaded.get()
        let text = "x<|endoftext|><tool_call><|im_end|>한"
        let withSpecial: [Int32] = [87, 151643, 151657, 151645, 23573]
        // HF encode_special_tokens=True excludes special added matches and retains user-defined ones.
        let withoutSpecial: [Int32] = [87, 27, 91, 8691, 723, 427, 91, 29, 151657, 27, 91, 318, 6213, 91, 29, 23573]
        for addSpecial in [false, true] {
            #expect(tokenizer.encode(text, addSpecial: addSpecial, parseSpecial: true) == withSpecial)
            #expect(tokenizer.encode(text, addSpecial: addSpecial, parseSpecial: false) == withoutSpecial)
            #expect(tokenizer.encode("", addSpecial: addSpecial, parseSpecial: true).isEmpty)
        }
        #expect(tokenizer.encode("â½Ĺ") == [151642])
        #expect(tokenizer.encode("a\u{0302}½L\u{0301}") == [8835, 26062, 143735])
        #expect(tokenizer.encode("<tool_call>\u{0301}", addSpecial: false, parseSpecial: false) == [151657, 53839])
    }

    @Test func longestRawAddedMatchWinsBeforeSpecialFiltering() throws {
        let base = String(decoding: try Self.model.get(), as: UTF8.self)
        let key = #""added_tokens": ["#
        let added = """
        \(key)
          {"content":"[Q]","id":151669,"special":false,"normalized":false,"single_word":false,"lstrip":false,"rstrip":false},
          {"content":"[Q]end","id":151670,"special":true,"normalized":false,"single_word":false,"lstrip":false,"rstrip":false},
        """
        let text = base.replacingOccurrences(of: key, with: added)
        try #require(text != base)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try Data(text.utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let tokenizer = try QwenTokenizer(contentsOf: url)
        #expect(tokenizer.encode("a[Q]end[Q]z") == [64, 151670, 151669, 89])
        #expect(tokenizer.encode("a[Q]end[Q]z", addSpecial: false, parseSpecial: false)
            == [64, 58, 48, 60, 408, 151669, 89])
    }

    @Test func decodesByteLevelPiecesAndRawAddedTokens() throws {
        let tokenizer = try Self.loaded.get()
        #expect(tokenizer.encode("hello world") == [14990, 1879])
        #expect(tokenizer.pieceBytes(1879) == Array(" world".utf8))
        #expect(tokenizer.pieceBytes(151643) == Array("<|endoftext|>".utf8))
        #expect(tokenizer.pieceBytes(151657) == Array("<tool_call>".utf8))
        // The first special also has a vocabulary entry in the GPT-2 alphabet.
        #expect(tokenizer.pieceBytes(151642) == Array("⽗".utf8))

        let ids = tokenizer.encode("\u{0FF0}")
        #expect(ids == [126184, 108])
        #expect(try JSONValue.parse(TokenizerResponse.tokens(ids, pieces: tokenizer))
            == ["tokens": [["id": 126184, "piece": [224, 191]], ["id": 108, "piece": [176]]]])
        #expect(try TokenizerResponse.content(ids, tokenizer: tokenizer) == "\u{0FF0}")
        #expect(tokenizer.pieceBytes(-1) == nil)
        #expect(tokenizer.pieceBytes(151669) == nil)
        #expect(throws: ServiceError.self) {
            try TokenizerResponse.content([151669], tokenizer: tokenizer)
        }
    }

    @Test func boundsNormalizedBytesAndCountsAddedMatches() throws {
        let tokenizer: any TextTokenizer = try Self.loaded.get()
        let decomposed = String(repeating: "e\u{0301}", count: 43)
        let ids = tokenizer.encode(decomposed)
        #expect(ids.count > 1)
        #expect(tokenizer.encodeForEmbedding(decomposed, limit: 1) == .tooLong(atLeast: 2))
        #expect(tokenizer.encodeForEmbedding(decomposed, limit: ids.count) == .ids(ids))
        #expect(tokenizer.encodeForEmbedding("", limit: 0) == .ids([]))

        let text = "<|endoftext|>" + String(repeating: "x", count: 129) + "<tool_call>"
        let exact = tokenizer.encode(text)
        #expect(exact.count > 4)
        #expect(tokenizer.encodeForEmbedding(text, limit: 3) == .tooLong(atLeast: 4))
        #expect(tokenizer.encodeForEmbedding(text, limit: 4) == .tooLong(atLeast: 5))
        #expect(tokenizer.encodeForEmbedding(text, limit: exact.count) == .ids(exact))
        #expect(tokenizer.encodeForEmbedding(text, limit: Int.max) == .ids(exact))
        #expect(tokenizer.encodeForEmbedding("<|endoftext|><tool_call>", limit: 1) == .tooLong(atLeast: 2))
    }

    @Test func stopsAtTheFirstProvenBoundForRepeatedAddedTokens() throws {
        let tokenizer = try Self.loaded.get()
        let text = "e\u{0301}" + String(repeating: "<|endoftext|>", count: 100_000)
        #expect(tokenizer.encodeForEmbedding(text, limit: 16) == .tooLong(atLeast: 17))
    }

    @Test func rejectsGemmaAndIncompatibleQwenSchemas() throws {
        let gemma = try Fixtures.data("tokenizer.json.xz")
        #expect(throws: TokenizerError.self) { try QwenTokenizer(data: gemma) }

        let text = String(decoding: try Self.model.get(), as: UTF8.self)
        for (old, new) in [
            (#""type": "NFC""#, #""type": "NFD""#),
            (#""behavior": "Isolated""#, #""behavior": "Removed""#),
            (#""use_regex": false"#, #""use_regex": true"#),
            (#""byte_fallback": false"#, #""byte_fallback": true"#),
        ] {
            let incompatible = text.replacingOccurrences(of: old, with: new)
            try #require(incompatible != text)
            #expect(throws: TokenizerError.self) { try QwenTokenizer(data: Data(incompatible.utf8)) }
        }
    }
}
