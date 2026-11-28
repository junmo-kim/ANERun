import Foundation
import Testing

@testable import EmbedCore

/// A tiny Gemma-style tokenizer.json exercising every rule the real one relies on.
private func tinyTokenizer(merges: String = #"[["a", "a"], ["▁", "a"], ["aa", "a"], ["b", "c"], ["a", "b"]]"#,
                           edit: (String) -> String = { $0 }) throws -> GemmaTokenizer
{
    let json = """
        {"version": "1.0", "truncation": null, "padding": null,
         "added_tokens": [
           {"id": 0, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
           {"id": 1, "content": "<eos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
           {"id": 2, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
           {"id": 20, "content": "<|image>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
           {"id": 21, "content": "<|image|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
         "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
         "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": false},
         "post_processor": {"type": "TemplateProcessing",
           "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "<eos>", "type_id": 0}}],
           "special_tokens": {"<bos>": {"id": "<bos>", "ids": [2], "tokens": ["<bos>"]}, "<eos>": {"id": "<eos>", "ids": [1], "tokens": ["<eos>"]}}},
         "decoder": null,
         "model": {"type": "BPE", "dropout": null, "unk_token": "<unk>", "continuing_subword_prefix": null,
           "end_of_word_suffix": null, "fuse_unk": true, "byte_fallback": true, "ignore_merges": false,
           "vocab": {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "a": 4, "\\ufeffa": 5, "b": 6, "c": 7,
             "aa": 8, "▁": 9, "▁a": 10, "\u{F900}": 11, "\u{8C48}": 12, "aaa": 13, "bc": 14, "ab": 15,
             "\\ufeff": 16, "<|image>": 20, "<|image|>": 21,
             "<0xC3>": 30, "<0xA9>": 31, "<0xF0>": 32, "<0x9F>": 33, "<0x98>": 34, "<0x80>": 35,
             "<0xE2>": 36, "<0x96>": 37, "<0x81>": 38},
           "merges": \(merges)}}
        """
    // The fixture needs every byte token for full fallback; add the rest programmatically.
    var text = json
    var extra: [String] = []
    let present: Set<Int> = [0xC3, 0xA9, 0xF0, 0x9F, 0x98, 0x80, 0xE2, 0x96, 0x81]
    for byte in 0..<256 where !present.contains(byte) {
        extra.append(String(format: "\"<0x%02X>\": %d", byte, 100 + byte))
    }
    text = text.replacingOccurrences(of: "\"<0x81>\": 38}", with: "\"<0x81>\": 38, " + extra.joined(separator: ", ") + "}")
    return try GemmaTokenizer(data: Data(edit(text).utf8))
}

@Suite("Tokenizer rules")
struct TokenizerRuleTests {
    @Test func addsBosAndEos() throws {
        let tokenizer = try tinyTokenizer()
        #expect(tokenizer.prefixIDs == [2])
        #expect(tokenizer.suffixIDs == [1])
        #expect(tokenizer.encode("") == [2, 1])
        #expect(tokenizer.encode("a", addSpecialTokens: false) == [4])
    }

    @Test func appliesMergesByRankNotPosition() throws {
        let tokenizer = try tinyTokenizer()
        // "aaa": rank 0 (a,a) at position 0 first, then (aa,a) rank 2 -> "aaa"
        #expect(tokenizer.encode("aaa", addSpecialTokens: false) == [13])
        // "abc": (b,c) rank 3 beats (a,b) rank 4 -> a + bc
        #expect(tokenizer.encode("abc", addSpecialTokens: false) == [4, 14])
        // "aaaa": (a,a)@0 -> aa a a; then (aa,a) rank 2 vs (a,a)@2 rank 0 -> aa aa
        #expect(tokenizer.encode("aaaa", addSpecialTokens: false) == [8, 8])
    }

    @Test func boundedEncodingKeepsMergeableRunsIntact() throws {
        let tokenizer = try tinyTokenizer()
        let text = String(repeating: "a", count: 8193) + "<|image|> abc"
        let full = tokenizer.encode(text)
        for cap in [2, 64, 2048, full.count - 1] {
            #expect(tokenizer.encodeForEmbedding(text, limit: cap, truncate: true)
                    == .clipped(Array(full.prefix(cap - 1)) + [1]))
        }
        #expect(tokenizer.encodeForEmbedding(text, limit: full.count) == .ids(full))
    }

    @Test func replacesSpacesWithLowLine() throws {
        let tokenizer = try tinyTokenizer()
        #expect(tokenizer.encode(" a", addSpecialTokens: false) == [10])
        #expect(tokenizer.encode("a  a", addSpecialTokens: false) == [4, 9, 10])
    }

    @Test func keepsByteOrderMarkToken() throws {
        let tokenizer = try tinyTokenizer()
        // "\u{FEFF}a" is its own vocab entry but nothing merges into it: two symbols.
        #expect(tokenizer.encode("\u{FEFF}a", addSpecialTokens: false) == [16, 4])
        #expect(tokenizer.tokens([5]) == ["\u{FEFF}a"])
    }

    @Test func keepsCanonicallyEquivalentTokensApart() throws {
        let tokenizer = try tinyTokenizer()
        #expect(tokenizer.encode("\u{F900}\u{8C48}", addSpecialTokens: false) == [11, 12])
    }

    @Test func fallsBackToBytes() throws {
        let tokenizer = try tinyTokenizer()
        #expect(tokenizer.encode("é", addSpecialTokens: false) == [30, 31])
        #expect(tokenizer.encode("😀", addSpecialTokens: false) == [32, 33, 34, 35])
    }

    @Test func splitsAddedTokensLeftmostLongest() throws {
        let tokenizer = try tinyTokenizer()
        #expect(tokenizer.encode("<bos>a<eos>", addSpecialTokens: false) == [2, 4, 1])
        #expect(tokenizer.encode("<|image|>", addSpecialTokens: false) == [21])
        #expect(tokenizer.encode("<|image>", addSpecialTokens: false) == [20])
        // A partial look-alike is plain text (here: bytes, since '<' etc. are not in the vocab).
        #expect(tokenizer.encode("<bo", addSpecialTokens: false).count == 3)
    }

    /// llama-server's parse_special false: the text of a special token is encoded like any other text.
    @Test func encodesSpecialTokenTextAsTextWithoutParseSpecial() throws {
        let tokenizer = try tinyTokenizer()
        let plain = tokenizer.encode("<bos>a", addSpecialTokens: false, parseSpecial: false)
        #expect(!plain.contains(2))
        #expect(plain.last == 4 && plain.count == 6)  // "<", "b", "o", "s", ">" one token each, then "a"
        #expect(tokenizer.encode("<bos>a", addSpecialTokens: true, parseSpecial: false) == [2] + plain + [1])
    }

    /// tokenizer.json's decoder: "▁" -> " ", byte tokens -> their byte, other tokens (special too) -> their entry.
    @Test func decodesEachTokenToItsBytes() throws {
        let tokenizer = try tinyTokenizer()
        #expect(tokenizer.pieceBytes(10) == Array(" a".utf8))
        #expect(tokenizer.pieceBytes(9) == [0x20])
        #expect(tokenizer.pieceBytes(30) == [0xC3] && tokenizer.pieceBytes(31) == [0xA9])
        #expect(tokenizer.pieceBytes(36) == [0xE2])  // the byte token <0xE2>, not part of a "▁"
        #expect(tokenizer.pieceBytes(2) == Array("<bos>".utf8))
        #expect(tokenizer.pieceBytes(5) == Array("\u{FEFF}a".utf8))
        #expect(tokenizer.pieceBytes(-1) == nil && tokenizer.pieceBytes(1 << 20) == nil)
        for text in ["a  a", "é😀 aaa", "\u{F900}\u{8C48} bc", "<bos>a<eos>"] {
            let bytes = tokenizer.encode(text, addSpecialTokens: false).flatMap { tokenizer.pieceBytes($0)! }
            #expect(String(decoding: bytes, as: UTF8.self) == text)
        }
    }

    @Test func staleQueueEntriesUseProducedIdLikeTokenizers() throws {
        // Two merges producing the same token "aaa": (aa, a) rank 2 and (a, aa) rank 5.
        let tokenizer = try tinyTokenizer(merges: #"[["a", "a"], ["▁", "a"], ["aa", "a"], ["b", "c"], ["a", "b"], ["a", "aa"]]"#)
        #expect(tokenizer.encode("aaa", addSpecialTokens: false) == [13])
    }

    /// Ids are table indices: an added token or a template id below 0 is a malformed file.
    @Test func rejectsNegativeIds() {
        #expect(throws: Never.self) { try tinyTokenizer() }
        #expect(throws: TokenizerError.self) {
            try tinyTokenizer { $0.replacingOccurrences(of: #"{"id": 20, "content": "<|image>""#, with: #"{"id": -20, "content": "<|image>""#) }
        }
        #expect(throws: TokenizerError.self) {
            try tinyTokenizer { $0.replacingOccurrences(of: #""ids": [2]"#, with: #""ids": [-2]"#) }
        }
    }

    /// The vocabulary table grows past its initial capacity instead of trapping (a vocab over 524,288 entries).
    @Test func byteStringTableGrows() {
        var table = ByteStringTable(minimumCapacity: 1)
        for value in 0..<5000 { #expect(Array("token\(value)".utf8).withUnsafeBytes { table.insert($0, value: Int32(value)) }) }
        #expect(table.count == 5000)
        #expect((0..<5000).allSatisfy { table.lookup(Array("token\($0)".utf8)) == Int32($0) })
        #expect(!Array("token7".utf8).withUnsafeBytes { table.insert($0, value: 1) })
        #expect(table.lookup(Array("absent".utf8)) == nil)
    }

    /// The bound is the longest entry, added tokens included; without a byte token for every byte a run of unknown
    /// text can become one token, so there is none.
    @Test func boundsTheBytesOneTokenStandsFor() throws {
        let tokenizer = try tinyTokenizer()
        #expect(tokenizer.maxBytesPerToken == 9)  // "<|image|>"
        for text in ["", "aaaa bc", "é😀 ab", "<|image|><|image|>", "\u{F900}\u{8C48}"] {
            #expect(tokenizer.encode(text).count - 2 >= (text.utf8.count + 8) / 9)
        }
        let partial = try tinyTokenizer { $0.replacingOccurrences(of: #""<0x41>": 165, "#, with: "") }
        #expect(partial.maxBytesPerToken == nil)
    }

    @Test func rejectsUnsupportedConfiguration() {
        let json = #"{"model": {"type": "Unigram", "vocab": []}}"#
        #expect(throws: TokenizerError.self) { try GemmaTokenizer(data: Data(json.utf8)) }
    }
}
