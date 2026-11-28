import Foundation
import Testing

@testable import EmbedCore

@Suite("JSON")
struct JSONTests {
    /// JSON is UTF-8 (RFC 8259); a string with invalid bytes is an error, not silently repaired to U+FFFD.
    @Test(arguments: [[0x61, 0xFF, 0x62], [0xC3], [0xC0, 0xAF], [0xED, 0xA0, 0x80], [0xF4, 0x90, 0x80, 0x80]] as [[UInt8]])
    func rejectsInvalidUTF8InStrings(_ bad: [UInt8]) {
        let value = Data(#"{"input":""#.utf8) + Data(bad) + Data(#""}"#.utf8)
        #expect(throws: JSONParseError.self) { try JSONValue.parse(value) }
        let key = Data(#"{""#.utf8) + Data(bad) + Data(#"":1}"#.utf8)
        #expect(throws: JSONParseError.self) { try JSONValue.parse(key) }
    }

    @Test func validUTF8AndEscapedLoneSurrogatesStillParse() throws {
        #expect(try JSONValue.parse(Data(#"{"a":"é漢🙂"}"#.utf8))["a"] == "é漢🙂")
        #expect(try JSONValue.parse(Data(#"{"a":"\ud800x"}"#.utf8))["a"] == "\u{FFFD}x")
    }

    @Test func parsesScalarsAndContainers() throws {
        let value = try JSONValue.parse(#"{"a": [1, -2.5, true, false, null, "x"], "b": {"c": 1e3}}"#)
        #expect(value["a"] == [1, -2.5, true, false, nil, "x"])
        #expect(value["b"]?["c"]?.intValue == 1000)
        #expect(value["missing"] == nil)
    }

    @Test func keepsLeadingByteOrderMark() throws {
        // JSONSerialization drops this scalar; the tokenizer must see it.
        let value = try JSONValue.parse("[\"\u{FEFF}#\", \"\\ufeffx\"]")
        #expect(value.arrayValue?[0].stringValue?.unicodeScalars.first == "\u{FEFF}")
        #expect(value.arrayValue?[1].stringValue?.unicodeScalars.map(\.value) == [0xFEFF, 0x78])
    }

    @Test func keepsCanonicallyEquivalentStringsApart() throws {
        // U+F900 (CJK compatibility ideograph) and U+8C48 are canonically equivalent.
        let value = try JSONValue.parse("[\"\u{F900}\", \"\\u8c48\"]")
        let scalars = value.arrayValue!.map { $0.stringValue!.unicodeScalars.first!.value }
        #expect(scalars == [0xF900, 0x8C48])
    }

    @Test func decodesEscapesAndSurrogates() throws {
        let value = try JSONValue.parse(#"["\n\t\"\\\/", "\ud83d\ude00", "\ud800x", "\udc00"]"#)
        let strings = value.arrayValue!.map { $0.stringValue! }
        #expect(strings[0] == "\n\t\"\\/")
        #expect(strings[1] == "😀")
        #expect(strings[2] == "\u{FFFD}x")
        #expect(strings[3] == "\u{FFFD}")
    }

    @Test(arguments: ["", "{", "[1,]", "{\"a\" 1}", "01", "1.", "tru", "\"\u{01}\"", "[1] 2", "\"\\x\""])
    func rejectsMalformedInput(_ text: String) {
        #expect(throws: JSONParseError.self) { try JSONValue.parse(text) }
    }

    @Test func limitsNesting() {
        let deep = String(repeating: "[", count: 1000) + String(repeating: "]", count: 1000)
        #expect(throws: JSONParseError.self) { try JSONValue.parse(deep) }
    }

    @Test func roundTripsThroughWriter() throws {
        let value: JSONValue = ["s": "a\"b\\c\n\u{01}é😀", "n": 3, "d": 0.5, "z": nil, "l": [true, false]]
        #expect(try JSONValue.parse(value.jsonString()) == value)
        #expect(JSONValue.object(["k": "v", "a": 1]).jsonString() == #"{"k":"v","a":1}"#)
    }

    @Test func writesFloatsInShortestForm() throws {
        let floats: [Float] = [0.1, -1.5, 1e-7, 0]
        let text = JSONValue.floats(floats).jsonString()
        #expect(text == "[0.1,-1.5,1e-07,0.0]")
        let parsed = try JSONValue.parse(text).arrayValue!.map { Float($0.doubleValue!) }
        #expect(parsed == floats)
    }

    @Test func rendersYAML() {
        let value: JSONValue = [
            "model": "embeddinggemma-2", "count": 2, "flag": true, "text": "a: b", "empty": [],
            "data": [["index": 0, "embedding": .floats([0.5, 0.25])]],
        ]
        let yaml = value.render(as: .yaml)
        #expect(yaml == """
            model: embeddinggemma-2
            count: 2
            flag: true
            text: "a: b"
            empty: []
            data:
              - index: 0
                embedding: [0.5, 0.25]
            """)
    }

    @Test func quotesAmbiguousYAMLScalars() {
        #expect(JSONValue.object(["v": "true"]).render(as: .yaml) == #"v: "true""#)
        #expect(JSONValue.object(["v": "12"]).render(as: .yaml) == #"v: "12""#)
        #expect(JSONValue.object(["v": "\u{FEFF}x"]).render(as: .yaml) == "v: \"\u{FEFF}x\"")
    }
}
