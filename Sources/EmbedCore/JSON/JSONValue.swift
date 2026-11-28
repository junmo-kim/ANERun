// A small JSON value type with a parser and writer that work on raw bytes.
//
// Why not JSONSerialization: it drops a leading U+FEFF from strings ("\u{FEFF}#" becomes "#"), and
// a text that starts with a byte order mark is a different input to the tokenizer. Strings here are
// built from the UTF-8 bytes as written, which keeps every scalar (invalid UTF-8 is an error).
import Foundation

/// A JSON value. Objects keep their key order so that output is stable.
public enum JSONValue: Sendable, Equatable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object(JSONObject)
    /// Output only (never produced by the parser): a float32 vector written in its shortest
    /// round-trip form, which is both smaller and faster than going through Double.
    case floats([Float])

    public subscript(key: String) -> JSONValue? {
        if case .object(let object) = self { return object[key] }
        return nil
    }

    public var stringValue: String? {
        if case .string(let value) = self { return value }
        return nil
    }

    /// The number as an Int when it is integral and in range.
    public var intValue: Int? {
        if case .number(let value) = self, let int = Int(exactly: value) { return int }
        return nil
    }

    public var doubleValue: Double? {
        if case .number(let value) = self { return value }
        return nil
    }

    public var boolValue: Bool? {
        if case .bool(let value) = self { return value }
        return nil
    }

    public var arrayValue: [JSONValue]? {
        if case .array(let value) = self { return value }
        return nil
    }

    public var objectValue: JSONObject? {
        if case .object(let value) = self { return value }
        return nil
    }

    public var isNull: Bool {
        if case .null = self { return true }
        return false
    }

    /// The JSON type name, for error messages.
    public var typeName: String {
        switch self {
        case .null: return "null"
        case .bool: return "boolean"
        case .number: return "number"
        case .string: return "string"
        case .array, .floats: return "array"
        case .object: return "object"
        }
    }
}

/// An insertion-ordered JSON object. Lookup is linear: request and response objects are small.
public struct JSONObject: Sendable, Equatable, ExpressibleByDictionaryLiteral {
    public var pairs: [(key: String, value: JSONValue)]

    public init(_ pairs: [(key: String, value: JSONValue)] = []) {
        self.pairs = pairs
    }

    public init(dictionaryLiteral elements: (String, JSONValue)...) {
        pairs = elements.map { (key: $0.0, value: $0.1) }
    }

    /// The last value for `key` (JSON leaves duplicates undefined; last-wins matches most parsers).
    public subscript(key: String) -> JSONValue? {
        get { pairs.last(where: { $0.key == key })?.value }
        set {
            pairs.removeAll { $0.key == key }
            if let newValue { pairs.append((key: key, value: newValue)) }
        }
    }

    public var keys: [String] { pairs.map(\.key) }

    public static func == (lhs: JSONObject, rhs: JSONObject) -> Bool {
        lhs.pairs.count == rhs.pairs.count
            && zip(lhs.pairs, rhs.pairs).allSatisfy { $0.key == $1.key && $0.value == $1.value }
    }
}

extension JSONValue: ExpressibleByStringLiteral, ExpressibleByIntegerLiteral, ExpressibleByBooleanLiteral,
    ExpressibleByArrayLiteral, ExpressibleByDictionaryLiteral, ExpressibleByNilLiteral, ExpressibleByFloatLiteral
{
    public init(stringLiteral value: String) { self = .string(value) }
    public init(integerLiteral value: Int) { self = .number(Double(value)) }
    public init(floatLiteral value: Double) { self = .number(value) }
    public init(booleanLiteral value: Bool) { self = .bool(value) }
    public init(arrayLiteral elements: JSONValue...) { self = .array(elements) }
    public init(dictionaryLiteral elements: (String, JSONValue)...) {
        self = .object(JSONObject(elements.map { (key: $0.0, value: $0.1) }))
    }
    public init(nilLiteral: ()) { self = .null }

    public static func int(_ value: Int) -> JSONValue { .number(Double(value)) }
}

// MARK: - Parsing

public struct JSONParseError: Error, Sendable, Equatable, CustomStringConvertible {
    public let offset: Int
    public let message: String

    public var description: String { "invalid JSON at byte \(offset): \(message)" }
}

extension JSONValue {
    /// Parses one JSON document (RFC 8259). Trailing non-whitespace is an error.
    public static func parse(_ data: Data) throws(JSONParseError) -> JSONValue {
        do {
            return try data.withUnsafeBytes { raw in
                var reader = JSONReader(raw)
                let value = try reader.parseValue()
                reader.skipWhitespace()
                guard reader.atEnd else { throw reader.error("trailing characters after the JSON value") }
                return value
            }
        } catch let error as JSONParseError {
            throw error
        } catch {
            throw JSONParseError(offset: 0, message: "\(error)")
        }
    }

    public static func parse(_ text: String) throws(JSONParseError) -> JSONValue {
        try parse(Data(text.utf8))
    }
}

/// A forward-only reader over JSON bytes. Besides the generic `parseValue`, it exposes the
/// primitives the tokenizer loader uses to stream a 30 MB tokenizer.json without building a tree.
public struct JSONReader {
    let bytes: UnsafeRawBufferPointer
    public private(set) var position = 0
    /// Nesting limit, so that hostile input cannot exhaust the stack.
    public var maxDepth = 256
    private var depth = 0

    public init(_ bytes: UnsafeRawBufferPointer) {
        self.bytes = bytes
    }

    public var atEnd: Bool { position >= bytes.count }

    public func error(_ message: String) -> JSONParseError {
        JSONParseError(offset: position, message: message)
    }

    @inline(__always)
    public func peek() -> UInt8? {
        position < bytes.count ? bytes[position] : nil
    }

    @inline(__always)
    public mutating func skipWhitespace() {
        while position < bytes.count {
            switch bytes[position] {
            case 0x20, 0x0A, 0x0D, 0x09: position += 1
            default: return
            }
        }
    }

    /// Skips whitespace and consumes `byte`.
    public mutating func expect(_ byte: UInt8) throws(JSONParseError) {
        skipWhitespace()
        guard peek() == byte else {
            throw error("expected '\(Character(Unicode.Scalar(byte)))'")
        }
        position += 1
    }

    /// Skips whitespace and consumes `byte` if it is next.
    public mutating func consume(_ byte: UInt8) -> Bool {
        skipWhitespace()
        if peek() == byte {
            position += 1
            return true
        }
        return false
    }

    /// After an element of an array or object: true when another element follows (a comma was
    /// consumed), false when `close` was consumed.
    public mutating func nextElement(close: UInt8) throws(JSONParseError) -> Bool {
        skipWhitespace()
        guard let byte = peek() else { throw error("unexpected end of input") }
        position += 1
        if byte == UInt8(ascii: ",") { return true }
        if byte == close { return false }
        position -= 1
        throw error("expected ',' or '\(Character(Unicode.Scalar(close)))'")
    }

    /// Reads a string and appends its decoded UTF-8 bytes to `output`. Escapes are decoded;
    /// `😀` pairs become one scalar and lone surrogates become U+FFFD.
    public mutating func readString(into output: inout [UInt8]) throws(JSONParseError) {
        try expect(UInt8(ascii: "\""))
        while true {
            // Copy the run of plain bytes in one go.
            let start = position
            while position < bytes.count {
                let byte = bytes[position]
                if byte == UInt8(ascii: "\"") || byte == UInt8(ascii: "\\") || byte < 0x20 { break }
                position += 1
            }
            if position > start {
                output.append(contentsOf: UnsafeRawBufferPointer(rebasing: bytes[start..<position]))
            }
            guard position < bytes.count else { throw error("unterminated string") }
            let byte = bytes[position]
            position += 1
            if byte == UInt8(ascii: "\"") { return }
            if byte < 0x20 {
                position -= 1
                throw error("control character in string")
            }
            // Escape.
            guard position < bytes.count else { throw error("unterminated escape") }
            let escape = bytes[position]
            position += 1
            switch escape {
            case UInt8(ascii: "\""): output.append(0x22)
            case UInt8(ascii: "\\"): output.append(0x5C)
            case UInt8(ascii: "/"): output.append(0x2F)
            case UInt8(ascii: "b"): output.append(0x08)
            case UInt8(ascii: "f"): output.append(0x0C)
            case UInt8(ascii: "n"): output.append(0x0A)
            case UInt8(ascii: "r"): output.append(0x0D)
            case UInt8(ascii: "t"): output.append(0x09)
            case UInt8(ascii: "u"):
                var scalar = try readHex4()
                if (0xD800...0xDBFF).contains(scalar) {
                    // High surrogate: a low surrogate escape must follow.
                    if position + 1 < bytes.count, bytes[position] == UInt8(ascii: "\\"),
                        bytes[position + 1] == UInt8(ascii: "u")
                    {
                        let saved = position
                        position += 2
                        let low = try readHex4()
                        if (0xDC00...0xDFFF).contains(low) {
                            scalar = 0x10000 + ((scalar - 0xD800) << 10) + (low - 0xDC00)
                        } else {
                            position = saved
                            scalar = 0xFFFD
                        }
                    } else {
                        scalar = 0xFFFD
                    }
                } else if (0xDC00...0xDFFF).contains(scalar) {
                    scalar = 0xFFFD
                }
                appendUTF8(scalar, to: &output)
            default:
                position -= 1
                throw error("invalid escape")
            }
        }
    }

    private mutating func readHex4() throws(JSONParseError) -> UInt32 {
        guard position + 4 <= bytes.count else { throw error("truncated \\u escape") }
        var value: UInt32 = 0
        for _ in 0..<4 {
            let byte = bytes[position]
            let digit: UInt32
            switch byte {
            case UInt8(ascii: "0")...UInt8(ascii: "9"): digit = UInt32(byte - UInt8(ascii: "0"))
            case UInt8(ascii: "a")...UInt8(ascii: "f"): digit = UInt32(byte - UInt8(ascii: "a") + 10)
            case UInt8(ascii: "A")...UInt8(ascii: "F"): digit = UInt32(byte - UInt8(ascii: "A") + 10)
            default: throw error("invalid hex digit in \\u escape")
            }
            value = value << 4 | digit
            position += 1
        }
        return value
    }

    /// Reads a string as a Swift String (scalars preserved). JSON is UTF-8 (RFC 8259): invalid bytes are an error,
    /// as in JSONSerialization and OpenAI's API, not repaired to U+FFFD (that would embed a different input).
    public mutating func readString() throws(JSONParseError) -> String {
        var buffer: [UInt8] = []
        try readString(into: &buffer)
        guard let string = String(validating: buffer, as: UTF8.self) else { throw error("invalid UTF-8 in string") }
        return string
    }

    /// Reads a number token.
    public mutating func readNumber() throws(JSONParseError) -> Double {
        skipWhitespace()
        let start = position
        if peek() == UInt8(ascii: "-") { position += 1 }
        guard let first = peek(), (UInt8(ascii: "0")...UInt8(ascii: "9")).contains(first) else {
            throw error("invalid number")
        }
        if first == UInt8(ascii: "0") {
            position += 1
        } else {
            skipDigits()
        }
        var isInteger = true
        if peek() == UInt8(ascii: ".") {
            isInteger = false
            position += 1
            guard skipDigits() > 0 else { throw error("invalid number") }
        }
        if peek() == UInt8(ascii: "e") || peek() == UInt8(ascii: "E") {
            isInteger = false
            position += 1
            if peek() == UInt8(ascii: "+") || peek() == UInt8(ascii: "-") { position += 1 }
            guard skipDigits() > 0 else { throw error("invalid number") }
        }
        let token = UnsafeRawBufferPointer(rebasing: bytes[start..<position])
        if isInteger, token.count <= 18 {
            var value: Int64 = 0
            var negative = false
            for byte in token {
                if byte == UInt8(ascii: "-") { negative = true; continue }
                value = value * 10 + Int64(byte - UInt8(ascii: "0"))
            }
            return Double(negative ? -value : value)
        }
        guard let value = Double(String(decoding: token, as: UTF8.self)) else { throw error("invalid number") }
        return value
    }

    @discardableResult
    private mutating func skipDigits() -> Int {
        let start = position
        while let byte = peek(), (UInt8(ascii: "0")...UInt8(ascii: "9")).contains(byte) { position += 1 }
        return position - start
    }

    private mutating func literal(_ text: StaticString) throws(JSONParseError) {
        let count = text.utf8CodeUnitCount
        guard position + count <= bytes.count else { throw error("invalid literal") }
        let pointer = text.utf8Start
        for index in 0..<count where bytes[position + index] != pointer[index] {
            throw error("invalid literal")
        }
        position += count
    }

    /// Parses any JSON value into a tree.
    public mutating func parseValue() throws(JSONParseError) -> JSONValue {
        skipWhitespace()
        guard let byte = peek() else { throw error("unexpected end of input") }
        switch byte {
        case UInt8(ascii: "{"):
            depth += 1
            defer { depth -= 1 }
            guard depth <= maxDepth else { throw error("nesting deeper than \(maxDepth)") }
            position += 1
            var object = JSONObject()
            if consume(UInt8(ascii: "}")) { return .object(object) }
            repeat {
                skipWhitespace()
                let key = try readString()
                try expect(UInt8(ascii: ":"))
                let value = try parseValue()
                object.pairs.append((key: key, value: value))
            } while try nextElement(close: UInt8(ascii: "}"))
            return .object(object)
        case UInt8(ascii: "["):
            depth += 1
            defer { depth -= 1 }
            guard depth <= maxDepth else { throw error("nesting deeper than \(maxDepth)") }
            position += 1
            var array: [JSONValue] = []
            if consume(UInt8(ascii: "]")) { return .array(array) }
            repeat {
                array.append(try parseValue())
            } while try nextElement(close: UInt8(ascii: "]"))
            return .array(array)
        case UInt8(ascii: "\""):
            return .string(try readString())
        case UInt8(ascii: "t"):
            try literal("true")
            return .bool(true)
        case UInt8(ascii: "f"):
            try literal("false")
            return .bool(false)
        case UInt8(ascii: "n"):
            try literal("null")
            return .null
        default:
            return .number(try readNumber())
        }
    }

    /// Skips one value of any type without materializing it.
    public mutating func skipValue() throws(JSONParseError) {
        skipWhitespace()
        guard let byte = peek() else { throw error("unexpected end of input") }
        switch byte {
        case UInt8(ascii: "\""):
            var sink: [UInt8] = []
            try readString(into: &sink)
        case UInt8(ascii: "{"), UInt8(ascii: "["):
            _ = try parseValue()
        default:
            _ = try parseValue()
        }
    }
}

@inline(__always)
func appendUTF8(_ scalar: UInt32, to output: inout [UInt8]) {
    switch scalar {
    case 0..<0x80:
        output.append(UInt8(scalar))
    case 0x80..<0x800:
        output.append(UInt8(0xC0 | (scalar >> 6)))
        output.append(UInt8(0x80 | (scalar & 0x3F)))
    case 0x800..<0x10000:
        output.append(UInt8(0xE0 | (scalar >> 12)))
        output.append(UInt8(0x80 | ((scalar >> 6) & 0x3F)))
        output.append(UInt8(0x80 | (scalar & 0x3F)))
    default:
        output.append(UInt8(0xF0 | (scalar >> 18)))
        output.append(UInt8(0x80 | ((scalar >> 12) & 0x3F)))
        output.append(UInt8(0x80 | ((scalar >> 6) & 0x3F)))
        output.append(UInt8(0x80 | (scalar & 0x3F)))
    }
}
