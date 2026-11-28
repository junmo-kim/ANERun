// JSON and YAML output for JSONValue trees.
//
// Output contract (same as macvis): stdout carries only data, YAML by default and JSON with
// --format json; objects keep their insertion order so output is stable and diffable.
import Foundation

public enum OutputFormat: String, Sendable, CaseIterable {
    case yaml
    case json
}

extension JSONValue {
    /// Compact JSON bytes.
    public func jsonData() -> Data {
        var output: [UInt8] = []
        output.reserveCapacity(256)
        writeJSON(to: &output)
        return Data(output)
    }

    public func jsonString() -> String {
        var output: [UInt8] = []
        writeJSON(to: &output)
        return String(decoding: output, as: UTF8.self)
    }

    public func render(as format: OutputFormat) -> String {
        switch format {
        case .json: return jsonString()
        case .yaml:
            var lines: [String] = []
            renderYAML(indent: 0, into: &lines)
            return lines.joined(separator: "\n")
        }
    }

    public func writeJSON(to output: inout [UInt8]) {
        switch self {
        case .null: output.append(contentsOf: "null".utf8)
        case .bool(let value): output.append(contentsOf: (value ? "true" : "false").utf8)
        case .number(let value): writeJSONNumber(value, to: &output)
        case .string(let value): writeJSONString(value, to: &output)
        case .floats(let values): writeJSONFloats(values, to: &output)
        case .array(let values):
            output.append(UInt8(ascii: "["))
            for (index, value) in values.enumerated() {
                if index > 0 { output.append(UInt8(ascii: ",")) }
                value.writeJSON(to: &output)
            }
            output.append(UInt8(ascii: "]"))
        case .object(let object):
            output.append(UInt8(ascii: "{"))
            for (index, pair) in object.pairs.enumerated() {
                if index > 0 { output.append(UInt8(ascii: ",")) }
                writeJSONString(pair.key, to: &output)
                output.append(UInt8(ascii: ":"))
                pair.value.writeJSON(to: &output)
            }
            output.append(UInt8(ascii: "}"))
        }
    }

    // MARK: YAML

    private func renderYAML(indent: Int, into lines: inout [String]) {
        let pad = String(repeating: "  ", count: indent)
        switch self {
        case .array(let items) where !items.isEmpty:
            for item in items {
                if let scalar = item.yamlScalar {
                    lines.append("\(pad)- \(scalar)")
                } else {
                    var nested: [String] = []
                    item.renderYAML(indent: indent + 1, into: &nested)
                    // Put the first line of a nested mapping on the dash line ("- key: value").
                    if case .object = item, let first = nested.first {
                        lines.append("\(pad)- " + first.dropFirst(pad.count + 2))
                        lines.append(contentsOf: nested.dropFirst())
                    } else {
                        lines.append("\(pad)-")
                        lines.append(contentsOf: nested)
                    }
                }
            }
        case .object(let object) where !object.pairs.isEmpty:
            for (key, value) in object.pairs {
                if let scalar = value.yamlScalar {
                    lines.append("\(pad)\(yamlKey(key)): \(scalar)")
                } else {
                    lines.append("\(pad)\(yamlKey(key)):")
                    value.renderYAML(indent: indent + 1, into: &lines)
                }
            }
        default:
            lines.append(pad + (yamlScalar ?? "null"))
        }
    }

    /// The single-line YAML form of scalars, empty containers and float vectors (flow style).
    private var yamlScalar: String? {
        switch self {
        case .null: return "null"
        case .bool(let value): return value ? "true" : "false"
        case .number(let value):
            var output: [UInt8] = []
            writeJSONNumber(value, to: &output)
            return String(decoding: output, as: UTF8.self)
        case .string(let value): return yamlString(value)
        case .floats(let values):
            var output: [UInt8] = []
            writeJSONFloats(values, to: &output)
            return String(decoding: output, as: UTF8.self).replacingOccurrences(of: ",", with: ", ")
        case .array(let values) where values.isEmpty: return "[]"
        case .object(let object) where object.pairs.isEmpty: return "{}"
        default: return nil
        }
    }
}

func yamlKey(_ key: String) -> String {
    let plain = !key.isEmpty && key.unicodeScalars.allSatisfy {
        ($0 >= "a" && $0 <= "z") || ($0 >= "A" && $0 <= "Z") || ($0 >= "0" && $0 <= "9") || $0 == "_" || $0 == "-"
            || $0 == "."
    }
    return plain ? key : yamlString(key, forceQuotes: true)
}

/// A YAML scalar for `value`: plain when unambiguous, otherwise a JSON-style double-quoted string
/// (valid YAML 1.2).
func yamlString(_ value: String, forceQuotes: Bool = false) -> String {
    let reserved: Set<String> = [
        "", "null", "Null", "NULL", "~", "true", "True", "TRUE", "false", "False", "FALSE", "yes", "no", "on",
        "off", "Yes", "No", "On", "Off", "YES", "NO", "ON", "OFF",
    ]
    var needsQuotes = forceQuotes || reserved.contains(value) || Double(value) != nil
    if !needsQuotes, let first = value.unicodeScalars.first, let last = value.unicodeScalars.last {
        if " -?:,[]{}#&*!|>'\"%@`".unicodeScalars.contains(first) || last == " " { needsQuotes = true }
    }
    if !needsQuotes {
        for scalar in value.unicodeScalars {
            if scalar.value < 0x20 || scalar.value == 0x7F || scalar == "\u{FEFF}" || scalar.value >= 0x80 && scalar.properties.generalCategory == .format {
                needsQuotes = true
                break
            }
        }
        if value.contains(": ") || value.contains(" #") { needsQuotes = true }
    }
    guard needsQuotes else { return value }
    var output: [UInt8] = []
    writeJSONString(value, to: &output)
    return String(decoding: output, as: UTF8.self)
}

private let hexDigits = Array("0123456789abcdef".utf8)

public func writeJSONString(_ value: String, to output: inout [UInt8]) {
    output.append(UInt8(ascii: "\""))
    for byte in value.utf8 {
        switch byte {
        case UInt8(ascii: "\""): output.append(contentsOf: [0x5C, 0x22])
        case UInt8(ascii: "\\"): output.append(contentsOf: [0x5C, 0x5C])
        case 0x0A: output.append(contentsOf: [0x5C, UInt8(ascii: "n")])
        case 0x0D: output.append(contentsOf: [0x5C, UInt8(ascii: "r")])
        case 0x09: output.append(contentsOf: [0x5C, UInt8(ascii: "t")])
        case 0..<0x20, 0x7F:
            output.append(contentsOf: [0x5C, UInt8(ascii: "u"), UInt8(ascii: "0"), UInt8(ascii: "0")])
            output.append(hexDigits[Int(byte >> 4)])
            output.append(hexDigits[Int(byte & 0xF)])
        default: output.append(byte)
        }
    }
    output.append(UInt8(ascii: "\""))
}

public func writeJSONNumber(_ value: Double, to output: inout [UInt8]) {
    guard value.isFinite else {
        output.append(contentsOf: "null".utf8)  // JSON has no NaN/Infinity
        return
    }
    if let int = Int64(exactly: value), abs(int) < (1 << 53) {
        output.append(contentsOf: String(int).utf8)
    } else {
        output.append(contentsOf: value.description.utf8)
    }
}

/// A float32 vector as a JSON array, each value in its shortest round-trip form.
public func writeJSONFloats(_ values: [Float], to output: inout [UInt8]) {
    output.append(UInt8(ascii: "["))
    for (index, value) in values.enumerated() {
        if index > 0 { output.append(UInt8(ascii: ",")) }
        if value.isFinite {
            output.append(contentsOf: value.description.utf8)
        } else {
            output.append(contentsOf: "null".utf8)
        }
    }
    output.append(UInt8(ascii: "]"))
}
