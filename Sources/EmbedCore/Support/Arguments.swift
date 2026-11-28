// Hand-written argument parsing (no swift-argument-parser): positionals, `--name value`, `--name=value`,
// repeatable options, boolean flags and `--` to end options. Unknown options are usage errors (exit 64).
import Foundation

public struct ParsedArguments: Sendable, Equatable {
    public var positionals: [String] = []
    public var options: [String: [String]] = [:]
    public var flags: Set<String> = []

    public func value(_ name: String) -> String? { options[name]?.last }
    public func values(_ name: String) -> [String] { options[name] ?? [] }
    public func has(_ flag: String) -> Bool { flags.contains(flag) }

    /// The config options among the parsed ones (name -> last value).
    public var configFlags: [String: String] {
        var result: [String: String] = [:]
        for option in Config.options { if let value = value(option.name) { result[option.name] = value } }
        return result
    }
}

public struct ArgumentSpec: Sendable {
    public var valueOptions: Set<String>
    public var booleanFlags: Set<String>

    /// Command options plus every config option (accepted by all commands).
    public init(values: [String] = [], flags: [String] = []) {
        valueOptions = Set(values).union(Config.options.map(\.name))
        booleanFlags = Set(flags).union(["help", "print-config"])
    }

    public func parse(_ arguments: [String]) throws(ServiceError) -> ParsedArguments {
        var parsed = ParsedArguments()
        var index = 0
        var optionsEnded = false
        while index < arguments.count {
            let argument = arguments[index]
            index += 1
            if optionsEnded || !argument.hasPrefix("-") || argument == "-" {
                parsed.positionals.append(argument)
                continue
            }
            if argument == "--" {
                optionsEnded = true
                continue
            }
            if argument == "-h" {
                parsed.flags.insert("help")
                continue
            }
            guard argument.hasPrefix("--") else {
                throw .badRequest("unknown_option", "unknown option \(argument)")
            }
            var name = String(argument.dropFirst(2))
            var inlineValue: String?
            if let equals = name.firstIndex(of: "=") {
                inlineValue = String(name[name.index(after: equals)...])
                name = String(name[..<equals])
            }
            if booleanFlags.contains(name) {
                guard inlineValue == nil else { throw .badRequest("unexpected_value", "--\(name) takes no value") }
                parsed.flags.insert(name)
            } else if valueOptions.contains(name) {
                let value: String
                if let inlineValue {
                    value = inlineValue
                } else {
                    guard index < arguments.count else { throw .badRequest("missing_value", "--\(name) needs a value") }
                    value = arguments[index]
                    index += 1
                }
                parsed.options[name, default: []].append(value)
            } else {
                throw .badRequest("unknown_option", "unknown option --\(name)")
            }
        }
        return parsed
    }
}
