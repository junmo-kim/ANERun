// Configuration: every option is a flag and an environment variable with the same name
// (--max-tokens <-> ANERUN_MAX_TOKENS). Precedence: flag > ANERUN_* > ANEEMBED_* > default.
// `--print-config` shows each resolved value and where it came from.
import Foundation

public struct Config: Sendable, Equatable {
    public enum Source: String, Sendable {
        case flag
        case environment = "env"
        case `default`
    }

    public struct Option: Sendable {
        public let name: String
        public let help: String
        public let defaultValue: @Sendable () -> String
        public let validate: @Sendable (String) -> String?  // error message or nil

        public var flag: String { "--\(name)" }
        public var environmentName: String { "ANERUN_" + environmentSuffix }
        var legacyEnvironmentName: String { "ANEEMBED_" + environmentSuffix }

        private var environmentSuffix: String { name.uppercased().replacingOccurrences(of: "-", with: "_") }
    }

    public static let options: [Option] = [
        Option(name: "model", help: "model ID from the catalog",
               defaultValue: { "" }, validate: { _ in nil }),
        Option(name: "model-dir", help: "model package directory",
               defaultValue: { "" }, validate: { _ in nil }),
        Option(name: "data-dir", help: "where models and state live",
               defaultValue: { Config.defaultDataDirectory.path }, validate: { $0.isEmpty ? "must not be empty" : nil }),
        Option(name: "host", help: "listen address (loopback by default; anything else exposes an unauthenticated API)",
               defaultValue: { "127.0.0.1" }, validate: { $0.isEmpty ? "must not be empty" : nil }),
        Option(name: "idle-timeout", help: "decision server idle unload in seconds (0 disables)",
               defaultValue: { "0" }, validate: integer(0...Int.max)),
        Option(name: "port", help: "listen port", defaultValue: { "18180" }, validate: integer(1...65535)),
        Option(name: "max-tokens", help: "longest text accepted, in model tokens; only the functions needed for this cap are loaded",
               defaultValue: { "2048" }, validate: integer(1...32768)),
        Option(name: "model-name", help: "model alias reported by the API (defaults to the selected model ID)", defaultValue: { "" },
               validate: { $0.isEmpty ? "must not be empty" : nil }),
        Option(name: "max-body", help: "request body limit in MiB", defaultValue: { "72" }, validate: integer(1...1024)),
        Option(name: "max-connections", help: "open HTTP connections; more are answered 503 and closed",
               defaultValue: { "64" }, validate: integer(1...4096)),
        Option(name: "max-inputs", help: "texts per request", defaultValue: { "2048" }, validate: integer(1...100_000)),
        Option(name: "max-images", help: "images per request", defaultValue: { "8" }, validate: integer(1...64)),
        Option(name: "query-concurrency", help: "concurrent query-lane batches (Qwen3: <= 64 tokens, other models: <= 512)",
               defaultValue: { "8" }, validate: integer(1...64)),
        Option(name: "image-concurrency", help: "concurrent image decodes and encodes", defaultValue: { "2" },
               validate: integer(1...64)),
        Option(name: "long-concurrency", help: "concurrent texts above the query lane limit, and /tokenize and /detokenize requests", defaultValue: { "2" },
               validate: integer(1...64)),
        Option(name: "runtime", help: "coreml (Neural Engine) or fake (no model, for testing integrations)",
               defaultValue: { "coreml" }, validate: { ["coreml", "fake"].contains($0) ? nil : "must be coreml or fake" }),
        Option(name: "selftest",
               help: "require: serve only functions that pass the package's self-test; skip: serve every function unchecked (unsafe)",
               defaultValue: { "require" }, validate: { ["require", "skip"].contains($0) ? nil : "must be require or skip" }),
    ]

    public static func option(named name: String) -> Option? { options.first { $0.name == name } }

    private static func integer(_ range: ClosedRange<Int>) -> @Sendable (String) -> String? {
        { value in
            guard let int = Int(value) else { return "must be an integer" }
            return range.contains(int) ? nil : "must be in \(range.lowerBound)...\(range.upperBound)"
        }
    }

    // Keep the physical model and compile-cache paths stable across the product rename.
    public static var defaultDataDirectory: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/ANEEmbed", isDirectory: true)
    }

    public private(set) var values: [String: String] = [:]
    public private(set) var sources: [String: Source] = [:]
    public private(set) var modelID: ModelID?
    private var resolvedModelDirectory: URL?
    var modelResolutionError: ServiceError?

    /// Resolves every option from `flags` (name without dashes -> value) and `environment`.
    public init(flags: [String: String] = [:], environment: [String: String] = ProcessInfo.processInfo.environment)
        throws(ServiceError)
    {
        for name in flags.keys where Self.option(named: name) == nil {
            throw .badRequest("unknown_option", "unknown option --\(name)")
        }
        for option in Self.options {
            let value: String
            let source: Source
            let origin: String
            if let flag = flags[option.name] {
                origin = option.flag
                (value, source) = (flag, .flag)
            } else if let env = environment[option.environmentName], !env.isEmpty {
                origin = option.environmentName
                (value, source) = (env, .environment)
            } else if let env = environment[option.legacyEnvironmentName], !env.isEmpty {
                origin = option.legacyEnvironmentName
                (value, source) = (env, .environment)
            } else {
                origin = option.flag
                (value, source) = (option.defaultValue(), .default)
            }
            if source != .default || !value.isEmpty, let problem = option.validate(value) {
                throw .badRequest("invalid_option", "\(origin)=\(value): \(problem)", param: option.name)
            }
            values[option.name] = value
            sources[option.name] = source
        }
    }

    public func string(_ name: String) -> String { values[name] ?? "" }
    public func int(_ name: String) -> Int { Int(values[name] ?? "") ?? 0 }

    public var dataDirectory: URL { URL(fileURLWithPath: (string("data-dir") as NSString).expandingTildeInPath, isDirectory: true) }
    public var host: String { string("host") }
    public var port: Int { int("port") }
    public var maxTokens: Int { int("max-tokens") }
    public var modelName: String {
        let alias = string("model-name")
        return alias.isEmpty ? modelID?.rawValue ?? "" : alias
    }
    public var maxBodyBytes: Int { int("max-body") << 20 }
    public var maxConnections: Int { int("max-connections") }
    public var useFakeRuntime: Bool { string("runtime") == "fake" }
    /// `--selftest skip`: load without the package's self-test and serve every function unchecked (unsafe).
    public var skipSelfTest: Bool { string("selftest") == "skip" }

    /// Validates the selectors, then snapshots the model revision before a command uses the package.
    public func selectingModel(for command: String, requiresName: Bool = false, required: Bool = true)
        throws(ServiceError) -> Config
    {
        let name = string("model")
        let explicit = explicitModelDirectory
        if !name.isEmpty && explicit != nil {
            throw ServiceError(.badRequest, "conflicting_options", "--model \(name) and --model-dir \(string("model-dir")) are both set",
                               hint: "set only --model or --model-dir, including their ANEEMBED environment variables")
        }
        let installing = command == "models install" || command == "install"
        let needsName = requiresName || installing
        if name.isEmpty && (needsName || (required && explicit == nil)) {
            let example = "anerun \(command) --model \(ModelID.pplx.rawValue)"
            let selectors = needsName ? "--model" : "--model or --model-dir"
            throw ServiceError(.badRequest, "missing_model", "\(command) requires \(selectors)\n\(ModelID.catalog)",
                               param: "model", hint: "try `\(example)`")
        }

        if modelID != nil { return self }

        var selected = self
        if !name.isEmpty {
            guard let model = ModelID(rawValue: name) else {
                let valid = ModelID.allCases.map(\.rawValue).joined(separator: ", ")
                throw ServiceError(.badRequest, "unknown_model", "unknown model \(name)", param: "model",
                                   hint: "use --model with one of: \(valid)")
            }
            try validateTokenRange(for: model)
            selected.modelID = model
            let store = ModelStore(dataDirectory: dataDirectory, model: model)
            if installing || useFakeRuntime {
                try store.checkNamespace()
                return selected
            }
            let directory: URL
            do {
                directory = try store.resolveDirectory().standardizedFileURL.resolvingSymlinksInPath()
            } catch {
                if command == "doctor" {
                    selected.modelResolutionError = error
                    return selected
                }
                guard model == .qwen3Embedding, error.reason == "model_missing" else { throw error }
                throw ServiceError(error.kind, error.reason, error.message, param: error.param,
                                   hint: "install with `anerun models install --model \(model.rawValue) --from DIR`, "
                                       + "supply --repo and --revision, or pass --model-dir DIR")
            }
            if command != "doctor" { _ = try ModelID.identify(in: directory, expected: model) }
            selected.resolvedModelDirectory = directory
        } else if let explicit {
            let directory = explicit.standardizedFileURL.resolvingSymlinksInPath()
            let model = command == "doctor" ? try? ModelID.identify(in: directory) : try ModelID.identify(in: directory)
            if let model { try validateTokenRange(for: model) }
            selected.modelID = model
            selected.resolvedModelDirectory = directory
        }
        return selected
    }

    private func validateTokenRange(for model: ModelID) throws(ServiceError) {
        let range = model.tokenRange
        guard range.contains(maxTokens) else {
            throw ServiceError(.badRequest, "invalid_option",
                               "max-tokens \(maxTokens) for \(model.rawValue) must be in \(range.lowerBound)...\(range.upperBound)",
                               param: "max-tokens", hint: "set --max-tokens within the selected model's range")
        }
    }

    private var explicitModelDirectory: URL? {
        let path = string("model-dir")
        guard !path.isEmpty else { return nil }
        return URL(fileURLWithPath: (path as NSString).expandingTildeInPath, isDirectory: true)
    }

    /// Snapshots an installer-provided revision while preserving its existing typed model selection.
    public func withModelDirectory(_ directory: URL) -> Config {
        var copy = self
        copy.resolvedModelDirectory = directory.standardizedFileURL.resolvingSymlinksInPath()
        return copy
    }

    /// The selected path. Unresolved configurations retain the historical directory accessor for library callers.
    public var modelDirectory: URL {
        if let resolvedModelDirectory { return resolvedModelDirectory }
        if let modelID { return ModelStore(dataDirectory: dataDirectory, model: modelID).modelsDirectory }
        return explicitModelDirectory ?? ModelStore(dataDirectory: dataDirectory).currentDirectory
    }

    /// Pins the published revision and its repaired copies when the path came from a named selection.
    public var pinnedManifestSHA256: String? {
        guard let source = modelID?.source, let revision = resolvedModelDirectory?.lastPathComponent,
            let digest = source.manifestSHA256, explicitModelDirectory == nil
        else { return nil }
        if revision == source.revision { return digest }
        let repairPrefix = source.revision + "-r-"
        guard revision.hasPrefix(repairPrefix) else { return nil }
        let suffix = revision.dropFirst(repairPrefix.count)
        return suffix.count == 8 && suffix.allSatisfy { $0.isASCII && $0.isHexDigit } ? digest : nil
    }

    public var limits: ServiceLimits {
        var limits = ServiceLimits()
        limits.maxTokens = maxTokens
        if modelID == .qwen3Embedding { limits.shortTokens = 64 }
        limits.maxInputs = int("max-inputs")
        limits.maxImages = int("max-images")
        limits.queryConcurrency = int("query-concurrency")
        limits.imageConcurrency = int("image-concurrency")
        limits.longConcurrency = int("long-concurrency")
        return limits
    }

    /// `--print-config`: value and source of every option.
    public var printable: JSONValue {
        .object(JSONObject(Self.options.map { option in
            let value: String
            switch option.name {
            case "model": value = modelID?.rawValue ?? string(option.name)
            case "model-name": value = modelName
            case "model-dir": value = resolvedModelDirectory?.path ?? string(option.name)
            default: value = string(option.name)
            }
            return (key: option.name, value: ["value": .string(value), "source": .string(sources[option.name]!.rawValue),
                                              "env": .string(option.environmentName)])
        }))
    }
}
