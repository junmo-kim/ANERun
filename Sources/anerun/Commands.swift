import EmbedCore
import Foundation

enum CLI {
    static let usage = """
        anerun \(aneembedVersion): embeddings and decisions on the Apple Neural Engine

        USAGE
          anerun embed [TEXT ...] [--stdin] [--image PATH ...] [--prompt query|document]
                         [--no-truncate] [--dimensions N] [--encoding float|base64] [--format yaml|json]
          anerun similarity A B [--image-a PATH] [--image-b PATH] [--prompt query|document]
                         [--prompt-a ...] [--prompt-b ...] [--no-truncate] [--dimensions N] [--format yaml|json]
          anerun tokenize [TEXT ...] [--stdin] [--prompt query|document] [--show-tokens] [--format yaml|json]
          anerun decide --stdin    Decision request as JSON (state, questions, optional base64 image)
          anerun doctor [--verify] [--format yaml|json]
          anerun models path | verify | warm
          anerun models install --model ID [--from DIR | --repo R --revision COMMIT]
          anerun bench [--sizes 32,128,...] [--repeat N] [--image PATH] [--concurrency N] [--buckets]
                         [--breakdown] [--settle SECONDS] [--format yaml|json]
          anerun serve        HTTP: OpenAI /v1/embeddings, /v1/models, /health, POST /tokenize, /detokenize,
                                POST /mcp, QUERY /{tool}; decision models serve POST /v1/decisions
          anerun playground [--no-open] [--idle-timeout SECONDS]  Local browser UI (5 min idle by default)
          anerun mcp          MCP over stdio (newline-delimited JSON-RPC 2.0)
          anerun --version | --help | <command> --print-config

        Select a model with --model ID (or --model-dir DIR for commands that consume a package).
        \(ModelID.catalog)

        Every command takes these options. Legacy ANEEMBED_* variables are also accepted:
        \(Config.options.map { "  --\($0.name.padding(toLength: 18, withPad: " ", startingAt: 0)) \($0.environmentName.padding(toLength: 28, withPad: " ", startingAt: 0)) \($0.help)" }.joined(separator: "\n"))

        stdout carries data only (YAML by default, JSON with --format json; --stdin modes write JSONL).
        Text embeddings truncate at the model's context limit by default. Use --no-truncate to reject overlong texts.
        Errors go to stderr with an exit code: 64 bad arguments, 65 input the model cannot take,
        69 no model or Neural Engine, 75 still loading, 1 anything else.
        A one-off command loads the model each time (seconds); keep `serve` or `mcp` running for regular use.
        """

    static func run(_ arguments: [String]) async -> Int32 {
        guard let command = arguments.first, command != "--help", command != "-h", command != "help" else {
            Output.data(usage)
            return ExitCode.success.rawValue
        }
        if command == "--version" {
            Output.data("anerun \(aneembedVersion)")
            return ExitCode.success.rawValue
        }
        let rest = Array(arguments.dropFirst())
        var format = OutputFormat.yaml
        do throws(ServiceError) {
            let handler: Command
            switch command {
            case "decide": handler = DecideCommand()
            case "embed": handler = EmbedCommand()
            case "similarity": handler = SimilarityCommand()
            case "tokenize": handler = TokenizeCommand()
            case "doctor": handler = DoctorCommand()
            case "models": handler = ModelsCommand()
            case "serve": handler = ServeCommand()
            case "playground": handler = PlaygroundCommand()
            case "mcp": handler = MCPCommand()
            case "bench": handler = BenchCommand()
            default:
                throw ServiceError.badRequest("unknown_command", "unknown command \(command); see anerun --help")
            }
            let parsed = try handler.spec.parse(rest)
            format = parsed.value("format").flatMap(OutputFormat.init(rawValue:)) ?? .yaml
            if parsed.has("help") {
                Output.data(usage)
                return ExitCode.success.rawValue
            }
            if let value = parsed.value("format"), OutputFormat(rawValue: value) == nil {
                throw ServiceError.badRequest("invalid_format", "--format must be yaml or json", param: "format")
            }
            let config = try Config(flags: parsed.configFlags)
            let action = parsed.positionals.first ?? ""
            let commandName = command == "models" ? "models \(action)" : command
            let isInstall = command == "models" && action == "install" && parsed.positionals.count == 1
            let needsModel: Bool
            switch command {
            case "doctor": needsModel = parsed.has("verify")
            case "models": needsModel = parsed.positionals.count == 1 && ["path", "verify", "warm", "install"].contains(action)
            default: needsModel = true
            }
            if parsed.has("print-config") {
                let hasSelector = !config.string("model").isEmpty || !config.string("model-dir").isEmpty
                let context = isInstall && hasSelector ? commandName : "--print-config"
                let inspected = try config.selectingModel(for: context, required: false)
                Output.data(inspected.printable, format: format)
                return ExitCode.success.rawValue
            }
            let selected = try config.selectingModel(for: commandName, requiresName: isInstall, required: needsModel)
            if command != "doctor", command != "playground", let model = selected.modelID, model.isDecision {
                return try await DecisionCommands.run(commandName, arguments: parsed, config: selected, model: model, format: format).rawValue
            }
            return try await handler.run(parsed, config: selected, format: format).rawValue
        } catch {
            Output.write(error.structured.render(as: format) + "\n", to: FileHandle.standardError)
            return error.exitCode.rawValue
        }
    }
}

protocol Command {
    var spec: ArgumentSpec { get }
    func run(_ arguments: ParsedArguments, config: Config, format: OutputFormat) async throws(ServiceError) -> ExitCode
}

func inputType(_ value: String?, param: String) throws(ServiceError) -> InputType? {
    guard let value else { return nil }
    guard let type = InputType(rawValue: value) else {
        throw .badRequest("invalid_prompt", "--\(param) must be query or document", param: param)
    }
    return type
}

func readFile(_ path: String) throws(ServiceError) -> Data {
    do {
        return try Data(contentsOf: URL(fileURLWithPath: (path as NSString).expandingTildeInPath))
    } catch {
        throw .badRequest("unreadable_path", "cannot read \(path): \(error.localizedDescription)", param: "image")
    }
}

/// JSONL input: each line a JSON string or an object with "text" (and optional "id").
func parseInputLine(_ line: Data, number: Int) throws(ServiceError) -> (text: String, id: JSONValue?) {
    let value: JSONValue
    do { value = try JSONValue.parse(line) } catch {
        throw .badRequest("invalid_jsonl", "stdin line \(number): \(error.description)")
    }
    if let text = value.stringValue { return (text, nil) }
    if let text = value["text"]?.stringValue { return (text, value["id"]) }
    throw .badRequest("invalid_jsonl", "stdin line \(number): expected a JSON string or an object with \"text\"")
}

// MARK: - embed

struct EmbedCommand: Command {
    let spec = ArgumentSpec(values: ["image", "prompt", "dimensions", "encoding", "format"], flags: ["stdin", "no-truncate"])
    static let batchSize = 256

    func run(_ arguments: ParsedArguments, config: Config, format: OutputFormat) async throws(ServiceError) -> ExitCode {
        let images = arguments.values("image")
        let texts = arguments.positionals
        let fromStdin = arguments.has("stdin")
        let truncate = !arguments.has("no-truncate")
        guard [!images.isEmpty, !texts.isEmpty, fromStdin].filter({ $0 }).count == 1 else {
            throw .badRequest("missing_input", "give texts, --image paths or --stdin (one of them)")
        }
        var options: JSONObject = [:]
        if let dimensions = arguments.value("dimensions") { options["dimensions"] = .string(dimensions) }
        if let encoding = arguments.value("encoding") { options["encoding_format"] = .string(encoding) }
        let model = config.modelID ?? .embeddingGemma2
        let output = try Tools.outputOptions(options, modelID: model)
        let type = try inputType(arguments.value("prompt"), param: "prompt")
        if !images.isEmpty && !model.supportsImages {
            throw ServiceError(.invalidInput, "images_unsupported", "\(model.rawValue) accepts only text", param: "input",
                               hint: "select --model embeddinggemma-2 for image embeddings")
        }
        if !images.isEmpty && type != nil {
            throw .badRequest("invalid_prompt", "--prompt applies to texts only", param: "prompt")
        }
        let service = try await ModelLoader.makeService(config, log: Log.info)

        if fromStdin {
            var batch: [(text: String, id: JSONValue?)] = []
            var index = 0, number = 0
            func flush() async throws(ServiceError) {
                guard !batch.isEmpty else { return }
                let result = try await service.embedTexts(batch.map(\.text), inputType: type, truncate: truncate)
                let truncated = Set(result.truncatedIndices)
                var lines = ""
                for (offset, item) in batch.enumerated() {
                    let vector = Vectors.output(result.vectors[offset], to: output.dimensions, modelID: service.modelID)
                    var record = JSONObject()
                    if let id = item.id { record["id"] = id }
                    record["index"] = .int(index + offset)
                    record["tokens"] = .int(result.tokenCounts[offset])
                    if truncated.contains(offset) { record["truncated"] = true }
                    record["embedding"] = output.encoding == .base64 ? .string(Vectors.base64(vector)) : .floats(vector)
                    lines += JSONValue.object(record).jsonString() + "\n"
                }
                Output.write(lines, to: FileHandle.standardOutput)
                index += batch.count
                batch.removeAll(keepingCapacity: true)
            }
            let input = LineReader(fileDescriptor: 0, maxLineBytes: config.maxBodyBytes, maxPending: 4)
            for await item in input.lines {
                input.release()
                number += 1
                let line = try stdinLine(item, number: number, limit: config.maxBodyBytes)
                if line.allSatisfy({ $0 == 0x20 || $0 == 0x09 }) { continue }
                batch.append(try parseInputLine(line, number: number))
                if batch.count == Self.batchSize { try await flush() }
            }
            try await flush()
            return .success
        }

        let result: EmbeddingResult
        if !images.isEmpty {
            var data: [Data] = []
            for path in images { data.append(try readFile(path)) }
            result = try await service.embedImages(data)
        } else {
            result = try await service.embedTexts(texts, inputType: type, truncate: truncate)
        }
        Output.data(Tools.embeddingOutput(result, options: output, model: service.modelName, modelID: service.modelID), format: format)
        return .success
    }
}

// MARK: - similarity

struct SimilarityCommand: Command {
    let spec = ArgumentSpec(values: ["image-a", "image-b", "prompt", "prompt-a", "prompt-b", "dimensions", "format"],
                            flags: ["no-truncate"])

    func run(_ arguments: ParsedArguments, config: Config, format: OutputFormat) async throws(ServiceError) -> ExitCode {
        if let model = config.modelID, !model.supportsImages,
            arguments.value("image-a") != nil || arguments.value("image-b") != nil
        {
            throw ServiceError(.invalidInput, "images_unsupported", "\(model.rawValue) accepts only text", param: "input",
                               hint: "select --model embeddinggemma-2 for image embeddings")
        }
        var texts = arguments.positionals[...]
        var tool: JSONObject = [:]
        for side in ["a", "b"] {
            if let path = arguments.value("image-\(side)") {
                tool["\(side)_path"] = .string(path)
            } else if let text = texts.popFirst() {
                tool[side] = .string(text)
                if let prompt = arguments.value("prompt-\(side)") ?? arguments.value("prompt") {
                    tool["\(side)_input_type"] = .string(prompt)
                }
            } else {
                throw .badRequest("missing_input", "similarity needs two inputs: A B, or --image-a/--image-b")
            }
        }
        guard texts.isEmpty else { throw .badRequest("too_many_inputs", "similarity compares exactly two inputs") }
        if let dimensions = arguments.value("dimensions") { tool["dimensions"] = .string(dimensions) }
        if arguments.has("no-truncate") { tool["truncate"] = false }
        let service = try await ModelLoader.makeService(config, log: Log.info)
        let result = try await Tools.call("similarity", arguments: tool, context: ToolContext(service: service, allowPaths: true))
        Output.data(result, format: format)
        return .success
    }
}

// MARK: - tokenize

struct TokenizeCommand: Command {
    let spec = ArgumentSpec(values: ["prompt", "format"], flags: ["stdin", "show-tokens"])

    func run(_ arguments: ParsedArguments, config: Config, format: OutputFormat) async throws(ServiceError) -> ExitCode {
        let tokenizer = try ModelLoader.loadTokenizer(config)
        let type = try inputType(arguments.value("prompt"), param: "prompt")
        func encode(_ text: String) -> [Int32] {
            let input = type?.apply(text, modelID: config.modelID ?? .embeddingGemma2) ?? text
            return tokenizer.encode(input)
        }
        if arguments.has("stdin") {
            var number = 0
            let input = LineReader(fileDescriptor: 0, maxLineBytes: config.maxBodyBytes, maxPending: 4)
            for await item in input.lines {
                input.release()
                number += 1
                let (text, id) = try parseInputLine(try stdinLine(item, number: number, limit: config.maxBodyBytes), number: number)
                let ids = encode(text)
                var record: JSONObject = [:]
                if let id { record["id"] = id }
                record["count"] = .int(ids.count)
                record["ids"] = .array(ids.map { .int(Int($0)) })
                Output.data(JSONValue.object(record).jsonString())
            }
            return .success
        }
        guard !arguments.positionals.isEmpty else { throw .badRequest("missing_input", "give texts or --stdin") }
        let items: [JSONValue] = arguments.positionals.map { text in
            let ids = encode(text)
            var record: JSONObject = ["count": .int(ids.count), "ids": .array(ids.map { .int(Int($0)) })]
            if arguments.has("show-tokens") { record["tokens"] = .array(ids.map { .string(tokenizer.tokenLabel($0)) }) }
            return .object(record)
        }
        Output.data(["data": .array(items)], format: format)
        return .success
    }
}

// MARK: - doctor

struct DoctorCommand: Command {
    let spec = ArgumentSpec(values: ["format"], flags: ["verify"])

    func run(_ arguments: ParsedArguments, config: Config, format: OutputFormat) async throws(ServiceError) -> ExitCode {
        Output.data(Doctor.report(config, verify: arguments.has("verify")), format: format)
        return .success
    }
}

// MARK: - models

struct ModelsCommand: Command {
    let spec = ArgumentSpec(values: ["format", "repo", "revision", "from"], flags: [])

    func run(_ arguments: ParsedArguments, config: Config, format: OutputFormat) async throws(ServiceError) -> ExitCode {
        guard let action = arguments.positionals.first, arguments.positionals.count == 1 else {
            throw .badRequest("missing_action", "models needs one of: path, verify, warm, install")
        }
        switch action {
        case "path":
            Output.data(config.modelDirectory.path)
            return .success
        case "verify":
            return try verify(config, format: format)
        case "warm":
            Output.data(try await Self.warm(config), format: format)
            return .success
        case "install":
            guard !config.useFakeRuntime else {
                throw .badRequest("fake_runtime", "models install compiles, warms and self-tests the model on the Neural "
                                  + "Engine; it does not run with --runtime fake", param: "runtime")
            }
            guard let model = config.modelID else {
                throw .badRequest("missing_model", "models install requires --model", param: "model")
            }
            if let from = arguments.value("from") {
                guard arguments.value("repo") == nil && arguments.value("revision") == nil else {
                    throw .badRequest("conflicting_options", "--from installs a local package; it takes no --repo or --revision")
                }
                return try await installLocal(URL(fileURLWithPath: (from as NSString).expandingTildeInPath), config: config,
                                              format: format)
            }
            let repo = arguments.value("repo")
            let revision = arguments.value("revision")
            let source: ModelSource
            if repo == nil && revision == nil {
                source = model.source
            } else {
                guard let repo, ModelSource.isValidRepository(repo),
                      let revision, revision.utf8.count == 40,
                      revision.utf8.allSatisfy({ (48...57).contains($0) || (65...70).contains($0) || (97...102).contains($0) })
                else {
                    throw ServiceError(.badRequest, "invalid_model_source",
                                       "received --repo \(repo ?? "(missing)") and --revision \(revision ?? "(missing)")",
                                       param: "revision", hint: "supply --repo OWNER/NAME and --revision with a full 40-character commit")
                }
                source = ModelSource(repository: repo, revision: revision.lowercased())
            }
            return try await install(source, model: model, config: config, format: format)
        default:
            throw .badRequest("unknown_action", "unknown models action \(action); use path, verify, warm or install")
        }
    }

    func verify(_ config: Config, format: OutputFormat) throws(ServiceError) -> ExitCode {
        let directory = config.modelDirectory
        let manifest = try Manifest.load(from: directory)
        if let model = config.modelID { try manifest.checkModel(model, directory: directory) }
        try manifest.checkPinned(config.pinnedManifestSHA256)
        try manifest.checkContract()
        let problems = manifest.verify(in: directory, hash: true) { Log.info("hashing \($0)") }
        let compiled = manifest.compiledStatus(in: directory, hash: true, maxTokens: config.maxTokens)
        let ok = problems.isEmpty && !compiled.contains(where: \.problem)
        Output.data([
            "model_dir": .string(directory.path), "revision": .string(manifest.revision), "files": .int(manifest.files.count),
            "ok": .bool(ok), "problems": .array(problems.map { ["path": .string($0.path), "issue": .string($0.issue)] }),
            "compiled": .object(JSONObject(compiled.map { (key: $0.package + ".mlmodelc", value: .string($0.status)) })),
            "compiled_note": .string("compiled models are built on this Mac and are not byte-for-byte reproducible, so "
                + "their SHA-256 is not checked; the load-time self-test checks what they compute"),
        ], format: format)
        return ok ? .success : .dataError
    }

    /// Loads (compiling on first use) and self-tests every function; records the warm state.
    static func warm(_ config: Config, requireCap: Bool = false) async throws(ServiceError) -> JSONValue {
        let started = Date()
        Log.info("loading \(config.modelDirectory.path) (the first time compiles for the Neural Engine; this can take minutes)")
        let (service, report) = try await ModelLoader.load(config, log: Log.info)
        do { try await service.runtime.warm() } catch {
            throw ServiceError(.unavailable, "warm_failed", "\(error)")
        }
        let seconds = Date().timeIntervalSince(started)
        let state = WarmState(osBuild: operatingSystemBuild, binaryVersion: aneembedVersion, manifestSHA256: report.manifestSHA256,
                              maxTokens: service.maxTokens, functions: service.runtime.info.functions, warmedAt: Date(),
                              seconds: (seconds * 10).rounded() / 10)
        if !config.useFakeRuntime, config.modelID?.supportsImages == true {
            do { try state.save(to: config.modelDirectory) } catch {
                Log.info("could not record the warm state: \(error.localizedDescription)")
            }
        }
        if requireCap && service.maxTokens < config.maxTokens {
            throw ServiceError(.unavailable, "selftest_failed",
                               "installed candidate serves \(service.maxTokens) tokens, requested \(config.maxTokens)",
                               hint: "inspect the self-test log or retry with a verified package and --max-tokens \(service.maxTokens)")
        }
        var output = state.json
        if case .object(var object) = output {
            object["functions"] = .int(state.functions.count)
            if let selfTest = report.selfTest { object["selftest"] = .string(selfTest.summary) }
            output = .object(object)
        }
        return output
    }

    func install(_ source: ModelSource, model: ModelID, config: Config, format: OutputFormat)
        async throws(ServiceError) -> ExitCode {
        guard source.isPublished else {
            throw ServiceError(.unavailable, "model_not_published",
                               "no published revision of \(model.rawValue) is pinned in this build",
                               hint: "run anerun models install --model \(model.rawValue) --from DIR, or supply --repo and --revision")
        }
        let store = ModelStore(dataDirectory: config.dataDirectory, model: model)
        let target = try store.directory(for: source.revision)
        let lock = try Self.lock(store)
        defer { lock.release() }
        ModelInstaller.removeStaleStaging(for: target)
        let started = Date()
        let installed = try await ModelDownloader(source: source).install(into: target, store: store)
        let manifest = try Manifest.load(from: installed)
        return try await finishInstall(store: store, preferred: target, target: installed, manifest: manifest,
                                       fetched: Date().timeIntervalSince(started), config: config, format: format)
    }

    func installLocal(_ source: URL, config: Config, format: OutputFormat) async throws(ServiceError) -> ExitCode {
        let manifest = try Manifest.load(from: source)
        guard let model = config.modelID else {
            throw .badRequest("missing_model", "models install requires --model", param: "model")
        }
        try manifest.checkModel(model, directory: source)
        try manifest.checkContract()
        let store = ModelStore(dataDirectory: config.dataDirectory, model: model)
        let revision = ModelInstaller.localRevision(manifest)
        let target = try store.directory(for: revision)
        let lock = try Self.lock(store)
        defer { lock.release() }
        ModelInstaller.removeStaleStaging(for: target)
        let started = Date()
        let candidate = try ModelInstaller.installCandidate(target, in: store, manifestSHA256: manifest.sha256)
        let installed = candidate.directory
        if candidate.verified {
            Log.info("\(installed.lastPathComponent) is already installed and verified")
        } else {
            Log.info("installing \(source.path) as \(installed.lastPathComponent) (manifest sha256 \(manifest.sha256))")
            _ = try ModelInstaller.copy(from: source, into: installed, store: store, manifest: manifest, log: Log.info)
        }
        return try await finishInstall(store: store, preferred: target, target: installed, manifest: manifest,
                                       fetched: Date().timeIntervalSince(started), config: config, format: format)
    }

    /// The install lock: a second `models install` waits for the first one (download, compile, warm, switch).
    static func lock(_ store: ModelStore) throws(ServiceError) -> InstallLock {
        try InstallLock.acquire(in: store) { Log.info("another `anerun models install` is running; waiting for it") }
    }

    /// Compile, warm (with the self-test) and switch `current` to the new revision.
    func finishInstall(store: ModelStore, preferred: URL, target: URL, manifest: Manifest, fetched: Double, config: Config,
                       format: OutputFormat, rebuild: Bool = false) async throws(ServiceError) -> ExitCode
    {
        let revision = target.lastPathComponent
        let compileStarted = Date()
        let differing: [Manifest.Problem]
        let compiled: TimeInterval
        let warm: JSONValue
        do throws(ServiceError) {
            differing = try await ModelInstaller.compile(in: target, maxTokens: config.maxTokens, rebuild: rebuild, log: Log.info)
            compiled = Date().timeIntervalSince(compileStarted)
            if !differing.isEmpty {
                Log.info("note: \(differing.count) compiled files differ from the packager's compile (another Core ML compiler); "
                    + "the self-test checks the result")
            }
            warm = try await Self.warm(config.withModelDirectory(target), requireCap: config.modelID?.supportsImages == false)
        } catch {
            guard !rebuild, !Task.isCancelled,
                  ["compile_failed", "model_load_failed", "selftest_failed", "warm_failed"].contains(error.reason)
            else { throw error }
            let started = Date()
            guard ModelInstaller.isInstalled(at: target, manifestSHA256: manifest.sha256) else { throw error }
            if !store.isActive(target) {
                Log.info("rebuilding the compiled cache for \(revision) after \(error.reason)")
                return try await finishInstall(store: store, preferred: preferred, target: target, manifest: manifest,
                                               fetched: fetched + Date().timeIntervalSince(started), config: config,
                                               format: format, rebuild: true)
            }
            let repair = try ModelInstaller.repairCandidate(for: preferred, in: store, manifestSHA256: manifest.sha256)
            Log.info("retrying \(revision) at \(repair.directory.path) after \(error.reason)")
            if !repair.verified {
                _ = try ModelInstaller.copy(from: target, into: repair.directory, store: store, manifest: manifest, log: Log.info)
            }
            return try await finishInstall(store: store, preferred: preferred, target: repair.directory, manifest: manifest,
                                           fetched: fetched + Date().timeIntervalSince(started), config: config, format: format)
        }
        guard !Task.isCancelled else {
            throw ServiceError(.loading, "install_cancelled", "installation cancelled before activating \(revision)",
                               hint: "retry anerun models install")
        }
        do { try store.activate(revision: revision) } catch {
            throw ServiceError(.internal, "activate_failed", "cannot switch to \(revision): \(error.localizedDescription)")
        }
        func round(_ seconds: Double) -> JSONValue { .number((seconds * 10).rounded() / 10) }
        Output.data([
            "model_dir": .string(target.path), "revision": .string(revision),
            "manifest_sha256": .string((try? Manifest.load(from: target))?.sha256 ?? ""),
            "seconds": ["fetch_and_verify": round(fetched), "compile": round(compiled)],
            "compiled_matches_manifest": .bool(differing.isEmpty), "warm": warm,
        ], format: format)
        return .success
    }
}

// MARK: - bench

struct BenchCommand: Command {
    let spec = ArgumentSpec(values: ["format", "sizes", "repeat", "image", "concurrency", "settle"], flags: ["buckets", "breakdown"])

    func run(_ arguments: ParsedArguments, config: Config, format: OutputFormat) async throws(ServiceError) -> ExitCode {
        var options = Bench.Options()
        if let sizes = arguments.value("sizes") {
            let parts = sizes.split(separator: ",", omittingEmptySubsequences: false)
            let parsed = parts.compactMap { Int($0) }
            let model = config.modelID ?? .embeddingGemma2
            let range = model.supportsImages ? 3...8192 : model.tokenRange
            guard !parsed.isEmpty, parsed.count == parts.count, parsed.allSatisfy(range.contains) else {
                throw .badRequest("invalid_sizes", "--sizes \(sizes): expected token counts in \(range), comma separated",
                                  param: "sizes")
            }
            options.sizes = parsed
        }
        func integer(_ name: String, _ range: ClosedRange<Int>) throws(ServiceError) -> Int? {
            guard let value = arguments.value(name) else { return nil }
            guard let number = Int(value), range.contains(number) else {
                throw .badRequest("invalid_\(name)", "--\(name) must be in \(range.lowerBound)...\(range.upperBound)", param: name)
            }
            return number
        }
        options.repeats = try integer("repeat", 1...1000) ?? options.repeats
        options.concurrency = try integer("concurrency", 1...256) ?? 0
        options.settleSeconds = Double(try integer("settle", 0...3600) ?? 0)
        if let path = arguments.value("image") {
            if let model = config.modelID, !model.supportsImages {
                throw ServiceError(.invalidInput, "images_unsupported", "\(model.rawValue) accepts only text",
                                   param: "image", hint: "use --model embeddinggemma-2 to benchmark an image")
            }
            options.image = try readFile(path)
        }
        options.buckets = arguments.has("buckets")
        options.breakdown = arguments.has("breakdown")
        Output.data(try await Bench.run(config, options: options, log: Log.info), format: format)
        return .success
    }
}

// MARK: - serve / mcp

struct ServeCommand: Command {
    let spec = ArgumentSpec(values: ["format"])

    func run(_ arguments: ParsedArguments, config: Config, format: OutputFormat) async throws(ServiceError) -> ExitCode {
        let state = ServerState()
        let server = HTTPServer(host: config.host, port: config.port, maxBody: config.maxBodyBytes,
                                maxConnections: config.maxConnections,
                                router: Router(state: state, modelName: config.modelName,
                                               modelID: config.modelID ?? .embeddingGemma2,
                                               configuredMaxTokens: config.maxTokens, log: { Log.info($0) }))
        try server.start()
        if !["127.0.0.1", "::1", "localhost"].contains(config.host) {
            Log.info("warning: listening on \(config.host) without authentication; anyone who can reach this port can use it")
        }
        await load(into: state, config: config)
        return await server.wait()
    }
}

/// Loads and warms the model in the background of a long-running server, then flips `state` to ready.
func load(into state: ServerState, config: Config) async {
    let started = Date()
    do throws(ServiceError) {
        let directory = config.modelDirectory
        if !config.useFakeRuntime {
            let manifest = try? Manifest.load(from: directory)
            let required = manifest?.pplxContract?.requiredFunctions(for: config.maxTokens).map(\.function) ?? []
            let compiled = manifest?.compiledStatus(in: directory, maxTokens: config.maxTokens) ?? []
            if let warm = WarmState.load(from: directory), !compiled.contains(where: \.problem),
                warm.covers(osBuild: operatingSystemBuild, maxTokens: config.maxTokens, manifestSHA256: manifest?.sha256 ?? "",
                            requiredFunctions: required)
            {
                Log.info("loading \(directory.path) (warm for this macOS build)")
            } else {
                Log.info("loading \(directory.path): not warmed for this package, macOS build and max tokens, compiling first (minutes)")
            }
        }
        let (service, report) = try await ModelLoader.load(config, log: Log.info)
        do { try await service.runtime.warm() } catch {
            throw ServiceError(.unavailable, "warm_failed", "\(error)")
        }
        if !config.useFakeRuntime, config.modelID?.supportsImages == true,
            service.maxTokens < config.maxTokens
                || !(WarmState.load(from: directory)?.covers(osBuild: operatingSystemBuild, maxTokens: service.maxTokens,
                                                      manifestSHA256: report.manifestSHA256,
                                                      requiredFunctions: service.runtime.info.functions) ?? false)
        {
            let warm = WarmState(osBuild: operatingSystemBuild, binaryVersion: aneembedVersion, manifestSHA256: report.manifestSHA256,
                                 maxTokens: service.maxTokens, functions: service.runtime.info.functions,
                                 warmedAt: Date(), seconds: (Date().timeIntervalSince(started) * 10).rounded() / 10)
            try? warm.save(to: directory)
        }
        state.set(.ready(service))
        Log.info(String(format: "ready in %.1fs: max %d tokens, images %@, backend %@", Date().timeIntervalSince(started),
                        service.maxTokens, service.maxImageTokens > 0 ? "on" : "off", service.runtime.info.backend))
    } catch {
        Log.info("model failed to load: \(error.message)")
        state.set(.failed(error))
    }
}

/// A line of `--stdin` input, or the error for one over `--max-body`.
func stdinLine(_ item: InputLine, number: Int, limit: Int) throws(ServiceError) -> Data {
    switch item {
    case .line(let line): return line
    case .tooLong(let bytes):
        throw ServiceError(.payloadTooLarge, "line_too_long", "line \(number) has \(bytes) bytes; the limit is \(limit) (--max-body)",
                           param: "stdin")
    }
}

struct MCPCommand: Command {
    let spec = ArgumentSpec()
    /// Messages read ahead or being handled at once; stdin is not read further until one finishes.
    static let maxInFlight = 16

    func run(_ arguments: ParsedArguments, config: Config, format: OutputFormat) async throws(ServiceError) -> ExitCode {
        let state = ServerState()
        let loader = Task { await load(into: state, config: config) }
        let handler = MCPHandler(state: state, allowPaths: true, waitForModel: true,
                                 modelID: config.modelID ?? .embeddingGemma2,
                                 configuredMaxTokens: config.maxTokens)
        let input = LineReader(fileDescriptor: 0, maxLineBytes: config.maxBodyBytes, maxPending: Self.maxInFlight)
        await withDiscardingTaskGroup { group in
            for await item in input.lines {
                switch item {
                case .line(let line) where line.isEmpty:
                    input.release()
                case .line(let line):
                    group.addTask {
                        defer { input.release() }
                        guard let response = await handler.handle(raw: line) else { return }
                        Output.write(response.jsonString() + "\n", to: FileHandle.standardOutput)
                    }
                case .tooLong(let bytes):
                    input.release()
                    let error = MCPHandler.messageTooLarge(bytes: bytes, limit: config.maxBodyBytes)
                    Output.write(error.jsonString() + "\n", to: FileHandle.standardOutput)
                }
            }
        }
        loader.cancel()
        return .success
    }
}
