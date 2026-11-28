/// The supported Qwen3-family manifest ABI and the ordered set of functions needed to serve a token cap.
public struct PPLXPackageContract: Sendable, Equatable {
    public enum Role: String, Sendable, Equatable {
        case packed, fused, pre, post, attention
        case fusedPart = "fused_part"
    }

    public struct Tensor: Sendable, Equatable {
        public enum DType: String, Sendable, Equatable {
            case fp16, fp32
        }

        public let name: String
        public let shape: [Int]
        public let dtype: DType

        fileprivate init(_ name: String, _ shape: [Int], _ dtype: DType = .fp16) {
            self.name = name
            self.shape = shape
            self.dtype = dtype
        }
    }

    public struct Function: Sendable, Equatable {
        public let role: Role
        /// The bare package name, with no path or .mlpackage suffix.
        public let package: String
        public let function: String
        public let length: Int?
        public let layer: Int?
        public let part: Int?
        public let inputs: [Tensor]
        public let outputs: [Tensor]

        /// Single-function packages use the Core ML default function.
        public var runtimeFunctionName: String? { function == package ? nil : function }

        fileprivate init(_ role: Role, _ function: String, package: String? = nil, length: Int? = nil,
                         layer: Int? = nil, part: Int? = nil, inputs: [Tensor], outputs: [Tensor]) {
            self.role = role
            self.package = package ?? function
            self.function = function
            self.length = length
            self.layer = layer
            self.part = part
            self.inputs = inputs
            self.outputs = outputs
        }
    }

    public let model: ModelID
    public let revision: String
    public let created: String

    public static let requiredAssets: [String] = [
        "tokenizer.json", "tokenizer_config.json", "embeddings.f32", "final_norm.f32",
        "selftest.json", "selftest.bin", "LICENSE", "NOTICE",
    ]

    public static func parse(_ root: JSONValue, files: [ManifestEntry]) throws(ServiceError) -> Self {
        let keys: Set<String> = ["contract_version", "model", "revision", "created", "packages", "functions",
                                 "files", "source_model", "converter"]
        guard let envelope = root.objectValue, hasOnlyKeys(envelope, allowed: keys),
              envelope["contract_version"]?.intValue == 1,
              let model = envelope["model"]?.stringValue.flatMap(ModelID.init(rawValue:)),
              model == .pplx || model == .qwen3Embedding,
              let revision = envelope["revision"]?.stringValue, revision.utf8.count == 40,
              revision.allSatisfy({ $0.asciiValue != nil && $0.isHexDigit }),
              let created = envelope["created"]?.stringValue, created.contains(where: { !$0.isWhitespace })
        else { throw invalid("expected a supported version 1 text model, a source commit SHA and creation text") }

        // Manifest owns the file list. The source revision describes the same
        // snapshot in the contract and its producer metadata.
        if let fileList = envelope["files"], fileList.arrayValue == nil { throw invalid("files must be an array") }
        guard let source = envelope["source_model"]?.objectValue,
              nonemptyText(source["id"]), nonemptyText(source["license"]),
              source["revision"]?.stringValue?.lowercased() == revision.lowercased()
        else {
            throw invalid("source_model must identify the source, matching revision and license")
        }
        guard let converter = envelope["converter"]?.objectValue,
              nonemptyText(converter["script"]), nonemptyText(converter["precision"])
        else { throw invalid("converter must identify the build script and precision") }

        let groups = model == .qwen3Embedding ? qwenGroups : groups
        let functions = groups.flatMap(\.functions)
        let packageNames = Set(functions.map(\.package))
        var remainingPackages = packageNames
        guard let packages = envelope["packages"]?.arrayValue, packages.count == remainingPackages.count else {
            throw invalid("packages must list all \(remainingPackages.count) package names")
        }
        for entry in packages {
            guard let name = entry.stringValue, remainingPackages.remove(name) != nil else {
                throw invalid("packages contains an invalid, unknown or duplicate package name")
            }
        }

        var remainingFunctions = Dictionary(uniqueKeysWithValues: functions.map { ($0.function, $0) })
        guard let declared = envelope["functions"]?.arrayValue, declared.count == remainingFunctions.count else {
            throw invalid("functions must list all \(remainingFunctions.count) functions")
        }
        for entry in declared {
            guard let function = entry.objectValue, function.pairs.count <= 7,
                  let name = function["function"]?.stringValue,
                  let expected = remainingFunctions.removeValue(forKey: name)
            else { throw invalid("functions contains an invalid, unknown or duplicate function") }
            try checkFunction(function, expected: expected)
        }
        try checkFiles(files, packages: packageNames, model: model)
        return Self(model: model, revision: revision, created: created)
    }

    private static func nonemptyText(_ value: JSONValue?) -> Bool {
        value?.stringValue?.contains(where: { !$0.isWhitespace }) == true
    }

    /// Include every shorter group and stop after the first group that covers the cap.
    /// Callers validate the configured cap against the model's supported range.
    public func requiredFunctions(for maxTokens: Int) -> [Function] {
        var functions: [Function] = []
        for group in model == .qwen3Embedding ? Self.qwenGroups : Self.groups {
            functions += group.functions
            if maxTokens <= group.length { break }
        }
        return functions
    }

    public func requiredPackages(for maxTokens: Int) -> [String] {
        Self.packages(for: requiredFunctions(for: maxTokens))
    }

    private static func checkFunction(_ function: JSONObject, expected: Function) throws(ServiceError) {
        var keys: Set<String> = ["role", "package", "function", "inputs", "outputs"]
        if expected.length != nil { keys.insert("length") }
        if expected.layer != nil { keys.insert("layer") }
        if expected.part != nil { keys.insert("part") }
        guard function.pairs.count == keys.count, hasOnlyKeys(function, allowed: keys),
              function["role"]?.stringValue == expected.role.rawValue,
              function["package"]?.stringValue == expected.package,
              function["length"]?.intValue == expected.length,
              function["layer"]?.intValue == expected.layer,
              function["part"]?.intValue == expected.part
        else { throw invalid("\(expected.function) has incompatible role, package or routing fields") }

        try checkTensors(function["inputs"], expected: expected.inputs, context: "\(expected.function) inputs")
        try checkTensors(function["outputs"], expected: expected.outputs, context: "\(expected.function) outputs")
    }

    private static func checkTensors(_ value: JSONValue?, expected: [Tensor], context: String) throws(ServiceError) {
        guard let tensors = value?.arrayValue, tensors.count == expected.count else {
            throw invalid("\(context) must contain \(expected.count) tensors")
        }
        var remaining = Dictionary(uniqueKeysWithValues: expected.map { ($0.name, $0) })
        for value in tensors {
            guard let tensor = value.objectValue, tensor.pairs.count == 3,
                  hasOnlyKeys(tensor, allowed: ["name", "shape", "dtype"]),
                  let name = tensor["name"]?.stringValue, let expectedTensor = remaining.removeValue(forKey: name)
            else { throw invalid("\(context) contains a malformed, unknown or duplicate tensor") }

            guard tensor["dtype"]?.stringValue == expectedTensor.dtype.rawValue,
                  let shape = tensor["shape"]?.arrayValue, shape.count == expectedTensor.shape.count,
                  shape.elementsEqual(expectedTensor.shape, by: { $0.intValue == $1 })
            else { throw invalid("\(context) tensor \(name) must be \(expectedTensor.dtype.rawValue) \(expectedTensor.shape)") }
        }
    }

    private static func checkFiles(_ files: [ManifestEntry], packages: Set<String>, model: ModelID) throws(ServiceError) {
        var sizes: [String: Int] = [:]
        var packagesWithPayload: Set<String> = []
        let directories = Set(packages.map { $0 + ".mlpackage" })
        for entry in files {
            guard Manifest.isValidPath(entry.path), entry.size >= 0,
                  sizes.updateValue(entry.size, forKey: entry.path) == nil
            else { throw invalid("invalid or duplicate file entry \(entry.path)") }

            let components = entry.path.split(separator: "/", maxSplits: 1)
            let directory = String(components[0])
            if directory.lowercased().hasSuffix(".mlpackage") {
                guard directories.contains(directory), components.count == 2 else {
                    throw invalid("unexpected source package entry \(entry.path)")
                }
                if components[1].hasPrefix("Data/com.apple.CoreML/") {
                    packagesWithPayload.insert(directory)
                }
            }
        }

        for asset in requiredAssets where sizes[asset] == nil {
            throw invalid("missing required asset \(asset)")
        }
        for (path, bytes) in [("embeddings.f32", (model == .qwen3Embedding ? 151669 : 151936) * 4096), ("final_norm.f32", 4096)] {
            guard sizes[path] == bytes else { throw invalid("\(path) must be \(bytes) bytes") }
        }
        for package in packages {
            let directory = package + ".mlpackage"
            guard let size = sizes[directory + "/Manifest.json"], size > 0 else {
                throw invalid("\(directory)/Manifest.json must be listed and nonempty")
            }
            guard packagesWithPayload.contains(directory) else {
                throw invalid("\(directory) must list a file under Data/com.apple.CoreML/")
            }
        }
    }

    private static func hasOnlyKeys(_ object: JSONObject, allowed: Set<String>) -> Bool {
        guard object.pairs.count <= allowed.count else { return false }
        let keys = Set(object.keys)
        return keys.count == object.pairs.count && keys.isSubset(of: allowed)
    }

    private static func invalid(_ message: String) -> ServiceError {
        ServiceError(.unavailable, "invalid_manifest", "text manifest.json: \(message)")
    }

    private static let textPackage = "PPLXText"
    private static let hidden = 1024
    private static let queryRows = 2048

    private struct Group {
        let length: Int
        let functions: [Function]
    }

    private static let groups: [Group] = [
        Group(length: 64, functions: [pooled(64, role: .packed)]),
        Group(length: 512, functions: [pooled(512, role: .packed)]),
        Group(length: 1024, functions: [pooled(1024, role: .packed)]),
        Group(length: 2048, functions: [pooled(2048, role: .fused)]),
        Group(length: 4096, functions: [pooled(4096, role: .fused)]),
        Group(length: 8192, functions: (0..<4).map { partition(8192, part: $0) }),
        Group(length: 16384, functions: layerFunctions + [attention(16384)]),
        Group(length: 32768, functions: [attention(32768)]),
    ]

    private static let qwenGroups = [
        Group(length: 64, functions: [pooled(64, role: .packed, package: "embed_64")]),
        Group(length: 2048, functions: layerFunctions + [attention(2048)]),
        Group(length: 8192, functions: [attention(8192)]),
        Group(length: 32768, functions: [attention(32768)]),
    ]

    private static func packages(for functions: [Function]) -> [String] {
        var seen: Set<String> = []
        return functions.map(\.package).filter { seen.insert($0).inserted }
    }

    private static func pooled(_ length: Int, role: Role, package: String = textPackage) -> Function {
        let packed = role == .packed
        let slots = packed ? max(8, length / 32) : 1
        let name = (packed ? "embed_" : "fused_") + String(length)
        return Function(role, name, package: package, length: length, inputs: [
            Tensor("inputs_embeds", [1, length, hidden]),
            Tensor("bias", [1, 1, packed ? length : 1, length]),
            Tensor("pool", [slots, length]),
        ], outputs: [Tensor("pooled", [slots, hidden], .fp32)])
    }

    private static func partition(_ length: Int, part: Int) -> Function {
        let shape = [1, length, hidden]
        var inputs = [Tensor(part == 0 ? "inputs_embeds" : "hidden", shape), Tensor("bias", [1, 1, 1, length])]
        if part == 3 { inputs.append(Tensor("pool", [1, length])) }
        let output = part == 3 ? Tensor("pooled", [1, hidden], .fp32) : Tensor("hidden_out", shape)
        return Function(.fusedPart, "fused_\(length)_part\(part)", package: textPackage, length: length, part: part,
                        inputs: inputs, outputs: [output])
    }

    private static let layerFunctions: [Function] = (0..<28).flatMap { layer in
        let hiddenShape = [1, queryRows, hidden]
        let queryShape = [1, queryRows, 2048]
        let xs = Tensor("xs", hiddenShape)
        return [
            Function(.pre, "pre\(layer)", layer: layer, inputs: [xs], outputs: [
                Tensor("q", queryShape), Tensor("k", hiddenShape), Tensor("v", hiddenShape),
            ]),
            Function(.post, "post\(layer)", layer: layer, inputs: [xs, Tensor("o", queryShape)], outputs: [
                Tensor("a", hiddenShape), Tensor("d", hiddenShape),
            ]),
        ]
    }

    private static func attention(_ length: Int) -> Function {
        let queryShape = [1, 16, queryRows, 128]
        let keyShape = [1, 8, length, 128]
        return Function(.attention, "attn_S\(length)_Q\(queryRows)", length: length, inputs: [
            Tensor("q", queryShape), Tensor("k", keyShape), Tensor("v", keyShape), Tensor("bias", [1, 1, 1, length]),
        ], outputs: [Tensor("o", queryShape)])
    }
}
