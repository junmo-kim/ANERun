import Foundation
import Testing

@testable import EmbedCore

@Suite("PPLX package contract")
struct PPLXPackageContractTests {
    @Test func selectsEveryGroupThroughTheFirstCoveringLength() throws {
        let contract = try PPLXPackageContract.parse(.object(Self.root), files: Self.files)
        let partitions = (0..<4).map { "fused_8192_part\($0)" }
        let shared = (0..<28).flatMap { ["pre\($0)", "post\($0)"] }
        let groups = [
            ["embed_64"], ["embed_512"], ["embed_1024"], ["fused_2048"], ["fused_4096"], partitions,
            shared + ["attn_S16384_Q2048"], ["attn_S32768_Q2048"],
        ]
        let boundaries = [(1, 64), (65, 512), (513, 1024), (1025, 2048), (2049, 4096),
                          (4097, 8192), (8193, 16384), (16385, 32768)]
        for (index, boundary) in boundaries.enumerated() {
            for cap in [boundary.0, boundary.1] {
                let expectedFunctions = groups.prefix(index + 1).flatMap { $0 }
                let functions = contract.requiredFunctions(for: cap)
                #expect(functions.map(\.function) == expectedFunctions, "cap \(cap)")
                #expect(functions.map(\.runtimeFunctionName) == expectedFunctions.map {
                    Self.fusedNames.contains($0) ? $0 : nil
                })
                #expect(contract.requiredPackages(for: cap) ==
                        ["PPLXText"] + expectedFunctions.filter { !Self.fusedNames.contains($0) })
            }
        }
        #expect(Set(PPLXPackageContract.requiredAssets) == Set(Self.assets))
        #expect(PPLXPackageContract.requiredAssets.count == Self.assets.count)
    }

    @Test func canonicalPlanIgnoresManifestOrder() throws {
        let baseline = try PPLXPackageContract.parse(.object(Self.root), files: Self.files)
        var root = Self.root
        root["packages"] = .array(Self.packages.reversed().map(JSONValue.string))
        root["functions"] = .array(Self.functions.reversed().map { function in
            var function = function
            function["inputs"] = .array(Array(function["inputs"]!.arrayValue!.reversed()))
            function["outputs"] = .array(Array(function["outputs"]!.arrayValue!.reversed()))
            return .object(function)
        })
        #expect(try PPLXPackageContract.parse(.object(root), files: Array(Self.files.reversed())) == baseline)
    }

    @Test(arguments: [16384, 32768])
    func compiledAttentionRequiresANonemptyConstantBlob(length: Int) throws {
        let files = FileManager.default
        let directory = files.temporaryDirectory.appendingPathComponent("aneembed-compiled-\(UUID().uuidString)")
        defer { try? files.removeItem(at: directory) }
        let package = "attn_S\(length)_Q2048"
        let compiled = directory.appendingPathComponent(package + ".mlmodelc")
        let weights = compiled.appendingPathComponent("weights")
        try files.createDirectory(at: weights, withIntermediateDirectories: true)
        let placeholder = Data([1])
        try placeholder.write(to: compiled.appendingPathComponent("model.mil"))
        try placeholder.write(to: compiled.appendingPathComponent("coremldata.bin"))
        let blob = weights.appendingPathComponent("weight.bin")

        let manifest = try Manifest.parse(Self.descriptor)

        func status() throws -> (package: String, status: String, problem: Bool) {
            try #require(manifest.compiledStatus(in: directory, maxTokens: length).first { $0.package == package })
        }
        let missing = try status()
        #expect(missing.problem && missing.status.contains("weights/weight.bin"))
        try Data().write(to: blob)
        #expect(try status().problem)
        try placeholder.write(to: blob)
        let complete = try status()
        #expect(!complete.problem && complete.status == "compiled")
    }

    @Test func plainDoctorUsesTheEmbeddedReportForItsRequestedCap() throws {
        let files = FileManager.default
        let directory = files.temporaryDirectory.appendingPathComponent("aneembed-pplx-doctor-\(UUID().uuidString)")
        try files.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? files.removeItem(at: directory) }
        try Self.descriptor.write(to: directory.appendingPathComponent(ModelLayout.manifest))
        let config = try Config(flags: ["model-dir": directory.path, "max-tokens": "2048", "runtime": "fake"], environment: [:])
            .selectingModel(for: "doctor")
        let short = SelfTestResult(function: "embed_64", kind: "pplx_64", passed: true, cosine: 1)
        let wide = SelfTestResult(function: "attn_S32768_Q2048", kind: "pplx_32768", passed: false, cosine: 0)
        let test = SelfTestReport(osBuild: operatingSystemBuild, ranAt: Date(), seconds: 0, results: [short, wide])
        try WarmState(osBuild: test.osBuild, binaryVersion: "test", manifestSHA256: SHA256Hasher.hex(Self.descriptor),
                      maxTokens: 2048, functions: [short.function], warmedAt: Date(), seconds: 0, selfTest: test)
            .save(to: directory)
        // A report left by an earlier build is independent of the new snapshot.
        var legacy = test
        legacy.results = [SelfTestResult(function: "embed_64", kind: "legacy", passed: false, cosine: 0)]
        try legacy.save(to: directory)

        let plain = Doctor.report(config)["selftest"]
        #expect(plain?["summary"] == "1/1 passed")
        #expect(plain?["failed"] == nil)
        let full = Doctor.report(config, verify: true)["selftest"]
        #expect(full?["failed"]?.arrayValue == [wide.json])
    }

    @Test func rejectsMalformedEnvelopeAndPackageLists() {
        Self.reject([])
        for (key, value): (String, JSONValue?) in [
            ("contract_version", nil), ("contract_version", 2), ("contract_version", "1"),
            ("model", "embeddinggemma-2"), ("revision", "moving-tag"), ("revision", .string(String(repeating: "Ｆ", count: 40))),
            ("created", nil), ("created", ""), ("created", 20261118), ("unknown", true),
            ("packages", .array(Self.packages.dropLast().map(JSONValue.string))),
            ("packages", .array((Self.packages + ["PPLXText"]).map(JSONValue.string))),
            ("packages", .array((["PPLXText.mlpackage"] + Self.packages.dropFirst()).map(JSONValue.string))),
            ("packages", .array((["../PPLXText"] + Self.packages.dropFirst()).map(JSONValue.string))),
        ] {
            var root = Self.root
            root[key] = value
            Self.reject(.object(root), label: key)
        }
        var duplicate = Self.root
        duplicate.pairs.append((key: "model", value: "pplx-embed-v1-0.6b"))
        Self.reject(.object(duplicate), label: "duplicate envelope key")
    }

    @Test func rejectsMissingOrInvalidProducerProvenance() {
        for key in ["source_model", "converter"] {
            for value: JSONValue? in [nil, .null, "producer", [:]] {
                var root = Self.root
                root[key] = value
                Self.reject(.object(root), label: key)
            }
        }
        for (key, field, value): (String, String, JSONValue?) in [
            ("source_model", "id", " "), ("source_model", "revision", .string(String(repeating: "a", count: 40))),
            ("source_model", "license", nil), ("converter", "script", nil), ("converter", "precision", 16),
        ] {
            var root = Self.root
            var object = root[key]!.objectValue!
            object[field] = value
            root[key] = .object(object)
            Self.reject(.object(root), label: "\(key).\(field)")
        }
    }

    @Test func rejectsMissingDuplicateAndSurplusFunctions() {
        var root = Self.root
        root["functions"] = .array(Self.functions.dropLast().map(JSONValue.object))
        Self.reject(.object(root), label: "missing attention function")
        root["functions"] = .array((Self.functions.dropLast() + [Self.functions[0]]).map(JSONValue.object))
        Self.reject(.object(root), label: "duplicate packed function")
        root["functions"] = .array((Self.functions + [Self.functions[0]]).map(JSONValue.object))
        Self.reject(.object(root), label: "surplus function")
    }

    @Test func rejectsIncompatibleRolesAndRoleSpecificFields() {
        for (index, field, value): (Int, String, JSONValue?) in [
            (0, "role", "fused"), (0, "role", "future"), (0, "function", "embed_128"), (0, "package", "pre0"),
            (0, "length", 65), (0, "length", 64.5), (0, "length", nil), (0, "length", .number(1e300)),
            (0, "part", 0), (0, "layer", .null), (0, "extra", 0),
            (5, "part", nil), (5, "part", 1), (5, "layer", 0),
            (9, "layer", nil), (9, "layer", 28), (9, "length", 2048), (9, "part", 0),
            (10, "layer", .null), (10, "package", "post1"), (65, "length", 32768), (65, "layer", 0),
        ] {
            var function = Self.functions[index]
            function[field] = value
            Self.rejectReplacingFunction(at: index, with: function, label: field)
        }
        var duplicate = Self.functions[0]
        duplicate.pairs.append((key: "role", value: "packed"))
        Self.rejectReplacingFunction(at: 0, with: duplicate, label: "duplicate function key")
    }

    @Test func rejectsIncompatibleTensors() {
        for (index, direction, tensorIndex, field, value): (Int, String, Int, String, JSONValue?) in [
            (0, "inputs", 0, "shape", [1, 64, 512]), (0, "inputs", 1, "shape", [1, 1, 1, 64]),
            (0, "inputs", 2, "name", "bias"), (0, "inputs", 1, "dtype", "fp32"),
            (0, "inputs", 0, "shape", [1, 64, 1024, 1]), (0, "inputs", 0, "shape", [1, -64, 1024]),
            (0, "inputs", 0, "shape", [1, .number(1e300), 1024]), (0, "inputs", 0, "shape", [1, 64.5, 1024]),
            (0, "inputs", 0, "shape", [1, "64", 1024]), (0, "inputs", 0, "shape", nil),
            (0, "inputs", 0, "extra", 0), (0, "outputs", 0, "dtype", "float32"),
            (2, "inputs", 2, "shape", [8, 1024]), (3, "outputs", 0, "dtype", "fp16"),
            (5, "outputs", 0, "name", "hidden"), (6, "inputs", 0, "name", "inputs_embeds"),
            (8, "inputs", 2, "shape", [8, 8192]), (9, "outputs", 0, "shape", [1, 2048, 1024]),
            (10, "inputs", 1, "shape", [1, 2048, 1024]), (65, "inputs", 1, "shape", [1, 16, 16384, 128]),
            (66, "outputs", 0, "dtype", "fp32"),
        ] {
            var function = Self.functions[index]
            var tensors = function[direction]!.arrayValue!
            var tensor = tensors[tensorIndex].objectValue!
            tensor[field] = value
            tensors[tensorIndex] = .object(tensor)
            function[direction] = .array(tensors)
            Self.rejectReplacingFunction(at: index, with: function, label: "\(direction).\(field)")
        }
        var function = Self.functions[0]
        let inputs = function["inputs"]!.arrayValue!
        function["inputs"] = .array(Array(inputs.dropLast()))
        Self.rejectReplacingFunction(at: 0, with: function, label: "missing pool")
        function["inputs"] = .array(inputs + [inputs[0]])
        Self.rejectReplacingFunction(at: 0, with: function, label: "surplus input")
        var duplicate = inputs[0].objectValue!
        duplicate.pairs.append((key: "dtype", value: "fp16"))
        function["inputs"] = .array([.object(duplicate)] + inputs.dropFirst())
        Self.rejectReplacingFunction(at: 0, with: function, label: "duplicate tensor key")
    }

    @Test func rejectsMissingAssetsAndInvalidSourcePackageEntries() {
        for asset in Self.assets {
            Self.reject(.object(Self.root), files: Self.files.filter { $0.path != asset }, label: asset)
        }
        for (path, size) in [("embeddings.f32", 622_329_855), ("final_norm.f32", 4_097),
                             ("attn_S32768_Q2048.mlpackage/Manifest.json", 0)] {
            let files = Self.files.map { $0.path == path ? Self.entry(path, size: size) : $0 }
            Self.reject(.object(Self.root), files: files, label: path)
        }

        let model = "attn_S32768_Q2048.mlpackage/Data/com.apple.CoreML/model.mlmodel"
        let withoutModel = Self.files.filter { $0.path != model }
        Self.reject(.object(Self.root), files: withoutModel, label: "missing package payload")
        for path in ["attn_S32768_Q2048.mlmodelc/model.mil",
                     "attn_S32768_Q2048.mlpackage/Data/com.apple.CoreMLish/model.mlmodel",
                     "attn_S32768_Q2048.mlpackage/Data/com.apple.CoreML/",
                     "attn_S32768_Q2048.mlpackage/Data/com.apple.CoreML/../model.mlmodel"] {
            Self.reject(.object(Self.root), files: withoutModel + [Self.entry(path)], label: path)
        }

        let manifest = "PPLXText.mlpackage/Manifest.json"
        Self.reject(.object(Self.root), files: Self.files.filter { $0.path != manifest }, label: "missing inner Manifest")
        Self.reject(.object(Self.root), files: Self.files + [Self.entry(manifest)], label: "duplicate file")
        Self.reject(.object(Self.root), files: Self.files + [Self.entry("unknown.mlpackage/Manifest.json")],
                    label: "undeclared source package")
    }

    private static func reject(_ root: JSONValue, files: [ManifestEntry] = files, label: String = "root",
                               sourceLocation: SourceLocation = #_sourceLocation) {
        let error = #expect(throws: ServiceError.self, Comment(rawValue: label), sourceLocation: sourceLocation) {
            try PPLXPackageContract.parse(root, files: files)
        }
        #expect(error?.reason == "invalid_manifest", sourceLocation: sourceLocation)
    }

    private static func rejectReplacingFunction(at index: Int, with function: JSONObject, label: String,
                                                sourceLocation: SourceLocation = #_sourceLocation) {
        var functions = Self.functions
        functions[index] = function
        var root = Self.root
        root["functions"] = .array(functions.map(JSONValue.object))
        reject(.object(root), label: "\(function["function"]?.stringValue ?? "?") \(label)",
               sourceLocation: sourceLocation)
    }

    private static let assets = [
        "tokenizer.json", "tokenizer_config.json", "embeddings.f32", "final_norm.f32",
        "selftest.json", "selftest.bin", "LICENSE", "NOTICE",
    ]
    private static let fusedNames = [
        "embed_64", "embed_512", "embed_1024", "fused_2048", "fused_4096",
        "fused_8192_part0", "fused_8192_part1", "fused_8192_part2", "fused_8192_part3",
    ]
    private static let packages = ["PPLXText"] + (0..<28).flatMap { ["pre\($0)", "post\($0)"] }
        + ["attn_S16384_Q2048", "attn_S32768_Q2048"]

    private static func entry(_ path: String, size: Int = 1) -> ManifestEntry {
        ManifestEntry(path: path, size: size, sha256: String(repeating: "0", count: 64))
    }

    private static let files = assets.map {
        entry($0, size: ["embeddings.f32": 622_329_856, "final_norm.f32": 4096][$0] ?? 1)
    } + packages.flatMap {
        [entry("\($0).mlpackage/Manifest.json"), entry("\($0).mlpackage/Data/com.apple.CoreML/model.mlmodel")]
    }

    private static let root: JSONObject = [
        "contract_version": 1, "model": "pplx-embed-v1-0.6b", "revision": "2c4d510dd4a732063c31a0f70193e35067b51fd8",
        "created": "2026-11-18", "packages": .array(packages.map(JSONValue.string)),
        "source_model": ["id": "perplexity-ai/pplx-embed-v1-0.6b", "revision": "2c4d510dd4a732063c31a0f70193e35067b51fd8", "license": "MIT"],
        "converter": ["script": "tools/convert/build.py", "precision": "fp16 ML Program, macOS 15"],
        "functions": .array(functions.map(JSONValue.object)),
    ]

    private static let descriptor: Data = {
        var root = Self.root
        root["files"] = .array(Self.files.map {
            ["path": .string($0.path), "size": .int($0.size), "sha256": .string($0.sha256)]
        })
        return JSONValue.object(root).jsonData()
    }()

    private static let functions: [JSONObject] = {
        func tensor(_ name: String, _ shape: [Int], _ dtype: JSONValue = "fp16") -> JSONValue {
            ["name": .string(name), "shape": .array(shape.map(JSONValue.int)), "dtype": dtype]
        }
        func declaration(_ role: String, _ name: String, package: String? = nil, inputs: [JSONValue],
                         outputs: [JSONValue], length: Int? = nil, layer: Int? = nil, part: Int? = nil) -> JSONObject {
            var result: JSONObject = ["role": .string(role), "function": .string(name), "package": .string(package ?? name),
                                      "inputs": .array(inputs), "outputs": .array(outputs)]
            result["length"] = length.map(JSONValue.int)
            result["layer"] = layer.map(JSONValue.int)
            result["part"] = part.map(JSONValue.int)
            return result
        }

        var result: [JSONObject] = []
        for (s, slots) in [(64, 8), (512, 16), (1024, 32), (2048, 1), (4096, 1)] {
            let packed = s <= 1024
            result.append(declaration(packed ? "packed" : "fused", packed ? "embed_\(s)" : "fused_\(s)",
                package: "PPLXText", inputs: [tensor("inputs_embeds", [1, s, 1024]),
                    tensor("bias", [1, 1, packed ? s : 1, s]), tensor("pool", [slots, s])],
                outputs: [tensor("pooled", [slots, 1024], "fp32")], length: s))
        }
        for part in 0..<4 {
            var inputs = [tensor(part == 0 ? "inputs_embeds" : "hidden", [1, 8192, 1024]),
                          tensor("bias", [1, 1, 1, 8192])]
            if part == 3 { inputs.append(tensor("pool", [1, 8192])) }
            let output = part == 3 ? tensor("pooled", [1, 1024], "fp32") : tensor("hidden_out", [1, 8192, 1024])
            result.append(declaration("fused_part", "fused_8192_part\(part)", package: "PPLXText",
                                      inputs: inputs, outputs: [output], length: 8192, part: part))
        }
        for layer in 0..<28 {
            let xs = tensor("xs", [1, 2048, 1024])
            result.append(declaration("pre", "pre\(layer)", inputs: [xs], outputs: [
                tensor("q", [1, 2048, 2048]), tensor("k", [1, 2048, 1024]), tensor("v", [1, 2048, 1024]),
            ], layer: layer))
            result.append(declaration("post", "post\(layer)", inputs: [xs, tensor("o", [1, 2048, 2048])],
                                      outputs: [tensor("a", [1, 2048, 1024]), tensor("d", [1, 2048, 1024])], layer: layer))
        }
        for s in [16384, 32768] {
            result.append(declaration("attention", "attn_S\(s)_Q2048", inputs: [
                tensor("q", [1, 16, 2048, 128]), tensor("k", [1, 8, s, 128]), tensor("v", [1, 8, s, 128]),
                tensor("bias", [1, 1, 1, s]),
            ], outputs: [tensor("o", [1, 16, 2048, 128])], length: s))
        }
        return result
    }()
}
