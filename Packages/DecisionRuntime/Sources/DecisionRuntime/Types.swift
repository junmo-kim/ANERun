import DecisionProfiles
import Foundation

#if canImport(CoreAI)
import CoreAI
#endif

/// A specialized compiled model threw the observed Core AI nilError while loading this entry.
public struct CompiledFunctionLoadFailure: Error, Sendable, Hashable, Codable, CustomStringConvertible {
  public let asset: String
  public let function: String

  public init(asset: String, function: String) {
    self.asset = asset
    self.function = function
  }

  public var description: String { "loadFunction(\(function)) threw nilError for \(asset)" }
}

public struct DecisionError: Error, CustomStringConvertible {
  public let description: String
  public let isInputError: Bool
  init(_ text: String, input: Bool = false) {
    description = text
    isInputError = input
  }
}
public struct DecisionCandidate: Codable, Sendable {
  public let key: String
  public let description: String
  public init(key: String, description: String) {
    self.key = key
    self.description = description
  }
}
public struct DecisionQuestion: Codable, Sendable {
  public enum Kind: String, Codable, Sendable { case choice, noul, score }
  public let name: String
  public let kind: Kind
  public let instructions: String
  public let candidates: [DecisionCandidate]
  public init(from decoder: Decoder) throws {
    let values = try decoder.container(keyedBy: CodingKeys.self)
    name = try values.decode(String.self, forKey: .name)
    kind = try values.decode(Kind.self, forKey: .kind)
    instructions = try values.decode(String.self, forKey: .instructions)
    candidates = try values.decodeIfPresent([DecisionCandidate].self, forKey: .candidates) ?? []
  }
  public init(name: String, kind: Kind, instructions: String, candidates: [DecisionCandidate] = [])
  {
    self.name = name
    self.kind = kind
    self.instructions = instructions
    self.candidates = candidates
  }
}
public struct DecisionInput: Sendable {
  public let state: String
  public let questions: [DecisionQuestion]
  public let image: Data?
  public let truncate: Bool
  public init(
    state: String, questions: [DecisionQuestion], image: Data? = nil, truncate: Bool = true
  ) {
    self.state = state
    self.questions = questions
    self.image = image
    self.truncate = truncate
  }
}
public struct DecisionAnswer: Codable, Sendable {
  enum CodingKeys: String, CodingKey {
    case type
    case choice
    case noul
    case score
    case expectedLevel = "expected_level"
    case confidence
    case probabilities
    case legend
  }

  public let type: DecisionQuestion.Kind
  public let choice: String?
  public let noul: Double?
  public let score: Double?
  public let expectedLevel: Double?
  public let confidence: Double?
  public let probabilities: [String: Double]?
  public let legend: [String: String]?
}
public struct DecisionResponse: Codable, Sendable {
  enum CodingKeys: String, CodingKey {
    case model
    case answers
    case usage
    case internalUsage = "internal_usage"
  }

  public struct Usage: Codable, Sendable {
    enum CodingKeys: String, CodingKey {
      case inputTokens = "input_tokens"
      case outputTokens = "output_tokens"
      case truncated
      case promptTokensPerSecond = "prompt_tokens_per_second"
    }

    public let inputTokens: Int
    public let outputTokens: Int
    public let truncated: Bool?
    public let promptTokensPerSecond: Double?

    init(
      inputTokens: Int, outputTokens: Int, truncated: Bool = false,
      promptTokensPerSecond: Double? = nil
    ) {
      self.promptTokensPerSecond = promptTokensPerSecond
      self.inputTokens = inputTokens
      self.outputTokens = outputTokens
      self.truncated = truncated ? true : nil
    }
  }
  public struct InternalUsage: Codable, Sendable {
    enum CodingKeys: String, CodingKey {
      case computeTokens = "compute_tokens"
    }
    public let computeTokens: Int
  }
  /// Preserve the engine's scores and usage when the serving process advertises a model alias.
  public func reportingModel(_ model: String) -> Self {
    Self(model: model, answers: answers, usage: usage, internalUsage: internalUsage)
  }

  public let model: String
  public let answers: [String: DecisionAnswer]
  public let usage: Usage
  public let internalUsage: InternalUsage?
}

struct Manifest: Decodable {
  enum CodingKeys: String, CodingKey {
    case architecture
    case source
    case assets
    case execution
    case stageKinds = "stage_kinds"
    case hidden
    case width
    case context
    case tiers
    case rotaryDim = "rotary_dim"
    case prompts
    case heads
    case kvHeads = "kv_heads"
    case headDim = "head_dim"
    case blockMask = "block_mask"
    case rotaryTables = "rotary_tables"
    case rounding = "residual_rounding"
    case embeddingUnscale = "embedding_unscale"
    case visionFractions = "vision_fractions"
    case visionPosition = "vision_position"
    case featureScale = "feature_scale"
    case embeddingLookup = "embedding_lookup"
    case rolePool = "role_pool"
    case roleMixer = "role_mixer"
    case residualBlend = "residual_blend"
    case resultStats = "result_stats"
    case decisionHead = "decision_head"
    case smallVision = "vision_256"
    case visionAsset = "vision_asset"
    case visionSourceHash = "vision_source_hash"
    case visionFunctions = "vision_functions"
  }

  struct Asset: Decodable {
    enum CodingKeys: String, CodingKey {
      case asset
      case functions
      case sourceHash = "source_hash"
    }

    let asset: String
    let functions: [String]
    let sourceHash: String
  }
  struct Block: Decodable {
    let stages: [Int]
    let passes: Int
  }
  struct Source: Decodable {
    let repo: String
    let revision: String
  }
  let architecture: String
  let source: Source
  let assets: [Asset], execution: [Block], stageKinds: [String]
  let hidden: Int, width: Int, context: Int, tiers: [Int], rotaryDim: Int
  let prompts: [Int]?
  let heads: Int, kvHeads: Int, headDim: Int, blockMask: Bool
  struct Lookup: Decodable {
    enum CodingKeys: String, CodingKey {
      case bankSize = "bank_size"
      case laneWidth = "lane_width"
      case vocabSize = "vocab_size"
      case assets
    }
    let bankSize: Int, laneWidth: Int, vocabSize: Int, assets: [Asset]
  }
  let embeddingLookup: Lookup?, residualBlend: Asset?, resultStats: Asset?
  let rotaryTables: [String: String]?, rounding: [Asset]?, embeddingUnscale: Asset?
  let visionFractions: String?, visionPosition: Asset?, featureScale: Asset?
  static func logicalFunction(_ name: String) -> String {
    guard let split = name.lastIndex(of: "_") else { return name }
    let digest = name[name.index(after: split)...].utf8
    let logical = String(name[..<split])
    let shortVision = digest.count == 8 && (logical == "p256" || logical == "p1024")
    guard digest.count == 64 || shortVision,
      digest.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) })
    else { return name }
    return logical
  }
  var deviceExtras: [Asset] {
    [decisionHead, residualBlend, resultStats, embeddingUnscale, visionPosition, featureScale, rolePool, roleMixer, smallVision]
      .compactMap { $0 } + (rounding ?? []) + (embeddingLookup?.assets ?? [])
  }
  let decisionHead: Asset?, rolePool: Asset?, roleMixer: Asset?
  let smallVision: Asset?
  let visionAsset: String?, visionSourceHash: String?, visionFunctions: [String]?
  var kDecision: Bool { DecisionProfile(repository: source.repo) == .kDecision9B }
  var knowLine: Bool { DecisionProfile(repository: source.repo) == .knowLine9B }
  var jeffan: Bool { DecisionProfile(repository: source.repo) == .jeffan9B }
  var causal: Bool { kDecision || knowLine }
  var elephant: Bool { DecisionProfile(repository: source.repo) == .elephant1 }
  var valen: Bool { DecisionProfile(repository: source.repo) == .valen4B }
  #if canImport(CoreAI)
  @available(macOS 27.0, *)
  static func validated(at directory: URL) throws -> Manifest {
    let manifest = try directory.decode("model.json", as: Manifest.self)
    try manifest.validate()
    return manifest
  }

  @available(macOS 27.0, *)
  func validate() throws {
    if !valen && (rolePool != nil || roleMixer != nil || smallVision != nil) {
      throw DecisionError("Unexpected Valen role operators")
    }
    if valen {
      if embeddingLookup == nil && rolePool == nil && roleMixer == nil {
        let chip = AIModel.deviceArchitectureName
        let migration = chip == "h16g"
          ? "Reinstall with anerun models install --model valen-4b."
          : "Build a device-operator bundle for \(chip) and install it with anerun models install --model valen-4b --from DIR."
        throw DecisionError("Legacy Valen bundle requires migration. " + migration)
      }
      let expected = [(rolePool, "valenPool"), (roleMixer, "valenMixer"), (decisionHead, "readout"),
                      (resultStats, "stats"), (visionPosition, "visionpos"), (featureScale, "featurescale")]
      guard hidden == 2560, context == 4096, width == 128, visionFunctions?.map(Self.logicalFunction) == ["p1024"],
        smallVision.map({ $0.functions.map(Self.logicalFunction) == ["p256"] }) ?? true,
        visionAsset != nil, visionFractions == "vision-fractions.fp16",
        rotaryTables == ["text": "text-rotary.fp16", "vision": "vision-rotary.fp16"],
        rounding == nil, residualBlend == nil, embeddingUnscale == nil,
        stageKinds == Array(repeating: ["linear", "linear", "linear", "attention"], count: 8).flatMap({ $0 }),
        execution.map(\.stages) == Array(0..<32).map({ [$0] }), execution.allSatisfy({ $0.passes == 1 }),
        expected.allSatisfy({ $0.0?.functions.map(Self.logicalFunction) == [$0.1] }),
        let lookup = embeddingLookup, lookup.bankSize == 4096, lookup.laneWidth == 16,
        lookup.vocabSize == 248320, lookup.assets.count == 61,
        lookup.assets.enumerated().allSatisfy({ $0.element.functions.map(Self.logicalFunction) == ["lookup\($0.offset)"] }),
        Set((assets + deviceExtras).map(\.asset) + [visionAsset!]).count == assets.count + deviceExtras.count + 1
      else { throw DecisionError("Incomplete Valen device operator manifest") }
    } else if elephant {
      guard
        stageKinds
          == Array(repeating: ["linear", "linear", "attention"], count: 8).flatMap({ $0 }),
        context <= 1024, execution.allSatisfy({ $0.passes == 1 }), residualBlend == nil,
        rounding?.count == 3, Set(rotaryTables?.keys.map { $0 } ?? []) == ["text", "vision"],
        rotaryTables?.values.allSatisfy({
          $0.hasSuffix(".fp16") && URL(fileURLWithPath: $0).lastPathComponent == $0
        }) == true,
        let banks = embeddingLookup, banks.bankSize == 4096, banks.laneWidth == 16,
        banks.vocabSize == 248320, banks.assets.count == 61,
        rounding?.map(\.functions) == [["relusc"], ["two"], ["finite"]],
        resultStats?.functions.map(Self.logicalFunction) == ["stats"],
        decisionHead?.functions.map(Self.logicalFunction) == ["readout"],
        visionPosition?.functions.map(Self.logicalFunction) == ["visionpos"],
        featureScale?.functions.map(Self.logicalFunction) == ["featurescale"],
        embeddingUnscale?.functions.map(Self.logicalFunction) == ["unscale"],
        let fractions = visionFractions,
        fractions == URL(fileURLWithPath: fractions).lastPathComponent,
        fractions.hasSuffix(".fp16"),
        banks.assets.enumerated().allSatisfy({
          $0.element.functions.map(Self.logicalFunction) == ["lookup\($0.offset)"]
        }),
        deviceExtras.allSatisfy({
          $0.asset == URL(fileURLWithPath: $0.asset).lastPathComponent
            && $0.asset.hasSuffix(".aimodelc") && $0.sourceHash.count == 64
            && $0.sourceHash.allSatisfy(\.isHexDigit)
        }),
        Set((assets + deviceExtras).map(\.asset)).count == assets.count + deviceExtras.count
      else { throw DecisionError("Invalid Elephant device assets") }
    } else if DecisionProfile(repository: source.repo) == .glide9B {
      guard rounding == nil, embeddingUnscale == nil, visionPosition == nil, featureScale == nil,
        rotaryTables == nil, visionFractions == nil, prompts?.isEmpty != false, visionAsset == nil,
        stageKinds
          == Array(repeating: ["linear", "linear", "linear", "attention"], count: 8).flatMap({
            $0
          }),
        assets.count == 32,
        let lookup = embeddingLookup, lookup.bankSize == 4096, lookup.vocabSize == 248320,
        lookup.laneWidth == 16, lookup.assets.count == 61,
        lookup.assets.enumerated().allSatisfy({ i, a in
          a.asset == "lookup" + String(i) + "." + architecture + ".aimodelc"
            && a.functions == ["lookup" + String(i)]
        }),
        residualBlend?.asset == "blend2." + architecture + ".aimodelc",
        residualBlend?.functions == ["blend2"],
        resultStats?.asset == "readoutStats." + architecture + ".aimodelc",
        resultStats?.functions == ["readoutStats"],
        execution.filter({ $0.passes > 1 }).map(\.stages) == [Array(12...17), Array(18...23)],
        execution.filter({ $0.passes > 1 }).allSatisfy({ $0.passes == 2 }),
        deviceExtras.allSatisfy({
          $0.sourceHash.count == 64 && $0.sourceHash.allSatisfy(\.isHexDigit)
        })
      else { throw DecisionError("Incomplete device execution manifest") }
      guard let head = decisionHead, head.functions == ["readout"],
        head.asset == "head." + architecture + ".aimodelc",
        hidden == 4096
      else { throw DecisionError("Missing or incompatible GLiDE-9B head") }
    } else if embeddingLookup != nil || !deviceExtras.isEmpty {
      throw DecisionError("Unexpected decision head")
    }
    // Jeffan repeats three four-layer blocks, regardless of the physical asset boundaries.
    let splitKinds = Array(repeating: ["linear", "linear", "linear", "attention"], count: 8).flatMap
    { $0 }
    let jeffanSplit = stageKinds == splitKinds
    let jeffanGroups =
      jeffanSplit
      ? (0..<4).map { [$0] } + [Array(4..<8), Array(8..<12), Array(12..<16)]
        + (16..<32).map { [$0] }
      : (0..<8).map { [$0] }
    let jeffanPasses =
      jeffanSplit
      ? Array(repeating: 1, count: 4) + [2, 2, 2] + Array(repeating: 1, count: 16)
      : [1, 2, 2, 2, 1, 1, 1, 1]
    guard let profile = DecisionProfile(repository: source.repo),
      architecture == AIModel.deviceArchitectureName, profile.revision == source.revision,
      width == 128, (1...8192).contains(hidden),
      rotaryDim == 64, context == tiers.last, context <= 16384, !tiers.isEmpty,
      tiers == tiers.sorted(), tiers.allSatisfy({ $0 >= width && $0 % width == 0 }),
      (1...64).contains(kvHeads), (1...256).contains(headDim),
      blockMask == (valen || elephant || causal),
      !causal
        || (prompts?.isEmpty != false && visionAsset == nil
          && execution.allSatisfy { $0.passes == 1 }),
      !elephant
        || (hidden == 4096 && visionAsset != nil && visionFunctions == ["p1024"]
          && prompts?.isEmpty == true),
      !jeffan
        || (visionAsset == nil && prompts?.isEmpty != false
          && (jeffanSplit || stageKinds == Array(repeating: "attention", count: 8))
          && execution.map(\.stages) == jeffanGroups
          && execution.map(\.passes) == jeffanPasses),
      stageKinds.count == assets.count, stageKinds.contains("attention"),
      stageKinds.allSatisfy({ ["linear", "attention"].contains($0) }),
      execution.flatMap(\.stages) == Array(assets.indices),
      execution.allSatisfy({ (1...8).contains($0.passes) }),
      ((assets + deviceExtras).map(\.asset) + [visionAsset].compactMap { $0 }).allSatisfy({
        $0.hasSuffix(".aimodelc") && URL(fileURLWithPath: $0).lastPathComponent == $0
      })
    else { throw DecisionError("Unsupported decision bundle manifest") }
  }
  #endif
}
extension URL {
  func bundleChild(_ name: String) throws -> URL {
    let base = standardizedFileURL.resolvingSymlinksInPath()
    let child = base.appendingPathComponent(name).standardizedFileURL.resolvingSymlinksInPath()
    guard !name.isEmpty, child.lastPathComponent == name,
      child.deletingLastPathComponent().path == base.path
    else { throw DecisionError("Expected a bundle-relative filename") }
    return child
  }
  func decode<T: Decodable>(_ name: String, as: T.Type) throws -> T {
    try JSONDecoder().decode(T.self, from: Data(contentsOf: appendingPathComponent(name)))
  }
}
