#if canImport(CoreAI)
import CoreAI
import DecisionProfiles
import Foundation

@available(macOS 27.0, *)
final class Session {
  let manifest: Manifest, builder: PromptBuilder, backbone: Backbone,
    readout: Linear?, vision: Vision?
  let temperature: Double
  init(_ root: URL) throws {
    guard root.isFileURL else { throw DecisionError("Expected a resolved local bundle URL") }
    let m = try Manifest.validated(at: root)
    let weights =
      m.embeddingLookup == nil
      ? try Weights(root.appendingPathComponent("model.safetensors")) : nil
    manifest = m
    backbone = try Backbone(root, m, weights)
    var codes = [String]()
    if m.elephant {

      let labels = try root.decode("decision_config.json", as: ElephantLabels.self).codes
      guard labels.count == 256, Set(labels).count == labels.count else {
        throw DecisionError("Invalid Elephant slot table")
      }
      codes = labels
      temperature = 1
      vision = Vision(elephant: true)
      readout = nil
    } else if m.valen {
      readout = nil
      temperature = 1
      vision = Vision(elephant: false)
    } else {
      struct Config: Decodable {
        let codes: [String]
        let temperature: Double
      }
      let c = try root.decode("decision_config.json", as: Config.self)
      guard c.temperature.isFinite, c.temperature > 0 else {
        throw DecisionError("Invalid readout calibration")
      }
      if m.kDecision, c.codes != KDecisionPrompt.alphabet || c.temperature != 1 {
        throw DecisionError("Invalid K-Decision symbol readout")
      }
      if m.knowLine || m.jeffan, c.codes.count != 255 || c.temperature != 1 {
        throw DecisionError("Invalid text label readout")
      }
      if m.decisionHead != nil && c.temperature != 1 {
        throw DecisionError("Head temperature differs from configuration")
      }
      codes = c.codes
      temperature = c.temperature
      vision = nil
      readout =
        m.decisionHead != nil
        ? nil
        : try Linear(
          Weights(root.appendingPathComponent("readout.safetensors")), "", input: m.hidden,
          output: m.kDecision ? 26 : 255, hasBias: false)
    }
    builder = try PromptBuilder(
      tokenizer: BPETokenizer(root), valen: m.valen, codes: codes, context: m.context)
  }
  func decide(_ request: DecisionInput) async throws -> DecisionResponse {
    backbone.trace.removeAll(keepingCapacity: true)
    let timer = InputProcessingTimer()
    var phases = [String: Double]()
    let timed = manifest.valen && backbone.tracing
    var last = timed ? ProcessInfo.processInfo.systemUptime : 0
    func mark(_ phase: String) {
      guard timed else { return }
      let now = ProcessInfo.processInfo.systemUptime
      phases[phase, default: 0] += now - last
      last = now
    }
    let visual: VisualFeatures?
    if let image = request.image {
      guard let vision else {
        throw DecisionError("This model bundle does not support images", input: true)
      }
      visual = try await backbone.encodeImage(image, with: vision)
    } else {
      visual = nil
    }
    mark("vision")
    let visualTokens = visual?.count ?? 0
    let prepared: PreparedDecision
    if manifest.kDecision {
      prepared = try KDecisionPrompt.build(
        request, tokenizer: builder.tokenizer, context: manifest.context)
    } else if manifest.elephant {
      prepared = try ElephantPrompt(
        tokenizer: builder.tokenizer, context: manifest.context, labels: builder.codes
      )
      .build(request, visualTokens: visualTokens)
    } else if manifest.jeffan {
      prepared = try JeffanPrompt.build(
        request, tokenizer: builder.tokenizer, codes: builder.codes,
        context: manifest.context)
    } else if manifest.knowLine {
      prepared = try KnowLinePrompt.build(
        request, tokenizer: builder.tokenizer, codes: builder.codes,
        context: manifest.context)
    } else {
      prepared = try builder.build(request, visualTokens: visualTokens)
    }
    mark("prompt")
    let prompts = prepared.prompts
    var answers = [String: DecisionAnswer]()
    for p in prompts {
      let chunks = try await backbone.run(p, vision: visual)
      mark("backbone")
      func row(_ index: Int) -> [Float] {
        chunks[index / manifest.width].read { data, strides in
          (0..<manifest.hidden).map { Float(data[$0 * strides[1] + index % manifest.width * strides[3]]) }
        }
      }
      for b in p.branches {
        if manifest.valen {
          answers[b.question.name] = try await valenAnswer(chunks, branch: b)
          mark("readout")
          continue
        }
        if let head = manifest.decisionHead {
          if manifest.elephant {
            answers[b.question.name] = try await elephantDeviceAnswer(
              hidden: row(p.ids.count - 1), branch: b, head: head)
            continue
          }
          answers[b.question.name] = try await deviceAnswer(
            hidden: row(p.ids.count - 1), branch: b, head: head)
          continue
        }
        let logits: [Float]
        if let readout {
          let projected = readout(row(manifest.kDecision ? b.roles[0].lowerBound : p.ids.count - 1))
          // K-Decision predicates use the vocabulary symbols Y and N.
          logits =
            manifest.kDecision && b.question.kind == .noul
            ? [projected[24], projected[13]] : Array(projected.prefix(b.candidates.count))
        } else {
          throw DecisionError("Missing decision head")
        }
        answers[b.question.name] = try answer(logits, b)
      }
    }
    if backbone.tracing {
      var trace: [String: Any] = [
        "device_calls": backbone.trace, "tokens": prepared.prompts.map { $0.ids.count },
      ]
      if timed { trace["seconds"] = phases }
      if let json = try? JSONSerialization.data(withJSONObject: trace) {
        FileHandle.standardError.write(json + Data([10]))
      }
    }
    let tokens = prepared.tokenCounts(sharingPrefix: manifest.elephant)
    let outputTokens = manifest.elephant ? answers.count : manifest.knowLine ? prompts.count : 0
    return DecisionResponse(
      model: manifest.valen
        ? "Valen" : manifest.source.repo.split(separator: "/").last!.lowercased(),
      answers: answers,
      usage: .init(
        inputTokens: tokens.input, outputTokens: outputTokens, truncated: prepared.truncated,
        promptTokensPerSecond: timer.tokensPerSecond(tokens.compute)),
      internalUsage: manifest.valen || manifest.elephant
        ? .init(computeTokens: tokens.compute) : nil)
  }
  func answer(_ logits: [Float], _ b: Branch) throws -> DecisionAnswer {
    guard let peak = logits.max(), logits.allSatisfy(\.isFinite) else {
      throw DecisionError("Nonfinite decision logits")
    }
    let expScores = logits.map { exp((Double($0) - Double(peak)) / temperature) }
    let total = expScores.reduce(0, +)
    return format(expScores.map { $0 / total }, b)
  }

  func elephantDeviceAnswer(hidden: [Float], branch: Branch, head: Manifest.Asset) async throws
    -> DecisionAnswer
  {
    let count = branch.candidates.count
    guard hidden.count == manifest.hidden, (1...256).contains(count) else {
      throw DecisionError("Invalid readout input shape")
    }
    var input = NDArray.zeros([1, manifest.hidden, 1, 1])
    input.write { p, strides in
      for i in hidden.indices { p[i * strides[1]] = Float16(hidden[i]) }
    }
    var mask = NDArray.zeros([1, 256, 1, 1])
    mask.write { p, strides in
      for i in count..<256 { p[i * strides[1]] = -30000 }
    }
    let f = try await backbone.function("readout", in: head.asset)
    var out = try await f.run(inputs: ["hidden": input, "mask": mask])
    let output = try out.tensor("probabilities")
    return try await formatDeviceAnswer(output, branch: branch)
  }

  func formatDeviceAnswer(_ output: NDArray, branch: Branch) async throws -> DecisionAnswer {
    let count = branch.candidates.count
    guard output.shape == [1, 256, 1, 1] else { throw DecisionError("Invalid probability shape") }
    let values = output.read { p, strides in (0..<256).map { Double(p[$0 * strides[1]]) } }
    guard values.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
      values.dropFirst(count).allSatisfy({ $0 == 0 })
    else {
      throw DecisionError("Invalid device probabilities")
    }
    let p = Array(values.prefix(count))
    let k = branch.question.kind
    let top = p.firstIndex(of: p.max()!)!
    func scalar(_ value: Float16) -> NDArray {
      var tensor = NDArray.zeros([1, 1, 1, 1])
      tensor.write { p, _ in p[0] = value }
      return tensor
    }
    guard let stats = manifest.resultStats else {
      throw DecisionError("Missing Elephant statistics")
    }
    let sf = try await backbone.function("stats", in: stats.asset)
    let constants = ElephantDeviceConstants.stats[count - 1]
    var result = try await sf.run(inputs: [
      "p": output, "top_index": scalar(Float16(top)),
      "choice_scale": scalar(constants[0]), "choice_bias": scalar(constants[1]),
      "mad_scale": scalar(constants[2]),
    ])
    let tensor = try result.tensor("statistics")
    guard tensor.shape == [1, 3, 1, 1] else { throw DecisionError("Invalid decision statistics") }
    let v = tensor.read { a, s in (0..<3).map { Double(a[$0 * s[1]]) } }
    guard v.allSatisfy({ $0.isFinite }) else { throw DecisionError("Nonfinite device statistics") }
    return DecisionAnswer(
      type: k, choice: k == .choice ? branch.candidates[top].key : nil,
      noul: k == .noul ? p[0] : nil, score: k == .score ? v[0] : nil, expectedLevel: nil,
      confidence: k == .score ? v[2] : k == .choice ? v[1] : nil,
      probabilities: k == .noul
        ? nil : Dictionary(uniqueKeysWithValues: zip(branch.candidates.map(\.key), p)),
      legend: k == .score
        ? Dictionary(uniqueKeysWithValues: branch.candidates.map { ($0.key, $0.description) }) : nil
    )
  }

  func deviceAnswer(hidden: [Float], branch: Branch, head: Manifest.Asset) async throws
    -> DecisionAnswer
  {
    let count = branch.candidates.count
    guard hidden.count == manifest.hidden, (1...255).contains(count) else {
      throw DecisionError("Invalid readout input shape")
    }
    var input = NDArray.zeros([1, manifest.hidden, 1, 1])
    input.write { p, strides in
      for i in hidden.indices { p[i * strides[1]] = Float16(hidden[i]) }
    }
    var mask = NDArray.zeros([1, 255, 1, 1])
    mask.write { p, strides in
      for i in count..<255 { p[i * strides[1]] = -30000 }
    }
    let f = try await backbone.function("readout", in: head.asset)
    var out = try await f.run(inputs: ["hidden": input, "mask": mask])
    let output = try out.tensor("probabilities")
    guard output.shape == [1, 255, 1, 1] else { throw DecisionError("Invalid probability shape") }
    let values = output.read { p, strides in (0..<255).map { Double(p[$0 * strides[1]]) } }
    guard values.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
      values.dropFirst(count).allSatisfy({ $0 == 0 })
    else {
      throw DecisionError("Invalid device probabilities")
    }
    let p = Array(values.prefix(count))
    guard let stats = manifest.resultStats, let maximum = p.max(),
      let top = p.firstIndex(of: maximum)
    else { throw DecisionError("Missing device statistics") }
    var excluded = NDArray.zeros([1, 255, 1, 1])
    excluded.write { ptr, stride in ptr[top * stride[1]] = -30000 }
    let statsFunction = try await backbone.function("readoutStats", in: stats.asset)
    var statsOut = try await statsFunction.run(inputs: ["p": output, "exclude": excluded])
    let fields = try statsOut.tensor("statistics")
    guard fields.shape == [1, 3, 1, 1] else { throw DecisionError("Invalid device statistics") }
    let numbers = fields.read { ptr, stride in (0..<3).map { Double(ptr[$0 * stride[1]]) } }
    guard numbers.allSatisfy(\.isFinite) else {
      throw DecisionError("Nonfinite device statistics")
    }
    let k = branch.question.kind
    return DecisionAnswer(
      type: k, choice: k == .choice ? branch.candidates[top].key : nil,
      noul: k == .noul ? p[0] : nil, score: k == .score ? Double(top) : nil,
      expectedLevel: k == .score ? numbers[2] : nil,
      confidence: k == .noul ? numbers[1] : numbers[0],
      probabilities: k == .noul
        ? nil : Dictionary(uniqueKeysWithValues: zip(branch.candidates.map(\.key), p)),
      legend: nil)
  }

  func format(_ p: [Double], _ b: Branch) -> DecisionAnswer {
    let top = p.firstIndex(of: p.max()!)!
    let k = b.question.kind
    let n = Double(p.count)
    let pairs = zip(b.candidates.map(\.key), p)
    let expected = p.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element }
    var conf: Double?
    if manifest.valen || manifest.elephant {
      if k == .choice { conf = n == 1 ? 1 : max(0, (p[top] - 1 / n) / (1 - 1 / n)) }
      if k == .score {
        let mad = p.indices.reduce(0.0) { $0 + abs(Double($1) - (n - 1) / 2) } / n
        conf =
          n == 1
          ? 1 : max(0, 1 - p.indices.reduce(0.0) { $0 + p[$1] * Double(abs($1 - top)) } / mad)
      }
    } else if manifest.knowLine {
      if k != .noul {
        let entropy = -p.filter { $0 > 0 }.reduce(0) { $0 + $1 * log($1) }
        conf = min(1, max(0, 1 - entropy / log(n)))
      }
    } else if !manifest.kDecision && !manifest.jeffan {
      conf = k == .noul ? abs(p[0] - 0.5) * 2 : p[top] - (p.sorted(by: >).dropFirst().first ?? 0)
    }
    let expectedScore = manifest.valen || manifest.elephant || manifest.knowLine || manifest.jeffan
    return DecisionAnswer(
      type: k, choice: k == .choice ? b.candidates[top].key : nil,
      noul: k == .noul ? p[manifest.jeffan ? 1 : 0] : nil,
      score: k == .score ? (expectedScore ? expected : Double(top)) : nil,
      expectedLevel: k == .score && !expectedScore ? expected : nil, confidence: conf,
      probabilities: k == .noul ? nil : Dictionary(uniqueKeysWithValues: pairs),
      legend: k == .score && (manifest.valen || manifest.elephant || manifest.knowLine)
        ? Dictionary(uniqueKeysWithValues: b.candidates.map { ($0.key, $0.description) }) : nil)
  }
}

/// Serializes the whole request, including Core AI suspension points. Unload waits for the
/// running decision; subsequent requests recreate the session at the same stable asset path.
@available(macOS 27.0, *)
public actor DecisionEngine {
  public let bundleURL: URL
  let idleTimeout: Duration?
  private var session: Session?, busy = false, waiters = [CheckedContinuation<Void, Never>]()
  private var timer: Task<Void, Never>?, generation: UInt = 0
  public init(bundleURL: URL, idleTimeout: Duration? = nil) {
    self.bundleURL = bundleURL
    self.idleTimeout = idleTimeout
  }
  /// Check the descriptor without creating a session or opening model assets.
  public static func validateManifest(at directory: URL) throws {
    _ = try Manifest.validated(at: directory)
  }
  public var isLoaded: Bool { session != nil }
  private func acquire() async {
    if busy { await withCheckedContinuation { waiters.append($0) } } else { busy = true }
    generation &+= 1
    timer?.cancel()
    timer = nil
  }
  private func release() {
    if !waiters.isEmpty {
      waiters.removeFirst().resume()
      return
    }
    busy = false
    guard let idleTimeout, session != nil else { return }
    let current = generation
    timer = Task { [weak self] in
      do { try await Task.sleep(for: idleTimeout) } catch { return }
      await self?.expire(current)
    }
  }
  private func expire(_ current: UInt) {
    if !busy, current == generation {
      session = nil
      timer = nil
    }
  }
  private func resident() throws -> Session {
    if let session { return session }
    let new = try Session(bundleURL)
    if new.manifest.embeddingLookup != nil { try new.backbone.verifyPlacement() }
    session = new
    return new
  }
  public func prepare() async throws {
    await acquire()
    defer { release() }
    let temporary = try Session(bundleURL)
    try temporary.backbone.verifyPlacement()
    try await temporary.backbone.prepare()
  }
  public func load() async throws {
    await acquire()
    defer { release() }
    _ = try resident()
  }
  public func decide(_ request: DecisionInput) async throws -> DecisionResponse {
    await acquire()
    defer { release() }
    try Task.checkCancellation()
    return try await resident().decide(request)
  }
  public func unload() async {
    await acquire()
    defer { release() }
    session = nil
  }
}
#endif
