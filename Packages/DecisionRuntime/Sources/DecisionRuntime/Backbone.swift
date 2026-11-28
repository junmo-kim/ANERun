#if canImport(CoreAI)
import CoreAI
import Foundation

@available(macOS 27.0, *)
extension NDArray {
  /// All bundled functions use rank-four FP16 tensors. Core AI may pad individual axes.
  mutating func write(_ f: (UnsafeMutablePointer<Float16>, [Int]) -> Void) {
    let s = strides
    mutableView(as: Float16.self).withUnsafeMutablePointer { p, _, _ in f(p, s) }
  }
  func read<T>(_ f: (UnsafePointer<Float16>, [Int]) -> T) -> T {
    let s = strides
    return rawView().withUnsafeBytes { raw, _, _ in f(raw.assumingMemoryBound(to: Float16.self), s)
    }
  }
  static func zeros(_ shape: [Int]) -> NDArray {
    var a = NDArray(shape: shape, scalarType: .float16)
    let span = zip(shape, a.strides).reduce(1) { $0 + ($1.0 - 1) * $1.1 }
    _ = a.mutableRawView().withUnsafeMutableBytes { raw, _, _ in
      raw.initializeMemory(as: Float16.self, repeating: 0, count: span)
    }
    return a
  }
}
@available(macOS 27.0, *)
extension InferenceFunction.Outputs {
  mutating func tensor(_ name: String) throws -> NDArray {
    guard let t = remove(name)?.ndArray, t.scalarType == .float16, t.shape.count == 4 else {
      throw DecisionError("Invalid output " + name)
    }
    return t
  }
}

@available(macOS 27.0, *)
final class Backbone {
  let manifest: Manifest, root: URL, embeddings: Tensor?
  let positionsTable: Data?
  let rotary: [String: [UInt16]]
  let tracing = ProcessInfo.processInfo.environment["ANERUN_TRACE_DECISION"] == "1"
  var trace = [String: Int]()
  private var initialStates = [[Int]: NDArray]()
  var models = [String: AIModel](), functions = [String: InferenceFunction]()
  init(_ root: URL, _ m: Manifest, _ w: Weights?) throws {
    var tables = [String: [UInt16]]()
    if m.elephant || m.valen {
      for (kind, width) in [("text", 32), ("vision", m.valen ? 16 : 18)] {
        guard let name = m.rotaryTables?[kind], name == URL(fileURLWithPath: name).lastPathComponent
        else {
          throw DecisionError("Missing rotary constant table")
        }
        let data = try Data(contentsOf: root.bundleChild(name))
        let limit = m.valen ? m.context : 2048
        guard data.count == limit * width * 4 else {
          throw DecisionError("Invalid rotary table size")
        }
        tables[kind] = data.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
      }
    }
    rotary = tables
    self.root = root
    manifest = m
    if m.elephant || m.valen {
      positionsTable = nil
      embeddings = nil
      return
    }
    if m.embeddingLookup != nil {
      let data = try Data(contentsOf: root.bundleChild("rotary.fp16"), options: .mappedIfSafe)
      guard data.count == m.context * m.rotaryDim * 2 else {
        throw DecisionError("Invalid rotary table size")
      }
      positionsTable = data
      embeddings = nil
      return
    }
    positionsTable = nil
    guard let w else { throw DecisionError("Missing host embedding") }
    let table = try w.get(
      (m.valen ? "backbone." : "model.") + "language_model.embed_tokens.weight")
    embeddings = table
    guard table.shape.count == 2, table.shape[1] == m.hidden else {
      throw DecisionError("Invalid token embedding matrix")
    }
  }
  func function(_ name: String, in asset: String) async throws -> InferenceFunction {
    if tracing { trace[name, default: 0] += 1 }
    let entries =
      (manifest.assets + manifest.deviceExtras).first { $0.asset == asset }?.functions ?? manifest
      .visionFunctions ?? []
    let matches = entries.filter { $0 == name || Manifest.logicalFunction($0) == name }
    guard matches.count == 1, let physicalName = matches.first else {
      throw DecisionError("Missing or ambiguous function " + name + " in " + asset)
    }
    if let f = functions[physicalName] { return f }
    let model: AIModel
    if let cached = models[asset] {
      model = cached
    } else {
      model = try await Self.specialize(root.bundleChild(asset))
      models[asset] = model
    }
    let f = try Self.load(physicalName, from: model, asset: asset)
    functions[physicalName] = f
    return f
  }
  private static func specialize(_ url: URL) async throws -> AIModel {
    var options = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
    options.expectFrequentReshapes = false
    return try await AIModel.specialize(contentsOf: url, options: options, cachePolicy: .persistent)
  }

  private static func load(_ entry: String, from model: AIModel, asset: String) throws -> InferenceFunction {
    do {
      guard let f = try model.loadFunction(named: entry) else { throw DecisionError("Missing function " + entry) }
      return f
    } catch where String(describing: error) == "nilError" {
      throw CompiledFunctionLoadFailure(asset: asset, function: entry)
    }
  }

  /// Overlap independent model loads inside the first request. The recurrent evaluations stay serial.
  private func loadValenFunctions(for ids: [Int], capacity: Int) async throws {
    guard let layout = manifest.embeddingLookup else { throw DecisionError("Missing Valen embedding banks") }
    var needed = [(Manifest.Asset, [String])]()
    for (i, asset) in manifest.assets.enumerated() where models[asset.asset] == nil {
      let names = ["scan\(i)", "attend\(i)c\(capacity)"]
      needed.append((asset, asset.functions.filter { names.contains(Manifest.logicalFunction($0)) }))
    }
    let bankIndices = Set(ids.map { $0 / layout.bankSize })
    for i in bankIndices.sorted() where layout.assets.indices.contains(i) {
      let bank = layout.assets[i]
      if models[bank.asset] == nil { needed.append((bank, bank.functions)) }
    }
    for asset in [manifest.rolePool, manifest.roleMixer, manifest.decisionHead, manifest.resultStats].compactMap({ $0 })
    where models[asset.asset] == nil { needed.append((asset, asset.functions)) }
    if !needed.isEmpty { cache(try await Self.loadAssets(needed, in: root)) }
  }

  func encodeImage(_ data: Data, with vision: Vision) async throws -> VisualFeatures {
    guard manifest.valen else { return try await vision.encode(data, using: self) }
    // All text scans are needed regardless of the image grid or selected attention tier.
    let scans = manifest.assets.enumerated().compactMap { index, asset -> (Manifest.Asset, [String])? in
      guard models[asset.asset] == nil else { return nil }
      return (asset, asset.functions.filter { Manifest.logicalFunction($0) == "scan\(index)" })
    }
    guard !scans.isEmpty else { return try await vision.encode(data, using: self) }
    async let prepared = Self.loadAssets(scans, in: root)
    let features = try await vision.encode(data, using: self)
    cache(try await prepared)
    return features
  }

  private typealias LoadedAsset = (String, AIModel, [String: InferenceFunction])

  private func cache(_ assets: [LoadedAsset]) {
    for (name, model, entries) in assets {
      models[name] = model
      functions.merge(entries) { existing, _ in existing }
    }
  }

  private static func loadAssets(_ needed: [(Manifest.Asset, [String])], in directory: URL)
    async throws -> [LoadedAsset]
  {
    try await withThrowingTaskGroup(of: LoadedAsset.self) { group in
      var completed = [LoadedAsset]()
      for (index, item) in needed.enumerated() {
        if index >= 4, let next = try await group.next() { completed.append(next) }
        group.addTask {
          try Task.checkCancellation()
          let (asset, names) = item
          let model = try await specialize(directory.bundleChild(asset.asset))
          var entries = [String: InferenceFunction]()
          for name in names { entries[name] = try load(name, from: model, asset: asset.asset) }
          return (asset.asset, model, entries)
        }
      }
      for try await result in group { completed.append(result) }
      return completed
    }
  }

  func prepare() async throws {
    for asset in manifest.assets + manifest.deviceExtras {
      for name in asset.functions { _ = try await function(name, in: asset.asset) }
    }
    if let asset = manifest.visionAsset {
      for name in manifest.visionFunctions ?? [] { _ = try await function(name, in: asset) }
    }
  }
  func run(_ prompt: Prompt, vision: VisualFeatures?) async throws -> [NDArray] {
    let m = manifest
    let w = m.width
    let h = m.hidden
    let n = prompt.ids.count
    guard let capacity = m.tiers.first(where: { $0 >= n }) else {
      throw DecisionError("Context limit exceeded")
    }
    if m.valen { try await loadValenFunctions(for: prompt.ids, capacity: capacity) }
    let causal = m.causal
    let count = (n + w - 1) / w
    let imageID = 248056
    var chunks: [NDArray]
    if let lookup = m.embeddingLookup {
      chunks = try await lookupTokens(
        prompt.ids, using: lookup, skipping: vision == nil ? nil : imageID)
    } else {
      chunks = (0..<count).map { _ in NDArray.zeros([1, h, 1, w]) }
    }
    var rope = [[String: NDArray]]()
    var masks = [NDArray]()
    var position = 0
    var imageIndex = 0
    for chunk in 0..<count {
      var x = chunks[chunk]
      var cos = NDArray.zeros([1, 1, m.rotaryDim / 2, w])
      var sin = cos
      if let table = positionsTable {
        let plane = m.rotaryDim / 2 * w
        table.withUnsafeBytes { raw in
          let ptr = raw.bindMemory(to: Float16.self)
          cos.write { pc, sc in
            sin.write { ps, ss in
              for d in 0..<(m.rotaryDim / 2) {
                for t in 0..<w {
                  let offset = 2 * chunk * plane + d * w + t
                  pc[d * sc[2] + t * sc[3]] = ptr[offset]
                  ps[d * ss[2] + t * ss[3]] = ptr[offset + plane]
                }
              }
            }
          }
        }
      } else {

        var positions = [(Int, Int, Int)]()
        for t in 0..<w {
          let i = chunk * w + t
          if i < n, prompt.ids[i] == imageID, let vision {
            guard imageIndex < vision.count else {
              throw DecisionError("Image slots mismatch")
            }
            positions.append(
              (
                position, position + imageIndex / vision.columns,
                position + imageIndex % vision.columns
              ))
            let imageRow = imageIndex
            imageIndex += 1
            if imageIndex == vision.count {
              position += max(vision.columns, vision.count / vision.columns)
            }
            vision.tensor.read { source, stride in
              x.write { p, s in
                for c in 0..<h { p[c * s[1] + t * s[3]] = source[c * stride[1] + imageRow * stride[3]] }
              }
            }
          } else {
            positions.append((position, position, position))
            position += 1
            if i < n && m.embeddingLookup == nil {
              guard let embeddings else { throw DecisionError("Missing token embeddings") }
              let id = prompt.ids[i]
              guard id >= 0, id < embeddings.shape[0] else {
                throw DecisionError("Token outside embedding table")
              }
              let row = embeddings.floats(id * h..<(id + 1) * h)
              x.write { p, s in for c in 0..<h { p[c * s[1] + t * s[3]] = Float16(row[c] * 64) } }
            }
          }
        }
        cos.write { pc, sc in
          sin.write { ps, ss in
            for d in 0..<(m.rotaryDim / 2) {

              for (t, axes) in positions.enumerated() {
                let index =
                  d % 3 == 1 && d < 33 ? axes.1 : (d % 3 == 2 && d < 30 ? axes.2 : axes.0)
                if let table = rotary["text"] {
                  let offset = (index * 32 + d) * 2
                  pc[d * sc[2] + t * sc[3]] = Float16(bitPattern: table[offset])
                  ps[d * ss[2] + t * ss[3]] = Float16(bitPattern: table[offset + 1])
                  continue
                }
                let angle = Float(index) * (1 / powf(10_000_000, Float(2 * d) / Float(m.rotaryDim)))
                pc[d * sc[2] + t * sc[3]] = Float16(cosf(angle))
                ps[d * ss[2] + t * ss[3]] = Float16(sinf(angle))
              }
            }
          }
        }
      }
      rope.append(["cos": cos, "sin": sin])
      chunks[chunk] = x
      let queries = m.blockMask ? w : 1
      var mask = NDArray.zeros([1, 1, queries, capacity])
      mask.write { p, s in
        for q in 0..<queries {
          for k in 0..<capacity {
            let i = chunk * w + q
            let visible: Bool
            if causal {
              visible = k < n && k <= i
            } else {
              visible =
                k < n
                && (!m.blockMask
                  || (i >= n
                    ? k == 0
                    : k < prompt.prefix
                      || (i >= prompt.prefix
                        && (prompt.segments[k] == 0 || prompt.segments[k] == prompt.segments[i]))))
            }
            p[q * s[2] + k * s[3]] = visible ? 0 : -30000
          }
        }
      }
      masks.append(mask)
    }
    // The 2B bundle exports a fused entry point for single-chunk prompts.
    if !m.valen, count == 1, m.prompts?.contains(w) == true {
      var x = chunks[0]
      var mask = NDArray.zeros([1, 1, 1, w])
      mask.write { p, s in for t in n..<w { p[t * s[3]] = -30000 } }
      for (i, asset) in m.assets.enumerated() {
        let f = try await function("prompt\(i)c\(w)", in: asset.asset)
        var inputs = rope[0]
        inputs["hidden_in"] = x
        inputs["mask"] = mask
        var out = try await f.run(inputs: inputs)
        x = try out.tensor("hidden_out")
      }
      guard x.shape == [1, h, 1, w] else { throw DecisionError("Invalid prompt output shape") }
      return [x]
    }
    var keys = NDArray.zeros([1, m.kvHeads, capacity, m.headDim])
    var values = keys
    for block in m.execution {
      for _ in 0..<block.passes {
        let previous = chunks
        for index in block.stages {
          let asset = m.assets[index].asset
          let scan = try await function("scan\(index)", in: asset)
          let attends = m.stageKinds[index] == "attention"
          guard case .ndArray(let desc)? = scan.descriptor.inputDescriptor(of: "packed_state"),
            desc.scalarType == .float16
          else { throw DecisionError("Invalid recurrence state") }
          var state = initialState(desc.shape)
          var saved = [[String: NDArray]]()
          for j in 0..<count {
            var inputs = attends ? rope[j] : [:]
            inputs["hidden_in"] = chunks[j]
            inputs["packed_state"] = state
            var out = try await scan.run(inputs: inputs)
            chunks[j] = try out.tensor("hidden_out")
            state = try out.tensor("packed_state_out")
            if attends {
              saved.append(["query": try out.tensor("query"), "gate": try out.tensor("gate")])
              try Self.copyKV(out.tensor("key"), into: &keys, offset: j * w, width: w)
              try Self.copyKV(out.tensor("value"), into: &values, offset: j * w, width: w)
            }
          }
          if attends {
            let f = try await function("attend\(index)c\(capacity)", in: asset)
            for j in 0..<count {
              var inputs = saved[j]
              inputs["key"] = keys
              inputs["value"] = values
              inputs["mask"] = masks[j]
              inputs["hidden_in"] = chunks[j]
              var out = try await f.run(inputs: inputs)
              chunks[j] = try out.tensor("hidden_out")
            }
          }
          if let r = m.rounding {
            let scale = try await function("relusc", in: r[0].asset)
            let bias = try await function("two", in: r[1].asset)
            let restore = try await function("finite", in: r[2].asset)
            for j in chunks.indices {
              var scaled = try await scale.run(inputs: ["x": chunks[j]])
              let sc = try scaled.tensor("s")
              var biased = try await bias.run(inputs: ["x": chunks[j], "s": sc])
              var result = try await restore.run(inputs: [
                "x": chunks[j], "b": try biased.tensor("b"), "c": try biased.tensor("c"), "s": sc,
              ])
              chunks[j] = try result.tensor("y")
            }
          }
          guard chunks.allSatisfy({ $0.shape == [1, h, 1, w] }) else {
            throw DecisionError("Invalid residual shape")
          }
        }
        if block.passes > 1 {
          for j in chunks.indices {
            if let asset = m.residualBlend {
              let f = try await function("blend2", in: asset.asset)
              var out = try await f.run(inputs: ["previous": previous[j], "current": chunks[j]])
              let next = try out.tensor("hidden_out")
              guard next.shape == [1, h, 1, w] else {
                throw DecisionError("Invalid residual blend shape")
              }
              chunks[j] = next
              continue
            }
            var blend = chunks[j]
            previous[j].read { a, sa in
              blend.write { b, sb in
                for c in 0..<h {
                  for t in 0..<w {
                    let i = c * sb[1] + t * sb[3]
                    let old = Float(a[c * sa[1] + t * sa[3]])
                    b[i] = Float16(old + (Float(b[i]) - old) / Float(block.passes))
                  }
                }
              }
            }
            chunks[j] = blend
          }
        }
      }
    }
    return chunks
  }

  private func initialState(_ shape: [Int]) -> NDArray {
    // Valen layers with the same state shape share an immutable initial input.
    // Each evaluation returns the subsequent state in a separate output.
    guard manifest.valen else { return NDArray.zeros(shape) }
    if let cached = initialStates[shape] { return cached }
    let zero = NDArray.zeros(shape)
    initialStates[shape] = zero
    return zero
  }

  func lookupTokens(_ ids: [Int], using layout: Manifest.Lookup, skipping imageID: Int?)
    async throws -> [NDArray]
  {
    let w = manifest.width
    let h = manifest.hidden
    let lanes = layout.laneWidth
    guard !ids.isEmpty, ids.allSatisfy({ (0..<layout.vocabSize).contains($0) })
    else { throw DecisionError("Token outside device lookup range") }
    var chunks = (0..<((ids.count + w - 1) / w)).map { _ in NDArray.zeros([1, h, 1, w]) }
    let grouped = Dictionary(
      grouping: ids.indices.filter { ids[$0] != imageID }, by: { ids[$0] / layout.bankSize })
    for bank in grouped.keys.sorted() {
      let asset = layout.assets[bank]
      let f = try await function(asset.functions[0], in: asset.asset)
      let columns = min(layout.bankSize, layout.vocabSize - bank * layout.bankSize)
      let positions = grouped[bank]!
      for start in stride(from: 0, to: positions.count, by: lanes) {
        let batch = positions[start..<min(start + lanes, positions.count)]
        var selector = NDArray.zeros([1, columns, 1, lanes])
        selector.write { p, s in
          for (lane, position) in batch.enumerated() {
            p[(ids[position] % layout.bankSize) * s[1] + lane * s[3]] = 1
          }
        }
        var out = try await f.run(inputs: ["one_hot": selector])
        var rows = try out.tensor("hidden_out")
        if let asset = manifest.embeddingUnscale {
          let unscale = try await function("unscale", in: asset.asset)
          var result = try await unscale.run(inputs: ["features": rows])
          rows = try result.tensor("scaled")
        }
        guard rows.shape == [1, h, 1, lanes] else {
          throw DecisionError("Invalid lookup output shape")
        }
        rows.read { q, qs in
          for (lane, position) in batch.enumerated() {
            let token = position % w
            chunks[position / w].write { p, ps in
              for c in 0..<h { p[c * ps[1] + token * ps[3]] = q[c * qs[1] + lane * qs[3]] }
            }
          }
        }
      }
    }
    return chunks
  }
  private static func copyKV(_ src: NDArray, into dst: inout NDArray, offset: Int, width: Int)
    throws
  {
    let shape = dst.shape
    guard src.shape == [1, shape[1], width, shape[3]], offset + width <= shape[2] else {
      throw DecisionError("Invalid attention cache shape")
    }
    src.read { p, a in
      dst.write { q, b in
        for h in 0..<shape[1] {
          for t in 0..<width {
            let from = p + h * a[1] + t * a[2]
            let to = q + h * b[1] + (offset + t) * b[2]
            if a[3] == 1 && b[3] == 1 {
              to.update(from: from, count: shape[3])
            } else {
              for d in 0..<shape[3] { to[d * b[3]] = from[d * a[3]] }
            }
          }
        }
      }
    }
  }
}

/// Inspect the compiled graph, since a preferred compute unit permits fallback.
@available(macOS 27.0, *)
extension Backbone {
  func verifyPlacement() throws {
    let assets =
      (manifest.assets + manifest.deviceExtras).map { ($0.asset, $0.sourceHash) }
      + (manifest.visionAsset.map { [($0, manifest.visionSourceHash ?? "")] } ?? [])
    struct Metadata: Decodable { let sourceHash: String }
    for (asset, digest) in assets {
      let url = try root.bundleChild(asset)
      let meta = try url.decode("metadata.json", as: Metadata.self)
      guard meta.sourceHash.lowercased() == digest.lowercased(),
        let files = FileManager.default.enumerator(at: url, includingPropertiesForKeys: nil),
        let path = files.compactMap({ $0 as? URL }).first(where: {
          $0.lastPathComponent == "manifest.plist"
        })
      else { throw DecisionError("Missing compiled identity for " + asset) }
      let raw = try PropertyListSerialization.propertyList(
        from: Data(contentsOf: path), format: nil)
      guard let plist = raw as? [String: Any],
        let versions = plist["Package Version"] as? [String: [String: Any]],
        !versions.isEmpty
      else { throw DecisionError("Invalid placement manifest for " + asset) }
      var placedNames = Set<String>()
      for v in versions.values {
        guard let modules = v["Optimized Modules"] as? [String: [String: Any]], !modules.isEmpty
        else {
          throw DecisionError("Missing compiled modules in " + asset)
        }
        for module in modules.values {
          guard let entries = module["Entry Function Attributes"] as? [String: [Any]],
            !entries.isEmpty,
            entries.values.allSatisfy({ attrs in
              attrs.contains(where: { ($0 as? String) == "mps.fullyPlacedOnANE" })
            })
          else { throw DecisionError("Incomplete ANE placement in " + asset) }
          placedNames.formUnion(entries.keys)
        }
      }
      if manifest.embeddingLookup != nil {
        let expected =
          (manifest.assets + manifest.deviceExtras).first { $0.asset == asset }?.functions
          ?? manifest.visionFunctions ?? []
        guard !expected.isEmpty, placedNames.count == expected.count,
          expected.allSatisfy({ name in
            placedNames.contains { $0 == name || $0.hasPrefix(name + "_") }
          })
        else { throw DecisionError("Unexpected compiled functions in " + asset) }
      }
    }
  }
}
#endif
