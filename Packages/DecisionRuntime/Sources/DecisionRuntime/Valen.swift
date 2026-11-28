#if canImport(CoreAI)
import CoreAI
import Foundation

@available(macOS 27.0, *)
extension Session {
  func valenAnswer(_ chunks: [NDArray], branch: Branch) async throws -> DecisionAnswer {
    guard let pool = manifest.rolePool, let mixer = manifest.roleMixer,
      let head = manifest.decisionHead, branch.roles.count == branch.candidates.count + 3
    else { throw DecisionError("Missing Valen role layout") }
    let width = manifest.width, hidden = manifest.hidden
    let poolFunction = try await backbone.function("valenPool", in: pool.asset)
    var pooled = [NDArray]()
    for span in branch.roles {
      guard !span.isEmpty, span.lowerBound >= 0, span.upperBound <= chunks.count * width else {
        throw DecisionError("Invalid Valen role span")
      }
      var length = NDArray.zeros([1, 1, 1, 1])
      length.write { p, _ in p[0] = Float16(span.count) }
      var previous = NDArray.zeros([1, hidden, 1, 1])
      for index in (span.lowerBound / width)...((span.upperBound - 1) / width) {
        var mask = NDArray.zeros([1, 1, 1, width])
        mask.write { p, s in
          for t in 0..<width where span.contains(index * width + t) { p[t * s[3]] = 1 }
        }
        var result = try await poolFunction.run(inputs: [
          "hidden": chunks[index], "valid": mask, "length": length, "previous": previous,
        ])
        previous = try result.tensor("pooled")
        guard previous.shape == [1, hidden, 1, 1] else { throw DecisionError("Invalid pooled role shape") }
      }
      pooled.append(previous)
    }
    let mix = try await backbone.function("valenMixer", in: mixer.asset)
    var logits = NDArray.zeros([1, 256, 1, 1])
    var mask = NDArray.zeros([1, 256, 1, 1])
    for i in branch.candidates.indices {
      var roles = NDArray.zeros([1, hidden, 1, 4])
      for (lane, index) in [0, 1, 2 + i, pooled.count - 1].enumerated() {
        pooled[index].read { a, sa in
          roles.write { b, sb in
            for c in 0..<hidden { b[c * sb[1] + lane * sb[3]] = a[c * sa[1]] }
          }
        }
      }
      var result = try await mix.run(inputs: ["pooled": roles])
      let value = try result.tensor("logit")
      guard value.shape == [1, 1, 1, 1] else { throw DecisionError("Invalid Valen logit shape") }
      value.read { a, _ in logits.write { b, s in b[i * s[1]] = a[0] } }
    }
    mask.write { p, s in for i in branch.candidates.count..<256 { p[i * s[1]] = -30000 } }
    let readout = try await backbone.function("readout", in: head.asset)
    var result = try await readout.run(inputs: ["logits": logits, "mask": mask])
    return try await formatDeviceAnswer(result.tensor("probabilities"), branch: branch)
  }
}
#endif
