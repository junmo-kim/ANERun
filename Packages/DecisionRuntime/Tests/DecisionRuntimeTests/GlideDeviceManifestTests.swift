#if canImport(CoreAI)
import CoreAI
import Foundation
import XCTest

@testable import DecisionRuntime

@available(macOS 27.0, *)
final class GlideDeviceManifestTests: XCTestCase {
  private func asset(_ name: String, _ functions: [String]) -> [String: Any] {
    ["asset": "\(name).\(AIModel.deviceArchitectureName).aimodelc",
     "functions": functions, "source_hash": String(repeating: "a", count: 64)]
  }

  func testCompleteDeviceScheduleIsRequiredBeforeOpeningWeights() throws {
    let kinds = Array(repeating: ["linear", "linear", "linear", "attention"], count: 8).flatMap { $0 }
    let groups = (0..<12).map { [$0] } + [Array(12...17), Array(18...23)] + (24..<32).map { [$0] }
    let banks = (0..<61).map { asset("lookup\($0)", ["lookup\($0)"]) }
    let base: [String: Any] = [
      "architecture": AIModel.deviceArchitectureName,
      "source": ["repo": "fastino/GLiDE-9B", "revision": "ab54df5acba0eebf7df45560ffdaacd64e32df3b"],
      "assets": kinds.indices.map { i in
        asset("stage\(i)", ["scan\(i)"] + (kinds[i] == "attention" ? ["attend\(i)c256"] : []))
      },
      "stage_kinds": kinds, "execution": groups.map { ["stages": $0, "passes": $0.count == 1 ? 1 : 2] },
      "width": 128, "context": 256, "tiers": [256], "rotary_dim": 64, "hidden": 4096,
      "kv_heads": 4, "head_dim": 256, "heads": 16, "block_mask": false, "prompts": [],
      "embedding_lookup": ["bank_size": 4096, "lane_width": 16, "vocab_size": 248320, "assets": banks],
      "decision_head": asset("head", ["readout"]), "residual_blend": asset("blend2", ["blend2"]),
      "result_stats": asset("readoutStats", ["readoutStats"])
    ]
    func validate(_ json: [String: Any]) throws {
      try JSONDecoder().decode(Manifest.self, from: JSONSerialization.data(withJSONObject: json)).validate()
    }
    try validate(base)
    for key in ["embedding_lookup", "decision_head", "residual_blend", "result_stats"] {
      XCTAssertThrowsError(try validate(base.filter { $0.key != key }))
    }
    var wrong = base
    wrong["decision_head"] = asset("../head", ["readout"])
    XCTAssertThrowsError(try validate(wrong))
    wrong = base
    wrong["prompts"] = [128]
    XCTAssertThrowsError(try validate(wrong))
  }
}
#endif
