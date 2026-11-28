// Store cos and sin planes in the same chunk order and FP16 encoding as the runtime.
import Foundation

let context = Int(CommandLine.arguments[2])!
precondition(context > 0 && context.isMultiple(of: 128))
var data = Data()
for chunk in 0..<(context / 128) {
  for plane in 0..<2 {
    for d in 0..<32 {
      let frequency = 1 / powf(10_000_000, Float(2 * d) / 64)
      for token in 0..<128 {
        let angle = Float(chunk * 128 + token) * frequency
        var bits = Float16(plane == 0 ? cosf(angle) : sinf(angle)).bitPattern.littleEndian
        withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
      }
    }
  }
}
try data.write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
