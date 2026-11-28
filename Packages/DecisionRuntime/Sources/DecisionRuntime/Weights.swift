import Accelerate
import Foundation

/// Each view retains the mapped file. Only the requested embedding rows are faulted in.
struct Weights {
  struct Entry: Decodable {
    enum CodingKeys: String, CodingKey {
      case dtype
      case shape
      case dataOffsets = "data_offsets"
    }
    let dtype: String, shape: [Int], dataOffsets: [Int]
  }
  let file: Data, entries: [String: Entry], start: Int
  init(_ url: URL) throws {
    file = try Data(contentsOf: url, options: .alwaysMapped)
    guard file.count >= 8 else { throw DecisionError("Truncated tensor header") }
    let size = file.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
    guard size <= min(100_000_000, file.count - 8) else {
      throw DecisionError("Invalid tensor header length")
    }
    start = 8 + Int(size)
    guard var header = try JSONSerialization.jsonObject(with: file[8..<start]) as? [String: Any]
    else {
      throw DecisionError("Invalid tensor header")
    }
    header.removeValue(forKey: "__metadata__")
    entries = try JSONDecoder().decode(
      [String: Entry].self, from: JSONSerialization.data(withJSONObject: header))
  }
  func get(_ name: String) throws -> Tensor {
    guard let e = entries[name], ["F16", "BF16", "F32"].contains(e.dtype),
      e.dataOffsets.count == 2,
      e.dataOffsets[0] >= 0, e.dataOffsets[1] >= e.dataOffsets[0],
      e.dataOffsets[1] <= file.count - start
    else { throw DecisionError("Invalid tensor: \(name)") }
    let stride = e.dtype == "F32" ? 4 : 2
    var count = stride
    for dim in e.shape {
      guard dim > 0, count <= Int.max / dim else { throw DecisionError("Invalid tensor shape") }
      count *= dim
    }
    guard e.dataOffsets[1] - e.dataOffsets[0] == count else {
      throw DecisionError("Tensor size mismatch")
    }
    return Tensor(
      file: file, offset: start + e.dataOffsets[0], dtype: e.dtype, shape: e.shape,
      count: count / stride)
  }
}
struct Tensor {
  let file: Data, offset: Int, dtype: String, shape: [Int], count: Int
  func floats(_ range: Range<Int>) -> [Float] {
    file.withUnsafeBytes { buf in
      let p = buf.baseAddress!.advanced(by: offset)
      switch dtype {
      case "F32": return range.map { p.loadUnaligned(fromByteOffset: $0 * 4, as: Float.self) }
      case "F16":
        return range.map { Float(p.loadUnaligned(fromByteOffset: $0 * 2, as: Float16.self)) }
      default:
        return range.map {
          Float(bitPattern: UInt32(p.loadUnaligned(fromByteOffset: $0 * 2, as: UInt16.self)) << 16)
        }
      }
    }
  }
}

struct Linear {
  let matrix: [Float], bias: [Float], input: Int
  init(_ w: Weights, _ name: String, input: Int, output: Int, hasBias: Bool = true) throws {
    let tensor = try w.get(name + "weight")
    guard tensor.shape == [output, input] else {
      throw DecisionError("Invalid projection dimensions in " + name)
    }
    self.input = input
    matrix = tensor.floats(0..<tensor.count)
    if hasBias {
      let b = try w.get(name + "bias")
      guard b.shape == [output] else { throw DecisionError("Invalid bias in " + name) }
      bias = b.floats(0..<output)
    } else {
      bias = Array(repeating: 0, count: output)
    }
  }
  func callAsFunction(_ x: [Float]) -> [Float] {
    var y = [Float](repeating: 0, count: bias.count)
    vDSP_mmul(matrix, 1, x, 1, &y, 1, vDSP_Length(y.count), 1, vDSP_Length(input))
    vDSP_vadd(y, 1, bias, 1, &y, 1, vDSP_Length(y.count))
    return y
  }
}
