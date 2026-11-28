#if canImport(CoreAI)
import CoreAI
import CoreGraphics
import Foundation
import ImageIO

@available(macOS 27.0, *)
struct VisualFeatures { let tensor: NDArray, count: Int, columns: Int }
@available(macOS 27.0, *)
struct Vision {
  // FP16 encoding of each decoded sRGB byte after normalization to [-1, 1].
  static let pixelBits: [UInt16] = [
    0xbc00, 0xbbf0, 0xbbe0, 0xbbd0, 0xbbc0, 0xbbb0, 0xbba0, 0xbb90,
    0xbb7f, 0xbb6f, 0xbb5f, 0xbb4f, 0xbb3f, 0xbb2f, 0xbb1f, 0xbb0f,
    0xbaff, 0xbaef, 0xbadf, 0xbacf, 0xbabf, 0xbaaf, 0xba9f, 0xba8f,
    0xba7e, 0xba6e, 0xba5e, 0xba4e, 0xba3e, 0xba2e, 0xba1e, 0xba0e,
    0xb9fe, 0xb9ee, 0xb9de, 0xb9ce, 0xb9be, 0xb9ae, 0xb99e, 0xb98e,
    0xb97d, 0xb96d, 0xb95d, 0xb94d, 0xb93d, 0xb92d, 0xb91d, 0xb90d,
    0xb8fd, 0xb8ed, 0xb8dd, 0xb8cd, 0xb8bd, 0xb8ad, 0xb89d, 0xb88d,
    0xb87c, 0xb86c, 0xb85c, 0xb84c, 0xb83c, 0xb82c, 0xb81c, 0xb80c,
    0xb7f8, 0xb7d8, 0xb7b8, 0xb798, 0xb777, 0xb757, 0xb737, 0xb717,
    0xb6f7, 0xb6d7, 0xb6b7, 0xb697, 0xb676, 0xb656, 0xb636, 0xb616,
    0xb5f6, 0xb5d6, 0xb5b6, 0xb596, 0xb575, 0xb555, 0xb535, 0xb515,
    0xb4f5, 0xb4d5, 0xb4b5, 0xb495, 0xb474, 0xb454, 0xb434, 0xb414,
    0xb3e8, 0xb3a8, 0xb367, 0xb327, 0xb2e7, 0xb2a7, 0xb266, 0xb226,
    0xb1e6, 0xb1a6, 0xb165, 0xb125, 0xb0e5, 0xb0a5, 0xb064, 0xb024,
    0xafc8, 0xaf47, 0xaec7, 0xae46, 0xadc6, 0xad45, 0xacc5, 0xac44,
    0xab88, 0xaa87, 0xa986, 0xa885, 0xa707, 0xa505, 0xa206, 0x9c04,
    0x1c04, 0x2206, 0x2505, 0x2707, 0x2885, 0x2986, 0x2a87, 0x2b88,
    0x2c44, 0x2cc5, 0x2d45, 0x2dc6, 0x2e46, 0x2ec7, 0x2f47, 0x2fc8,
    0x3024, 0x3064, 0x30a5, 0x30e5, 0x3125, 0x3165, 0x31a6, 0x31e6,
    0x3226, 0x3266, 0x32a7, 0x32e7, 0x3327, 0x3367, 0x33a8, 0x33e8,
    0x3414, 0x3434, 0x3454, 0x3474, 0x3495, 0x34b5, 0x34d5, 0x34f5,
    0x3515, 0x3535, 0x3555, 0x3575, 0x3596, 0x35b6, 0x35d6, 0x35f6,
    0x3616, 0x3636, 0x3656, 0x3676, 0x3697, 0x36b7, 0x36d7, 0x36f7,
    0x3717, 0x3737, 0x3757, 0x3777, 0x3798, 0x37b8, 0x37d8, 0x37f8,
    0x380c, 0x381c, 0x382c, 0x383c, 0x384c, 0x385c, 0x386c, 0x387c,
    0x388d, 0x389d, 0x38ad, 0x38bd, 0x38cd, 0x38dd, 0x38ed, 0x38fd,
    0x390d, 0x391d, 0x392d, 0x393d, 0x394d, 0x395d, 0x396d, 0x397d,
    0x398e, 0x399e, 0x39ae, 0x39be, 0x39ce, 0x39de, 0x39ee, 0x39fe,
    0x3a0e, 0x3a1e, 0x3a2e, 0x3a3e, 0x3a4e, 0x3a5e, 0x3a6e, 0x3a7e,
    0x3a8f, 0x3a9f, 0x3aaf, 0x3abf, 0x3acf, 0x3adf, 0x3aef, 0x3aff,
    0x3b0f, 0x3b1f, 0x3b2f, 0x3b3f, 0x3b4f, 0x3b5f, 0x3b6f, 0x3b7f,
    0x3b90, 0x3ba0, 0x3bb0, 0x3bc0, 0x3bd0, 0x3be0, 0x3bf0, 0x3c00,
  ]
  let channels: Int, rotaryQuarter: Int
  init(elephant: Bool) {
    channels = elephant ? 1152 : 1024
    rotaryQuarter = elephant ? 18 : 16
  }
  func encode(_ data: Data, using backbone: Backbone) async throws -> VisualFeatures {
    guard let source = CGImageSourceCreateWithData(data as CFData, nil),
      let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
      let width = props[kCGImagePropertyPixelWidth] as? Int,
      let height = props[kCGImagePropertyPixelHeight] as? Int,
      width > 0, height > 0, Double(max(width, height)) / Double(min(width, height)) <= 200,
      let fullAsset = backbone.manifest.visionAsset
    else { throw DecisionError("Unsupported image", input: true) }
    let area = Double(width) * Double(height)
    var w = max(32, Int((Double(width) / 32).rounded(.toNearestOrEven)) * 32)
    var h = max(32, Int((Double(height) / 32).rounded(.toNearestOrEven)) * 32)
    if Double(w) * Double(h) > 262144 {
      if backbone.manifest.elephant {
        throw DecisionError("Image exceeds bundle patch capacity", input: true)
      }
      let scale = sqrt(262144 / area)
      w = max(32, Int(floor(Double(width) * scale / 32)) * 32)
      h = max(32, Int(floor(Double(height) * scale / 32)) * 32)
    } else if w * h < 65536 {
      let scale = sqrt(65536 / area)
      w = max(32, Int(ceil(Double(width) * scale / 32)) * 32)
      h = max(32, Int(ceil(Double(height) * scale / 32)) * 32)
    }
    let gw = w / 16
    let gh = h / 16
    let patches = gw * gh
    let small = patches <= 256 ? backbone.manifest.smallVision : nil
    let capacity = small == nil ? 1024 : 256
    guard patches <= capacity,
      let image = CGImageSourceCreateThumbnailAtIndex(
        source, 0,
        [
          kCGImageSourceCreateThumbnailFromImageAlways: true,
          kCGImageSourceThumbnailMaxPixelSize: max(w, h),
        ] as CFDictionary)
    else { throw DecisionError("Image exceeds the exported patch budget", input: true) }
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    guard let color = CGColorSpace(name: CGColorSpace.sRGB) else {
      throw DecisionError("Missing sRGB color space")
    }
    try rgba.withUnsafeMutableBytes { buffer in
      guard
        let ctx = CGContext(
          data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
          space: color, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
      else {
        throw DecisionError("Image resize failed")
      }
      ctx.interpolationQuality = .high
      ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    }
    var pixels = NDArray.zeros([1, 1536, 1, capacity])
    var pos = NDArray.zeros([1, channels, 1, capacity])
    var cos = NDArray.zeros([1, 1, rotaryQuarter * 2, capacity])
    var sin = cos
    var mask = NDArray.zeros([1, 1, 1, capacity])
    mask.write { p, s in for i in patches..<capacity { p[i * s[3]] = -30000 } }
    for i in 0..<patches {
      let group = i / 4
      let row = (group / (gw / 2)) * 2 + (i % 4) / 2
      let col = (group % (gw / 2)) * 2 + i % 2
      pixels.write { p, s in
        for c in 0..<3 {
          for y in 0..<16 {
            for x in 0..<16 {
              let channel = c * 512 + y * 16 + x
              let byte = rgba[((row * 16 + y) * w + col * 16 + x) * 4 + c]
              let value = Float16(bitPattern: Self.pixelBits[Int(byte)])
              p[channel * s[1] + i * s[3]] = value
              p[(channel + 256) * s[1] + i * s[3]] = value
            }
          }
        }
      }
      cos.write { pc, sc in
        sin.write { ps, ss in
          for d in 0..<(rotaryQuarter * 2) {
            if let lookup = backbone.rotary["vision"] {
              let index = d < rotaryQuarter ? row : col
              let offset = (index * rotaryQuarter + d % rotaryQuarter) * 2
              pc[d * sc[2] + i * sc[3]] = Float16(bitPattern: lookup[offset])
              ps[d * ss[2] + i * ss[3]] = Float16(bitPattern: lookup[offset + 1])
              continue
            }
            let angle =
              Float(d < rotaryQuarter ? row : col)
              * powf(10000, -Float(d % rotaryQuarter) / Float(rotaryQuarter))
            pc[d * sc[2] + i * sc[3]] = Float16(cosf(angle))
            ps[d * ss[2] + i * ss[3]] = Float16(sinf(angle))
          }
        }
      }
    }
    if let asset = backbone.manifest.visionPosition {
      let project = try await backbone.function("visionpos", in: asset.asset)
      guard let name = backbone.manifest.visionFractions else {
        throw DecisionError("Missing vision fraction table")
      }
      let data = try Data(contentsOf: backbone.root.bundleChild(name))
      guard data.count == 511 * 512, (2...512).contains(gh), (2...512).contains(gw) else {
        throw DecisionError("Invalid vision fraction table or grid")
      }
      let fractions = data.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
      func fraction(_ remainder: Int, _ denominator: Int) -> Float16 {
        let i = denominator * (denominator - 1) / 2 + remainder
        return Float16(bitPattern: UInt16(littleEndian: fractions[i]))
      }
      for start in stride(from: 0, to: patches, by: 16) {
        var inputs = [String: NDArray]()
        for key in ["row", "column", "fy", "fx"] {
          var arr = NDArray.zeros([1, 1, 1, 16])
          arr.write { a, s in
            for t in 0..<16 {
              let i = min(start + t, patches - 1)
              let group = i / 4
              let row = (group / (gw / 2)) * 2 + (i % 4) / 2
              let col = (group % (gw / 2)) * 2 + i % 2
              let value: Float16
              switch key {
              case "row": value = Float16(row * 47 / (gh - 1))
              case "column": value = Float16(col * 47 / (gw - 1))
              case "fy": value = fraction(row * 47 % (gh - 1), gh - 1)
              default: value = fraction(col * 47 % (gw - 1), gw - 1)
              }
              a[t * s[3]] = value
            }
          }
          inputs[key] = arr
        }
        var result = try await project.run(inputs: inputs)
        let lanes = try result.tensor("pos")
        guard lanes.shape == [1, channels, 1, 16] else {
          throw DecisionError("Vision position lane shape")
        }
        lanes.read { a, sa in
          pos.write { b, sb in
            for t in 0..<min(16, patches - start) {
              for c in 0..<channels {
                b[c * sb[1] + (start + t) * sb[3]] = a[c * sa[1] + t * sa[3]]
              }
            }
          }
        }
      }
    }
    let f = try await backbone.function("p\(capacity)", in: small?.asset ?? fullAsset)
    var out = try await f.run(inputs: [
      "pixel_values": pixels, "cos": cos, "sin": sin, "attn_mask": mask, "pos_embeds": pos,
    ])
    var hidden = try out.tensor("features")
    if small == nil, let scale = backbone.manifest.featureScale {
      let f = try await backbone.function("featurescale", in: scale.asset)
      var result = try await f.run(inputs: ["features": hidden])
      hidden = try result.tensor("scaled")
    }
    let hsize = backbone.manifest.hidden
    guard hidden.shape == [1, hsize, 1, capacity / 4] else {
      throw DecisionError("Invalid visual features")
    }
    return VisualFeatures(tensor: hidden, count: patches / 4, columns: gw / 2)
  }
}
#endif
