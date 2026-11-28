// Image preprocessing of the official EmbeddingGemma 2 recipe: transformers' Gemma4ImageProcessor (torchvision
// backend, the one Hugging Face uses when torchvision is installed) with the model's processor_config.json
// (max_soft_tokens 280, patch 16, pooling 3, bicubic, rescale 1/255, no normalization).
//   1. decode like PIL: JPEG with JPEGDecoder (libjpeg-turbo's algorithms), other formats with ImageIO without
//      colour management (the raw decoded values); drop alpha and apply the EXIF orientation, like transformers'
//      load_image / load_image_as_tensor for an image given as a path, URL or base64 string
//   2. get_aspect_ratio_preserving_size: scale up or down to the largest size whose sides are multiples of 48 px
//      with at most 280 * 9 patches of 16 px (no padding, no minimum)
//   3. torchvision resize of the uint8 image: bicubic (a = -0.5), antialias=True, which runs torch's separable
//      uint8 kernel (UpSampleKernel.cpp: int16 fixed-point weights, horizontal pass then vertical pass, uint8
//      intermediate); ported exactly below
import CoreGraphics
import Foundation
import ImageIO

/// A resized canvas ready for the vision encoder.
public struct PreparedImage: Sendable {
    /// RGB, interleaved, row-major
    public let rgb: [UInt8]
    public let width: Int
    public let height: Int
    /// Soft tokens: one per 48 x 48 cell (3 x 3 patches of 16 px).
    public var tokens: Int { (width / 48) * (height / 48) }

    public init(rgb: [UInt8], width: Int, height: Int) {
        self.rgb = rgb
        self.width = width
        self.height = height
    }
}

public enum ImageError: Error, CustomStringConvertible, Sendable {
    case decode(String)

    public var description: String {
        switch self {
        case .decode(let message): return message
        }
    }
}

public enum ImagePrep {
    public static let patchSize = 16
    public static let poolingKernel = 3
    public static let maxSoftTokens = 280
    /// Decoded images above this are rejected (an 8 MiB URL can hold a decompression bomb).
    public static let maxSourcePixels = 40_000_000

    public static func prepare(_ data: Data) throws -> PreparedImage {
        let (source, orientation) = try open(data)
        // A JPEG goes from the decoder to the resize row by row; the full-size image never exists.
        let sink = ResizingSink(orientation: orientation)
        if try JPEGDecoder.decodeRows(data, maxPixels: maxSourcePixels, into: sink), let resizer = sink.resizer {
            return PreparedImage(rgb: resizer.finish(), width: resizer.width, height: resizer.height)
        }
        let (rgb, width, height) = try imageIODecode(source, orientation: orientation)
        let (targetHeight, targetWidth) = try targetSize(height: height, width: width)
        let resized = resize(rgb, width: width, height: height, toWidth: targetWidth, toHeight: targetHeight)
        return PreparedImage(rgb: resized, width: targetWidth, height: targetHeight)
    }

    /// The resize of a `width` x `height` source stored in EXIF `orientation`, to the canvas of the oriented image.
    static func resizer(width: Int, height: Int, orientation: Int) throws -> RowResizer {
        let transposed = (5...8).contains(orientation)
        let (targetHeight, targetWidth) = try targetSize(height: transposed ? width : height, width: transposed ? height : width)
        return RowResizer(sourceWidth: width, sourceHeight: height, orientation: orientation, toWidth: targetWidth,
                          toHeight: targetHeight)
    }

    // MARK: - target size (transformers get_aspect_ratio_preserving_size)

    public static func targetSize(height: Int, width: Int) throws -> (Int, Int) {
        let maxPatches = maxSoftTokens * poolingKernel * poolingKernel
        let targetPixels = maxPatches * patchSize * patchSize
        let factor = (Double(targetPixels) / Double(height * width)).squareRoot()
        let idealHeight = factor * Double(height)
        let idealWidth = factor * Double(width)
        let side = poolingKernel * patchSize
        var targetHeight = Int((idealHeight / Double(side)).rounded(.down)) * side
        var targetWidth = Int((idealWidth / Double(side)).rounded(.down)) * side
        if targetHeight == 0 && targetWidth == 0 { throw ImageError.decode("image resizes to 0 x 0") }
        let maxSide = (maxPatches / (poolingKernel * poolingKernel)) * side
        if targetHeight == 0 {
            targetHeight = side
            targetWidth = min(Int((Double(width) / Double(height)).rounded(.down)) * side, maxSide)
        } else if targetWidth == 0 {
            targetWidth = side
            targetHeight = min(Int((Double(height) / Double(width)).rounded(.down)) * side, maxSide)
        }
        if targetHeight * targetWidth > targetPixels {
            throw ImageError.decode("\(width) x \(height) resizes beyond \(maxPatches) patches")
        }
        return (targetHeight, targetWidth)
    }

    // MARK: - resize (torch uint8 antialiased bicubic, separable)
    //
    // torch resizes the oriented image horizontally, then vertically, with a uint8 intermediate. RowResizer runs
    // the same two passes, with the same weights and integer arithmetic, on rows as a decoder produces them.

    /// Source range and int16 weights of every output index along one axis
    /// (HelperInterpBase::_compute_index_ranges_int16_weights with HelperInterpCubic::aa_filter<double, true>).
    struct AxisWeights {
        var start: [Int]
        var count: [Int]
        var weights: [Int16]  // `stride` per output index
        var stride: Int
        var precision: Int32
    }

    static func cubic(_ value: Double) -> Double {
        let a = -0.5
        let x = abs(value)
        if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2 { return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a }
        return 0
    }

    static func axisWeights(input: Int, output: Int) -> AxisWeights {
        let scale = Double(input) / Double(output)
        let support = scale >= 1 ? 2 * scale : 2.0
        let stride = Int(support.rounded(.up)) * 2 + 1
        let invscale = scale >= 1 ? 1 / scale : 1.0
        var start = [Int](repeating: 0, count: output)
        var count = [Int](repeating: 0, count: output)
        var real = [Double](repeating: 0, count: output * stride)
        var maxWeight = 0.0
        for i in 0..<output {
            let center = scale * (Double(i) + 0.5)
            // C++ static_cast<int64_t> truncates toward zero
            let lo = max(Int((center - support + 0.5).rounded(.towardZero)), 0)
            let size = min(max(min(Int((center + support + 0.5).rounded(.towardZero)), input) - lo, 0), stride)
            var total = 0.0
            for j in 0..<size {
                let w = cubic((Double(j + lo) - center + 0.5) * invscale)
                real[i * stride + j] = w
                total += w
            }
            if total != 0 {
                for j in 0..<size {
                    real[i * stride + j] /= total
                    maxWeight = max(maxWeight, real[i * stride + j])
                }
            }
            start[i] = lo
            count[i] = size
        }
        var precision: Int32 = 0
        while precision < 22 {
            let next = Int32((0.5 + maxWeight * Double(Int32(1) << (precision + 1))).rounded(.towardZero))
            if next >= 1 << 15 { break }
            precision += 1
        }
        let unit = Double(Int32(1) << precision)
        let weights = real.map { w -> Int16 in
            let v = w * unit
            return Int16(truncatingIfNeeded: Int32((v < 0 ? -0.5 + v : 0.5 + v).rounded(.towardZero)))
        }
        return AxisWeights(start: start, count: count, weights: weights, stride: stride, precision: precision)
    }

    public static func resize(_ rgb: [UInt8], width: Int, height: Int, toWidth: Int, toHeight: Int) -> [UInt8] {
        let resizer = RowResizer(sourceWidth: width, sourceHeight: height, orientation: 1, toWidth: toWidth, toHeight: toHeight)
        rgb.withUnsafeBufferPointer { pixels in
            for y in 0..<height { resizer.push(pixels.baseAddress! + y * width * 3) }
        }
        return resizer.finish()
    }

    /// One output pixel line of the horizontal kernel: `axis.start.count` RGB pixels from `axis` input pixels.
    static func resample(_ src: UnsafePointer<UInt8>, _ axis: AxisWeights, into dst: UnsafeMutablePointer<UInt8>) {
        let half = Int32(1) << (axis.precision - 1), shift = axis.precision
        axis.weights.withUnsafeBufferPointer { w in
            for x in 0..<axis.start.count {
                let lo = axis.start[x], size = axis.count[x], wBase = x * axis.stride
                for c in 0..<3 {
                    var sum = half
                    for j in 0..<size { sum += Int32(src[(lo + j) * 3 + c]) * Int32(w[wBase + j]) }
                    dst[x * 3 + c] = UInt8(clamping: min(max(sum >> shift, 0), 255))
                }
            }
        }
    }

    // MARK: - decode

    /// The decoded image, oriented, at full size.
    public static func decode(_ data: Data) throws -> ([UInt8], Int, Int) {
        let (source, orientation) = try open(data)
        // JPEG as libjpeg-turbo decodes it (PIL); other formats and unsupported JPEG variants through ImageIO
        let collector = RGBCollector()
        if try JPEGDecoder.decodeRows(data, maxPixels: maxSourcePixels, into: collector) {
            return orient(collector.rgb, width: collector.width, height: collector.height, orientation: orientation)
        }
        return try imageIODecode(source, orientation: orientation)
    }

    /// The image source and its EXIF orientation; refuses images over `maxSourcePixels` before decoding.
    static func open(_ data: Data) throws -> (CGImageSource, Int) {
        let options = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithData(data as CFData, options), CGImageSourceGetCount(source) > 0 else {
            throw ImageError.decode("not an image ImageIO can read")
        }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, options) as? [CFString: Any] ?? [:]
        if let w = properties[kCGImagePropertyPixelWidth] as? Int, let h = properties[kCGImagePropertyPixelHeight] as? Int,
            w * h > maxSourcePixels
        {
            throw ImageError.decode("\(w) x \(h) exceeds \(maxSourcePixels) pixels")
        }
        return (source, properties[kCGImagePropertyOrientation] as? Int ?? 1)
    }

    static func imageIODecode(_ source: CGImageSource, orientation: Int) throws -> ([UInt8], Int, Int) {
        let options = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let image = CGImageSourceCreateImageAtIndex(source, 0, options) else {
            throw ImageError.decode("ImageIO could not decode the image")
        }
        let width = image.width, height = image.height
        guard width > 0, height > 0, width * height <= maxSourcePixels else {
            throw ImageError.decode("\(width) x \(height) is empty or exceeds \(maxSourcePixels) pixels")
        }
        guard let rgb = rawRGB(image) ?? drawnRGB(image) else {
            throw ImageError.decode("unsupported pixel format")
        }
        return orient(rgb, width: width, height: height, orientation: orientation)
    }

    /// The decoded 8-bit values as stored (RGB, RGBA/ARGB in either byte order, grey, grey + alpha), alpha dropped.
    private static func rawRGB(_ image: CGImage) -> [UInt8]? {
        guard image.bitsPerComponent == 8, let space = image.colorSpace,
            let provider = image.dataProvider, let data = provider.data
        else { return nil }
        let colours: Int
        switch space.model {
        case .rgb: colours = 3
        case .monochrome: colours = 1
        default: return nil
        }
        let alpha = image.alphaInfo
        let hasAlpha = alpha != .none
        let premultiplied = alpha == .premultipliedFirst || alpha == .premultipliedLast
        let alphaFirst = alpha == .premultipliedFirst || alpha == .first || alpha == .noneSkipFirst
        let pixelBytes = colours + (hasAlpha ? 1 : 0)
        guard alpha != .alphaOnly, image.bitsPerPixel == pixelBytes * 8 else { return nil }
        let order = image.bitmapInfo.intersection(.byteOrderMask)
        var reversed = false
        switch order {
        case [], .byteOrder32Big, .byteOrderDefault: reversed = false
        case .byteOrder32Little where pixelBytes == 4: reversed = true
        case .byteOrder16Little where pixelBytes == 2: reversed = true
        case .byteOrder16Big where pixelBytes == 2: reversed = false
        default: return nil
        }
        // byte offset of each colour channel and of alpha inside one pixel, in memory order
        var logical = Array(0..<pixelBytes)  // channel order as described by alphaInfo (big endian)
        if reversed { logical.reverse() }
        func memoryOffset(_ channel: Int) -> Int { logical.firstIndex(of: channel)! }
        let firstColour = hasAlpha && alphaFirst ? 1 : 0
        let colourOffsets = (0..<colours).map { memoryOffset(firstColour + $0) }
        let alphaOffset = hasAlpha ? memoryOffset(alphaFirst ? 0 : colours) : -1
        let width = image.width, height = image.height, rowBytes = image.bytesPerRow
        guard CFDataGetLength(data) >= rowBytes * (height - 1) + width * pixelBytes else { return nil }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        let base = CFDataGetBytePtr(data)!
        rgb.withUnsafeMutableBufferPointer { out in
            for y in 0..<height {
                let row = base + y * rowBytes
                for x in 0..<width {
                    let pixel = row + x * pixelBytes
                    let o = (y * width + x) * 3
                    for c in 0..<3 {
                        var v = pixel[colourOffsets[colours == 3 ? c : 0]]
                        if premultiplied {
                            let a = Int(pixel[alphaOffset])
                            v = a == 0 ? 0 : UInt8(min(255, (Int(v) * 255 + a / 2) / a))
                        }
                        out[o + c] = v
                    }
                }
            }
        }
        return rgb
    }

    /// Any other format (CMYK, indexed, 16-bit, ...): drawn 1:1 into an 8-bit RGB context in the image's own RGB
    /// colour space (sRGB for non-RGB models), alpha dropped.
    private static func drawnRGB(_ image: CGImage) -> [UInt8]? {
        let width = image.width, height = image.height
        let space = image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil } ?? CGColorSpace(name: CGColorSpace.sRGB)!
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: width * 4, space: space,
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            context.interpolationQuality = .none
            context.setBlendMode(.copy)
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { return nil }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for p in 0..<(width * height) {
            let a = Int(rgba[p * 4 + 3])
            for c in 0..<3 {
                let v = Int(rgba[p * 4 + c])
                rgb[p * 3 + c] = a == 255 ? UInt8(v) : (a == 0 ? 0 : UInt8(min(255, (v * 255 + a / 2) / a)))
            }
        }
        return rgb
    }

    /// PIL ImageOps.exif_transpose for EXIF orientations 2-8.
    public static func orient(_ rgb: [UInt8], width: Int, height: Int, orientation: Int) -> ([UInt8], Int, Int) {
        guard (2...8).contains(orientation) else { return (rgb, width, height) }
        let swap = orientation >= 5
        let outWidth = swap ? height : width, outHeight = swap ? width : height
        var out = [UInt8](repeating: 0, count: rgb.count)
        for y in 0..<outHeight {
            for x in 0..<outWidth {
                let (sx, sy): (Int, Int)
                switch orientation {
                case 2: (sx, sy) = (width - 1 - x, y)  // FLIP_LEFT_RIGHT
                case 3: (sx, sy) = (width - 1 - x, height - 1 - y)  // ROTATE_180
                case 4: (sx, sy) = (x, height - 1 - y)  // FLIP_TOP_BOTTOM
                case 5: (sx, sy) = (y, x)  // TRANSPOSE
                case 6: (sx, sy) = (y, height - 1 - x)  // ROTATE_270
                case 7: (sx, sy) = (width - 1 - y, height - 1 - x)  // TRANSVERSE
                default: (sx, sy) = (width - 1 - y, x)  // 8: ROTATE_90
                }
                let s = (sy * width + sx) * 3, d = (y * outWidth + x) * 3
                out[d] = rgb[s]
                out[d + 1] = rgb[s + 1]
                out[d + 2] = rgb[s + 2]
            }
        }
        return (out, outWidth, outHeight)
    }
}

/// Resizes a decoder's rows as they arrive.
final class ResizingSink: JPEGRowSink {
    let orientation: Int
    private(set) var resizer: RowResizer?

    init(orientation: Int) {
        self.orientation = orientation
    }

    func start(width: Int, height: Int) throws {
        resizer = try ImagePrep.resizer(width: width, height: height, orientation: orientation)
    }

    func row(_ rgb: UnsafePointer<UInt8>) { resizer!.push(rgb) }
}
