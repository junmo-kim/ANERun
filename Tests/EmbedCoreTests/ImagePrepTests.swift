import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers

@testable import EmbedCore

/// Golden data from scripts/golden/make_image_golden.py (synthetic images; PIL / transformers hashes).
private struct ImageGolden {
    struct JPEG { let file: String; let width: Int; let height: Int; let sha256: String }
    struct Resize {
        let file: String; let sourceWidth: Int; let sourceHeight: Int; let sourceSHA256: String
        let width: Int; let height: Int; let tokens: Int; let sha256: String
    }
    let jpeg: [JPEG]
    let resize: [Resize]

    static let shared: ImageGolden = {
        let root = try! JSONValue.parse(Data(contentsOf: Fixtures.url("images/golden.json")))
        return ImageGolden(
            jpeg: root["jpeg"]!.arrayValue!.map {
                JPEG(file: $0["file"]!.stringValue!, width: $0["width"]!.intValue!, height: $0["height"]!.intValue!,
                     sha256: $0["sha256"]!.stringValue!)
            },
            resize: root["resize"]!.arrayValue!.map {
                Resize(file: $0["file"]!.stringValue!, sourceWidth: $0["source_width"]!.intValue!,
                       sourceHeight: $0["source_height"]!.intValue!, sourceSHA256: $0["source_sha256"]!.stringValue!,
                       width: $0["width"]!.intValue!, height: $0["height"]!.intValue!, tokens: $0["tokens"]!.intValue!,
                       sha256: $0["sha256"]!.stringValue!)
            })
    }()
}

@Suite("Image preprocessing (HF Gemma4ImageProcessor, PIL)")
struct ImagePrepTests {
    @Test func jpegDecoderMatchesLibjpegTurbo() throws {
        var handled = 0, fallback = 0
        for item in ImageGolden.shared.jpeg {
            let data = try Data(contentsOf: Fixtures.url("images/\(item.file)"))
            if let (rgb, width, height) = JPEGDecoder.decode(data, maxPixels: ImagePrep.maxSourcePixels) {
                #expect(width == item.width && height == item.height, "\(item.file)")
                #expect(SHA256Hasher.hex(rgb) == item.sha256, "\(item.file) differs from PIL")
                handled += 1
            } else {
                // Not a JPEGDecoder format (CMYK): ImageIO must still decode it.
                let (_, width, height) = try ImagePrep.decode(data)
                #expect(width == item.width && height == item.height, "\(item.file)")
                fallback += 1
            }
        }
        #expect(handled >= 50)
        #expect(fallback == 1)
    }

    @Test func targetSizesMatchTheProcessor() throws {
        for item in ImageGolden.shared.resize {
            let (height, width) = try ImagePrep.targetSize(height: item.sourceHeight, width: item.sourceWidth)
            #expect(width == item.width && height == item.height, "\(item.file)")
            #expect(item.tokens <= ImagePrep.maxSoftTokens)
        }
    }

    @Test func resizedCanvasesAreBitExact() throws {
        for item in ImageGolden.shared.resize {
            let data = try Data(contentsOf: Fixtures.url("images/\(item.file)"))
            let (source, width, height) = try ImagePrep.decode(data)
            #expect(SHA256Hasher.hex(source) == item.sourceSHA256, "\(item.file): decode differs from PIL")
            let resized = ImagePrep.resize(source, width: width, height: height, toWidth: item.width, toHeight: item.height)
            #expect(SHA256Hasher.hex(resized) == item.sha256, "\(item.file): resize differs from torchvision")
            let prepared = try ImagePrep.prepare(data)
            #expect(prepared.tokens == item.tokens)
            #expect(SHA256Hasher.hex(prepared.rgb) == item.sha256, "\(item.file): end to end")
        }
    }

    @Test func rejectsNonImages() {
        #expect(throws: ImageError.self) { try ImagePrep.prepare(Data("not an image".utf8)) }
        #expect(throws: ImageError.self) { try ImagePrep.prepare(Data([0, 0, 0, 0])) }
    }

    @Test func appliesExifOrientation() {
        // 2 x 1 image [A, B]; orientation 6 (rotate 90 cw) gives a 1 x 2 image [A; B].
        let rgb: [UInt8] = [1, 2, 3, 4, 5, 6]
        let (out, width, height) = ImagePrep.orient(rgb, width: 2, height: 1, orientation: 6)
        #expect(width == 1 && height == 2)
        #expect(out == [1, 2, 3, 4, 5, 6])
        let (flipped, _, _) = ImagePrep.orient(rgb, width: 2, height: 1, orientation: 2)
        #expect(flipped == [4, 5, 6, 1, 2, 3])
    }

    /// The resize as torch runs it on a whole image: the horizontal pass over all rows, then the vertical pass.
    /// The production code streams rows (RowResizer); this is the reference it must match.
    static func referenceResize(_ rgb: [UInt8], width: Int, height: Int, toWidth: Int, toHeight: Int) -> [UInt8] {
        var image = rgb, currentWidth = width
        if toWidth != width {
            let axis = ImagePrep.axisWeights(input: width, output: toWidth)
            var out = [UInt8](repeating: 0, count: toWidth * height * 3)
            for y in 0..<height {
                for x in 0..<toWidth {
                    for c in 0..<3 {
                        var sum = Int32(1) << (axis.precision - 1)
                        for j in 0..<axis.count[x] {
                            sum += Int32(image[(y * width + axis.start[x] + j) * 3 + c]) * Int32(axis.weights[x * axis.stride + j])
                        }
                        out[(y * toWidth + x) * 3 + c] = UInt8(clamping: min(max(sum >> axis.precision, 0), 255))
                    }
                }
            }
            image = out
            currentWidth = toWidth
        }
        if toHeight != height {
            let axis = ImagePrep.axisWeights(input: height, output: toHeight), row = currentWidth * 3
            var out = [UInt8](repeating: 0, count: row * toHeight)
            for y in 0..<toHeight {
                for i in 0..<row {
                    var sum = Int32(1) << (axis.precision - 1)
                    for j in 0..<axis.count[y] {
                        sum += Int32(image[(axis.start[y] + j) * row + i]) * Int32(axis.weights[y * axis.stride + j])
                    }
                    out[y * row + i] = UInt8(clamping: min(max(sum >> axis.precision, 0), 255))
                }
            }
            image = out
        }
        return image
    }

    /// Streaming the rows and folding the EXIF orientation into the resize gives exactly what orienting the whole
    /// image and then resizing it gives: every orientation, scaling down, up or not at all along either axis.
    @Test(arguments: 1...8) func streamedResizeWithOrientationIsBitExact(_ orientation: Int) {
        var random = UInt64(orientation)
        func byte() -> UInt8 {
            random = random &* 6364136223846793005 &+ 1442695040888963407
            return UInt8(truncatingIfNeeded: random >> 33)
        }
        for (width, height) in [(37, 23), (101, 67), (16, 90), (1, 5)] {
            let rgb = (0..<(width * height * 3)).map { _ in byte() }
            let (oriented, orientedWidth, orientedHeight) = ImagePrep.orient(rgb, width: width, height: height,
                                                                           orientation: orientation)
            let targets = [(orientedWidth, orientedHeight * 2 + 1), (orientedWidth / 2 + 1, orientedHeight),
                           (orientedWidth * 3, max(orientedHeight / 3, 1)), (orientedWidth, orientedHeight), (48, 48)]
            for (toWidth, toHeight) in targets {
                let expected = Self.referenceResize(oriented, width: orientedWidth, height: orientedHeight,
                                                    toWidth: toWidth, toHeight: toHeight)
                let resizer = RowResizer(sourceWidth: width, sourceHeight: height, orientation: orientation,
                                         toWidth: toWidth, toHeight: toHeight)
                rgb.withUnsafeBufferPointer { pixels in
                    for y in 0..<height { resizer.push(pixels.baseAddress! + y * width * 3) }
                }
                #expect(resizer.finish() == expected, "\(width) x \(height), orientation \(orientation) -> \(toWidth) x \(toHeight)")
            }
        }
    }

    /// A 3-frame animation (GIF or APNG) of solid frames in `colours`, 8 x 8 px, written by ImageIO.
    static func animation(_ type: UTType, colours: [(UInt8, UInt8, UInt8)]) throws -> Data {
        let output = NSMutableData()
        let destination = try #require(CGImageDestinationCreateWithData(output, type.identifier as CFString, colours.count, nil))
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        for (r, g, b) in colours {
            var pixels = [UInt8](repeating: 255, count: 8 * 8 * 4)
            for p in 0..<64 { (pixels[p * 4], pixels[p * 4 + 1], pixels[p * 4 + 2]) = (r, g, b) }
            let image = try #require(pixels.withUnsafeMutableBytes { raw in
                CGContext(data: raw.baseAddress, width: 8, height: 8, bitsPerComponent: 8, bytesPerRow: 32, space: space,
                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)?.makeImage()
            })
            let properties = type == .gif
                ? [kCGImagePropertyGIFDictionary: [kCGImagePropertyGIFDelayTime: 0.1]]
                : [kCGImagePropertyPNGDictionary: [kCGImagePropertyAPNGDelayTime: 0.1]]
            CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        }
        #expect(CGImageDestinationFinalize(destination))
        return output as Data
    }

    /// Animated images embed their first frame, as transformers' load_image (PIL) does.
    @Test(arguments: [UTType.gif, UTType.png])
    func animatedImagesUseTheFirstFrame(_ type: UTType) throws {
        let colours: [(UInt8, UInt8, UInt8)] = [(200, 30, 40), (10, 220, 30), (20, 40, 230)]
        let data = try Self.animation(type, colours: colours)
        let source = try #require(CGImageSourceCreateWithData(data as CFData, nil))
        #expect(CGImageSourceGetCount(source) == 3, "\(type): not animated")
        let (rgb, width, height) = try ImagePrep.decode(data)
        #expect(width == 8 && height == 8)
        for p in 0..<(width * height) {
            let pixel = (rgb[p * 3], rgb[p * 3 + 1], rgb[p * 3 + 2])
            guard abs(Int(pixel.0) - 200) <= 2, abs(Int(pixel.1) - 30) <= 2, abs(Int(pixel.2) - 40) <= 2 else {
                Issue.record("\(type): pixel \(p) is \(pixel), want the first frame (200, 30, 40)")
                return
            }
        }
    }
}
