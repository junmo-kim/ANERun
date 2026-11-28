import Foundation
import ImageIO
import UniformTypeIdentifiers

@testable import EmbedCore

enum TestSupport {
    static func service(maxTokens: Int = 2048, images: Bool = true, limits: ServiceLimits? = nil) -> EmbeddingService {
        var configured = limits ?? ServiceLimits()
        configured.maxTokens = maxTokens
        return EmbeddingService(tokenizer: ByteTokenizer(), runtime: FakeRuntime.make(maxTokens: maxTokens, images: images),
                                limits: configured)
    }

    /// A solid-colour PNG (ImageIO), like blackbox.py's png_url.
    static func png(width: Int, height: Int, rgb: (UInt8, UInt8, UInt8) = (200, 30, 90)) -> Data {
        var pixels = [UInt8]()
        pixels.reserveCapacity(width * height * 4)
        for _ in 0..<(width * height) { pixels += [rgb.0, rgb.1, rgb.2, 255] }
        let data = NSMutableData()
        pixels.withUnsafeMutableBytes { raw in
            let context = CGContext(data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                    bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
            let image = context.makeImage()!
            let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil)!
            CGImageDestinationAddImage(destination, image, nil)
            CGImageDestinationFinalize(destination)
        }
        return data as Data
    }

    static func dataURL(_ png: Data) -> String { "data:image/png;base64," + png.base64EncodedString() }

    static func norm(_ vector: [Float]) -> Float { vector.reduce(0) { $0 + $1 * $1 }.squareRoot() }
}
