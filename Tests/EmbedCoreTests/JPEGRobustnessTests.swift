import Foundation
import Testing

@testable import EmbedCore

/// Hand-built JPEGs whose tables or entropy data are hostile. The decoder must reject them with an error
/// (nil from `decode`); a trap would take the whole server down.
enum CraftedJPEG {
    /// An 8 x 8 greyscale baseline JPEG with one-code Huffman tables: the DC code `0` decodes to `dcSymbol`,
    /// the AC code `0` to `acSymbol`. `entropy` is the scan data.
    static func make(dcSymbol: UInt8 = 0, acSymbol: UInt8 = 0x00, entropy: [UInt8] = [0x00], width: Int = 8,
                     height: Int = 8) -> Data
    {
        var bytes: [UInt8] = [0xFF, 0xD8]
        bytes += [0xFF, 0xDB, 0x00, 0x43, 0x00] + [UInt8](repeating: 1, count: 64)  // DQT 0
        bytes += [0xFF, 0xC0, 0x00, 0x0B, 0x08, UInt8(height >> 8), UInt8(height & 255), UInt8(width >> 8),
                  UInt8(width & 255), 0x01, 0x01, 0x11, 0x00]  // SOF0, one component
        let oneCode = [UInt8(1)] + [UInt8](repeating: 0, count: 15)
        bytes += [0xFF, 0xC4, 0x00, 0x14, 0x00] + oneCode + [dcSymbol]  // DHT DC 0
        bytes += [0xFF, 0xC4, 0x00, 0x14, 0x10] + oneCode + [acSymbol]  // DHT AC 0
        bytes += [0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00]  // SOS
        bytes += entropy + [0xFF, 0xD9]
        return Data(bytes)
    }
}

/// Hand-built progressive JPEGs: 8 x 8 greyscale, one-code tables, and for each scan (Ss, Se, Ah, Al) one zero byte of
/// data, which is a zero DC difference, an end of band or a zero refinement bit. Only the scan order differs.
enum CraftedProgressive {
    static func make(_ scans: [(Int, Int, Int, Int)]) -> Data {
        var bytes: [UInt8] = [0xFF, 0xD8]
        bytes += [0xFF, 0xDB, 0x00, 0x43, 0x00] + [UInt8](repeating: 1, count: 64)  // DQT 0
        bytes += [0xFF, 0xC2, 0x00, 0x0B, 0x08, 0x00, 0x08, 0x00, 0x08, 0x01, 0x01, 0x11, 0x00]  // SOF2, one component
        let oneCode = [UInt8(1)] + [UInt8](repeating: 0, count: 15)
        bytes += [0xFF, 0xC4, 0x00, 0x14, 0x00] + oneCode + [0x00]  // DHT DC 0: size 0
        bytes += [0xFF, 0xC4, 0x00, 0x14, 0x10] + oneCode + [0x00]  // DHT AC 0: end of band
        for (ss, se, ah, al) in scans {
            bytes += [0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, UInt8(ss), UInt8(se), UInt8(ah << 4 | al), 0x00]
        }
        bytes += [0xFF, 0xD9]
        return Data(bytes)
    }

    /// libjpeg's progression for one component (jcparam.c jpeg_simple_progression).
    static let standard = [(0, 0, 0, 1), (1, 5, 0, 2), (6, 63, 0, 2), (1, 63, 2, 1), (0, 0, 1, 0), (1, 63, 1, 0)]

    /// A valid progression of `count` scans (at most 128): DC in two steps, then each AC coefficient on its own.
    static func valid(scans count: Int) -> [(Int, Int, Int, Int)] {
        let all = [(0, 0, 0, 1), (0, 0, 1, 0)] + (1...63).map { ($0, $0, 0, 1) } + (1...63).map { ($0, $0, 1, 0) }
        return Array(all.prefix(count))
    }
}

/// What a refusal looks like: an error from `decodeRows` (so ImagePrep does not hand the file to ImageIO).
func refusal(_ data: Data) -> String? {
    do {
        _ = try JPEGDecoder.decodeRows(data, maxPixels: 1 << 20, into: RGBCollector())
        return nil
    } catch {
        return "\(error)"
    }
}

@Suite("JPEG decoder robustness")
struct JPEGRobustnessTests {
    @Test func theCraftedFileDecodes() throws {
        let (rgb, width, height) = try #require(JPEGDecoder.decode(CraftedJPEG.make(), maxPixels: 100))
        #expect(width == 8 && height == 8 && rgb.count == 8 * 8 * 3)
    }

    /// A DC symbol is a bit count; 255, 64 or 63 used to shift past the 64-bit buffer and trap.
    @Test(arguments: [UInt8(12), 15, 16, 17, 63, 64, 65, 255])
    func rejectsDCSymbolsBeyondEightBitPrecision(_ symbol: UInt8) {
        #expect(JPEGDecoder.decode(CraftedJPEG.make(dcSymbol: symbol), maxPixels: 100) == nil)
        #expect(JPEGDecoder.decode(CraftedJPEG.make(dcSymbol: symbol, entropy: [0x00, 0x00, 0x00]), maxPixels: 100) == nil)
    }

    @Test(arguments: [UInt8(0x0B), 0x0F, 0x1B, 0xFF])
    func rejectsACSymbolsBeyondEightBitPrecision(_ symbol: UInt8) {
        #expect(JPEGDecoder.decode(CraftedJPEG.make(acSymbol: symbol), maxPixels: 100) == nil)
    }

    @Test func acceptsTheLargestValidSymbols() {
        // DC size 11 and AC size 10 are the 8-bit maxima; with an all-zero scan they decode to some image.
        #expect(JPEGDecoder.decode(CraftedJPEG.make(dcSymbol: 11, acSymbol: 0x0A, entropy: [UInt8](repeating: 0, count: 64)),
                                   maxPixels: 100) != nil)
    }

    @Test func bitCountsOutsideZeroToSixteenThrow() throws {
        let bytes = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: 64)
        defer { bytes.deallocate() }
        bytes.initialize(repeating: 0xA5)
        var reader = JPEGDecoder.BitReader(bytes: UnsafeBufferPointer(bytes), position: 0)
        #expect(try reader.bits(0) == 0)
        #expect(try reader.bits(16) == 0xA5A5)
        for n in [-1, 17, 32, 63, 64, 65, 255] {
            #expect(throws: JPEGDecoder.DecodeError.self) { try reader.bits(n) }
            #expect(throws: JPEGDecoder.DecodeError.self) { try JPEGDecoder.BitReader.extend(1, n) }
        }
        #expect(try JPEGDecoder.BitReader.extend(0, 11) == -2047)
        #expect(try JPEGDecoder.BitReader.extend(0xFFFF, 16) == 0xFFFF)
    }

    @Test func aStandardProgressionDecodes() throws {
        let (rgb, width, height) = try #require(JPEGDecoder.decode(CraftedProgressive.make(CraftedProgressive.standard), maxPixels: 100))
        #expect(width == 8 && height == 8 && rgb.count == 8 * 8 * 3)
    }

    /// A scan that codes what an earlier scan already coded, out of order, or before the DC, is refused: decoding it
    /// again is work for nothing, and repeating such scans is how a small file asks for unbounded work.
    @Test(arguments: [
        [(0, 0, 0, 0), (0, 0, 0, 0)],  // the DC first scan again
        [(0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 0)],  // the same DC refinement twice
        [(1, 63, 0, 0)],  // AC before DC
        [(0, 0, 0, 0), (1, 5, 0, 0), (3, 8, 0, 0)],  // a first scan of coefficients 3...5 again
        [(0, 0, 0, 0), (1, 63, 0, 2), (1, 63, 1, 0)],  // a refinement that skips a bit
        [(0, 0, 0, 0), (1, 63, 1, 0)],  // a refinement before the first scan
    ])
    func aScanOutOfTheProgressionIsRefused(_ scans: [(Int, Int, Int, Int)]) throws {
        let data = CraftedProgressive.make(scans)
        let message = try #require(refusal(data))
        #expect(message.hasPrefix("JPEG refused: scan \(scans.count) "), "\(message)")
        #expect(JPEGDecoder.decode(data, maxPixels: 100) == nil)
        #expect(throws: ImageError.self) { try ImagePrep.prepare(data) }
    }

    @Test func noMoreThanMaxScans() throws {
        let limit = JPEGDecoder.maxScans
        #expect(JPEGDecoder.decode(CraftedProgressive.make(CraftedProgressive.valid(scans: limit)), maxPixels: 100) != nil)
        let over = CraftedProgressive.make(CraftedProgressive.valid(scans: limit + 1))
        #expect(refusal(over) == "JPEG refused: more than \(limit) scans")
    }

    /// 4096 copies of the same DC scan in 41 KB (64 x 64): refused at the second, and not handed to ImageIO.
    @Test func repeatedScansAreRefusedWithoutDecodingThemAll() throws {
        let data = try Fixtures.data("repeated-scans.jpg")
        #expect(refusal(data)?.hasPrefix("JPEG refused: scan 2 ") == true)
        #expect(throws: ImageError.self) { try ImagePrep.prepare(data) }
        #expect(throws: ImageError.self) { try ImagePrep.decode(data) }
    }

    @Test func imagePrepRejectsTheCrashFileWithAnError() {
        #expect(throws: ImageError.self) { try ImagePrep.prepare(CraftedJPEG.make(dcSymbol: 255)) }
    }
}
