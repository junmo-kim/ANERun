// JPEG decoder that reproduces libjpeg-turbo's default output, i.e. what PIL (and so Hugging Face) decodes:
// 8-bit Huffman JPEG (baseline, extended sequential, progressive), ISLOW integer IDCT (jidctint.c), fancy
// "triangle" chroma upsampling with replicated edges (jdsample.c, jdmainct.c) and the integer YCbCr -> RGB tables
// (jdcolor.c). ImageIO decodes the same files with different chroma upsampling (differences up to ~50 levels at
// block edges), which moves embeddings by up to ~0.02 cosine, so producer JPEGs are decoded here.
// Returns nil for anything else (arithmetic coding, 12-bit, lossless, CMYK / YCCK, odd sampling, corrupt data);
// the caller then falls back to ImageIO.
//
// The image is produced row by row (decodeRows), never as a whole. A sequential JPEG whose first scan holds every
// component, which libjpeg also decodes in a single pass, is entropy-decoded one MCU row at a time into a strip of
// coefficients, and each MCU row goes through IDCT, upsampling and colour conversion at once. Every other JPEG
// (progressive, or sequential with a scan per component) needs all coefficients before any pixel, so those are kept
// (2 bytes per sample) and the pixels are then produced the same way, one MCU row at a time. Upsampling keeps a few
// sample rows of each component: those of the current MCU row and the rows above it that the fancy upsampler reads.
import Foundation

public enum JPEGDecoder {
    struct Component {
        let id: Int
        let h: Int
        let v: Int
        let quantTable: Int
        var quant = [Int32](repeating: 0, count: 64)  // natural order, latched at the component's first scan
        var latched = false
        var width = 0  // downsampled size in samples
        var height = 0
        var blocksPerLine = 0  // MCU-padded block grid
        var blocksPerColumn = 0
        /// All blocks, or (single-pass decoding) the blocks of one MCU row; shared, not copied, by copies of the
        /// Component.
        var coefficients: Coefficients?
    }

    /// Coefficients of `blocks` 8 x 8 blocks, zeroed.
    final class Coefficients {
        let blocks: Int
        let values: UnsafeMutablePointer<Int16>

        init(blocks: Int) {
            self.blocks = blocks
            values = .allocate(capacity: blocks * 64)
            clear()
        }

        func clear() { values.initialize(repeating: 0, count: blocks * 64) }

        deinit { values.deallocate() }
    }

    struct Huffman {
        var maxcode = [Int32](repeating: -1, count: 18)
        var valptr = [Int32](repeating: 0, count: 17)
        var mincode = [Int32](repeating: 0, count: 17)
        var values: [UInt8] = []

        init?(counts: [Int], values: [UInt8]) {
            guard counts.reduce(0, +) == values.count, values.count <= 256 else { return nil }
            self.values = values
            var code: Int32 = 0, k: Int32 = 0
            for length in 1...16 {
                let n = Int32(counts[length - 1])
                if n > 0 {
                    valptr[length] = k
                    mincode[length] = code
                    code += n
                    k += n
                    maxcode[length] = code - 1
                } else {
                    maxcode[length] = -1
                }
                guard code <= (1 << length) else { return nil }
                code <<= 1
            }
            maxcode[17] = Int32.max
        }
    }

    struct DecodeError: Error {}

    /// Scans in one file, at most. Each scan is a pass over every block of its components, however few bytes it
    /// takes (a scan of end-of-block runs is a handful of bytes for millions of blocks), so the number of scans, not
    /// the file size, bounds the work. libjpeg-turbo leaves the limit to the application, as a guard against the
    /// attack in its report "Two Issues with the JPEG Standard": TurboJPEG's TJFLAG_LIMITSCANS sets 500
    /// (src/turbojpeg.c), and its own fuzzers decode with TJPARAM_SCANLIMIT 100 (fuzz/decompress.cc). The standard
    /// progressive script (cjpeg -progressive) writes 10 scans for colour and 6 for greyscale, so 100 leaves ten
    /// times that.
    static let maxScans = 100

    /// A file refused outright rather than handed to ImageIO, which would repeat the work it asks for: more than
    /// `maxScans` scans, or a progressive scan that repeats or skips part of the progression.
    static func refused(_ reason: String) -> ImageError { .decode("JPEG refused: \(reason)") }

    /// zigzag index -> natural index, with 16 extra entries so corrupt run lengths stay in bounds (jpeg_natural_order)
    static let naturalOrder: [Int] = [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
        28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
        54, 47, 55, 62, 63,
    ] + [Int](repeating: 63, count: 16)

    // MARK: - entropy-coded segment reader

    struct BitReader {
        let bytes: UnsafeBufferPointer<UInt8>
        var position: Int
        var buffer: UInt64 = 0
        var count = 0
        var hitMarker = false

        mutating func fill() {
            while count <= 56 {
                var byte: UInt8 = 0
                if !hitMarker && position < bytes.count {
                    let b = bytes[position]
                    if b == 0xFF {
                        let next = position + 1 < bytes.count ? bytes[position + 1] : 0xD9
                        if next == 0x00 {
                            position += 2
                            byte = 0xFF
                        } else {
                            hitMarker = true  // zeros from here on, like libjpeg
                        }
                    } else {
                        position += 1
                        byte = b
                    }
                }
                buffer |= UInt64(byte) << UInt64(56 - count)
                count += 8
            }
        }

        /// The next `n` bits, 0 <= n <= 16 (the longest code or magnitude an 8-bit JPEG has). Anything else comes
        /// from a corrupt table and is an error: `n` past the 64-bit buffer would trap.
        mutating func bits(_ n: Int) throws -> Int {
            if n == 0 { return 0 }
            guard n > 0, n <= 16 else { throw DecodeError() }
            if count < n { fill() }
            let value = Int(buffer >> UInt64(64 - n))
            buffer <<= UInt64(n)
            count -= n
            return value
        }

        mutating func bit() -> Int {
            if count < 1 { fill() }
            let value = Int(buffer >> 63)
            buffer <<= 1
            count -= 1
            return value
        }

        mutating func decode(_ table: Huffman) throws -> Int {
            if count < 16 { fill() }
            var code = Int32(buffer >> 63)
            var length = 1
            while code > table.maxcode[length] {
                code = (code << 1) | Int32((buffer >> UInt64(63 - length)) & 1)
                length += 1
                if length > 16 { throw DecodeError() }
            }
            buffer <<= UInt64(length)
            count -= length
            let index = Int(table.valptr[length] + code - table.mincode[length])
            guard index >= 0, index < table.values.count else { throw DecodeError() }
            return Int(table.values[index])
        }

        static func extend(_ value: Int, _ size: Int) throws -> Int {
            guard size >= 0, size <= 16 else { throw DecodeError() }
            return size == 0 ? 0 : (value < (1 << (size - 1)) ? value - (1 << size) + 1 : value)
        }

        mutating func received(_ size: Int) throws -> Int { try Self.extend(try bits(size), size) }

        /// Discard buffered bits and consume the RSTn marker that must follow.
        mutating func restart() throws {
            buffer = 0
            count = 0
            hitMarker = false
            while position + 1 < bytes.count {
                if bytes[position] == 0xFF {
                    let m = bytes[position + 1]
                    if m >= 0xD0 && m <= 0xD7 {
                        position += 2
                        return
                    }
                    if m == 0xFF {
                        position += 1
                        continue
                    }
                    throw DecodeError()
                }
                position += 1
            }
            throw DecodeError()
        }
    }

    // MARK: - decode

    /// An error thrown by the row sink, passed on rather than taken for a decode failure.
    struct SinkError: Error {
        let error: any Error
    }

    /// The whole image as interleaved RGB (tests, the fuzzer, ImagePrep.decode); nil when it is not a JPEG this
    /// decoder handles.
    public static func decode(_ data: Data, maxPixels: Int) -> ([UInt8], Int, Int)? {
        let collector = RGBCollector()
        guard (try? decodeRows(data, maxPixels: maxPixels, into: collector)) == true else { return nil }
        return (collector.rgb, collector.width, collector.height)
    }

    /// Decodes `data` row by row into `sink`. True once every row has been delivered; false for a file this decoder
    /// does not handle or that turns out to be corrupt, possibly after some rows. Throws what `sink.start` throws,
    /// and an ImageError for a file no decoder should spend more time on (see `refused`).
    static func decodeRows(_ data: Data, maxPixels: Int, into sink: some JPEGRowSink) throws -> Bool {
        try data.withUnsafeBytes { raw in
            do {
                return try decode(raw.bindMemory(to: UInt8.self), maxPixels: maxPixels, sink: sink)
            } catch let failure as SinkError {
                throw failure.error
            } catch let refusal as ImageError {
                throw refusal
            } catch {
                return false
            }
        }
    }

    private static func decode(_ bytes: UnsafeBufferPointer<UInt8>, maxPixels: Int, sink: some JPEGRowSink) throws -> Bool {
        guard bytes.count > 4, bytes[0] == 0xFF, bytes[1] == 0xD8 else { return false }
        var quant = [[Int32]?](repeating: nil, count: 4)
        var dcTables = [Huffman?](repeating: nil, count: 4)
        var acTables = [Huffman?](repeating: nil, count: 4)
        var components: [Component] = []
        var width = 0, height = 0, progressive = false
        var restartInterval = 0
        var sawJFIF = false, adobeTransform: Int? = nil
        var hmax = 1, vmax = 1, mcusX = 0, mcusY = 0
        var position = 2
        var scans = 0, rgbInput = false
        /// Progressive: per component and coefficient (zigzag index, as Ss and Se count), the point transform Al of
        /// the last scan that coded it, -1 before its first scan (libjpeg's coef_bits).
        var coefficientBits: [[Int]] = []
        /// Set when the first scan holds every component of a sequential image: rows are produced during that scan.
        var singlePass: RowAssembler?

        func u16(_ p: Int) -> Int { Int(bytes[p]) << 8 | Int(bytes[p + 1]) }

        while true {
            // next marker (skip fill bytes)
            while position < bytes.count && bytes[position] != 0xFF { position += 1 }
            while position < bytes.count && bytes[position] == 0xFF { position += 1 }
            guard position < bytes.count else { break }
            let marker = bytes[position]
            position += 1
            if marker == 0xD9 { break }
            if (0xD0...0xD7).contains(marker) || marker == 0x01 { continue }
            guard position + 2 <= bytes.count else { throw DecodeError() }
            let length = u16(position)
            guard length >= 2, position + length <= bytes.count else { throw DecodeError() }
            let start = position + 2, end = position + length
            position = end
            switch marker {
            case 0xC0, 0xC1, 0xC2:  // SOF0 baseline, SOF1 extended, SOF2 progressive (Huffman)
                guard components.isEmpty, end - start >= 6, bytes[start] == 8 else { return false }
                progressive = marker == 0xC2
                height = u16(start + 1)
                width = u16(start + 3)
                let n = Int(bytes[start + 5])
                guard width > 0, height > 0, width * height <= maxPixels, n == 1 || n == 3,
                    end - start >= 6 + 3 * n
                else { return false }
                for i in 0..<n {
                    let p = start + 6 + 3 * i
                    let h = Int(bytes[p + 1] >> 4), v = Int(bytes[p + 1] & 15), q = Int(bytes[p + 2])
                    guard (1...4).contains(h), (1...4).contains(v), q < 4 else { return false }
                    components.append(Component(id: Int(bytes[p]), h: h, v: v, quantTable: q))
                }
                hmax = components.map(\.h).max()!
                vmax = components.map(\.v).max()!
                mcusX = (width + 8 * hmax - 1) / (8 * hmax)
                mcusY = (height + 8 * vmax - 1) / (8 * vmax)
                for i in components.indices {
                    var c = components[i]
                    guard hmax % c.h == 0, vmax % c.v == 0 else { return false }
                    c.width = (width * c.h + hmax - 1) / hmax
                    c.height = (height * c.v + vmax - 1) / vmax
                    c.blocksPerLine = mcusX * c.h
                    c.blocksPerColumn = mcusY * c.v
                    components[i] = c
                }
            case 0xC3, 0xC5...0xC7, 0xC9...0xCB, 0xCD...0xCF:
                return false  // lossless, hierarchical, arithmetic
            case 0xC4:  // DHT
                var p = start
                while p < end {
                    guard p + 17 <= end else { throw DecodeError() }
                    let info = Int(bytes[p])
                    let counts = (0..<16).map { Int(bytes[p + 1 + $0]) }
                    let total = counts.reduce(0, +)
                    guard p + 17 + total <= end, info & 15 < 4, info >> 4 <= 1 else { throw DecodeError() }
                    let values = Array(bytes[(p + 17)..<(p + 17 + total)])
                    // Symbols are bit counts: a DC difference has at most 11 bits and an AC coefficient (low nibble
                    // of run/size) at most 10 at 8-bit precision. Larger ones only come from corrupt or hostile files.
                    let valid = info >> 4 == 0 ? values.allSatisfy { $0 <= 11 } : values.allSatisfy { $0 & 15 <= 10 }
                    guard valid, let table = Huffman(counts: counts, values: values) else { throw DecodeError() }
                    if info >> 4 == 0 { dcTables[info & 15] = table } else { acTables[info & 15] = table }
                    p += 17 + total
                }
            case 0xDB:  // DQT
                var p = start
                while p < end {
                    let precision = Int(bytes[p] >> 4), id = Int(bytes[p] & 15)
                    guard id < 4, p + 1 + 64 * (precision + 1) <= end else { throw DecodeError() }
                    var table = [Int32](repeating: 0, count: 64)
                    for k in 0..<64 {
                        let value = precision == 0 ? Int32(bytes[p + 1 + k]) : Int32(u16(p + 1 + 2 * k))
                        table[naturalOrder[k]] = value
                    }
                    quant[id] = table
                    p += 1 + 64 * (precision + 1)
                }
            case 0xDD:  // DRI
                guard end - start >= 2 else { throw DecodeError() }
                restartInterval = u16(start)
            case 0xE0:  // APP0: JFIF
                if end - start >= 14, Array(bytes[start..<(start + 5)]) == [0x4A, 0x46, 0x49, 0x46, 0] { sawJFIF = true }
            case 0xEE:  // APP14: Adobe
                if end - start >= 12, Array(bytes[start..<(start + 5)]) == [0x41, 0x64, 0x6F, 0x62, 0x65] {
                    adobeTransform = Int(bytes[start + 11])
                }
            case 0xDA:  // SOS
                guard !components.isEmpty, end - start >= 1 else { throw DecodeError() }
                let n = Int(bytes[start])
                guard n >= 1, n <= components.count, end - start >= 1 + 2 * n + 3 else { throw DecodeError() }
                var scan: [(Int, Int, Int)] = []  // component index, dc table, ac table
                for i in 0..<n {
                    let id = Int(bytes[start + 1 + 2 * i]), tables = Int(bytes[start + 2 + 2 * i])
                    guard let index = components.firstIndex(where: { $0.id == id }) else { throw DecodeError() }
                    scan.append((index, tables >> 4 & 3, tables & 3))
                }
                let ss = Int(bytes[start + 1 + 2 * n]), se = Int(bytes[start + 2 + 2 * n])
                let ah = Int(bytes[start + 3 + 2 * n] >> 4), al = Int(bytes[start + 3 + 2 * n] & 15)
                if !progressive { guard ss == 0, se == 63, ah == 0, al == 0 else { throw DecodeError() } }
                guard ss <= se, se <= 63, al <= 13 else { throw DecodeError() }
                if progressive, ss > 0, n != 1 { throw DecodeError() }
                // libjpeg's other parameter checks (jdphuff.c): a DC scan codes coefficient 0 only, and a refinement
                // scan one more bit than the scan before it.
                if progressive, (ss == 0 && se != 0) || (ah != 0 && al != ah - 1) { throw DecodeError() }
                guard scans < maxScans else { throw refused("more than \(maxScans) scans") }
                if progressive {
                    // Every coefficient gets one first scan (Ah 0) and then refinements, each continuing from the
                    // last (Ah = the previous Al), and AC coefficients only after the component's DC. libjpeg decodes
                    // a scan out of this order anyway (with a warning at most; a repeated first scan with Al 0 gets
                    // none); refused here, so that no scan repeats work already done.
                    if coefficientBits.isEmpty {
                        coefficientBits = [[Int]](repeating: [Int](repeating: -1, count: 64), count: components.count)
                    }
                    for (index, _, _) in scan {
                        if ss > 0, coefficientBits[index][0] < 0 {
                            throw refused("scan \(scans + 1) codes AC coefficients of component \(components[index].id) before its DC")
                        }
                        for k in ss...se {
                            guard ah == 0 ? coefficientBits[index][k] < 0 : coefficientBits[index][k] == ah else {
                                throw refused("scan \(scans + 1) repeats or skips a step of component \(components[index].id)'s progression")
                            }
                            coefficientBits[index][k] = al
                        }
                    }
                }
                for (index, _, _) in scan where !components[index].latched {
                    guard let table = quant[components[index].quantTable] else { throw DecodeError() }
                    components[index].quant = table
                    components[index].latched = true
                }
                if scans == 0 {
                    // Decided at the first scan, as libjpeg does: the colour space from the markers before it, and
                    // whether the image is decoded in a single pass.
                    rgbInput = isRGB(components: components, sawJFIF: sawJFIF, adobeTransform: adobeTransform)
                    if !progressive && scan.count == components.count {
                        for i in components.indices {
                            components[i].coefficients = Coefficients(blocks: components[i].blocksPerLine * components[i].v)
                        }
                        try startSink(sink, width: width, height: height)
                        singlePass = RowAssembler(components: components, width: width, height: height, hmax: hmax,
                                                  vmax: vmax, rgbInput: rgbInput, sink: sink)
                    } else {
                        for i in components.indices {
                            components[i].coefficients = Coefficients(
                                blocks: components[i].blocksPerLine * components[i].blocksPerColumn)
                        }
                    }
                } else if singlePass != nil {
                    // libjpeg ends such an image after its one scan; another scan is an error (JERR_EOI_EXPECTED).
                    throw DecodeError()
                }
                scans += 1
                position = try decodeScan(bytes: bytes, position: end, components: components, scan: scan,
                                          dcTables: dcTables, acTables: acTables, progressive: progressive,
                                          ss: ss, se: se, ah: ah, al: al, restartInterval: restartInterval,
                                          mcusX: mcusX, mcusY: mcusY, output: singlePass)
            default:
                break
            }
        }
        guard !components.isEmpty, components.allSatisfy(\.latched) else { return false }
        if let singlePass { return singlePass.isComplete }
        try startSink(sink, width: width, height: height)
        let rows = RowAssembler(components: components, width: width, height: height, hmax: hmax, vmax: vmax,
                                rgbInput: rgbInput, sink: sink)
        for my in 0..<mcusY { rows.add(mcuRow: my, stripped: false) }
        return rows.isComplete
    }

    /// Whether three components hold RGB rather than YCbCr (jdapimin.c default_decompress_parms).
    static func isRGB(components: [Component], sawJFIF: Bool, adobeTransform: Int?) -> Bool {
        guard components.count == 3 else { return false }
        if sawJFIF { return false }
        if let transform = adobeTransform { return transform == 0 }
        return components.map(\.id) == [82, 71, 66]
    }

    static func startSink(_ sink: some JPEGRowSink, width: Int, height: Int) throws {
        do { try sink.start(width: width, height: height) } catch { throw SinkError(error: error) }
    }

    /// Decodes one scan into the components' coefficients. With `output` (single-pass decoding), the coefficients
    /// hold one MCU row, which goes to `output` as soon as it is complete.
    private static func decodeScan(bytes: UnsafeBufferPointer<UInt8>, position: Int, components: [Component],
                                   scan: [(Int, Int, Int)], dcTables: [Huffman?], acTables: [Huffman?],
                                   progressive: Bool, ss: Int, se: Int, ah: Int, al: Int, restartInterval: Int,
                                   mcusX: Int, mcusY: Int, output: RowAssembler?) throws -> Int {
        var reader = BitReader(bytes: bytes, position: position)
        var eobrun = 0
        var predictors = [Int](repeating: 0, count: components.count)  // last_dc_val, reset per scan
        let needDC = !progressive || ss == 0, needAC = !progressive || se > 0
        func table(_ tables: [Huffman?], _ i: Int) throws -> Huffman {
            guard let t = tables[i] else { throw DecodeError() }
            return t
        }
        let dc = try scan.map { needDC && ah == 0 ? try table(dcTables, $0.1) : nil }
        let ac = try scan.map { needAC && (!progressive || ss > 0) ? try table(acTables, $0.2) : nil }

        let storage = scan.map { components[$0.0].coefficients!.values }
        /// The block row each component's coefficients start at: 0, or the current MCU row's first.
        var firstRow = [Int](repeating: 0, count: components.count)

        func decodeBlock(_ s: Int, _ offset: Int) throws {
            let ci = scan[s].0
            let coef = storage[s]
            do {
                if !progressive {
                    let t = try reader.decode(dc[s]!)
                    let diff = try reader.received(t)
                    predictors[ci] += diff
                    coef[offset] = Int16(truncatingIfNeeded: predictors[ci])
                    var k = 1
                    let acTable = ac[s]!
                    while k < 64 {
                        let rs = try reader.decode(acTable)
                        let r = rs >> 4, size = rs & 15
                        if size != 0 {
                            k += r
                            coef[offset + naturalOrder[k]] = Int16(truncatingIfNeeded: try reader.received(size))
                        } else {
                            if r != 15 { break }
                            k += 15
                        }
                        k += 1
                    }
                } else if ss == 0 {
                    if ah == 0 {
                        let t = try reader.decode(dc[s]!)
                        predictors[ci] += try reader.received(t)
                        coef[offset] = Int16(truncatingIfNeeded: predictors[ci] << al)
                    } else if reader.bit() == 1 {
                        coef[offset] |= Int16(truncatingIfNeeded: 1 << al)
                    }
                } else if ah == 0 {
                    if eobrun > 0 {
                        eobrun -= 1
                        return
                    }
                    var k = ss
                    let acTable = ac[s]!
                    while k <= se {
                        let rs = try reader.decode(acTable)
                        let r = rs >> 4, size = rs & 15
                        if size != 0 {
                            k += r
                            coef[offset + naturalOrder[k]] = Int16(truncatingIfNeeded: try reader.received(size) << al)
                        } else if r == 15 {
                            k += 15
                        } else {
                            eobrun = 1 << r
                            if r != 0 { eobrun += try reader.bits(r) }
                            eobrun -= 1
                            break
                        }
                        k += 1
                    }
                } else {
                    let p1 = Int16(truncatingIfNeeded: 1 << al), m1 = Int16(truncatingIfNeeded: -1 << al)
                    var k = ss
                    let acTable = ac[s]!
                    if eobrun == 0 {
                        while k <= se {
                            let rs = try reader.decode(acTable)
                            var r = rs >> 4
                            var value: Int16 = 0
                            if rs & 15 != 0 {
                                value = reader.bit() == 1 ? p1 : m1
                            } else if r != 15 {
                                eobrun = 1 << r
                                if r != 0 { eobrun += try reader.bits(r) }
                                break
                            }
                            repeat {
                                let z = offset + naturalOrder[k]
                                if coef[z] != 0 {
                                    if reader.bit() == 1, coef[z] & p1 == 0 { coef[z] &+= coef[z] >= 0 ? p1 : m1 }
                                } else {
                                    if r == 0 { break }
                                    r -= 1
                                }
                                k += 1
                            } while k <= se
                            if value != 0 { coef[offset + naturalOrder[k]] = value }
                            k += 1
                        }
                    }
                    if eobrun > 0 {
                        while k <= se {
                            let z = offset + naturalOrder[k]
                            if coef[z] != 0, reader.bit() == 1, coef[z] & p1 == 0 { coef[z] &+= coef[z] >= 0 ? p1 : m1 }
                            k += 1
                        }
                        eobrun -= 1
                    }
                }
            }
        }

        func blockOffset(_ ci: Int, _ row: Int, _ column: Int) -> Int {
            ((row - firstRow[ci]) * components[ci].blocksPerLine + column) * 64
        }

        /// Single pass: hands MCU row `my` over and clears the coefficients for the next one.
        func finishRow(_ my: Int) {
            guard let output else { return }
            output.add(mcuRow: my, stripped: true)
            for (ci, _, _) in scan {
                components[ci].coefficients!.clear()
                firstRow[ci] = (my + 1) * components[ci].v
            }
        }

        var unitsLeft = restartInterval
        func restartIfNeeded() throws {
            guard restartInterval > 0 else { return }
            if unitsLeft == 0 {
                try reader.restart()
                for (ci, _, _) in scan { predictors[ci] = 0 }
                eobrun = 0
                unitsLeft = restartInterval
            }
            unitsLeft -= 1
        }

        if scan.count == 1 {
            // non-interleaved: the component's own block grid, not padded to MCUs
            let ci = scan[0].0
            let columns = (components[ci].width + 7) / 8, rows = (components[ci].height + 7) / 8
            let v = components[ci].v
            for row in 0..<rows {
                for column in 0..<columns {
                    try restartIfNeeded()
                    try decodeBlock(0, blockOffset(ci, row, column))
                }
                if row % v == v - 1 || row == rows - 1 { finishRow(row / v) }
            }
        } else {
            // Only the sampling factors: a copy of the Component would share its coefficient array, and decodeBlock's
            // write into that array would then copy all of it, for every block.
            let factors = scan.map { (h: components[$0.0].h, v: components[$0.0].v) }
            for my in 0..<mcusY {
                for mx in 0..<mcusX {
                    try restartIfNeeded()
                    for s in scan.indices {
                        let c = factors[s]
                        for v in 0..<c.v {
                            for h in 0..<c.h {
                                try decodeBlock(s, blockOffset(scan[s].0, my * c.v + v, mx * c.h + h))
                            }
                        }
                    }
                }
                finishRow(my)
            }
        }
        // resume marker parsing at the marker that ended the entropy-coded segment
        var p = reader.position
        while p + 1 < bytes.count && !(bytes[p] == 0xFF && bytes[p + 1] != 0x00 && !(0xD0...0xD7).contains(bytes[p + 1])) {
            p += 1
        }
        return p
    }

    // MARK: - IDCT

    /// jpeg_idct_islow into `plane` (stride `stride`) at (x, y).
    static func idctIslow(_ coef: UnsafeBufferPointer<Int16>, _ offset: Int, _ quant: [Int32],
                          _ plane: UnsafeMutableBufferPointer<UInt8>, _ stride: Int, _ x: Int, _ y: Int) {
        let constBits = 13, pass1Bits = 2
        func descale(_ v: Int, _ n: Int) -> Int { (v + (1 << (n - 1))) >> n }
        func limit(_ v: Int) -> UInt8 {
            let i = v & 1023
            let s = i < 512 ? i : i - 1024
            return UInt8(min(max(s + 128, 0), 255))
        }
        var ws = [Int](repeating: 0, count: 64)
        for c in 0..<8 {
            func dq(_ r: Int) -> Int { Int(coef[offset + r * 8 + c]) * Int(quant[r * 8 + c]) }
            if coef[offset + 8 + c] == 0 && coef[offset + 16 + c] == 0 && coef[offset + 24 + c] == 0
                && coef[offset + 32 + c] == 0 && coef[offset + 40 + c] == 0 && coef[offset + 48 + c] == 0
                && coef[offset + 56 + c] == 0
            {
                let dcval = dq(0) << pass1Bits
                for r in 0..<8 { ws[r * 8 + c] = dcval }
                continue
            }
            var z2 = dq(2), z3 = dq(6)
            var z1 = (z2 + z3) * 4433
            var tmp2 = z1 + z3 * -15137
            var tmp3 = z1 + z2 * 6270
            z2 = dq(0)
            z3 = dq(4)
            var tmp0 = (z2 + z3) << constBits
            var tmp1 = (z2 - z3) << constBits
            let tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2
            tmp0 = dq(7); tmp1 = dq(5); tmp2 = dq(3); tmp3 = dq(1)
            z1 = tmp0 + tmp3
            z2 = tmp1 + tmp2
            z3 = tmp0 + tmp2
            var z4 = tmp1 + tmp3
            let z5 = (z3 + z4) * 9633
            tmp0 *= 2446; tmp1 *= 16819; tmp2 *= 25172; tmp3 *= 12299
            z1 *= -7373; z2 *= -20995; z3 *= -16069; z4 *= -3196
            z3 += z5; z4 += z5
            tmp0 += z1 + z3; tmp1 += z2 + z4; tmp2 += z2 + z3; tmp3 += z1 + z4
            let n = constBits - pass1Bits
            ws[0 * 8 + c] = descale(tmp10 + tmp3, n); ws[7 * 8 + c] = descale(tmp10 - tmp3, n)
            ws[1 * 8 + c] = descale(tmp11 + tmp2, n); ws[6 * 8 + c] = descale(tmp11 - tmp2, n)
            ws[2 * 8 + c] = descale(tmp12 + tmp1, n); ws[5 * 8 + c] = descale(tmp12 - tmp1, n)
            ws[3 * 8 + c] = descale(tmp13 + tmp0, n); ws[4 * 8 + c] = descale(tmp13 - tmp0, n)
        }
        for r in 0..<8 {
            let w = r * 8, out = (y + r) * stride + x
            if ws[w + 1] == 0 && ws[w + 2] == 0 && ws[w + 3] == 0 && ws[w + 4] == 0 && ws[w + 5] == 0
                && ws[w + 6] == 0 && ws[w + 7] == 0
            {
                let dcval = limit(descale(ws[w], pass1Bits + 3))
                for k in 0..<8 { plane[out + k] = dcval }
                continue
            }
            var z2 = ws[w + 2], z3 = ws[w + 6]
            var z1 = (z2 + z3) * 4433
            var tmp2 = z1 + z3 * -15137
            var tmp3 = z1 + z2 * 6270
            var tmp0 = (ws[w] + ws[w + 4]) << constBits
            var tmp1 = (ws[w] - ws[w + 4]) << constBits
            let tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2
            tmp0 = ws[w + 7]; tmp1 = ws[w + 5]; tmp2 = ws[w + 3]; tmp3 = ws[w + 1]
            z1 = tmp0 + tmp3
            z2 = tmp1 + tmp2
            z3 = tmp0 + tmp2
            var z4 = tmp1 + tmp3
            let z5 = (z3 + z4) * 9633
            tmp0 *= 2446; tmp1 *= 16819; tmp2 *= 25172; tmp3 *= 12299
            z1 *= -7373; z2 *= -20995; z3 *= -16069; z4 *= -3196
            z3 += z5; z4 += z5
            tmp0 += z1 + z3; tmp1 += z2 + z4; tmp2 += z2 + z3; tmp3 += z1 + z4
            let n = constBits + pass1Bits + 3
            plane[out + 0] = limit(descale(tmp10 + tmp3, n)); plane[out + 7] = limit(descale(tmp10 - tmp3, n))
            plane[out + 1] = limit(descale(tmp11 + tmp2, n)); plane[out + 6] = limit(descale(tmp11 - tmp2, n))
            plane[out + 2] = limit(descale(tmp12 + tmp1, n)); plane[out + 5] = limit(descale(tmp12 - tmp1, n))
            plane[out + 3] = limit(descale(tmp13 + tmp0, n)); plane[out + 4] = limit(descale(tmp13 - tmp0, n))
        }
    }

    // MARK: - upsampling and colour conversion, row by row

    /// Turns MCU rows of coefficients into RGB rows: the IDCT writes into a few sample rows per component, from
    /// which each image row is upsampled (jdsample.c, with jdmainct.c's replicated edge rows), colour-converted
    /// (jdcolor.c) and handed to the sink as soon as every sample row it reads has been decoded.
    final class RowAssembler {
        enum Upsampling { case none, h2v1, h1v2, h2v2, replicate }

        struct Plane {
            let quant: [Int32]
            let coefficients: Coefficients
            let blocksPerLine: Int
            let v: Int
            let fh: Int
            let fv: Int
            let upsampling: Upsampling
            /// Component size in samples, and the blocks that cover it (the rest of the MCU padding is never read).
            let width: Int
            let height: Int
            let blockColumns: Int
            let blockRows: Int
            /// Sample rows kept: this MCU row's plus at least the two above it that fancy upsampling may still read.
            let rows: Int
            let stride: Int
            let samples: UnsafeMutablePointer<UInt8>
            /// Sample rows decoded so far.
            var decoded = 0

            func row(_ y: Int) -> UnsafeMutablePointer<UInt8> { samples + (y % rows) * stride }
            func clampedRow(_ y: Int) -> UnsafeMutablePointer<UInt8> { row(min(max(y, 0), height - 1)) }

            /// The last sample row that image row `y` reads.
            func lastRow(for y: Int) -> Int {
                switch upsampling {
                case .none, .h2v1: return y
                case .h1v2, .h2v2: return min(y / 2 + 1, height - 1)
                case .replicate: return y / fv
                }
            }
        }

        let width: Int
        let height: Int
        private var planes: [Plane]
        private let rgbInput: Bool
        private let sink: any JPEGRowSink
        private let line: UnsafeMutablePointer<UInt8>
        private var nextRow = 0
        // jdcolor.c build_ycc_rgb_table, SCALEBITS 16
        private var crR = [Int](repeating: 0, count: 256), cbB = [Int](repeating: 0, count: 256)
        private var crG = [Int](repeating: 0, count: 256), cbG = [Int](repeating: 0, count: 256)

        init(components: [Component], width: Int, height: Int, hmax: Int, vmax: Int, rgbInput: Bool, sink: some JPEGRowSink) {
            self.width = width
            self.height = height
            self.rgbInput = rgbInput
            self.sink = sink
            line = .allocate(capacity: width * 3)
            planes = components.map { c in
                let fh = hmax / c.h, fv = vmax / c.v
                let upsampling: Upsampling
                if fh == 1 && fv == 1 {
                    upsampling = .none
                } else if fh == 2 && fv == 1 && c.width > 2 {  // h2v1_fancy_upsample
                    upsampling = .h2v1
                } else if fh == 1 && fv == 2 {  // h1v2_fancy_upsample
                    upsampling = .h1v2
                } else if fh == 2 && fv == 2 && c.width > 2 {  // h2v2_fancy_upsample
                    upsampling = .h2v2
                } else {  // h2v1_upsample / h2v2_upsample / int_upsample: replication
                    upsampling = .replicate
                }
                let stride = c.blocksPerLine * 8, rows = 8 * (c.v + 1)
                return Plane(quant: c.quant, coefficients: c.coefficients!, blocksPerLine: c.blocksPerLine, v: c.v,
                             fh: fh, fv: fv, upsampling: upsampling, width: c.width, height: c.height,
                             blockColumns: (c.width + 7) / 8, blockRows: (c.height + 7) / 8, rows: rows,
                             stride: stride, samples: .allocate(capacity: stride * rows))
            }
            let half = 1 << 15
            for i in 0..<256 {
                let x = i - 128
                crR[i] = (91881 * x + half) >> 16
                cbB[i] = (116130 * x + half) >> 16
                crG[i] = -46802 * x
                cbG[i] = -22554 * x + half
            }
        }

        deinit {
            line.deallocate()
            for plane in planes { plane.samples.deallocate() }
        }

        /// Every image row has gone to the sink.
        var isComplete: Bool { nextRow == height }

        /// Decodes MCU row `my` into the sample rows, then hands over every image row that is complete.
        /// `stripped`: the coefficients hold only this MCU row; otherwise all of them.
        func add(mcuRow my: Int, stripped: Bool) {
            for p in planes.indices {
                let plane = planes[p]
                let coefficients = UnsafeBufferPointer(start: plane.coefficients.values, count: plane.coefficients.blocks * 64)
                let samples = UnsafeMutableBufferPointer(start: plane.samples, count: plane.stride * plane.rows)
                for v in 0..<plane.v {
                    let blockRow = my * plane.v + v
                    guard blockRow < plane.blockRows else { break }
                    let storedRow = stripped ? v : blockRow
                    for column in 0..<plane.blockColumns {
                        idctIslow(coefficients, (storedRow * plane.blocksPerLine + column) * 64, plane.quant, samples,
                                  plane.stride, column * 8, (blockRow * 8) % plane.rows)
                    }
                }
                planes[p].decoded = (my + 1) * plane.v * 8
            }
            while nextRow < height, planes.allSatisfy({ $0.lastRow(for: nextRow) < $0.decoded }) {
                produce(nextRow)
                sink.row(line)
                nextRow += 1
            }
        }

        private func produce(_ y: Int) {
            for (channel, plane) in planes.enumerated() {
                let w = plane.width
                func put(_ x: Int, _ value: Int) { line[x * 3 + channel] = UInt8(truncatingIfNeeded: value) }
                switch plane.upsampling {
                case .none:
                    let row = plane.row(y)
                    for x in 0..<width { line[x * 3 + channel] = row[x] }
                case .h2v1:
                    let row = plane.clampedRow(y)
                    func at(_ x: Int) -> Int { Int(row[min(max(x, 0), w - 1)]) }
                    for x in 0..<width {
                        let cx = x >> 1, v3 = at(cx) * 3
                        put(x, x & 1 == 0 ? (v3 + at(cx - 1) + 1) >> 2 : (v3 + at(cx + 1) + 2) >> 2)
                    }
                case .h1v2:
                    let cy = y >> 1
                    let this = plane.clampedRow(cy), other = plane.clampedRow(y & 1 == 0 ? cy - 1 : cy + 1)
                    for x in 0..<width {
                        let cx = min(x, w - 1), v3 = Int(this[cx]) * 3
                        put(x, y & 1 == 0 ? (v3 + Int(other[cx]) + 1) >> 2 : (v3 + Int(other[cx]) + 2) >> 2)
                    }
                case .h2v2:
                    let cy = y >> 1
                    let this = plane.clampedRow(cy), next = plane.clampedRow(y & 1 == 0 ? cy - 1 : cy + 1)
                    func at(_ row: UnsafeMutablePointer<UInt8>, _ x: Int) -> Int { Int(row[min(max(x, 0), w - 1)]) }
                    for x in 0..<width {
                        let cx = x >> 1
                        let sum = at(this, cx) * 3 + at(next, cx)
                        if x & 1 == 0 {
                            put(x, (sum * 3 + at(this, cx - 1) * 3 + at(next, cx - 1) + 8) >> 4)
                        } else {
                            put(x, (sum * 3 + at(this, cx + 1) * 3 + at(next, cx + 1) + 7) >> 4)
                        }
                    }
                case .replicate:
                    let row = plane.row(y / plane.fv)
                    for x in 0..<width { line[x * 3 + channel] = row[x / plane.fh] }
                }
            }
            if planes.count == 1 {
                for x in 0..<width { (line[x * 3 + 1], line[x * 3 + 2]) = (line[x * 3], line[x * 3]) }
                return
            }
            if rgbInput { return }
            func clamp(_ v: Int) -> UInt8 { UInt8(min(max(v, 0), 255)) }
            for x in 0..<width {
                let y = Int(line[x * 3]), cb = Int(line[x * 3 + 1]), cr = Int(line[x * 3 + 2])
                line[x * 3] = clamp(y + crR[cr])
                line[x * 3 + 1] = clamp(y + ((cbG[cb] + crG[cr]) >> 16))
                line[x * 3 + 2] = clamp(y + cbB[cb])
            }
        }
    }
}

/// Receives a decoded image row by row.
protocol JPEGRowSink: AnyObject {
    /// Called once with the image size, before the first row; an error thrown here ends the decode with that error.
    func start(width: Int, height: Int) throws
    /// The next row, top to bottom: `width` RGB pixels.
    func row(_ rgb: UnsafePointer<UInt8>)
}

/// Collects the rows into one interleaved RGB image.
final class RGBCollector: JPEGRowSink {
    private(set) var rgb: [UInt8] = []
    private(set) var width = 0
    private(set) var height = 0
    private var filled = 0

    func start(width: Int, height: Int) {
        self.width = width
        self.height = height
        rgb = [UInt8](repeating: 0, count: width * height * 3)
    }

    func row(_ row: UnsafePointer<UInt8>) {
        let bytes = width * 3
        rgb.withUnsafeMutableBufferPointer { ($0.baseAddress! + filled).update(from: row, count: bytes) }
        filled += bytes
    }
}
