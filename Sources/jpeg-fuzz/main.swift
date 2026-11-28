// jpeg-fuzz: deterministic mutation fuzzing of JPEGDecoder (a development tool, not part of the release).
//
// Takes JPEG files (or directories of them), derives every iteration's input from (seed, iteration) alone,
// applies byte flips, truncations and edits of segment lengths, Huffman symbols, frame and scan headers, and
// decodes each input in process. A decode must return nil or a well-formed image; anything else exits 1. A trap
// cannot be caught, so the harness is meant to run as a child process: a crash is a nonzero exit (signal), and
// the input that crashed is written to --crash-file first so it can be replayed with --replay.
//
//   jpeg-fuzz [--iterations N] [--seed S] [--start I] [--max-pixels P] [--max-seconds T] [--crash-file PATH] CORPUS...
//   jpeg-fuzz --replay FILE [--max-pixels P]
//   jpeg-fuzz --write-input I [--seed S] --out FILE CORPUS...
//   jpeg-fuzz --flat-jpeg WIDTHxHEIGHT [--sampling 444|420] [--progressive] --out FILE   (a few hundred bytes,
//                                               any pixel count)
//   jpeg-fuzz --peak FILE [--max-peak-mib M] [--max-seconds T]   (ImagePrep.prepare once; prints the peak
//                                               physical footprint it added, and exits 1 above M)
//
// --max-seconds bounds one decode (default 10 s per fuzz input, 60 s for --peak). A watchdog thread enforces it while
// the decode runs, so a decode that never finishes fails with a message instead of hanging until a CI timeout.
// --peak --hang replaces the decode with one that never returns, to test exactly that.
import Darwin
import EmbedCore
import Foundation
import Synchronization

// MARK: - arguments

var iterations = 20_000
var seed: UInt64 = 1
var start = 0
var maxPixels = 1_000_000
var maxSeconds: Double?
var crashPath: String?
var replayPath: String?
var writeInput: Int?
var outPath: String?
var flatSize: String?
var sampling = "444"
var progressive = false
var peakPath: String?
var maxPeakMiB: Int?
var hang = false
var corpusPaths: [String] = []

func usage(_ message: String) -> Never {
    FileHandle.standardError.write(Data("jpeg-fuzz: \(message)\n".utf8))
    exit(64)
}

var argumentList = CommandLine.arguments.dropFirst()
while let argument = argumentList.popFirst() {
    func value() -> String {
        guard let next = argumentList.popFirst() else { usage("\(argument) needs a value") }
        return next
    }
    func number<T: FixedWidthInteger>(_ type: T.Type) -> T {
        let raw = value()
        guard let parsed = T(raw), parsed >= 0 else { usage("\(argument): not a nonnegative integer: \(raw)") }
        return parsed
    }
    switch argument {
    case "--iterations": iterations = number(Int.self)
    case "--seed": seed = number(UInt64.self)
    case "--start": start = number(Int.self)
    case "--max-pixels": maxPixels = number(Int.self)
    case "--max-seconds": maxSeconds = Double(number(Int.self))
    case "--crash-file": crashPath = value()
    case "--replay": replayPath = value()
    case "--write-input": writeInput = number(Int.self)
    case "--out": outPath = value()
    case "--flat-jpeg": flatSize = value()
    case "--sampling": sampling = value()
    case "--progressive": progressive = true
    case "--peak": peakPath = value()
    case "--max-peak-mib": maxPeakMiB = number(Int.self)
    case "--hang": hang = true
    case "-h", "--help": usage("see the header of Sources/jpeg-fuzz/main.swift")
    default:
        if argument.hasPrefix("--") { usage("unknown option \(argument)") }
        corpusPaths.append(argument)
    }
}

// MARK: - watchdog

/// Uptime in nanoseconds at which the current decode started; 0 while none runs.
let decodeStarted = Atomic<UInt64>(0)

/// Checks twice a second whether the current decode has run longer than `limit` seconds; if so, prints `message()`,
/// runs `onTimeout` and exits 1. `exit` does not wait for the decode, which may never return.
func startWatchdog(limit: Double, message: @escaping @Sendable () -> String, onTimeout: @escaping @Sendable () -> Void = {}) {
    let thread = Thread {
        while true {
            Thread.sleep(forTimeInterval: 0.5)
            let started = decodeStarted.load(ordering: .acquiring)
            guard started != 0 else { continue }
            if Double(DispatchTime.now().uptimeNanoseconds &- started) / 1e9 > limit {
                FileHandle.standardError.write(Data((message() + "\n").utf8))
                onTimeout()
                exit(1)
            }
        }
    }
    thread.start()
}

func decodeBegins() { decodeStarted.store(max(DispatchTime.now().uptimeNanoseconds, 1), ordering: .releasing) }
func decodeEnds() { decodeStarted.store(0, ordering: .releasing) }

// MARK: - expectations

/// nil when the result is acceptable (rejected, or a well-formed image within the pixel limit).
func problem(_ result: ([UInt8], Int, Int)?, maxPixels: Int) -> String? {
    guard let (rgb, width, height) = result else { return nil }
    if width <= 0 || height <= 0 { return "decoded to \(width) x \(height)" }
    if width * height > maxPixels { return "decoded \(width) x \(height) beyond --max-pixels \(maxPixels)" }
    if rgb.count != width * height * 3 { return "\(rgb.count) bytes for \(width) x \(height)" }
    return nil
}

// MARK: - memory measurement

/// A YCbCr JPEG of one flat colour: every block is "DC difference 0, end of block", which the decoder reads from
/// the zero bits after the end of the (empty) scan data. Tiny, and as large as its header says. Baseline, or
/// progressive with a single DC scan, which makes the decoder keep every coefficient as for any progressive file.
func flatJPEG(width: Int, height: Int, chromaSubsampled: Bool, progressive: Bool) -> Data {
    var bytes: [UInt8] = [0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x43, 0x00] + [UInt8](repeating: 1, count: 64)
    let luma: UInt8 = chromaSubsampled ? 0x22 : 0x11
    bytes += [0xFF, progressive ? 0xC2 : 0xC0, 0x00, 0x11, 0x08, UInt8(height >> 8), UInt8(height & 255),
              UInt8(width >> 8), UInt8(width & 255), 0x03, 0x01, luma, 0x00, 0x02, 0x11, 0x00, 0x03, 0x11, 0x00]
    let oneCode = [UInt8(1)] + [UInt8](repeating: 0, count: 15)
    bytes += [0xFF, 0xC4, 0x00, 0x14, 0x00] + oneCode + [0x00]  // DC: code 0 -> size 0
    bytes += [0xFF, 0xC4, 0x00, 0x14, 0x10] + oneCode + [0x00]  // AC: code 0 -> end of block
    bytes += [0xFF, 0xDA, 0x00, 0x0C, 0x03, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x00, progressive ? 0x00 : 0x3F, 0x00]
    bytes += [0xFF, 0xD9]
    return Data(bytes)
}

func peakFootprint() -> UInt64 {
    var info = rusage_info_v4()
    let status = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0) }
    }
    return status == 0 ? info.ri_lifetime_max_phys_footprint : 0
}

if let flatSize {
    let parts = flatSize.split(separator: "x").compactMap { Int($0) }
    guard parts.count == 2, (1...65535).contains(parts[0]), (1...65535).contains(parts[1]), ["444", "420"].contains(sampling),
        let outPath
    else { usage("--flat-jpeg WIDTHxHEIGHT [--sampling 444|420] [--progressive] --out FILE") }
    let flat = flatJPEG(width: parts[0], height: parts[1], chromaSubsampled: sampling == "420", progressive: progressive)
    guard FileManager.default.createFile(atPath: outPath, contents: flat)
    else { usage("cannot write \(outPath)") }
    exit(0)
}

if let peakPath {
    guard let data = FileManager.default.contents(atPath: peakPath) else { usage("cannot read \(peakPath)") }
    let limit = maxSeconds ?? 60
    startWatchdog(limit: limit) { "peak \(peakPath): no result after \(limit) s (--max-seconds); giving up" }
    let before = peakFootprint()
    let started = Date()
    decodeBegins()
    while hang { Thread.sleep(forTimeInterval: 1) }
    do {
        let image = try ImagePrep.prepare(data)
        decodeEnds()
        let after = peakFootprint()
        let added = Int((after - before) >> 20)
        print("peak \(peakPath): \(image.width) x \(image.height) canvas; footprint peak \(added) MiB above the "
              + "\(before >> 20) MiB before (process peak \(after >> 20) MiB); \(Int(Date().timeIntervalSince(started) * 1000)) ms")
        if let maxPeakMiB, added > maxPeakMiB {
            print("peak \(peakPath): \(added) MiB is over --max-peak-mib \(maxPeakMiB)")
            exit(1)
        }
        exit(0)
    } catch {
        print("peak \(peakPath): rejected (\(error)); process peak \(peakFootprint() >> 20) MiB")
        exit(0)
    }
}

if let replayPath {
    guard let data = FileManager.default.contents(atPath: replayPath) else { usage("cannot read \(replayPath)") }
    let result = JPEGDecoder.decode(data, maxPixels: maxPixels)
    if let failure = problem(result, maxPixels: maxPixels) {
        print("replay \(replayPath): \(failure)")
        exit(1)
    }
    print("replay \(replayPath): \(result.map { "\($0.1) x \($0.2)" } ?? "rejected")")
    exit(0)
}

// MARK: - corpus

var corpus: [[UInt8]] = []
for path in corpusPaths {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) else { usage("no such file \(path)") }
    let files = isDirectory.boolValue
        ? ((try? FileManager.default.contentsOfDirectory(atPath: path)) ?? []).sorted()
            .filter { $0.lowercased().hasSuffix(".jpg") || $0.lowercased().hasSuffix(".jpeg") }
            .map { (path as NSString).appendingPathComponent($0) }
        : [path]
    for file in files {
        if let data = FileManager.default.contents(atPath: file) { corpus.append([UInt8](data)) }
    }
}
guard !corpus.isEmpty else { usage("no JPEG files in the corpus") }

// MARK: - deterministic mutations

/// SplitMix64: tiny, fast and identical on every platform.
struct Random {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func below(_ bound: Int) -> Int { bound <= 1 ? 0 : Int(next() % UInt64(bound)) }
    mutating func byte() -> UInt8 { UInt8(truncatingIfNeeded: next()) }
    mutating func chance(_ percent: Int) -> Bool { below(100) < percent }
}

/// Marker segments with a length field: offset of the 0xFF, marker, length (including the two length bytes).
func segments(_ bytes: [UInt8]) -> [(offset: Int, marker: UInt8, length: Int)] {
    var found: [(Int, UInt8, Int)] = []
    var p = 2
    while p + 3 < bytes.count {
        guard bytes[p] == 0xFF else { p += 1; continue }
        let marker = bytes[p + 1]
        if marker == 0xFF || marker == 0x00 || marker == 0x01 || marker == 0xD8 || (0xD0...0xD7).contains(marker) {
            p += marker == 0xFF ? 1 : 2
            continue
        }
        if marker == 0xD9 { break }
        let length = Int(bytes[p + 2]) << 8 | Int(bytes[p + 3])
        found.append((p, marker, length))
        p += 2 + max(length, 2)
        if marker == 0xDA {  // entropy-coded data up to the next real marker
            while p + 1 < bytes.count, !(bytes[p] == 0xFF && bytes[p + 1] != 0 && !(0xD0...0xD7).contains(bytes[p + 1])) {
                p += 1
            }
        }
    }
    return found
}

let interesting: [UInt8] = [0x00, 0x01, 0x02, 0x07, 0x08, 0x0B, 0x0C, 0x0F, 0x10, 0x11, 0x3F, 0x40, 0x7F, 0x80, 0xC0,
                            0xC2, 0xC4, 0xD0, 0xD9, 0xDA, 0xDD, 0xFE, 0xFF]

func mutate(_ source: [UInt8], _ random: inout Random) -> [UInt8] {
    var bytes = source
    let edits = 1 + random.below(4)
    for _ in 0..<edits {
        guard bytes.count > 4 else { break }
        let segs = segments(bytes)
        func pick(_ markers: ClosedRange<UInt8>) -> (offset: Int, marker: UInt8, length: Int)? {
            let matching = segs.filter { markers.contains($0.marker) && $0.offset + 4 <= bytes.count }
            return matching.isEmpty ? nil : matching[random.below(matching.count)]
        }
        func set(_ offset: Int, _ value: UInt8) { if offset >= 0 && offset < bytes.count { bytes[offset] = value } }
        func choice() -> UInt8 { random.chance(50) ? interesting[random.below(interesting.count)] : random.byte() }
        switch random.below(13) {
        case 0:  // flip bits
            for _ in 0..<(1 + random.below(8)) { bytes[random.below(bytes.count)] ^= 1 << UInt8(random.below(8)) }
        case 1:  // overwrite bytes
            for _ in 0..<(1 + random.below(4)) { bytes[random.below(bytes.count)] = choice() }
        case 2:  // truncate
            bytes.removeSubrange((2 + random.below(bytes.count - 2))...)
        case 3:  // segment length
            if let seg = pick(0x00...0xFF) {
                let deltas = [0, 1, 2, 3, seg.length - 1, seg.length + 1, seg.length - 17, seg.length + 64, 0xFFFF,
                              random.below(0x10000)]
                let length = max(0, min(0xFFFF, deltas[random.below(deltas.count)]))
                set(seg.offset + 2, UInt8(length >> 8))
                set(seg.offset + 3, UInt8(length & 0xFF))
            }
        case 4:  // Huffman table: class/id, counts or symbols
            if let seg = pick(0xC4...0xC4), seg.length > 3 {
                set(seg.offset + 4 + random.below(seg.length - 2), choice())
            }
        case 5:  // frame header: precision, size, components, sampling, quant tables
            if let seg = pick(0xC0...0xC2) {
                let fields = [4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17]
                set(seg.offset + fields[random.below(fields.count)], choice())
            }
        case 6:  // scan header: components, tables, spectral selection, approximation
            if let seg = pick(0xDA...0xDA), seg.length > 2 {
                set(seg.offset + 4 + random.below(seg.length - 2), choice())
            }
        case 7:  // quantization tables, restart interval
            if let seg = pick(0xDB...0xDD), seg.length > 2 {
                set(seg.offset + 4 + random.below(min(seg.length - 2, 8)), choice())
            }
        case 8:  // insert bytes or a marker
            let at = 2 + random.below(bytes.count - 1)
            let insert = random.chance(50) ? [0xFF, choice()] : (0..<(1 + random.below(16))).map { _ in random.byte() }
            bytes.insert(contentsOf: insert, at: min(at, bytes.count))
        case 9:  // delete a range
            let from = 2 + random.below(bytes.count - 2)
            bytes.removeSubrange(from..<min(bytes.count, from + 1 + random.below(64)))
        case 10:  // duplicate a segment (two frames, repeated tables, extra scans)
            if let seg = pick(0x00...0xFF), seg.offset + 2 + seg.length <= bytes.count {
                let copy = Array(bytes[seg.offset..<(seg.offset + 2 + seg.length)])
                let at = segs[random.below(segs.count)].offset
                bytes.insert(contentsOf: copy, at: at)
            }
        case 11:  // restart interval before a scan, RSTn markers inside the entropy-coded data
            if let sos = pick(0xDA...0xDA) {
                let from = sos.offset + 2 + sos.length
                if random.chance(50), from < bytes.count - 2 {
                    for _ in 0..<(1 + random.below(4)) {
                        let at = from + random.below(bytes.count - 2 - from)
                        bytes.insert(contentsOf: [0xFF, 0xD0 + UInt8(random.below(8))], at: at)
                    }
                }
                let interval = [0, 1, 2, 3, 7, 64, 0xFFFF, random.below(0x10000)][random.below(8)]
                bytes.insert(contentsOf: [0xFF, 0xDD, 0x00, 0x04, UInt8(interval >> 8), UInt8(interval & 0xFF)], at: sos.offset)
            }
        default:  // entropy-coded data
            if let sos = pick(0xDA...0xDA) {
                let from = sos.offset + 2 + sos.length
                if from < bytes.count - 2 {
                    for _ in 0..<(1 + random.below(16)) { set(from + random.below(bytes.count - 2 - from), random.byte()) }
                }
            }
        }
    }
    return bytes
}

func input(_ iteration: Int, seed: UInt64, corpus: [[UInt8]]) -> [UInt8] {
    var random = Random(state: seed &* 0x2545_F491_4F6C_DD1D &+ UInt64(iteration))
    _ = random.next()
    return mutate(corpus[random.below(corpus.count)], &random)
}

if let writeInput {
    guard let outPath else { usage("--write-input needs --out") }
    guard FileManager.default.createFile(atPath: outPath, contents: Data(input(writeInput, seed: seed, corpus: corpus))) else { usage("cannot write \(outPath)") }
    exit(0)
}

// MARK: - crash capture

/// The input being decoded, kept where a signal handler can reach it without allocating.
nonisolated(unsafe) var currentInput = UnsafeMutablePointer<UInt8>.allocate(capacity: 1)
nonisolated(unsafe) var currentCapacity = 1
nonisolated(unsafe) var currentLength = 0
nonisolated(unsafe) var currentIteration = 0
nonisolated(unsafe) var crashDescriptor: Int32 = -1

func writeRaw(_ descriptor: Int32, _ text: StaticString) {
    _ = Darwin.write(descriptor, text.utf8Start, text.utf8CodeUnitCount)
}

/// Digits for the signal handler, allocated up front (a handler must not allocate).
nonisolated(unsafe) let digitBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: 24)

func writeNumber(_ descriptor: Int32, _ value: Int) {
    var n = value, i = 24
    repeat {
        i -= 1
        digitBuffer[i] = UInt8(48 + n % 10)
        n /= 10
    } while n > 0 && i > 0
    _ = Darwin.write(descriptor, digitBuffer + i, 24 - i)
}

if let crashPath {
    crashDescriptor = open(crashPath, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    if crashDescriptor < 0 { usage("cannot open \(crashPath)") }
}

let handler: @convention(c) (Int32) -> Void = { signal in
    writeRaw(2, "jpeg-fuzz: CRASH at iteration ")
    writeNumber(2, currentIteration)
    writeRaw(2, " (signal ")
    writeNumber(2, Int(signal))
    writeRaw(2, ")\n")
    if crashDescriptor >= 0 {
        _ = ftruncate(crashDescriptor, 0)
        _ = pwrite(crashDescriptor, currentInput, currentLength, 0)
        close(crashDescriptor)
    }
    Darwin.signal(signal, SIG_DFL)
    raise(signal)  // keep the signal as the exit status
}
for sig in [SIGTRAP, SIGILL, SIGSEGV, SIGBUS, SIGABRT, SIGFPE] { signal(sig, handler) }

// MARK: - run

let inputLimit = maxSeconds ?? 10
startWatchdog(limit: inputLimit, message: {
    "jpeg-fuzz: iteration \(currentIteration) is still decoding after \(inputLimit) s (--max-seconds)"
}, onTimeout: {
    if crashDescriptor >= 0 { _ = pwrite(crashDescriptor, currentInput, currentLength, 0) }
})

var decoded = 0, rejected = 0
var slowest = 0.0, slowestIteration = 0
let started = Date()
let end = start + iterations
for iteration in start..<end {
    let bytes = input(iteration, seed: seed, corpus: corpus)
    if bytes.count > currentCapacity {
        currentInput.deallocate()
        currentCapacity = bytes.count * 2
        currentInput = .allocate(capacity: currentCapacity)
    }
    bytes.withUnsafeBufferPointer { currentInput.update(from: $0.baseAddress!, count: bytes.count) }
    currentLength = bytes.count
    currentIteration = iteration

    let t0 = DispatchTime.now().uptimeNanoseconds
    decodeBegins()
    let result = JPEGDecoder.decode(Data(bytes), maxPixels: maxPixels)
    decodeEnds()
    let seconds = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
    if seconds > slowest { (slowest, slowestIteration) = (seconds, iteration) }
    if let failure = problem(result, maxPixels: maxPixels) {
        FileHandle.standardError.write(Data("jpeg-fuzz: iteration \(iteration): \(failure)\n".utf8))
        if let crashPath { FileManager.default.createFile(atPath: crashPath, contents: Data(bytes)) }
        exit(1)
    }
    if seconds > inputLimit {
        FileHandle.standardError.write(Data("jpeg-fuzz: iteration \(iteration) took \(seconds) s (limit \(inputLimit) s)\n".utf8))
        if let crashPath { FileManager.default.createFile(atPath: crashPath, contents: Data(bytes)) }
        exit(1)
    }
    if result == nil { rejected += 1 } else { decoded += 1 }
}
let summary = "jpeg-fuzz: \(iterations) iterations (\(start)..<\(end), seed \(seed), \(corpus.count) corpus files, "
    + "max \(maxPixels) pixels): \(decoded) decoded, \(rejected) rejected, 0 failures; "
    + String(format: "slowest %.1f ms (iteration \(slowestIteration)); %.1f s", slowest * 1000, Date().timeIntervalSince(started))
print(summary)
if let crashPath { try? FileManager.default.removeItem(atPath: crashPath) }  // nothing crashed
