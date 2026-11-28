import Foundation
import Testing

@testable import EmbedCore

@Suite("Line input (stdio)")
struct LineReaderTests {
    func split(_ chunks: [String], max: Int = 10) -> [InputLine] {
        var splitter = LineSplitter(maxLineBytes: max)
        var lines = chunks.flatMap { splitter.feed(Data($0.utf8)) }
        if let last = splitter.finish() { lines.append(last) }
        return lines
    }

    static func line(_ text: String) -> InputLine { .line(Data(text.utf8)) }

    @Test func splitsAcrossChunksAndDropsCarriageReturns() {
        #expect(split(["ab", "c\nd\r\n\n", "ef"]) == [Self.line("abc"), Self.line("d"), Self.line(""), Self.line("ef")])
        #expect(split(["a\n"]) == [Self.line("a")])
        #expect(split([]) == [])
    }

    /// A line over the limit is reported with its size and not buffered; the lines around it are intact.
    @Test func reportsLinesOverTheLimitWithoutKeepingThem() {
        #expect(split(["0123456789\n"]) == [Self.line("0123456789")])
        #expect(split(["0123456789A\nok\n"]) == [.tooLong(bytes: 11), Self.line("ok")])
        #expect(split(["012345", "6789", "ABCDEF", "\nok"]) == [.tooLong(bytes: 16), Self.line("ok")])
        #expect(split(["ok\n", String(repeating: "x", count: 25)]) == [Self.line("ok"), .tooLong(bytes: 25)])
    }

    /// Each byte is looked at once: a long line in small chunks is not rescanned on every read.
    @Test func longLinesInSmallChunksStayLinear() {
        var splitter = LineSplitter(maxLineBytes: 1 << 22)
        let chunk = Data(repeating: UInt8(ascii: "a"), count: 1024)
        let started = ContinuousClock.now
        for _ in 0..<2048 { #expect(splitter.feed(chunk).isEmpty) }
        #expect(splitter.feed(Data("\n".utf8)) == [.line(Data(repeating: UInt8(ascii: "a"), count: 1 << 21))])
        #expect(ContinuousClock.now - started < .seconds(5))
    }

    /// The reader thread stops reading while `maxPending` lines are unreleased.
    @Test func readerStaysAtMostMaxPendingLinesAhead() async throws {
        var descriptors: [Int32] = [0, 0]
        #expect(pipe(&descriptors) == 0)
        let (readEnd, writeEnd) = (descriptors[0], descriptors[1])
        defer { close(readEnd) }
        let text = (0..<20).map { "line \($0)\n" }.joined() + String(repeating: "z", count: 100) + "\nlast"
        _ = text.utf8CString.withUnsafeBufferPointer { write(writeEnd, $0.baseAddress!, $0.count - 1) }
        close(writeEnd)

        let reader = LineReader(fileDescriptor: readEnd, maxLineBytes: 64, maxPending: 3)
        try await Task.sleep(for: .milliseconds(300))
        #expect(reader.outstanding == 3)
        var received: [InputLine] = []
        for await line in reader.lines {
            received.append(line)
            #expect(reader.outstanding <= 3)
            reader.release()
        }
        #expect(received.count == 22)
        #expect(received[0] == Self.line("line 0") && received[19] == Self.line("line 19"))
        #expect(received[20] == .tooLong(bytes: 100) && received[21] == Self.line("last"))
        #expect(reader.outstanding == 0)
    }
}
