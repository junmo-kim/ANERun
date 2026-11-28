// Newline-delimited input (stdio MCP, `embed --stdin`, `tokenize --stdin`) with bounded memory: a line longer
// than the limit is reported and skipped instead of buffered, and the reader stays at most `maxPending` lines
// ahead of its consumer (backpressure: further input is not read until a line is released).
import Foundation

/// One line of input, or the size of one that was over the limit (its bytes are discarded).
public enum InputLine: Sendable, Equatable {
    case line(Data)
    case tooLong(bytes: Int)
}

/// Splits bytes into lines without decoding them (a U+FEFF or any other byte sequence reaches the JSON parser
/// unchanged). Each byte is looked at once, however the input is chunked; a trailing "\r" is dropped.
public struct LineSplitter: Sendable {
    public let maxLineBytes: Int
    private var partial: [UInt8] = []
    /// Bytes of the current line seen so far once it is over the limit (they are not kept).
    private var overflow = 0

    public init(maxLineBytes: Int) {
        self.maxLineBytes = maxLineBytes
    }

    public mutating func feed(_ chunk: UnsafeBufferPointer<UInt8>) -> [InputLine] {
        var lines: [InputLine] = []
        var start = 0
        while start <= chunk.count {
            let newline = chunk[start...].firstIndex(of: UInt8(ascii: "\n"))
            let end = newline ?? chunk.count
            let segment = UnsafeBufferPointer(rebasing: chunk[start..<end])
            if overflow > 0 || partial.count + segment.count > maxLineBytes {
                overflow = (overflow > 0 ? overflow : partial.count) + segment.count
                partial = []
            } else {
                partial.append(contentsOf: segment)
            }
            guard newline != nil else { break }
            lines.append(take())
            start = end + 1
        }
        return lines
    }

    public mutating func feed(_ data: Data) -> [InputLine] {
        data.withUnsafeBytes { feed($0.bindMemory(to: UInt8.self)) }
    }

    /// The last line when the input ends without a newline.
    public mutating func finish() -> InputLine? {
        overflow > 0 || !partial.isEmpty ? take() : nil
    }

    private mutating func take() -> InputLine {
        defer {
            partial = []
            overflow = 0
        }
        if overflow > 0 { return .tooLong(bytes: overflow) }
        if partial.last == UInt8(ascii: "\r") { partial.removeLast() }
        return .line(Data(partial))
    }
}

/// Reads lines from a file descriptor on its own thread (blocking reads). The consumer calls `release()` once it is
/// done with a line; the reader blocks while `maxPending` lines are unreleased.
public final class LineReader: Sendable {
    public let lines: AsyncStream<InputLine>
    private let slots: DispatchSemaphore
    private let state: State

    final class State: @unchecked Sendable {
        let lock = NSLock()
        var outstanding = 0
    }

    public init(fileDescriptor: Int32, maxLineBytes: Int, maxPending: Int) {
        let (stream, continuation) = AsyncStream.makeStream(of: InputLine.self)
        let slots = DispatchSemaphore(value: maxPending)
        let state = State()
        lines = stream
        self.slots = slots
        self.state = state
        let thread = Thread {
            var splitter = LineSplitter(maxLineBytes: maxLineBytes)
            var buffer = [UInt8](repeating: 0, count: 1 << 16)
            func emit(_ line: InputLine) {
                slots.wait()
                state.lock.withLock { state.outstanding += 1 }
                continuation.yield(line)
            }
            while true {
                let count = buffer.withUnsafeMutableBytes { read(fileDescriptor, $0.baseAddress!, $0.count) }
                if count < 0 && errno == EINTR { continue }
                if count <= 0 { break }
                let lines = buffer.withUnsafeBufferPointer { splitter.feed(UnsafeBufferPointer(rebasing: $0[0..<count])) }
                lines.forEach(emit)
            }
            if let last = splitter.finish() { emit(last) }
            continuation.finish()
        }
        thread.stackSize = 1 << 20
        thread.start()
    }

    /// The consumer is done with one line (its memory and in-flight work are over).
    public func release() {
        state.lock.withLock { state.outstanding -= 1 }
        slots.signal()
    }

    /// Lines handed out and not released yet.
    public var outstanding: Int { state.lock.withLock { state.outstanding } }
}
