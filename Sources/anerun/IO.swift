import EmbedCore
import Foundation

/// stdout carries data only; everything else goes to stderr.
enum Output {
    private static let lock = NSLock()

    static func data(_ text: String) {
        write(text.hasSuffix("\n") ? text : text + "\n", to: FileHandle.standardOutput)
    }

    static func data(_ value: JSONValue, format: OutputFormat) {
        data(value.render(as: format))
    }

    static func write(_ text: String, to handle: FileHandle) {
        write(Data(text.utf8), to: handle)
    }

    /// POSIX write(2) under one lock, so concurrent writers (MCP responses) never interleave.
    static func write(_ bytes: Data, to handle: FileHandle) {
        let descriptor = handle.fileDescriptor
        lock.withLock {
            bytes.withUnsafeBytes { raw in
                var offset = 0
                while offset < raw.count {
                    let written = Darwin.write(descriptor, raw.baseAddress! + offset, raw.count - offset)
                    if written < 0 {
                        if errno == EINTR { continue }
                        return
                    }
                    offset += written
                }
            }
        }
    }
}

enum Log {
    static func info(_ message: String) {
        let stamp = Date().formatted(.iso8601)
        Output.write("\(stamp) \(message)\n", to: FileHandle.standardError)
    }
}

/// All of stdin as bytes.
func readStandardInput() -> Data {
    FileHandle.standardInput.readDataToEndOfFile()
}
