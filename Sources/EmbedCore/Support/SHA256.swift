// SHA-256 through CommonCrypto (libSystem), streaming so that large model files are hashed in chunks.
import CommonCrypto
import Foundation

public struct SHA256Hasher {
    private var context = CC_SHA256_CTX()

    public init() {
        CC_SHA256_Init(&context)
    }

    public mutating func update(_ bytes: UnsafeRawBufferPointer) {
        guard let base = bytes.baseAddress, bytes.count > 0 else { return }
        var offset = 0
        while offset < bytes.count {
            let chunk = min(bytes.count - offset, Int(Int32.max))
            CC_SHA256_Update(&context, base + offset, CC_LONG(chunk))
            offset += chunk
        }
    }

    public mutating func update(_ data: Data) {
        data.withUnsafeBytes { update($0) }
    }

    public mutating func finalize() -> String {
        var digest = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
        CC_SHA256_Final(&digest, &context)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    public static func hex(_ data: Data) -> String {
        var hasher = SHA256Hasher()
        hasher.update(data)
        return hasher.finalize()
    }

    public static func hex(_ bytes: [UInt8]) -> String {
        var hasher = SHA256Hasher()
        bytes.withUnsafeBytes { hasher.update($0) }
        return hasher.finalize()
    }

    /// Hashes a file in 4 MiB chunks.
    public static func hex(fileAt url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256Hasher()
        // FileHandle can autorelease its NSData. Drain each read even when the
        // caller hashes many assets from a single Swift concurrency task.
        var done = false
        repeat {
            done = try autoreleasepool {
                guard let chunk = try handle.read(upToCount: 4 << 20), !chunk.isEmpty else { return true }
                hasher.update(chunk)
                return false
            }
        } while !done
        return hasher.finalize()
    }
}
