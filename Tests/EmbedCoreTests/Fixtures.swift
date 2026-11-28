import Foundation

/// Repository test fixtures (Tests/Fixtures), located from this source file.
enum Fixtures {
    static let directory = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Fixtures")

    static func url(_ name: String) -> URL { directory.appendingPathComponent(name) }

    /// Contents of a fixture; `.xz` files are decompressed.
    static func data(_ name: String) throws -> Data {
        let raw = try Data(contentsOf: url(name))
        guard name.hasSuffix(".xz") else { return raw }
        return try (raw as NSData).decompressed(using: .lzma) as Data
    }
}
