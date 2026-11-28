// The token embedding table (embeddings.bf16: rows x 512 raw bfloat16, little endian), memory-mapped
// so that only the rows a request touches are paged in. Gathering converts bf16 -> fp16 and applies
// Gemma's embedding scale sqrt(512) in one pass, straight into the model's input buffer.
import Foundation

public enum TokenTableError: Error, CustomStringConvertible, Sendable {
    case unreadable(String)
    case badSize(Int)

    public var description: String {
        switch self {
        case .unreadable(let message): return "cannot map embeddings.bf16: \(message)"
        case .badSize(let size): return "embeddings.bf16 has \(size) bytes, not a multiple of 512 bf16 values"
        }
    }
}

public final class TokenTable: Sendable {
    public static let width = 512
    public let rows: Int
    /// sqrt(hidden size): Gemma multiplies token embeddings by it before the first layer.
    public let scale: Float
    private let data: Data

    public convenience init(contentsOf url: URL, scale: Float = Float(TokenTable.width).squareRoot()) throws {
        let data: Data
        do {
            data = try Data(contentsOf: url, options: .alwaysMapped)
        } catch {
            throw TokenTableError.unreadable("\(url.path): \(error.localizedDescription)")
        }
        try self.init(data: data, scale: scale)
    }

    public init(data: Data, scale: Float = Float(TokenTable.width).squareRoot()) throws {
        guard data.count > 0, data.count % (Self.width * 2) == 0 else { throw TokenTableError.badSize(data.count) }
        self.data = data
        rows = data.count / (Self.width * 2)
        self.scale = scale
    }

    /// Writes `ids.count` rows of 512 fp16 values (row i = table[ids[i]] * scale) to `destination`.
    /// Ids outside the table produce zero rows.
    public func gather(_ ids: UnsafeBufferPointer<Int32>, into destination: UnsafeMutablePointer<Float16>) {
        data.withUnsafeBytes { raw in
            let table = raw.baseAddress!
            for (position, id) in ids.enumerated() {
                let target = destination + position * Self.width
                guard id >= 0, Int(id) < rows else {
                    target.update(repeating: 0, count: Self.width)
                    continue
                }
                Self.convertRow(table + Int(id) * Self.width * 2, into: target, scale: scale)
            }
        }
    }

    public func gather(_ ids: [Int32], into destination: UnsafeMutablePointer<Float16>) {
        ids.withUnsafeBufferPointer { gather($0, into: destination) }
    }

    /// One row as Float (scaled), for tests and debugging.
    public func row(_ id: Int32) -> [Float] {
        var half = [Float16](repeating: 0, count: Self.width)
        half.withUnsafeMutableBufferPointer { gather([id], into: $0.baseAddress!) }
        return half.map(Float.init)
    }

    /// bf16 -> fp32 is a 16-bit left shift; 16 lanes at a time (NEON), then scale and narrow to fp16.
    @inline(__always)
    static func convertRow(_ source: UnsafeRawPointer, into target: UnsafeMutablePointer<Float16>, scale: Float) {
        let lanes = 16
        let scaleVector = SIMD16<Float>(repeating: scale)
        var column = 0
        while column < width {
            let bits = source.loadUnaligned(fromByteOffset: column * 2, as: SIMD16<UInt16>.self)
            let widened = SIMD16<UInt32>(truncatingIfNeeded: bits) &<< 16
            let floats = unsafeBitCast(widened, to: SIMD16<Float>.self) * scaleVector
            let halves = SIMD16<Float16>(floats)
            UnsafeMutableRawPointer(target + column).storeBytes(of: halves, as: SIMD16<Float16>.self)
            column += lanes
        }
    }
}
