import Foundation
import Testing

@testable import EmbedCore

@Suite("Token table")
struct TokenTableTests {
    /// 4 rows of pseudo-random bf16 bit patterns (finite, mixed signs and magnitudes).
    static func table() -> (Data, [UInt16]) {
        var state: UInt32 = 0x1234_5678
        var bits: [UInt16] = []
        for _ in 0..<(4 * TokenTable.width) {
            state = state &* 1_664_525 &+ 1_013_904_223
            // exponent 100...140 keeps values finite after scaling
            let exponent = UInt16(100 + (state >> 8) % 40)
            let mantissa = UInt16((state >> 16) & 0x7F)
            let sign = UInt16((state >> 30) & 1)
            bits.append(sign << 15 | exponent << 7 | mantissa)
        }
        return (bits.withUnsafeBytes { Data($0) }, bits)
    }

    @Test func gathersScaledRowsLikeTheScalarReference() throws {
        let (data, bits) = Self.table()
        let table = try TokenTable(data: data)
        #expect(table.rows == 4)
        let ids: [Int32] = [3, 0, 3, 1]
        var output = [Float16](repeating: .nan, count: ids.count * TokenTable.width)
        output.withUnsafeMutableBufferPointer { table.gather(ids, into: $0.baseAddress!) }
        let scale = Float(512).squareRoot()
        for (position, id) in ids.enumerated() {
            for column in 0..<TokenTable.width {
                let raw = bits[Int(id) * TokenTable.width + column]
                let expected = Float16(Float(bitPattern: UInt32(raw) << 16) * scale)
                #expect(output[position * TokenTable.width + column] == expected)
            }
        }
    }

    @Test func zeroesRowsForOutOfRangeIds() throws {
        let (data, _) = Self.table()
        let table = try TokenTable(data: data)
        #expect(table.row(99).allSatisfy { $0 == 0 })
        #expect(table.row(-1).allSatisfy { $0 == 0 })
    }

    @Test func rejectsTruncatedFiles() {
        #expect(throws: TokenTableError.self) { try TokenTable(data: Data(count: 1000)) }
    }
}
