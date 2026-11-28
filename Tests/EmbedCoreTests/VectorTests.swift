import Foundation
import Testing

@testable import EmbedCore

@Suite("Vectors")
struct VectorTests {
    @Test func pplxQuantizesWithTiesToEvenAndKeepsThePrefix() {
        // These are Float32 inputs for which tanh(x) * 127 lands on exact half-integers.
        let firstTie = atanh(Float(2.5) / 127)
        let secondTie = atanh(Float(3.5) / 127)
        let vector: [Float] = [firstTie, secondTie, -firstTie, -secondTie, 100, -100, 0]
        let full = Vectors.output(vector, to: 7, modelID: .pplx)
        #expect(full == [2, 4, -2, -4, 127, -127, 0])
        #expect(Vectors.output(vector, to: 3, modelID: .pplx) == Array(full.prefix(3)))
        #expect(Vectors.output([3, 4, 5], to: 2, modelID: .embeddingGemma2) == [0.6, 0.8])
    }

    @Test func pplxSerializesAQuantizedZeroAsPositiveFloat32Zero() {
        let output = Vectors.output([-0.001], to: 1, modelID: .pplx)
        #expect(Vectors.base64(output) == "AAAAAA==")
    }

    @Test func pplxMatchesThePinnedTorchBoundaryWhereDarwinTanhDiffers() {
        let below = Float(bitPattern: 0x3dea_d9ce)
        let boundary = Float(bitPattern: 0x3dea_d9cf)
        let expected: [Float] = [14, 15, -14, -15]
        let output = Vectors.output([below, boundary, -below, -boundary], to: 4, modelID: .pplx)
        #expect(output == expected)
        #expect(Vectors.decodeBase64(Vectors.base64(output)) == expected)
    }

    @Test func truncationIsTheRenormalizedPrefix() {
        let vector = Vectors.l2Normalized((0..<768).map { Float(($0 * 37) % 101) - 50 })
        for dimensions in Vectors.dimensions {
            let truncated = Vectors.truncated(vector, to: dimensions)
            #expect(truncated.count == dimensions)
            let norm = truncated.reduce(0) { $0 + $1 * $1 }.squareRoot()
            #expect(abs(norm - 1) < 1e-5)
            #expect(Vectors.cosine(truncated, Array(vector.prefix(dimensions))) > 0.99999)
        }
        let full = Vectors.truncated(vector, to: 768)
        #expect(zip(full, vector).allSatisfy { abs($0 - $1) < 1e-7 })
    }

    /// The full size is normalized as well: a vector whose norm is off comes out with norm 1, same direction.
    @Test func theFullSizeIsNormalizedToo() {
        let vector = (0..<768).map { Float(($0 * 37) % 101) - 50 }.map { $0 / 30 }
        let full = Vectors.truncated(vector, to: 768)
        #expect(abs(full.reduce(0) { $0 + $1 * $1 }.squareRoot() - 1) < 1e-5)
        #expect(Vectors.cosine(full, vector) > 0.99999)
    }

    @Test func base64IsLittleEndianFloat32() {
        let vector: [Float] = [1, -2.5, 0.1]
        let encoded = Vectors.base64(vector)
        #expect(encoded == "AACAPwAAIMDNzMw9")
        #expect(Vectors.decodeBase64(encoded) == vector)
        #expect(Vectors.decodeBase64("AAA") == nil)
    }

    /// Squares of tiny values underflow and of huge ones overflow in float32; scaling by the largest value first
    /// keeps both normalizable (a Matryoshka prefix can be tiny while the full vector is not).
    @Test func normalizesTinyAndHugeVectors() {
        func norm(_ v: [Float]) -> Float { v.reduce(0) { $0 + $1 * $1 }.squareRoot() }
        let tiny = Vectors.l2Normalized([Float](repeating: 1e-23, count: 128))
        #expect(abs(norm(tiny) - 1) < 1e-5)
        let huge = Vectors.l2Normalized([3e30, 4e30])
        #expect(abs(huge[0] - 0.6) < 1e-6 && abs(huge[1] - 0.8) < 1e-6)
        let prefix = Vectors.truncated([Float](repeating: 1e-23, count: 128) + [Float](repeating: 1, count: 640), to: 128)
        #expect(abs(norm(prefix) - 1) < 1e-5)
        #expect(Vectors.l2Normalized([0, 0]) == [0, 0])
    }

    @Test func normalizationLeavesZeroVectorsAlone() {
        #expect(Vectors.l2Normalized([0, 0]) == [0, 0])
        #expect(Vectors.l2Normalized([3, 4]) == [0.6, 0.8])
    }

    @Test func inputTypesApplyTheModelPrefixes() {
        #expect(InputType.query.apply("x") == "task: search result | query: x")
        #expect(InputType.document.apply("x") == "title: none | text: x")
    }
}
