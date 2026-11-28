import Foundation
import Testing

@testable import EmbedCore

@Suite("PPLX service contract")
struct PPLXServiceTests {
    static func service(maxTokens: Int = 2048, maxInputs: Int = 2048, modelName: String = "pplx-local") -> EmbeddingService {
        var limits = ServiceLimits()
        limits.maxTokens = maxTokens
        limits.maxInputs = maxInputs
        return EmbeddingService(tokenizer: ByteTokenizer(modelID: .pplx),
                                runtime: FakeRuntime.make(maxTokens: maxTokens, modelID: .pplx), limits: limits,
                                modelName: modelName, modelID: .pplx)
    }

    private struct BoundedTokenizer: TextTokenizer {
        // "composed" simulates a long source that becomes a single token after input preparation.
        var maxBytesPerToken: Int? { 1 }
        func encode(_ text: String) -> [Int32] { [99] }
        func encode(_ text: String, addSpecial: Bool, parseSpecial: Bool) -> [Int32] { encode(text) }
        func pieceBytes(_ id: Int32) -> [UInt8]? { nil }
        func encodeForEmbedding(_ text: String, limit: Int) -> BoundedTokenization {
            switch text {
            case "": return .ids([])
            case "exact": return .ids([1, 2, 3, 4])
            case "bounded": return .tooLong(atLeast: limit + 9)
            default: return .ids([7])
            }
        }
    }

    private actor RecordingRuntime: EmbeddingRuntime {
        nonisolated let info = RuntimeInfo(maxTokens: 3, maxImageTokens: 0, bucketLengths: [], backend: "test")
        private(set) var batches: [[[Int32]]] = []

        func embed(tokenized texts: [[Int32]]) async throws -> [[Float]] {
            batches.append(texts)
            return texts.map { _ in [0.5, -0.25] }
        }

        func embed(images: [PreparedImage]) async throws -> [[Float]] { [] }
        func warm() async throws {}
    }

    @Test func validatesWholeBatchBeforeRunningAndUsesBoundedOrExactCounts() async throws {
        let runtime = RecordingRuntime()
        let service = EmbeddingService(tokenizer: BoundedTokenizer(), runtime: runtime, modelName: "local", modelID: .pplx)
        for (text, reason, detail) in [
            ("", "empty_sequence", "input[1]"),
            ("exact", "context_length_exceeded", "input[1] has 4 tokens"),
            ("bounded", "context_length_exceeded", "input[1] has at least 12 tokens"),
        ] {
            do {
                _ = try await service.embedTexts(["composed", text], inputType: .document, truncate: false)
                Issue.record("accepted invalid input")
            } catch {
                #expect(error.reason == reason && error.param == "input" && error.exitCode == .dataError)
                #expect(error.message.contains(detail))
            }
        }
        #expect(await runtime.batches.isEmpty)
        let result = try await service.embedTexts(["composed"], inputType: .query, truncate: false)
        #expect(result == EmbeddingResult(vectors: [[0.5, -0.25]], tokenCounts: [1]))
        #expect(await runtime.batches == [[[7]]])
    }

    @Test func fakeHonorsModelTokenLimitAndOmitsGemmaFraming() async throws {
        let service = Self.service(maxTokens: 32768)
        #expect(service.modelID == .pplx && service.maxTokens == 32768 && service.maxImageTokens == 0)
        #expect(FakeRuntime.make(maxTokens: 40000, modelID: .pplx).info.maxTokens == 32768)
        #expect(Self.service(maxTokens: 1).maxTokens == 1)
        let plain = service.tokenize("é")
        #expect(plain.count == 2 && service.tokenize("é", inputType: .document) == plain)
        #expect(service.tokenizer.encode("é", addSpecial: true, parseSpecial: true) == plain)
        #expect(service.tokenizer.pieceBytes(2) == nil)
        #expect(service.tokenize("").isEmpty)
        #expect(ByteTokenizer().encode("é").count == 4)
        let result = try await service.embedTexts(["a", " "])
        #expect(result.tokenCounts == [1, 1])
        #expect(result.vectors.allSatisfy { $0.count == 1024 && $0.allSatisfy { $0.magnitude < 1 } })
        #expect(result.vectors[0] != result.vectors[1])
    }
}
