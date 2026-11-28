import Accelerate
@preconcurrency import CoreML
import Foundation

/// Test fixtures use raw Float32 means so a wrong scaling or pooling path
/// cannot be hidden by the API quantizer or by the display encoding.
struct PPLXSelfTest {
    private struct Route {
        let sequences: [[Int32]]
        let expected: [[Float]]
    }

    struct Metrics {
        let passed: Bool
        let cosine: Double
        let error: String?

        static let pass = Metrics(passed: true, cosine: 1, error: nil)
        static func failed(_ error: String) -> Self { .init(passed: false, cosine: 0, error: error) }
    }

    private let routes: [String: Route]
    private let attention: Set<String>
    private static let packedRoutes: [String: (length: Int, slots: Int)] = [
        "embed_64": (64, 8), "embed_512": (512, 16), "embed_1024": (1024, 32),
    ]
    private static let oneSequence: [String: Int] = [
        "fused_2048": 2048, "fused_4096": 4096, "fused_8192_part0": 8192,
        "attn_S16384_Q2048": 16384, "attn_S32768_Q2048": 32768,
    ]
    private static let attentionKeys: Set<String> = ["attn_S16384_Q2048", "attn_S32768_Q2048"]

    static func load(from directory: URL, manifest: Manifest) throws(ServiceError) -> Self {
        do {
            let descriptor = try SmallFile.read(directory.appendingPathComponent(ModelLayout.selfTest), limit: ModelLayout.selfTestLimit)
            guard let entry = manifest.sourceFiles.first(where: { $0.path == ModelLayout.selfTestData }),
                  entry.size <= ModelLayout.selfTestLimit
            else { throw invalid("selftest.bin is absent or exceeds the size limit") }
            let blob = try SmallFile.read(directory.appendingPathComponent(ModelLayout.selfTestData), limit: entry.size)
            let json = try JSONValue.parse(descriptor)
            let causal = manifest.modelID == .qwen3Embedding
            let format = causal ? "anerun-qwen-selftest/1" : "aneembed-pplx-selftest/1"
            let packedRoutes = causal ? ["embed_64": (length: 64, slots: 8)] : Self.packedRoutes
            let oneSequence = causal ? ["attn_S2048_Q2048": 2048, "attn_S8192_Q2048": 8192,
                                        "attn_S32768_Q2048": 32768] : Self.oneSequence
            let attentionKeys: Set<String> = causal ? [] : Self.attentionKeys
            guard json["format"]?.stringValue == format, let cases = json["cases"]?.arrayValue else {
                throw invalid("expected \(format) and its cases")
            }
            var routes: [String: Route] = [:]
            var attention: Set<String> = []
            for entry in cases {
                guard let name = entry["function"]?.stringValue else { throw invalid("case is missing its function") }
                if entry["kind"]?.stringValue == "attention" {
                    guard attentionKeys.contains(name), attention.insert(name).inserted else {
                        throw invalid("unexpected or duplicate attention case \(name)")
                    }
                    continue
                }
                let length = packedRoutes[name]?.length ?? oneSequence[name]
                guard entry["kind"]?.stringValue == "route", let length, routes[name] == nil,
                      let list = entry["sequences"]?.arrayValue, !list.isEmpty,
                      list.count <= (packedRoutes[name]?.slots ?? 1), let offset = entry["offset"]?.intValue,
                      offset >= 0, offset % 4 == 0
                else { throw invalid("malformed or duplicate route case \(name)") }
                var sequences: [[Int32]] = []
                var rows = 0
                for input in list {
                    guard let tokens = input.arrayValue, !tokens.isEmpty, tokens.count <= length - rows else {
                        throw invalid("case \(name) input exceeds its shape")
                    }
                    let ids = try tokens.map { token in
                        guard let id = token.intValue, (0..<PPLXTokenTable.rowCount).contains(id) else {
                            throw invalid("case \(name) contains an invalid token id")
                        }
                        return Int32(id)
                    }
                    if causal && ids.last != 151643 { throw invalid("Qwen route must end in EOS") }
                    rows += ids.count
                    sequences.append(ids)
                }
                let vectorBytes = PPLXTokenTable.width * MemoryLayout<Float>.stride
                guard offset <= blob.count, list.count * vectorBytes <= blob.count - offset else {
                    throw invalid("case \(name) exceeds selftest.bin")
                }
                let expected = (0..<list.count).map {
                    floats(blob, offset: offset + $0 * vectorBytes, count: PPLXTokenTable.width)
                }
                guard expected.joined().allSatisfy(\.isFinite) else { throw invalid("case \(name) has nonfinite reference values") }
                routes[name] = Route(sequences: sequences, expected: expected)
            }
            guard Set(routes.keys) == Set(packedRoutes.keys).union(oneSequence.keys), attention == attentionKeys else {
                throw invalid("the suite must cover every pplx route and both attention lengths")
            }
            return Self(routes: routes, attention: attention)
        } catch let error as ServiceError {
            throw error
        } catch {
            throw invalid(String(describing: error))
        }
    }

    static func readNorm(_ url: URL) throws -> [Float] {
        let width = PPLXTokenTable.width
        let data = try SmallFile.read(url, limit: width * MemoryLayout<Float>.stride)
        guard data.count == width * MemoryLayout<Float>.stride else { throw invalid("invalid final_norm.f32 size") }
        let norm = floats(data, offset: 0, count: width)
        guard norm.allSatisfy(\.isFinite) else { throw invalid("invalid final_norm.f32 values") }
        return norm
    }

    private static func floats(_ data: Data, offset: Int, count: Int) -> [Float] {
        data.withUnsafeBytes { bytes in
            (0..<count).map { index in
                let value = bytes.loadUnaligned(fromByteOffset: offset + index * 4, as: UInt32.self)
                return Float(bitPattern: UInt32(littleEndian: value))
            }
        }
    }

    private static func invalid(_ detail: String) -> ServiceError {
        ServiceError(.unavailable, "selftest_invalid", detail,
                     hint: "verify and reinstall the pplx-embed-v1-0.6b package")
    }

    func run(_ name: String, length: Int, runner: any PPLXPredicting, table: PPLXTokenTable, norm: [Float], causal: Bool = false) async -> Metrics {
        guard let route = routes[name] else { return .failed("missing pplx self-test for \(name)") }
        do {
            if attention.contains(name) { try await Self.runAttention(name, length: length, runner: runner) }
            let actual = try await PPLXTextRuntime.run(route.sequences, length: length, runner: runner, table: table, norm: norm, causal: causal)
            return Self.compare(actual, with: route.expected, name: name)
        } catch {
            return .failed(String(describing: error))
        }
    }

    static func compare(_ actual: [[Float]], with expected: [[Float]], name: String) -> Metrics {
        let width = PPLXTokenTable.width
        guard !actual.isEmpty, actual.count == expected.count,
              actual.allSatisfy({ $0.count == width }),
              expected.allSatisfy({ $0.count == width })
        else { return .failed("unexpected \(name) output shape") }

        let minimumCosine: Float = 0.995
        // The angular tolerance permits a 10% L2 difference at equal norm.
        // Apply the same tolerance to magnitude: the int8 transform uses raw means.
        let normTolerance = sqrt(2 * (1 - minimumCosine))
        var minCos: Float = 1
        var maxNormError: Float = 0
        for (mean, reference) in zip(actual, expected) {
            let cosine = Vectors.cosine(mean, reference)
            guard cosine.isFinite else { return .failed("\(name) returned a nonfinite cosine") }
            minCos = min(minCos, cosine)
            var meanSq: Float = 0
            var referenceSq: Float = 0
            vDSP_svesq(mean, 1, &meanSq, vDSP_Length(width))
            vDSP_svesq(reference, 1, &referenceSq, vDSP_Length(width))
            let relativeError = abs(sqrt(meanSq / referenceSq) - 1)
            guard relativeError.isFinite else { return .failed("\(name) returned an invalid mean magnitude") }
            maxNormError = max(maxNormError, relativeError)
        }
        let error: String?
        if minCos < minimumCosine {
            error = "\(name) raw cosine \(minCos) < \(minimumCosine)"
        } else if maxNormError > normTolerance {
            error = "\(name) raw mean L2 magnitude differs by \(maxNormError); tolerance \(normTolerance)"
        } else {
            error = nil
        }
        return Metrics(passed: error == nil, cosine: Double(minCos), error: error)
    }

    private static func runAttention(_ name: String, length: Int, runner: any PPLXPredicting) async throws {
        let heads = PPLXDecomposition.keyHeads
        let width = PPLXDecomposition.headWidth
        let rows = PPLXDecomposition.tileRows
        let qHeads = PPLXDecomposition.queryHeads
        let q = try PPLXTensor.half([1, qHeads, rows, width], zero: true)
        let k = try PPLXTensor.half([1, heads, length, width], zero: true)
        let v = try PPLXTensor.half([1, heads, length, width], zero: true)
        let bias = try PPLXTensor.half([1, 1, 1, length])
        let target = try PPLXTensor.half([1, qHeads, rows, width])
        let mask = PPLXHostMath.halfPointer(bias)
        mask.update(repeating: -10_000, count: length)
        mask[0] = 0
        mask[length - 1] = 0
        let values = PPLXHostMath.halfPointer(v)
        for head in 0..<heads {
            (values + head * length * width).update(repeating: 1, count: width)
            (values + ((head + 1) * length - 1) * width).update(repeating: 3, count: width)
        }
        let inputs = ["q": q, "k": k, "v": v, "bias": bias]
        let sparse = try await runner.predict(name, inputs: inputs, outputs: ["o": target])
        try checkAttention(sparse, rows: rows, width: width, heads: qHeads, expected: 2, name: name, caseName: "key blocks")

        // Uniform 4s have a representable mean but a sum that exceeds fp16
        // at both supported lengths. Keep the same storage for the second canary.
        mask.update(repeating: 0, count: length)
        values.update(repeating: 4, count: heads * length * width)
        let dense = try await runner.predict(name, inputs: inputs, outputs: ["o": target])
        try checkAttention(dense, rows: rows, width: width, heads: qHeads, expected: 4, name: name, caseName: "fp16 range")
    }

    private static func checkAttention(_ result: MLFeatureProvider, rows: Int, width: Int, heads: Int,
                                       expected: Float, name: String, caseName: String) throws
    {
        let out = try PPLXTensor.output(result, "o")
        let pointer = PPLXHostMath.halfPointer(out)
        let strides = out.strides
        guard out.dataType == .float16, strides.count == 4, strides[3].intValue == 1 else {
            throw RuntimeError.model("\(name) canary received an invalid tensor")
        }
        var floats = [Float](repeating: 0, count: rows * width)
        var offset = -expected
        try floats.withUnsafeMutableBufferPointer { scratch in
            let count = vDSP_Length(scratch.count)
            for head in 0..<heads {
                let values = scratch.baseAddress!
                try PPLXHostMath.toFloat(pointer + head * strides[1].intValue, values,
                                         rows: rows, columns: width, inputStride: strides[2].intValue)
                vDSP_vsadd(values, 1, &offset, values, 1, count)
                var maxError: Float = 0
                vDSP_maxmgv(values, 1, &maxError, count)
                guard maxError.isFinite, maxError <= 0.01 else {
                    throw RuntimeError.model("\(name) \(caseName) canary failed (max error \(maxError))")
                }
            }
        }
    }
}
