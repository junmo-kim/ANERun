// The winl{S}_b512 functions of the text package on the Neural Engine: one call per long text.
@preconcurrency import CoreML
import Foundation

public final class CoreMLLongRunner: LongFunctionRunner, @unchecked Sendable {
    public let lengths: [Int]
    private let models: [Int: MLModel]

    private init(models: [Int: MLModel]) {
        self.models = models
        lengths = models.keys.sorted()
    }

    public var functionNames: [String] { lengths.map(WindowContract.longName) }

    /// Loads winl for every length in `lengths`.
    public static func load(from compiled: URL, lengths: [Int], computeUnits: MLComputeUnits = .cpuAndNeuralEngine)
        async throws -> CoreMLLongRunner
    {
        let models = try await loadFunctions(compiled, lengths.map { ($0, WindowContract.longName($0)) },
                                             computeUnits: computeUnits)
        for (length, model) in models {
            try checkSignature(model, name: WindowContract.longName(length), inputs: [
                (BucketContract.inputsEmbeds, [1, length, BucketContract.hidden]),
                (WindowContract.bandBias, [1, 1, length, WindowContract.bandColumns]),
                (WindowContract.keyBias, [1, 1, 1, length]), (BucketContract.pool, [1, length]),
            ], output: BucketContract.embedding)
        }
        return CoreMLLongRunner(models: models)
    }

    /// The same runner with only the functions named in `allowed`.
    public func keeping(_ allowed: Set<String>) -> CoreMLLongRunner {
        CoreMLLongRunner(models: models.filter { allowed.contains(WindowContract.longName($0.key)) })
    }

    public func run(_ buffers: LongBuffers) async throws -> [Float] {
        let length = buffers.length
        guard let model = models[length] else { throw RuntimeError.model("\(WindowContract.longName(length)) is not loaded") }
        let input: MLFeatureProvider
        if let cached = buffers.modelInput as? MLFeatureProvider {
            input = cached
        } else {
            input = try MLDictionaryFeatureProvider(dictionary: [
                BucketContract.inputsEmbeds: MLFeatureValue(
                    multiArray: try halfArray(buffers.inputsEmbeds, [1, length, BucketContract.hidden])),
                WindowContract.bandBias: MLFeatureValue(
                    multiArray: try halfArray(buffers.bandBias, [1, 1, length, WindowContract.bandColumns])),
                WindowContract.keyBias: MLFeatureValue(multiArray: try halfArray(buffers.keyBias, [1, 1, 1, length])),
                BucketContract.pool: MLFeatureValue(multiArray: try halfArray(buffers.pool, [1, length])),
            ])
            buffers.modelInput = input
        }
        let output = try await model.prediction(from: input)
        guard let embedding = output.featureValue(for: BucketContract.embedding)?.multiArrayValue else {
            throw RuntimeError.model("\(WindowContract.longName(length)) returned no \(BucketContract.embedding)")
        }
        return try rows(of: embedding, count: 1, width: BucketContract.dimension)[0]
    }
}
