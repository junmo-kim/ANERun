@preconcurrency import CoreML
import Foundation

protocol PPLXPredicting: Sendable {
    /// Each caller owns its inputs and backings through completion. The returned features identify
    /// the actual outputs, which may differ from the supplied backings.
    func predict(_ function: String, inputs: [String: MLMultiArray], outputs: [String: MLMultiArray])
        async throws -> MLFeatureProvider
}

extension PPLXPredicting {
    func predict(_ function: String, inputs: [String: MLMultiArray]) async throws -> MLFeatureProvider {
        try await predict(function, inputs: inputs, outputs: [:])
    }
}

final class PPLXCoreMLRunner: PPLXPredicting, @unchecked Sendable {
    private let models: [String: MLModel]
    private let permit: AsyncSemaphore

    private init(models: [String: MLModel], permit: AsyncSemaphore) {
        self.models = models
        self.permit = permit
    }

    static func load(_ functions: [PPLXPackageContract.Function], from directory: URL,
                     unavailablePackages: [String: String])
        async -> (runner: PPLXCoreMLRunner, failures: [String: String])
    {
        var packages: [String: Result<URL, any Error>] = [:]
        var models: [String: MLModel] = [:]
        var failures: [String: String] = [:]
        for function in functions {
            if let failure = unavailablePackages[function.package] {
                failures[function.function] = failure
                continue
            }
            do {
                try Task.checkCancellation()
                let package: Result<URL, any Error>
                if let cached = packages[function.package] {
                    package = cached
                } else {
                    let source = directory.appendingPathComponent(function.package).appendingPathExtension("mlpackage")
                    do { package = .success(try await compiledModelURL(source)) }
                    catch { package = .failure(error) }
                    packages[function.package] = package
                }
                let url = try package.get()
                let configuration = MLModelConfiguration()
                configuration.computeUnits = .cpuAndNeuralEngine
                configuration.functionName = function.runtimeFunctionName
                let model = try await MLModel.load(contentsOf: url, configuration: configuration)
                try Task.checkCancellation()
                let description = model.modelDescription
                try check(description.inputDescriptionsByName, against: function.inputs, direction: "input")
                try check(description.outputDescriptionsByName, against: function.outputs, direction: "output")
                models[function.function] = model
            } catch {
                let reason = (error as? RuntimeError)?.description ?? error.localizedDescription
                failures[function.function] = "\(function.package).mlpackage / \(function.function): \(reason)"
            }
        }
        return (PPLXCoreMLRunner(models: models, permit: AsyncSemaphore(1)), failures)
    }

    func keeping(_ names: Set<String>) -> PPLXCoreMLRunner {
        PPLXCoreMLRunner(models: models.filter { names.contains($0.key) }, permit: permit)
    }

    func predict(_ function: String, inputs: [String: MLMultiArray], outputs: [String: MLMultiArray])
        async throws -> MLFeatureProvider
    {
        guard let model = models[function] else {
            throw RuntimeError.model("pplx function \(function) is not loaded")
        }
        await permit.wait()
        let result: MLFeatureProvider
        do {
            try Task.checkCancellation()
            let features = try MLDictionaryFeatureProvider(dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) })
            let options = MLPredictionOptions()
            options.outputBackings = outputs
            try Task.checkCancellation()
            result = try await model.prediction(from: features, options: options)
        } catch {
            await permit.signal()
            try Task.checkCancellation()
            throw error
        }
        await permit.signal()
        try Task.checkCancellation()
        return result
    }

    private static func check(_ descriptions: [String: MLFeatureDescription], against expected: [PPLXPackageContract.Tensor],
                              direction: String) throws(RuntimeError)
    {
        let names = expected.map(\.name).sorted()
        guard descriptions.keys.sorted() == names else {
            throw .model("\(direction) names \(descriptions.keys.sorted()), expected \(names)")
        }
        for tensor in expected {
            let dtype: MLMultiArrayDataType = tensor.dtype == .fp16 ? .float16 : .float32
            guard let feature = descriptions[tensor.name], feature.type == .multiArray,
                  let array = feature.multiArrayConstraint,
                  array.dataType == dtype, array.shape.map(\.intValue) == tensor.shape,
                  matches(array.shapeConstraint, shape: tensor.shape)
            else {
                throw .model("\(direction) \(tensor.name) must be \(tensor.dtype.rawValue) \(tensor.shape)")
            }
        }
    }

    private static func matches(_ constraint: MLMultiArrayShapeConstraint, shape: [Int]) -> Bool {
        switch constraint.type {
        case .unspecified:
            return true
        case .enumerated:
            return constraint.enumeratedShapes.map { $0.map(\.intValue) } == [shape]
        case .range:
            let ranges = constraint.sizeRangeForDimension
            return ranges.count == shape.count && zip(ranges, shape).allSatisfy {
                $0.0.rangeValue == NSRange(location: $0.1, length: 1)
            }
        @unknown default:
            return false
        }
    }
}

enum PPLXTensor {
    static func half(_ shape: [Int], zero: Bool = false) throws(RuntimeError) -> MLMultiArray {
        let array = try contiguous(shape, dtype: .float16)
        if zero {
            array.withUnsafeMutableBufferPointer(ofType: Float16.self) { buffer, _ in
                buffer.initialize(repeating: 0)
            }
        }
        return array
    }

    static func output(_ features: MLFeatureProvider, _ name: String) throws(RuntimeError) -> MLMultiArray {
        guard let value = features.featureValue(for: name), let array = value.multiArrayValue else {
            throw .model("pplx prediction has no multi-array output named \(name)")
        }
        return array
    }

    static func contiguous(_ shape: [Int], dtype: MLMultiArrayDataType) throws(RuntimeError) -> MLMultiArray {
        var strides: [Int] = []
        var count = 1
        guard !shape.isEmpty else { throw .model("pplx tensor needs at least one dimension") }
        for dimension in shape.reversed() {
            strides.append(count)
            let size = count.multipliedReportingOverflow(by: dimension)
            guard dimension > 0, !size.overflow else { throw .model("invalid pplx tensor shape \(shape)") }
            count = size.partialValue
        }
        let scalarBytes = dtype == .float16 ? MemoryLayout<Float16>.stride : MemoryLayout<Float>.stride
        guard count <= Int.max / scalarBytes else { throw .model("invalid pplx tensor shape \(shape)") }
        let array: MLMultiArray
        do {
            array = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: dtype)
        } catch {
            throw .model("cannot allocate pplx tensor \(shape): \(error.localizedDescription)")
        }
        guard array.dataType == dtype, array.count == count, array.strides.map(\.intValue) == Array(strides.reversed())
        else { throw .model("pplx tensor \(shape) must have contiguous storage") }
        return array
    }
}
