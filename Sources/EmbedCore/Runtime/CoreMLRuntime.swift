// Core ML backends: the packing text functions (embed_S, winp) and the vision encoder, on the Neural Engine
// (MLComputeUnits.cpuAndNeuralEngine; the GPU is never used, so a local LLM keeps it to itself).
//
// Inputs are MLMultiArrays over the BucketBuffers memory (no copy on our side). IOSurface-backed inputs
// and MLPredictionOptions.outputBackings were measured against this on an M4 and changed nothing
// (within 1%): the Neural Engine's own time is over 98% of a call.
@preconcurrency import CoreML
import Foundation

public enum NeuralEngine {
    /// Whether Core ML reports a Neural Engine on this Mac, and its core count.
    public static var device: (available: Bool, cores: Int) {
        for device in MLComputeDevice.allComputeDevices {
            if case .neuralEngine(let engine) = device { return (true, engine.totalCoreCount) }
        }
        return (false, 0)
    }
}

/// Publishes a complete compiled model atomically. Concurrent warm/serve processes share this
/// cache by revision and package path, which are also part of the OS's ANE cache key.
func compiledModelURL(_ url: URL, rebuild: Bool = false) async throws -> URL {
    let base = url.deletingPathExtension()
    let source = base.appendingPathExtension("mlpackage")
    let target = base.appendingPathExtension("mlmodelc")
    let parent = base.deletingLastPathComponent()
    let stage = parent.appendingPathComponent(".\(base.lastPathComponent)\(ModelLayout.compileStageSuffix)", isDirectory: false)
    let files = Manifest.compiledCoreFiles

    func complete(_ directory: URL) -> Bool {
        let parent = directory.deletingLastPathComponent()
        return files.allSatisfy { name in
            guard case .success(let size) = Manifest.regularFileSize("\(directory.lastPathComponent)/\(name)", in: parent)
            else { return false }
            return size > 0
        }
    }

    func reclaimStage() throws {
        var info = stat()
        if lstat(stage.path, &info) != 0 {
            if errno == ENOENT { return }
            throw RuntimeError.model("cannot inspect compiled staging slot at \(stage.path): \(String(cString: strerror(errno)))")
        }
        if (info.st_mode & S_IFMT) == S_IFDIR {
            try FileManager.default.removeItem(at: stage)
        } else if unlink(stage.path) != 0 {
            throw RuntimeError.model("cannot remove compiled staging slot at \(stage.path): \(String(cString: strerror(errno)))")
        }
    }

    // A complete cache needs the lock only when a staging entry needs attention.
    var stageInfo = stat()
    if !rebuild, complete(target), lstat(stage.path, &stageInfo) != 0, errno == ENOENT { return target }

    let lock = parent.appendingPathComponent(".\(base.lastPathComponent)\(ModelLayout.compileLockSuffix)")
    let descriptor = open(lock.path, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK, 0o644)
    guard descriptor >= 0 else { throw RuntimeError.model("cannot open compile lock at \(lock.path)") }
    defer { close(descriptor) }
    var info = stat()
    guard fstat(descriptor, &info) == 0, SmallFile.isRegular(info) else {
        throw RuntimeError.model("compile lock is not a regular file at \(lock.path)")
    }
    while flock(descriptor, LOCK_EX | LOCK_NB) != 0 {
        if errno == EINTR { try Task.checkCancellation(); continue }
        guard errno == EWOULDBLOCK else { throw RuntimeError.model("cannot lock \(lock.path)") }
        try await Task.sleep(for: .milliseconds(20))
    }
    try Task.checkCancellation()
    try reclaimStage()
    if !rebuild, complete(target) { return target }
    guard FileManager.default.fileExists(atPath: source.path) else {
        throw RuntimeError.model("incomplete compiled model at \(target.path), and no source package")
    }

    let compiled = try await MLModel.compileModel(at: source)
    defer { try? FileManager.default.removeItem(at: compiled) }
    guard complete(compiled) else { throw RuntimeError.model("compiler produced an incomplete model at \(source.path)") }
    defer { try? reclaimStage() }
    try FileManager.default.moveItem(at: compiled, to: stage)
    try Task.checkCancellation()
    var targetInfo = stat()
    let replacing = lstat(target.path, &targetInfo) == 0
    guard replacing || errno == ENOENT else {
        throw RuntimeError.model("cannot inspect compiled model at \(target.path): \(String(cString: strerror(errno)))")
    }
    let flags = replacing ? UInt32(RENAME_SWAP) : UInt32(RENAME_EXCL)
    let result = renamex_np(stage.path, target.path, flags)
    guard result == 0 else {
        throw RuntimeError.model("cannot publish compiled model at \(target.path): \(String(cString: strerror(errno)))")
    }
    return target
}

func loadFunction(_ url: URL, name: String, computeUnits: MLComputeUnits) async throws -> MLModel {
    let configuration = MLModelConfiguration()
    configuration.computeUnits = computeUnits
    configuration.functionName = name
    do {
        return try await MLModel.load(contentsOf: url, configuration: configuration)
    } catch {
        throw RuntimeError.model("cannot load \(name) from \(url.lastPathComponent): \(error.localizedDescription)")
    }
}

/// An fp16 MLMultiArray over memory we own (no copy on our side).
func halfArray(_ pointer: UnsafeMutablePointer<Float16>, _ shape: [Int]) throws -> MLMultiArray {
    var strides = [Int](repeating: 1, count: shape.count)
    for index in stride(from: shape.count - 2, through: 0, by: -1) { strides[index] = strides[index + 1] * shape[index + 1] }
    return try MLMultiArray(dataPointer: pointer, shape: shape.map(NSNumber.init(value:)), dataType: .float16,
                            strides: strides.map(NSNumber.init(value:)), deallocator: nil)
}

/// Checks that `model` has the named fp16 inputs with these shapes and the named output.
func checkSignature(_ model: MLModel, name: String, inputs: [(String, [Int])], output: String,
                    layout: String = "the windowed package layout") throws
{
    let available = model.modelDescription.inputDescriptionsByName
    for (input, shape) in inputs {
        guard let constraint = available[input]?.multiArrayConstraint else {
            throw RuntimeError.model("\(name) has no \(input) input (this anerun expects \(layout))")
        }
        guard constraint.shape.map(\.intValue) == shape else {
            throw RuntimeError.model("\(name).\(input) is \(constraint.shape), want \(shape)")
        }
    }
    guard model.modelDescription.outputDescriptionsByName[output] != nil else {
        throw RuntimeError.model("\(name) has no \(output) output")
    }
}

/// Loads several functions of one compiled model, `width` at a time. Loading a cached function is mostly CPU work
/// that grows with the size of the whole program (every function of the package is parsed), so overlapping the
/// loads shortens a start.
func loadFunctions<Key: Hashable & Sendable>(_ url: URL, _ names: [(Key, String)], computeUnits: MLComputeUnits,
                                             width: Int = 4) async throws -> [Key: MLModel]
{
    let loaded = try await windowedMap(names, width: width) { key, name in
        (key, try await loadFunction(url, name: name, computeUnits: computeUnits))
    }
    return Dictionary(loaded, uniquingKeysWith: { _, last in last })
}

/// Copies rows of a 2-D (or batched) fp16/fp32 MLMultiArray into Float arrays.
func rows(of array: MLMultiArray, count: Int, width: Int) throws -> [[Float]] {
    let shape = array.shape.map(\.intValue)
    let strides = array.strides.map(\.intValue)
    guard shape.count >= 2, shape[shape.count - 1] == width, shape[shape.count - 2] >= count else {
        throw RuntimeError.model("unexpected output shape \(shape), want [\(count), \(width)]")
    }
    let rowStride = strides[strides.count - 2], columnStride = strides[strides.count - 1]
    var output: [[Float]] = []
    output.reserveCapacity(count)
    switch array.dataType {
    case .float32:
        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        for row in 0..<count { output.append((0..<width).map { pointer[row * rowStride + $0 * columnStride] }) }
    case .float16:
        let pointer = array.dataPointer.assumingMemoryBound(to: Float16.self)
        for row in 0..<count { output.append((0..<width).map { Float(pointer[row * rowStride + $0 * columnStride]) }) }
    default:
        throw RuntimeError.model("unexpected output type \(array.dataType.rawValue)")
    }
    return output
}

public final class CoreMLBucketRunner: BucketFunctionRunner, @unchecked Sendable {
    public let lengths: [Int]
    private let models: [Int: MLModel]

    private init(models: [Int: MLModel]) {
        self.models = models
        lengths = models.keys.sorted()
    }

    /// The same runner with only the functions named in `allowed`.
    public func keeping(_ allowed: Set<String>) -> CoreMLBucketRunner {
        CoreMLBucketRunner(models: models.filter { allowed.contains(BucketContract.functionName(forLength: $0.key)) })
    }

    public var functionNames: [String] { lengths.map(BucketContract.functionName(forLength:)) }

    /// Loads every packing function (embed_S, winp) up to the smallest one that holds `maxTokens` (at most 2048).
    public static func load(from url: URL, maxTokens: Int, computeUnits: MLComputeUnits = .cpuAndNeuralEngine)
        async throws -> CoreMLBucketRunner
    {
        let compiled = try await compiledModelURL(url)
        let models = try await loadFunctions(
            compiled, lengths(forMaxTokens: maxTokens).map { ($0, BucketContract.functionName(forLength: $0)) },
            computeUnits: computeUnits)
        for (length, model) in models { try check(model, length: length) }
        return CoreMLBucketRunner(models: models)
    }

    /// The packing lengths a `maxTokens` limit loads.
    public static func lengths(forMaxTokens maxTokens: Int) -> [Int] {
        let all = BucketContract.lengths
        let cap = all.first { $0 >= min(maxTokens, all.last!) }!
        return all.filter { $0 <= cap }
    }

    /// Fails early with a readable message when the package does not follow the contract.
    static func check(_ model: MLModel, length: Int) throws {
        let sliding = BucketContract.isWindowed(length)
            ? (WindowContract.bandBias, [1, 1, length, WindowContract.bandColumns])
            : (BucketContract.localBias, [1, 1, length, length])
        try checkSignature(model, name: BucketContract.functionName(forLength: length), inputs: [
            (BucketContract.inputsEmbeds, [1, length, BucketContract.hidden]),
            (BucketContract.globalBias, [1, 1, length, length]), sliding,
            (BucketContract.pool, [BucketContract.slots(forLength: length), length]),
        ], output: BucketContract.embedding)
    }

    public func run(_ buffers: BucketBuffers) async throws -> [[Float]] {
        let (model, input) = try features(buffers)
        let output = try await model.prediction(from: input)
        return try embeddings(output, buffers)
    }

    /// The function for the buffers' length and its inputs: MLMultiArrays over the buffer memory, made once
    /// per buffer (the memory never moves, the values are read at prediction time).
    func features(_ buffers: BucketBuffers) throws -> (MLModel, MLFeatureProvider) {
        guard let model = models[buffers.length] else {
            throw RuntimeError.model("\(BucketContract.functionName(forLength: buffers.length)) is not loaded")
        }
        if let input = buffers.modelInput as? MLFeatureProvider { return (model, input) }
        let length = buffers.length, hidden = BucketContract.hidden
        let slidingName = buffers.windowed ? WindowContract.bandBias : BucketContract.localBias
        let input = try MLDictionaryFeatureProvider(dictionary: [
            BucketContract.inputsEmbeds: MLFeatureValue(multiArray: try halfArray(buffers.inputsEmbeds, [1, length, hidden])),
            BucketContract.globalBias: MLFeatureValue(multiArray: try halfArray(buffers.globalBias, [1, 1, length, length])),
            slidingName: MLFeatureValue(
                multiArray: try halfArray(buffers.slidingBias, [1, 1, length, buffers.slidingColumns])),
            BucketContract.pool: MLFeatureValue(multiArray: try halfArray(buffers.pool, [buffers.slots, length])),
        ])
        buffers.modelInput = input
        return (model, input)
    }

    /// Every slot's row of the output (rows of unused slots are 0).
    func embeddings(_ output: MLFeatureProvider, _ buffers: BucketBuffers) throws -> [[Float]] {
        guard let embedding = output.featureValue(for: BucketContract.embedding)?.multiArrayValue else {
            throw RuntimeError.model("\(BucketContract.functionName(forLength: buffers.length)) returned no \(BucketContract.embedding)")
        }
        return try rows(of: embedding, count: buffers.slots, width: BucketContract.dimension)
    }

    /// One call per function so the first request does not pay for paging each one in.
    public func warm() async throws {
        for length in lengths {
            let buffers = BucketBuffers(length: length)
            buffers.setLayout([PackedSegment(item: 0, offset: 0, length: 2)])
            _ = try await run(buffers)
        }
    }
}

/// Reusable fp16 input buffers for one vision budget. Steady-state calls allocate nothing, and
/// when consecutive images share the same resized dimensions, geometry-only inputs (`positions`,
/// `position_embeddings`, `valid`, `pool`) are already populated.
final class VisionBuffers: @unchecked Sendable {
    let budget: Int
    let total: Int
    private let patchesArray: MLMultiArray
    private let positionsArray: MLMultiArray
    private let embeddingsArray: MLMultiArray
    private let validArray: MLMultiArray
    private let poolArray: MLMultiArray
    let patches: UnsafeMutablePointer<Float16>
    let positions: UnsafeMutablePointer<Float16>
    let embeddings: UnsafeMutablePointer<Float16>
    let valid: UnsafeMutablePointer<Float16>
    let pool: UnsafeMutablePointer<Float16>
    let provider: MLDictionaryFeatureProvider

    private var dirtyPatches = 0
    private var lastColumns = 0
    private var lastRows = 0

    static let pixelLUT: [Float16] = (0..<256).map { Float16(Float($0) / 255) }

    init(budget: Int) throws {
        self.budget = budget
        let total = budget * 9
        self.total = total
        let patchWidth = CoreMLVisionEncoder.patchWidth
        let count = NSNumber(value: total)
        let patchesArray = try MLMultiArray(shape: [1, count, NSNumber(value: patchWidth)], dataType: .float16)
        let positionsArray = try MLMultiArray(shape: [1, count, 2], dataType: .float16)
        let embeddingsArray = try MLMultiArray(shape: [1, count, NSNumber(value: patchWidth)], dataType: .float16)
        let validArray = try MLMultiArray(shape: [1, count], dataType: .float16)
        let poolArray = try MLMultiArray(shape: [NSNumber(value: budget), count], dataType: .float16)
        let patches = patchesArray.dataPointer.assumingMemoryBound(to: Float16.self)
        let positions = positionsArray.dataPointer.assumingMemoryBound(to: Float16.self)
        let embeddings = embeddingsArray.dataPointer.assumingMemoryBound(to: Float16.self)
        let valid = validArray.dataPointer.assumingMemoryBound(to: Float16.self)
        let pool = poolArray.dataPointer.assumingMemoryBound(to: Float16.self)
        patches.update(repeating: 0, count: total * patchWidth)
        positions.update(repeating: -1, count: total * 2)
        embeddings.update(repeating: 0, count: total * patchWidth)
        valid.update(repeating: 0, count: total)
        pool.update(repeating: 0, count: budget * total)
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "patches": MLFeatureValue(multiArray: patchesArray),
            "positions": MLFeatureValue(multiArray: positionsArray),
            "position_embeddings": MLFeatureValue(multiArray: embeddingsArray),
            "valid": MLFeatureValue(multiArray: validArray),
            "pool": MLFeatureValue(multiArray: poolArray),
        ])
        self.patchesArray = patchesArray
        self.positionsArray = positionsArray
        self.embeddingsArray = embeddingsArray
        self.validArray = validArray
        self.poolArray = poolArray
        self.patches = patches
        self.positions = positions
        self.embeddings = embeddings
        self.valid = valid
        self.pool = pool
        self.provider = provider
    }

    func fill(_ image: PreparedImage, positionTable: Data) -> MLDictionaryFeatureProvider {
        let patchWidth = CoreMLVisionEncoder.patchWidth
        let tableRows = CoreMLVisionEncoder.tableRows
        let width = image.width, columns = width / CoreMLVisionEncoder.patch, rows = image.height / CoreMLVisionEncoder.patch
        let usedPatches = rows * columns
        precondition(usedPatches <= total && width % 48 == 0 && image.height % 48 == 0)

        let geometryChanged = columns != lastColumns || rows != lastRows
        if geometryChanged {
            if usedPatches < dirtyPatches {
                let stale = dirtyPatches - usedPatches
                (patches + usedPatches * patchWidth).update(repeating: 0, count: stale * patchWidth)
                (embeddings + usedPatches * patchWidth).update(repeating: 0, count: stale * patchWidth)
                (positions + usedPatches * 2).update(repeating: -1, count: stale * 2)
                (valid + usedPatches).update(repeating: 0, count: stale)
            }
            if lastColumns > 0 && lastRows > 0 {
                let prevGroupsPerRow = lastColumns / 3
                for row in 0..<lastRows {
                    for column in 0..<lastColumns {
                        let patch = row * lastColumns + column
                        let group = (column / 3) + prevGroupsPerRow * (row / 3)
                        pool[group * total + patch] = 0
                    }
                }
            }
        }

        let pp = patches, po = positions, pe = embeddings, pv = valid, pl = pool
        image.rgb.withUnsafeBufferPointer { rgb in
            Self.pixelLUT.withUnsafeBufferPointer { lut in
                let rgbBase = rgb.baseAddress!
                let lutBase = lut.baseAddress!
                for row in 0..<rows {
                    let rowPixelBase = row * 16 * width
                    for column in 0..<columns {
                        let patch = row * columns + column
                        let base = patch * patchWidth
                        let colPixelBase = column * 16
                        for y in 0..<16 {
                            let srcRow = rgbBase + ((rowPixelBase + y * width + colPixelBase) * 3)
                            let dstRow = pp + (base + y * 48)
                            for x in 0..<48 {
                                dstRow[x] = lutBase[Int(srcRow[x])]
                            }
                        }
                    }
                }
            }
        }

        if geometryChanged {
            let groupsPerRow = columns / 3
            let poolWeight = Float16(1.0 / 9.0)
            positionTable.withUnsafeBytes { raw in
                let table = raw.bindMemory(to: Float16.self).baseAddress!
                for row in 0..<rows {
                    let yRow = (tableRows + row) * patchWidth
                    let rowGroupOffset = groupsPerRow * (row / 3)
                    let rowF = Float16(row)
                    for column in 0..<columns {
                        let patch = row * columns + column
                        let base = patch * patchWidth
                        po[patch * 2] = Float16(column)
                        po[patch * 2 + 1] = rowF
                        pv[patch] = 1
                        let xRow = column * patchWidth
                        for c in 0..<patchWidth {
                            pe[base + c] = Float16(Float(table[xRow + c]) + Float(table[yRow + c]))
                        }
                        let group = (column / 3) + rowGroupOffset
                        pl[group * total + patch] = poolWeight
                    }
                }
            }
            dirtyPatches = usedPatches
            lastColumns = columns
            lastRows = rows
        }

        return provider
    }
}

final class VisionBufferPool: @unchecked Sendable {
    private let lock = NSLock()
    private var idle: [Int: [VisionBuffers]] = [:]
    private let maxIdlePerBudget: Int

    init(maxIdlePerBudget: Int = 3) {
        self.maxIdlePerBudget = maxIdlePerBudget
    }

    func checkout(budget: Int) throws -> VisionBuffers {
        if let reused = lock.withLock({ idle[budget]?.popLast() }) {
            return reused
        }
        return try VisionBuffers(budget: budget)
    }

    func checkin(_ buffers: VisionBuffers) {
        lock.withLock {
            var list = idle[buffers.budget, default: []]
            if list.count < maxIdlePerBudget { list.append(buffers) }
            idle[buffers.budget] = list
        }
    }
}

/// EmbeddingGemma vision package functions vision_70, vision_140 and vision_280:
/// 16 px patches scaled to [0, 1], 2-D position embeddings from position_embeddings.f16 (x row + y row),
/// and a 3 x 3 average pool of the patch grid into soft tokens.
public final class CoreMLVisionEncoder: VisionEncoder, @unchecked Sendable {
    public static let allBudgets = [70, 140, 280]
    static let patch = 16
    static let patchWidth = 768  // 16 * 16 * 3
    static let tableRows = 1024
    /// Loaded budgets (soft tokens per function), ascending.
    public let budgets: [Int]
    public var maxTokens: Int { budgets.last ?? 0 }
    private let models: [Int: MLModel]
    private let positionTable: Data
    let buffers = VisionBufferPool()

    private init(models: [Int: MLModel], positionTable: Data) {
        self.models = models
        self.positionTable = positionTable
        budgets = models.keys.sorted()
    }

    /// The same encoder with only the functions named in `allowed`.
    public func keeping(_ allowed: Set<String>) -> CoreMLVisionEncoder {
        CoreMLVisionEncoder(models: models.filter { allowed.contains("vision_\($0.key)") }, positionTable: positionTable)
    }

    public var functionNames: [String] { budgets.map { "vision_\($0)" } }

    public static func load(model url: URL, positionTable: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine)
        async throws -> CoreMLVisionEncoder
    {
        guard let table = try? Data(contentsOf: positionTable, options: .alwaysMapped),
            table.count == 2 * tableRows * patchWidth * 2
        else { throw RuntimeError.model("missing or malformed \(positionTable.lastPathComponent)") }
        let compiled = try await compiledModelURL(url)
        let models = try await loadFunctions(compiled, allBudgets.map { ($0, "vision_\($0)") }, computeUnits: computeUnits)
        for (budget, model) in models { try check(model, budget: budget) }
        return CoreMLVisionEncoder(models: models, positionTable: table)
    }

    /// Fails early with a readable message when a vision function does not take the inputs `inputs(_:budget:)` builds.
    static func check(_ model: MLModel, budget: Int) throws {
        let total = budget * 9
        try checkSignature(model, name: "vision_\(budget)", inputs: [
            ("patches", [1, total, patchWidth]), ("positions", [1, total, 2]), ("position_embeddings", [1, total, patchWidth]),
            ("valid", [1, total]), ("pool", [budget, total]),
        ], output: "image_tokens", layout: "the EmbeddingGemma 2 vision package")
    }

    public func warm() async throws {
        for (budget, (columns, rows)) in zip(Self.allBudgets, [(10, 7), (14, 10), (20, 14)]) where models[budget] != nil {
            let image = PreparedImage(rgb: [UInt8](repeating: 0, count: columns * rows * 48 * 48 * 3),
                                      width: columns * 48, height: rows * 48)
            precondition(image.tokens <= budget)
            _ = try await encode(image)
        }
    }

    public func encode(_ image: PreparedImage) async throws -> [Float16] {
        let tokens = image.tokens
        guard let budget = budgets.first(where: { $0 >= tokens }), let model = models[budget] else {
            throw RuntimeError.unsupported("\(tokens) image tokens do not fit")
        }
        let buffer = try buffers.checkout(budget: budget)
        defer { buffers.checkin(buffer) }
        let provider = buffer.fill(image, positionTable: positionTable)
        let output = try await model.prediction(from: provider)
        return try softTokens(output, count: tokens, budget: budget)
    }

    /// The vision function for `image` (nil when it does not fit).
    func model(for image: PreparedImage) -> (MLModel, Int)? {
        guard let budget = budgets.first(where: { $0 >= image.tokens }), let model = models[budget] else { return nil }
        return (model, budget)
    }

    /// The first `count` rows of image_tokens, copied directly as Float16 without a Float round-trip.
    func softTokens(_ output: MLFeatureProvider, count: Int, budget: Int) throws -> [Float16] {
        guard let soft = output.featureValue(for: "image_tokens")?.multiArrayValue else {
            throw RuntimeError.model("vision_\(budget) returned no image_tokens")
        }
        return try Self.extractSoftTokens(soft, count: count)
    }

    static func extractSoftTokens(_ soft: MLMultiArray, count: Int) throws -> [Float16] {
        let width = BucketContract.hidden
        let shape = soft.shape.map(\.intValue)
        let strides = soft.strides.map(\.intValue)
        guard shape.count >= 2, shape[shape.count - 1] == width, shape[shape.count - 2] >= count else {
            throw RuntimeError.model("unexpected output shape \(shape), want [\(count), \(width)]")
        }
        let rowStride = strides[strides.count - 2], columnStride = strides[strides.count - 1]
        let total = count * width
        switch soft.dataType {
        case .float16:
            let pointer = soft.dataPointer.assumingMemoryBound(to: Float16.self)
            return [Float16](unsafeUninitializedCapacity: total) { dest, initializedCount in
                let dst = dest.baseAddress!
                if columnStride == 1 && rowStride == width {
                    dst.initialize(from: pointer, count: total)
                } else if columnStride == 1 {
                    for row in 0..<count {
                        (dst + row * width).initialize(from: pointer + row * rowStride, count: width)
                    }
                } else {
                    for row in 0..<count {
                        let srcRow = pointer + row * rowStride
                        let dstRow = dst + row * width
                        for col in 0..<width { dstRow[col] = srcRow[col * columnStride] }
                    }
                }
                initializedCount = total
            }
        case .float32:
            let pointer = soft.dataPointer.assumingMemoryBound(to: Float.self)
            return [Float16](unsafeUninitializedCapacity: total) { dest, initializedCount in
                let dst = dest.baseAddress!
                for row in 0..<count {
                    let srcRow = pointer + row * rowStride
                    let dstRow = dst + row * width
                    for col in 0..<width { dstRow[col] = Float16(srcRow[col * columnStride]) }
                }
                initializedCount = total
            }
        default:
            throw RuntimeError.model("unexpected output type \(soft.dataType.rawValue)")
        }
    }

    func fill(_ buffer: VisionBuffers, with image: PreparedImage) -> MLDictionaryFeatureProvider {
        buffer.fill(image, positionTable: positionTable)
    }

    func inputs(_ image: PreparedImage, budget: Int) throws -> MLDictionaryFeatureProvider {
        let buffer = try VisionBuffers(budget: budget)
        return buffer.fill(image, positionTable: positionTable)
    }
}
