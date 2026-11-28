// `anerun bench`: latency, throughput and memory of this binary on this Mac, in process (no socket).
//
// Requests go through the same Router as `serve` (JSON parse, tokenize, schedule, Core ML, JSON encode),
// so the numbers are what a client sees minus the loopback round trip. Texts use a fixed bilingual
// sentence cut to the target token count, including the document prompt and framing for EmbeddingGemma 2.
import Darwin
import Foundation

public enum Bench {
    public struct Options: Sendable {
        public var sizes = [32, 128, 512, 1024, 2048, 4096, 8192]
        public var repeats = 10
        public var image: Data?
        public var buckets = false
        public var breakdown = false
        public var concurrency = 0
        public var settleSeconds = 0.0

        public init() {}
    }

    static let base = String(repeating: "임베딩 서버의 지연과 정확도를 측정하는 문장입니다. The quick brown fox jumps over the lazy dog. ",
                             count: 900)
    static let prompt = InputType.document

    /// Request text sized with the selected tokenizer, including any model-specific prompt.
    static func text(tokens: Int, service: EmbeddingService, offset: Int = 0) throws(ServiceError) -> String {
        let textOnly = !service.modelID.supportsImages
        let limit = tokens * (textOnly ? 8 : 4)
        let available = textOnly && limit > base.count ? String(repeating: base, count: (limit + base.count - 1) / base.count) : base
        let source = Array(available.dropFirst(offset % 50).prefix(limit))
        var lower = 1, upper = source.count
        while lower < upper {
            let middle = (lower + upper + 1) / 2
            if service.tokenize(String(source[..<middle]), inputType: prompt).count <= tokens { lower = middle } else { upper = middle - 1 }
        }
        let raw = String(source[..<lower])
        let input = prompt.apply(raw, modelID: service.modelID)
        let count = service.tokenize(input).count
        if textOnly, count != tokens {
            // These give exact counts with Qwen and the byte tokenizer, respectively.
            let framing = service.tokenize("").count
            for unit in [" x", "x"] {
                let repeated = String(repeating: unit, count: max(0, tokens - framing))
                if service.tokenize(repeated).count == tokens { return repeated }
            }
        }
        if count > 0, count <= tokens { return input }
        return try shortInput(service: service, limit: tokens)
    }

    private static func shortInput(service: EmbeddingService, limit: Int) throws(ServiceError) -> String {
        for input in ["x", ""] {
            let count = service.tokenize(input).count
            if count > 0, count <= limit { return input }
        }
        throw ServiceError(.internal, "bench_failed", "cannot create a benchmark input within \(limit) model tokens")
    }

    public static func run(_ config: Config, options: Options, log: @Sendable (String) -> Void) async throws(ServiceError) -> JSONValue {
        let config = try config.selectingModel(for: "bench")
        if options.image != nil, let model = config.modelID, !model.supportsImages {
            throw EmbeddingService.imagesUnsupported(model)
        }
        let loadStarted = Date()
        let (service, report) = try await ModelLoader.load(config, log: log)
        let loadSeconds = Date().timeIntervalSince(loadStarted)
        let router = Router(state: ServerState(.ready(service)), modelName: service.modelName, modelID: service.modelID)
        var output: JSONObject = [
            "aneembed": .string(aneembedVersion), "macos_build": .string(operatingSystemBuild), "chip": .string(chipName),
            "max_tokens": .int(service.maxTokens), "functions": .int(service.runtime.info.functions.count),
            "load_seconds": .number(round(loadSeconds, 2)),
            "selftest": .string(report.selfTest?.summary ?? "n/a"),
            "selftest_seconds": .number(round(report.selfTest?.seconds ?? 0, 2)),
            "footprint_after_load_mb": .number(round(Memory.footprintMB, 1)),
        ]

        func request(_ body: JSONValue) -> HTTPRequest {
            HTTPRequest(method: "POST", path: "/v1/embeddings", headers: [("Content-Type", "application/json")], body: body.jsonData())
        }
        func time(_ request: HTTPRequest, repeats: Int) async throws(ServiceError) -> (median: Double, p90: Double, tokens: Int) {
            var samples: [Double] = []
            var tokens = 0
            for index in 0...repeats {  // the first call is a warm-up
                let started = DispatchTime.now().uptimeNanoseconds
                let response = await router.handle(request)
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6
                guard response.status == 200 else {
                    throw ServiceError(.internal, "bench_failed", String(decoding: response.body, as: UTF8.self))
                }
                if index == 0 {
                    tokens = (try? JSONValue.parse(response.body))?["usage"]?["prompt_tokens"]?.intValue ?? 0
                } else {
                    samples.append(elapsed)
                }
            }
            samples.sort()
            return (samples[samples.count / 2], samples[min(samples.count - 1, Int(Double(samples.count) * 0.9))], tokens)
        }

        // Single texts
        var single: JSONObject = [:]
        for size in options.sizes where size <= service.maxTokens {
            let input = try text(tokens: size, service: service)
            let repeats = service.modelID == .embeddingGemma2 && size > 2048 ? max(3, options.repeats / 3) : options.repeats
            let result = try await time(request(["input": .string(input)]), repeats: repeats)
            single[String(size)] = ["median_ms": .number(round(result.median, 1)), "p90_ms": .number(round(result.p90, 1)),
                                    "tokens": .int(result.tokens)]
            log("single \(size): \(round(result.median, 1)) ms (\(result.tokens) tokens)")
        }
        output["single"] = .object(single)

        // Batches of 10 texts in one request
        var batches: JSONObject = [:]
        for size in [32, 2048] where size <= service.maxTokens {
            var texts: [JSONValue] = []
            for index in 0..<10 {
                texts.append(.string(try text(tokens: size, service: service, offset: index * 3)))
            }
            let result = try await time(request(["input": .array(texts)]), repeats: max(3, options.repeats / 2))
            batches[String(size)] = ["median_ms": .number(round(result.median, 1)), "tokens": .int(result.tokens)]
            log("batch of 10 x \(size): \(round(result.median, 1)) ms")
        }
        output["batch10"] = .object(batches)

        if let image = options.image, service.maxImageTokens > 0 {
            let url = "data:image/png;base64," + image.base64EncodedString()
            let item: JSONValue = ["content": [["type": "image_url", "image_url": ["url": .string(url)]]]]
            let result = try await time(request(["input": [item]]), repeats: options.repeats)
            output["image"] = ["median_ms": .number(round(result.median, 1)), "p90_ms": .number(round(result.p90, 1)),
                               "tokens": .int(result.tokens)]
            log("image: \(round(result.median, 1)) ms (\(result.tokens) tokens)")
        }

        if options.concurrency > 0 {
            output["concurrent"] = try await concurrent(router: router, service: service, clients: options.concurrency, log: log)
        }
        if options.buckets, let runtime = service.runtime as? BucketTextRuntime {
            output["bucket_call_ms"] = try await bucketCosts(runtime, repeats: max(10, options.repeats * 2), log: log)
        }
        if options.breakdown, let runtime = service.runtime as? BucketTextRuntime {
            output["breakdown_ms"] = try await breakdown(runtime, service: service, router: router, image: options.image, log: log)
        } else if !service.modelID.supportsImages {
            let qwen = service.modelID == .qwen3Embedding
            let fusedLimit = qwen ? 64 : 8192
            if options.buckets {
                output["routes"] = .array(service.runtime.info.bucketLengths.map { length in
                    let implementation: JSONValue = length > fusedLimit ? "decomposed" : (qwen ? "packed" : "fused")
                    return ["length": .int(length), "implementation": implementation]
                })
            }
            if options.breakdown, let runtime = service.runtime as? PPLXTextRuntime {
                output[qwen ? "qwen_breakdown_ms" : "pplx_breakdown_ms"] = .object(
                    try await textBreakdown(runtime, service: service, sizes: options.sizes, fusedLimit: fusedLimit, log: log))
            }
        }

        output["footprint_mb"] = .number(round(Memory.footprintMB, 1))
        output["footprint_peak_mb"] = .number(round(Memory.peakFootprintMB, 1))
        if options.settleSeconds > 0 {
            log("waiting \(Int(options.settleSeconds)) s for freed pages to be reclaimed")
            try? await Task.sleep(for: .seconds(options.settleSeconds))
            output["footprint_settled_mb"] = .number(round(Memory.footprintMB, 1))
        }
        return .object(output)
    }

    static func textBreakdown(_ runtime: PPLXTextRuntime, service: EmbeddingService, sizes: [Int],
                              fusedLimit: Int, log: @Sendable (String) -> Void) async throws(ServiceError) -> JSONObject
    {
        let model = service.modelID.rawValue
        var profiles = JSONObject()
        for size in sizes where size > fusedLimit && size <= service.maxTokens {
            let ids = service.tokenize(try text(tokens: size, service: service))
            do {
                let length = try runtime.bucket(for: ids.count)
                guard length > fusedLimit else { continue }
                let profile = try await runtime.profile(ids, length: length)
                let host = profile.layers.reduce(0) { $0 + $1.host } + profile.preparationSeconds + profile.poolingSeconds
                let attention = profile.layers.reduce(0) { $0 + $1.attention }
                let prepost = profile.layers.reduce(0) { $0 + $1.pre + $1.post }
                log("\(model) profile \(ids.count) tokens: host \(round(host, 3))s, attention \(round(attention, 3))s, pre/post \(round(prepost, 3))s")
                profiles[String(size)] = [
                    "tokens": .int(ids.count), "shape": .int(length),
                    "host_ms": .number(round(host * 1000, 3)),
                    "allocation_rope_ms": .number(round(profile.preparationSeconds * 1000, 3)),
                    "pool_ms": .number(round(profile.poolingSeconds * 1000, 3)),
                    "layers": .array(profile.layers.map { layer in
                        ["pre_ms": .number(round(layer.pre * 1000, 3)),
                         "attention_ms": .number(round(layer.attention * 1000, 3)),
                         "post_ms": .number(round(layer.post * 1000, 3)),
                         "host_ms": .number(round(layer.host * 1000, 3))]
                    }),
                ]
            } catch {
                throw ServiceError(.internal, "bench_failed", "\(model) profiling \(size) tokens: \(error)")
            }
        }
        return profiles
    }

    /// `clients` concurrent clients, each sending single short queries back to back.
    static func concurrent(router: Router, service: EmbeddingService, clients: Int, log: @Sendable (String) -> Void)
        async throws(ServiceError) -> JSONValue
    {
        let perClient = 32
        let limit = min(service.maxTokens, service.limits.shortTokens)
        var fallback: String?
        var batches: [[HTTPRequest]] = []
        for client in 0..<clients {
            var requests: [HTTPRequest] = []
            for index in 0..<perClient {
                let question = "question \(client * 1000 + index) about embeddings"
                var input = InputType.query.apply(question, modelID: service.modelID)
                let count = service.tokenize(input).count
                if count == 0 || count > limit {
                    if let fallback {
                        input = fallback
                    } else {
                        input = try shortInput(service: service, limit: limit)
                        fallback = input
                    }
                }
                requests.append(HTTPRequest(method: "POST", path: "/v1/embeddings", headers: [("Content-Type", "application/json")],
                                            body: (["input": .string(input)] as JSONValue).jsonData()))
            }
            batches.append(requests)
        }
        let started = DispatchTime.now().uptimeNanoseconds
        let measurement = await withTaskGroup(of: Result<[Double], ServiceError>.self) { group -> Result<[Double], ServiceError> in
            for requests in batches {
                group.addTask {
                    var samples: [Double] = []
                    samples.reserveCapacity(requests.count)
                    for request in requests {
                        guard !Task.isCancelled else {
                            return .failure(ServiceError(.internal, "bench_failed", "concurrent benchmark cancelled"))
                        }
                        let begin = DispatchTime.now().uptimeNanoseconds
                        let response = await router.handle(request)
                        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - begin) / 1e6
                        guard response.status == 200 else {
                            let detail = String(decoding: response.body, as: UTF8.self)
                            return .failure(ServiceError(.internal, "bench_failed",
                                                         "concurrent embedding returned HTTP \(response.status) (\(detail))"))
                        }
                        samples.append(elapsed)
                    }
                    return .success(samples)
                }
            }
            var all: [Double] = []
            for await result in group {
                switch result {
                case .success(let samples):
                    all += samples
                case .failure(let error):
                    group.cancelAll()
                    return .failure(error)
                }
            }
            return .success(all.sorted())
        }
        let latencies = try measurement.get()
        let seconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
        guard !latencies.isEmpty, seconds > 0 else {
            throw ServiceError(.internal, "bench_failed", "no concurrent embedding throughput could be measured")
        }
        let rate = round(Double(latencies.count) / seconds, 1)
        let result: JSONValue = [
            "clients": .int(clients), "requests": .int(latencies.count),
            "requests_per_s": .number(rate),
            "median_ms": .number(round(latencies[latencies.count / 2], 1)),
            "p99_ms": .number(round(latencies[min(latencies.count - 1, latencies.count * 99 / 100)], 1)),
        ]
        log("concurrent \(clients) clients: \(rate) req/s")
        return result
    }

    /// Median latency of one call per packing function (one text filling it): the planner's cost table; plus
    /// one call per loaded long function.
    static func bucketCosts(_ runtime: BucketTextRuntime, repeats: Int, log: @Sendable (String) -> Void)
        async throws(ServiceError) -> JSONValue
    {
        var costs: JSONObject = [:]
        for length in runtime.planner.lengths {
            let buffers = BucketBuffers(length: length)
            buffers.prepare(PackedCall(length: length, segments: [PackedSegment(item: 0, offset: 0, length: length)]),
                            ids: [[SpecialTokens.bos] + [Int32](repeating: 1000, count: length - 2) + [SpecialTokens.eos]],
                            rows: runtime.rows)
            var samples: [Double] = []
            for index in 0...repeats {
                let started = DispatchTime.now().uptimeNanoseconds
                do { _ = try await runtime.runner.run(buffers) } catch { throw ServiceError(.internal, "bench_failed", "\(error)") }
                if index > 0 { samples.append(Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6) }
            }
            samples.sort()
            costs[String(length)] = .number(round(samples[samples.count / 2], 2))
            log("\(BucketContract.functionName(forLength: length)): \(round(samples[samples.count / 2], 2)) ms per call")
        }
        if let engine = runtime.long as? LongTextEngine {
            for length in engine.lengths {
                let buffers = LongBuffers(length: length)
                buffers.prepare(ids: [SpecialTokens.bos] + [Int32](repeating: 1000, count: length - 2) + [SpecialTokens.eos],
                                rows: runtime.rows)
                var samples: [Double] = []
                for index in 0...max(3, repeats / 4) {
                    let started = DispatchTime.now().uptimeNanoseconds
                    do { _ = try await engine.runner.run(buffers) } catch { throw ServiceError(.internal, "bench_failed", "\(error)") }
                    if index > 0 { samples.append(Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6) }
                }
                samples.sort()
                costs[String(length)] = .number(round(samples[samples.count / 2], 2))
                log("\(WindowContract.longName(length)): \(round(samples[samples.count / 2], 2)) ms per call")
            }
        }
        return .object(costs)
    }

    /// Where the time of one request goes, stage by stage (medians over 20 runs).
    static func breakdown(_ runtime: BucketTextRuntime, service: EmbeddingService, router: Router, image: Data?,
                          log: @Sendable (String) -> Void) async throws(ServiceError) -> JSONValue
    {
        guard let runner = runtime.runner as? CoreMLBucketRunner else { return .null }
        var result: JSONObject = [:]
        for size in [32, 512, 2048] where size <= min(service.maxTokens, runtime.planner.maxLength) {
            let input = try text(tokens: size, service: service)
            let body = (["input": .string(input)] as JSONValue).jsonData()
            var stages: [String: [Double]] = [:]
            func measure<T>(_ name: String, _ work: () throws -> T) rethrows -> T {
                let started = DispatchTime.now().uptimeNanoseconds
                let value = try work()
                stages[name, default: []].append(Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6)
                return value
            }
            for _ in 0..<21 {
                do {
                    let parsed = try measure("parse_request") { try EmbeddingsRequest.parse(body) }
                    let ids = measure("tokenize") { service.tokenize(input) }
                    let call = measure("plan") { runtime.planner.plan([ids.count])[0] }
                    let buffers = runtime.buffers.checkout(length: call.length)
                    defer { runtime.buffers.checkin(buffers) }
                    measure("gather_bias_pool") { buffers.prepare(call, ids: [ids], rows: runtime.rows) }
                    let (model, input) = try measure("feature_provider") { try runner.features(buffers) }
                    let started = DispatchTime.now().uptimeNanoseconds
                    let output = try await model.prediction(from: input)
                    stages["predict", default: []].append(Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6)
                    let rows = try measure("read_output") { try runner.embeddings(output, buffers) }
                    let embedding = EmbeddingResult(vectors: [rows[0]], tokenCounts: [ids.count])
                    _ = measure("encode_json_float") { EmbeddingsResponse.body(embedding, request: parsed, model: service.modelName) }
                    var base64Request = parsed
                    base64Request.encodingFormat = .base64
                    _ = measure("encode_json_base64") {
                        EmbeddingsResponse.body(embedding, request: base64Request, model: service.modelName)
                    }
                } catch {
                    throw ServiceError(.internal, "bench_failed", "\(error)")
                }
                let started = DispatchTime.now().uptimeNanoseconds
                _ = await router.handle(HTTPRequest(method: "POST", path: "/v1/embeddings", body: body))
                stages["whole_request", default: []].append(Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6)
            }
            var row: JSONObject = [:]
            for name in ["parse_request", "tokenize", "plan", "gather_bias_pool", "feature_provider", "predict", "read_output",
                         "encode_json_float", "encode_json_base64", "whole_request"]
            {
                let samples = stages[name, default: [0]].dropFirst().sorted()
                row[name] = .number(round(samples[samples.count / 2], 3))
            }
            result[String(size)] = .object(row)
            log("breakdown \(size): \(JSONValue.object(row).jsonString())")
        }
        if let image, let vision = runtime.vision as? CoreMLVisionEncoder {
            var stages: [String: [Double]] = [:]
            func measure<T>(_ name: String, _ work: () throws -> T) rethrows -> T {
                let started = DispatchTime.now().uptimeNanoseconds
                let value = try work()
                stages[name, default: []].append(Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6)
                return value
            }
            do {
                for _ in 0..<11 {
                    let prepared = try measure("decode_resize") { try ImagePrep.prepare(image) }
                    guard let (model, budget) = vision.model(for: prepared) else { break }
                    let input = try measure("vision_inputs") { try vision.inputs(prepared, budget: budget) }
                    let started = DispatchTime.now().uptimeNanoseconds
                    let output = try await model.prediction(from: input)
                    stages["vision_predict", default: []].append(Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6)
                    let tokens = try measure("vision_output") { try vision.softTokens(output, count: prepared.tokens, budget: budget) }
                    let sequence = ModelSequence.image(rows: tokens)
                    let textStarted = DispatchTime.now().uptimeNanoseconds
                    _ = try await BucketTextRuntime.run(sequences: [sequence], planner: runtime.planner, buffers: runtime.buffers,
                                                        runner: runtime.runner, rows: runtime.rows)
                    stages["text_bucket", default: []].append(Double(DispatchTime.now().uptimeNanoseconds - textStarted) / 1e6)
                }
            } catch {
                throw ServiceError(.internal, "bench_failed", "\(error)")
            }
            var row: JSONObject = [:]
            for name in ["decode_resize", "vision_inputs", "vision_predict", "vision_output", "text_bucket"] {
                let samples = stages[name, default: [0]].dropFirst().sorted()
                row[name] = .number(round(samples[samples.count / 2], 3))
            }
            result["image"] = .object(row)
            log("breakdown image: \(JSONValue.object(row).jsonString())")
        }
        return .object(result)
    }

    static func round(_ value: Double, _ digits: Int) -> Double {
        let scale = pow(10, Double(digits))
        return (value * scale).rounded() / scale
    }

    static var chipName: String {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var buffer = [UInt8](repeating: 0, count: max(size, 1))
        sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0)
        return String(decoding: buffer.prefix { $0 != 0 }, as: UTF8.self)
    }
}

/// The process's physical footprint (what Activity Monitor shows as Memory), from task_info.
public enum Memory {
    static func vmInfo() -> task_vm_info_data_t? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? info : nil
    }

    public static var footprintMB: Double { Double(vmInfo()?.phys_footprint ?? 0) / 1_048_576 }
    public static var peakFootprintMB: Double { Double(vmInfo()?.ledger_phys_footprint_peak ?? 0) / 1_048_576 }
}
