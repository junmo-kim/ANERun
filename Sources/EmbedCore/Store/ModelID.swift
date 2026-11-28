import DecisionProfiles
import Foundation

public enum ModelID: RawRepresentable, CaseIterable, Hashable, Sendable {
    case embeddingGemma2
    case pplx
    case qwen3Embedding
    case decision(DecisionProfile)

    public var rawValue: String {
        switch self {
        case .embeddingGemma2: return "embeddinggemma-2"
        case .pplx: return "pplx-embed-v1-0.6b"
        case .qwen3Embedding: return "qwen3-embedding-0.6b"
        case .decision(let profile): return profile.rawValue
        }
    }

    public static var allCases: [Self] {
        [.embeddingGemma2, .pplx, .qwen3Embedding] + DecisionProfile.allCases.map(Self.decision)
    }

    public init?(rawValue: String) {
        guard let model = Self.allCases.first(where: { $0.rawValue == rawValue }) else { return nil }
        self = model
    }

    public var isDecision: Bool {
        if case .decision = self { return true }
        return false
    }

    public var unsupportedEmbedding: ServiceError {
        .badRequest("unsupported_model", "\(rawValue) is a decision model", param: "model")
    }

    /// A packed decision bundle identifies its checkpoint in model.json. Embedding packages
    /// keep their own manifest contract and validation errors.
    public static func identify(in directory: URL, expected: ModelID? = nil) throws(ServiceError) -> ModelID {
        let model: ModelID
        let url = directory.appendingPathComponent("model.json")
        if !FileManager.default.fileExists(atPath: directory.appendingPathComponent(ModelLayout.manifest).path),
           FileManager.default.fileExists(atPath: url.path) {
            struct Header: Decodable {
                struct Source: Decodable { let repo, revision: String }
                let source: Source
            }
            guard let data = try? SmallFile.read(url, limit: ModelLayout.manifestLimit),
                  let header = try? JSONDecoder().decode(Header.self, from: data),
                  let profile = DecisionProfile(repository: header.source.repo), profile.revision == header.source.revision else {
                throw ServiceError(.invalidInput, "unsupported_model", "unrecognized decision bundle at \(directory.path)")
            }
            model = .decision(profile)
        } else {
            let manifest = try Manifest.load(from: directory)
            if let expected { try manifest.checkModel(expected, directory: directory) }
            guard let found = manifest.modelID else { throw manifest.unsupportedModel(in: directory) }
            model = found
        }
        if let expected, expected != model {
            throw ServiceError(.invalidInput, "model_mismatch", "selected \(expected.rawValue), found \(model.rawValue)")
        }
        return model
    }

    public var supportsImages: Bool { self == .embeddingGemma2 }

    public var source: ModelSource {
        switch self {
        case .decision(let profile): return ModelSource(repository: profile.repository, revision: "UNPUBLISHED")
        case .embeddingGemma2: return .pinned
        case .qwen3Embedding:
            return ModelSource(repository: "junmo-k/qwen3-embedding-0.6b-ane", revision: "e3f5851431a491dd7a76436b82df048f798b2744",
                               manifestSHA256: "69c6571f9a726803f0d0b6edb055d372776e521d44e7b37d2c87ff89c55e4da8")
        case .pplx:
            return ModelSource(repository: "junmo-k/pplx-embed-v1-0.6b-ane", revision: "f07ba85d35ae742b0d9fb93bcbe0ec1384466b12",
                               manifestSHA256: "80a7f0e88dd74e6d7f552ebf68c7b27baaeb636a20bdab88e223242b84dd2017")
        }
    }

    public var tokenRange: ClosedRange<Int> {
        switch self {
        case .decision: return 1...32768 // Embedding token caps do not configure decision bundles.
        case .embeddingGemma2: return 2...8192
        case .pplx, .qwen3Embedding: return 1...32768
        }
    }

    public static var catalog: String {
        """
        On macOS 27, valen-4b (recommended for corporate use) has a pinned download for M4 h16g.
        elephant-1 and glide-9b have pinned downloads for h17d. Use --from DIR for a matching local decision bundle.
        embeddinggemma-2 downloads about 1 GB. First install took about 4–5 minutes on M4.
        pplx-embed-v1-0.6b downloads about 2.4 GB. The first local install at 2,048 tokens took about 1.5 minutes on M4.
        qwen3-embedding-0.6b returns normalized text embeddings (1024 dimensions by default, any width from 32 to 1024).
        Install it with --from DIR or an explicit --repo and --revision.
        """
    }
}
