/// Identity shared by the installer, explicit model selector and decision runtime.
public enum DecisionProfile: String, CaseIterable, Sendable {
    case valen4B = "valen-4b"
    case glide2B = "glide-2b"
    case glide9B = "glide-9b"
    case elephant1 = "elephant-1"
    case knowLine9B = "knowline-9b-gen1"
    case jeffan9B = "jeffan-9b-1211"
    case kDecision9B = "k-decision-9b"

    public var repository: String {
        switch self {
        case .knowLine9B: return "PelaAI/KnowLine-9B-Gen1"
        case .jeffan9B: return "tattrongvu/Jeffan-9B-1211"
        case .kDecision9B: return "ONYULTEA/K-Decision-9B"
        case .elephant1: return "baselabs/elephant-1"
        case .valen4B: return "Valen-Team/Valen-Decision-1.5-4B"
        case .glide2B: return "fastino/GLiDE-2B"
        case .glide9B: return "fastino/GLiDE-9B"
        }
    }

    public var revision: String {
        switch self {
        case .knowLine9B: return "325378d128e9f5eb5088196668f5874c5682b6c0"
        case .jeffan9B: return "862d1a7a8b64a8d708ddb0b0cd775a740f01841a"
        case .kDecision9B: return "29e7d96e37de0c29b498d390dfb38d1e38c17339"
        case .elephant1: return "b3f8a93462fb1051a9de031e94a7ff380e254072"
        case .valen4B: return "9b0cacc976d7fb72f252ee1adb1564f5f15e462c"
        case .glide2B: return "00f122c120cdeed82284106cb77d9bab5a50ad16"
        case .glide9B: return "ab54df5acba0eebf7df45560ffdaacd64e32df3b"
        }
    }

    public init?(repository: String) {
        guard let profile = Self.allCases.first(where: { $0.repository == repository }) else { return nil }
        self = profile
    }
}
