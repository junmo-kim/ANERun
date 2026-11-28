// swift-tools-version: 6.0
import PackageDescription

// ANERun serves embeddings on macOS 15+ and Core AI decision bundles on macOS 27+.
// Both use the same command-line executable.
//
// Release builds use the default -O. Unlike a short-lived CLI, this is a long-running server whose
// own code (tokenizer, gathers, bias building, JSON encoding) sits on the request path, so speed is
// the default; -Osize is to be measured against it before changing.

let package = Package(
    name: "ANERun",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "anerun", targets: ["anerun"]),
        .library(name: "EmbedCore", targets: ["EmbedCore"]),
    ],
    targets: [
        // Tokenizer, token table, image preprocessing, bias/pool builders, packing planner,
        // runtime protocols, API and MCP logic. Everything that does not open a socket.
        .target(name: "EmbedCore", dependencies: ["DecisionProfiles"]),
        // CLI, HTTP listener (Network.framework) and stdio MCP loop.
        .target(name: "DecisionProfiles", path: "Packages/DecisionRuntime/Sources/DecisionProfiles"),
        .target(name: "DecisionRuntime", dependencies: ["DecisionProfiles"], path: "Packages/DecisionRuntime/Sources/DecisionRuntime",
                swiftSettings: [.swiftLanguageMode(.v5)]),
        .executableTarget(name: "anerun", dependencies: ["EmbedCore", "DecisionRuntime"]),
        // Development tool, not a product: deterministic mutation fuzzing of the JPEG decoder. Tests run it as a
        // child process, because a trap cannot be caught in process.
        .executableTarget(name: "jpeg-fuzz", dependencies: ["EmbedCore"]),
        // Pure logic: runs anywhere, including CI runners without a Neural Engine.
        .testTarget(name: "EmbedCoreTests", dependencies: ["EmbedCore", "jpeg-fuzz", "anerun"]),
        // HTTP/OpenAI/MCP contracts end to end through the fake runtime (no model needed).
        .testTarget(name: "ServiceTests", dependencies: ["EmbedCore"]),
        // Real model on the Neural Engine. Skipped when CI is set or no model directory exists.
        .testTarget(name: "ModelTests", dependencies: ["EmbedCore"]),
    ]
)
