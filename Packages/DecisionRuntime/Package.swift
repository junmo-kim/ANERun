// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "DecisionRuntime",
  platforms: [.macOS("27.0")],
  products: [
    .library(name: "DecisionRuntime", targets: ["DecisionRuntime"]),
    .executable(name: "decision-bench", targets: ["decision-bench"]),
  ],
  targets: [
    .target(name: "DecisionProfiles"),
    .target(name: "DecisionRuntime", dependencies: ["DecisionProfiles"]),
    .executableTarget(name: "decision-bench", dependencies: ["DecisionRuntime"]),
    .testTarget(
      name: "DecisionRuntimeTests", dependencies: ["DecisionRuntime"],
      resources: [.copy("Fixtures")]),
  ],
  swiftLanguageModes: [.v5]
)
