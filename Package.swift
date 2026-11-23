// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "HaruKit",
    platforms: [.iOS(.v18)],
    products: [
        .library(name: "HaruKit", targets: ["HaruKit"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.33.0"),
    ],
    targets: [
        .binaryTarget(name: "HaruSpeechCpp", path: "Artifacts/HaruSpeechCpp.xcframework.zip"),
        .target(
            name: "HaruKit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                "HaruSpeechCpp",
            ],
            resources: [.process("Resources"), .copy("Licenses")],
            swiftSettings: [.define("HARU_ENABLED")]
        ),
    ],
    swiftLanguageModes: [.v5]
)
