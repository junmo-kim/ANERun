/// Float32 magnitude thresholds of the pinned PyTorch 2.7.1 pplx int8 quantizer.
enum PPLXQuantization {
    // Each threshold is the first positive Float32 bit pattern rounded to the
    // next code by torch.round(torch.tanh(x) * 127). Comparing bit patterns
    // keeps parity at the half-integer boundaries across macOS libm versions.
    private static let thresholds: [UInt32] = [
        0x3b810231, 0x3c418554, 0x3ca147db, 0x3ce1d22a, 0x3d1131d6, 0x3d317f32, 0x3d51d237, 0x3d722beb,
        0x3d8946ae, 0x3d997bca, 0x3da9b5d2, 0x3db9f54c, 0x3dca3ac3, 0x3dda86c0, 0x3dead9cf, 0x3dfb347d,
        0x3e05cbad, 0x3e0e0178, 0x3e163bed, 0x3e1e7b51, 0x3e26bff3, 0x3e2f0a1c, 0x3e375a1c, 0x3e3fb03d,
        0x3e480cd3, 0x3e50702b, 0x3e58da9d, 0x3e614c75, 0x3e69c610, 0x3e7247bf, 0x3e7ad1e1, 0x3e81b265,
        0x3e86006f, 0x3e8a533b, 0x3e8eaafc, 0x3e9307e1, 0x3e976a22, 0x3e9bd1ed, 0x3ea03f7c, 0x3ea4b303,
        0x3ea92cbf, 0x3eadace4, 0x3eb233b4, 0x3eb6c168, 0x3ebb5644, 0x3ebff284, 0x3ec49670, 0x3ec9424a,
        0x3ecdf65d, 0x3ed2b2f1, 0x3ed77856, 0x3edc46d7, 0x3ee11ecc, 0x3ee60084, 0x3eeaec5e, 0x3eefe2b0,
        0x3ef4e3df, 0x3ef9f049, 0x3eff085a, 0x3f02163d, 0x3f04ae8e, 0x3f074d59, 0x3f09f2db, 0x3f0c9f54,
        0x3f0f5308, 0x3f120e38, 0x3f14d133, 0x3f179c3f, 0x3f1a6fb2, 0x3f1d4bdb, 0x3f203116, 0x3f231fbc,
        0x3f261835, 0x3f291ae0, 0x3f2c2830, 0x3f2f4093, 0x3f326487, 0x3f359488, 0x3f38d123, 0x3f3c1ae4,
        0x3f3f726a, 0x3f42d853, 0x3f464d52, 0x3f49d21d, 0x3f4d6780, 0x3f510e49, 0x3f54c764, 0x3f5893c1,
        0x3f5c746c, 0x3f606a7c, 0x3f64772c, 0x3f689bc2, 0x3f6cd9af, 0x3f713274, 0x3f75a7c7, 0x3f7a3b75,
        0x3f7eef88, 0x3f81e317, 0x3f8460ef, 0x3f86f2a1, 0x3f8999b1, 0x3f8c57c7, 0x3f8f2ec7, 0x3f9220c7,
        0x3f95302b, 0x3f985fa4, 0x3f9bb24d, 0x3f9f2baf, 0x3fa2cfec, 0x3fa6a3cc, 0x3faaacfa, 0x3faef224,
        0x3fb37b60, 0x3fb8526a, 0x3fbd8356, 0x3fc31d27, 0x3fc93311, 0x3fcfddf8, 0x3fd73f2d, 0x3fdf8473,
        0x3fe8ef5d, 0x3ff3e26c, 0x40007e05, 0x4008a98a, 0x40138e94, 0x402407bc, 0x40475008,
    ]

    static func code(_ value: Float) -> Float {
        guard !value.isNaN else { return value }
        let magnitude = value.bitPattern & 0x7fff_ffff
        var lower = 0
        var upper = thresholds.count
        while lower < upper {
            let midpoint = (lower + upper) / 2
            if magnitude < thresholds[midpoint] {
                upper = midpoint
            } else {
                lower = midpoint + 1
            }
        }
        guard lower != 0 else { return 0 }
        return Float(value.sign == .minus ? -lower : lower)
    }
}
