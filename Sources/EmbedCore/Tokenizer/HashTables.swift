import Foundation

// Open-addressing tables for the tokenizer. Swift's Dictionary hashes with SipHash and would need a
// heap-allocated [UInt8] per lookup for byte keys; these tables hash with a cheap multiplicative or
// FNV-1a hash and never allocate on lookup.

/// UInt64 key to UInt64 value. Keys must not be `UInt64.max` (the empty marker).
struct PairTable: Sendable {
    private var keys: [UInt64]
    private var values: [UInt64]
    private let mask: Int
    private(set) var count = 0

    init(minimumCapacity: Int) {
        var capacity = 16
        while capacity < minimumCapacity * 2 { capacity <<= 1 }
        keys = [UInt64](repeating: .max, count: capacity)
        values = [UInt64](repeating: 0, count: capacity)
        mask = capacity - 1
    }

    @inline(__always)
    private func slot(_ key: UInt64) -> Int {
        Int(truncatingIfNeeded: (key &* 0x9E37_79B9_7F4A_7C15) >> 32) & mask
    }

    /// Inserts or overwrites (later entries win, like a HashMap built from an iterator).
    mutating func insert(_ key: UInt64, _ value: UInt64) {
        precondition(key != .max)
        var index = slot(key)
        while true {
            if keys[index] == .max {
                keys[index] = key
                values[index] = value
                count += 1
                precondition(count * 2 <= keys.count, "PairTable over capacity")
                return
            }
            if keys[index] == key {
                values[index] = value
                return
            }
            index = (index + 1) & mask
        }
    }

    @inline(__always)
    func lookup(_ key: UInt64) -> UInt64? {
        keys.withUnsafeBufferPointer { keys in
            var index = slot(key)
            while true {
                let found = keys[index]
                if found == key { return values[index] }
                if found == .max { return nil }
                index = (index + 1) & mask
            }
        }
    }
}

/// Byte-string keys to Int32 values, keys stored back to back in one buffer. Keys are compared by
/// bytes, never as Swift Strings (String equality is canonical: U+F900 == U+8C48).
struct ByteStringTable: Sendable {
    private(set) var storage: [UInt8] = []
    private(set) var offsets: [Int32] = [0]
    private var values: [Int32] = []
    private var slots: [Int32]
    private var mask: Int

    init(minimumCapacity: Int) {
        var capacity = 16
        while capacity < minimumCapacity * 2 { capacity <<= 1 }
        slots = [Int32](repeating: -1, count: capacity)
        mask = capacity - 1
        storage.reserveCapacity(minimumCapacity * 8)
        offsets.reserveCapacity(minimumCapacity + 1)
        values.reserveCapacity(minimumCapacity)
    }

    var count: Int { values.count }

    @inline(__always)
    private static func hash(_ bytes: UnsafeRawBufferPointer) -> UInt64 {
        var hash: UInt64 = 0xCBF2_9CE4_8422_2325
        for byte in bytes {
            hash = (hash ^ UInt64(byte)) &* 0x0000_0100_0000_01B3
        }
        return hash
    }

    func key(at entry: Int) -> ArraySlice<UInt8> {
        storage[Int(offsets[entry])..<Int(offsets[entry + 1])]
    }

    func value(at entry: Int) -> Int32 { values[entry] }

    private func equals(entry: Int, _ bytes: UnsafeRawBufferPointer) -> Bool {
        let start = Int(offsets[entry]), end = Int(offsets[entry + 1])
        guard end - start == bytes.count else { return false }
        return storage.withUnsafeBytes { all in
            bytes.count == 0 || memcmp(all.baseAddress! + start, bytes.baseAddress!, bytes.count) == 0
        }
    }

    /// Inserts `bytes` with `value`. Returns false (and keeps the old entry) for a duplicate key.
    @discardableResult
    mutating func insert(_ bytes: UnsafeRawBufferPointer, value: Int32) -> Bool {
        var index = Int(truncatingIfNeeded: Self.hash(bytes)) & mask
        while slots[index] >= 0 {
            if equals(entry: Int(slots[index]), bytes) { return false }
            index = (index + 1) & mask
        }
        if (values.count + 1) * 2 > slots.count {
            grow()
            index = Int(truncatingIfNeeded: Self.hash(bytes)) & mask
            while slots[index] >= 0 { index = (index + 1) & mask }
        }
        slots[index] = Int32(values.count)
        storage.append(contentsOf: bytes)
        offsets.append(Int32(storage.count))
        values.append(value)
        return true
    }

    /// Doubles the slots and re-inserts every entry (keys and values stay where they are).
    private mutating func grow() {
        let capacity = slots.count * 2
        var grown = [Int32](repeating: -1, count: capacity)
        let newMask = capacity - 1
        storage.withUnsafeBytes { all in
            for entry in 0..<values.count {
                let start = Int(offsets[entry]), end = Int(offsets[entry + 1])
                var index = Int(truncatingIfNeeded: Self.hash(UnsafeRawBufferPointer(rebasing: all[start..<end]))) & newMask
                while grown[index] >= 0 { index = (index + 1) & newMask }
                grown[index] = Int32(entry)
            }
        }
        slots = grown
        mask = newMask
    }

    func lookup(_ bytes: UnsafeRawBufferPointer) -> Int32? {
        var index = Int(truncatingIfNeeded: Self.hash(bytes)) & mask
        while slots[index] >= 0 {
            let entry = Int(slots[index])
            if equals(entry: entry, bytes) { return values[entry] }
            index = (index + 1) & mask
        }
        return nil
    }

    func lookup(_ bytes: [UInt8]) -> Int32? {
        bytes.withUnsafeBytes { lookup($0) }
    }
}
