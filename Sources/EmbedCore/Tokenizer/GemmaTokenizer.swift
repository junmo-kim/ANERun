// The EmbeddingGemma 2 tokenizer, written from the tokenizer.json specification and checked token
// for token against Hugging Face `tokenizers` (Tests/Fixtures/tokenizer-golden.jsonl).
//
// Pipeline (what `tokenizers` does for this file):
//   1. added tokens: the 24 special tokens (<bos>, <|image>, ...) are cut out of the raw text first,
//      leftmost-longest, and become their ids directly (they are not normalized)
//   2. normalizer: Replace " " with "▁" (U+2581) in every other piece
//   3. pre_tokenizer: Split on " " (MergedWithPrevious). After step 2 no " " is left, so each ordinary piece stays
//      intact; this is why BPE has to stay O(n log n) on a whole document
//   4. BPE: one symbol per Unicode scalar (its vocab id, or <0xXX> byte tokens when the scalar is not in
//      the vocab), then merges applied lowest rank first (leftmost on ties) with a priority queue
//   5. post_processor: <bos> ... <eos>
//
// Vocabulary keys are handled as bytes from end to end. Two traps this avoids: JSONSerialization drops a
// leading U+FEFF from keys (8 vocab entries start with one), and Swift String keys merge canonically
// equivalent entries such as U+F900 and U+8C48, which are different tokens.
import Foundation

public enum TokenizerError: Error, CustomStringConvertible, Sendable {
    case unreadable(String)
    case unsupported(String)
    case malformed(String)

    public var description: String {
        switch self {
        case .unreadable(let message): return "cannot read tokenizer.json: \(message)"
        case .unsupported(let message): return "unsupported tokenizer.json: \(message)"
        case .malformed(let message): return "malformed tokenizer.json: \(message)"
        }
    }
}

public final class GemmaTokenizer: Sendable {
    public let vocabularySize: Int
    public let mergeCount: Int
    /// Ids the post-processor puts before and after every text (<bos> and <eos> for EmbeddingGemma 2).
    public let prefixIDs: [Int32]
    public let suffixIDs: [Int32]
    public let unknownID: Int32?
    /// The longest vocabulary entry or added token in bytes (93 for EmbeddingGemma 2): no token stands for more input
    /// bytes than that, so a text of n bytes has at least n / maxBytesPerToken tokens. Why this holds:
    /// - an added token is cut from the raw text and stands for exactly its own bytes;
    /// - the normalizer only lengthens the rest (" " becomes the 3-byte "▁"), so the raw bytes of a piece are at most
    ///   its normalized bytes;
    /// - a BPE symbol starts as one normalized scalar whose entry is that scalar's bytes, or as one byte token
    ///   (`<0xXX>`, a 6-byte entry for 1 input byte), and a merge's entry is its two parts' entries joined (`init`
    ///   checks it is in the vocabulary), so every token's entry is at least as long as the input it stands for.
    /// nil when some byte has no `<0xXX>` token: then a scalar outside the vocabulary becomes `<unk>`, fused with its
    /// unknown neighbours, or is dropped, and one token can stand for any length of text.
    public let maxBytesPerToken: Int?

    /// Single-scalar vocabulary entries: BMP scalars in a flat table, the rest in a dictionary.
    private let bmpIDs: [Int32]
    private let astralIDs: [UInt32: Int32]
    /// <0x00> ... <0xFF>
    private let byteIDs: [Int32]
    /// (left id << 32 | right id) -> (rank << 32 | merged id)
    private let merges: PairTable
    private let mergeBoundaries: MergeBoundaries
    /// Added tokens, longest first within each leading byte.
    private let addedTokens: [[UInt8]]
    private let addedTokenIDs: [Int32]
    private let addedFirstBytes: [Bool]
    /// Whether each added token is still cut out of the text when special tokens are not parsed.
    private let addedIsUserDefined: [Bool]
    /// Token bytes by id (for `tokens(_:)`).
    private let tokenBytes: [[UInt8]]
    /// The byte each <0xXX> token stands for, by id (-1 for other tokens).
    private let byteValues: [Int16]

    public convenience init(contentsOf url: URL) throws {
        let data: Data
        do {
            data = try Data(contentsOf: url, options: .mappedIfSafe)
        } catch {
            throw TokenizerError.unreadable("\(url.path): \(error.localizedDescription)")
        }
        try self.init(data: data)
    }

    public init(data: Data) throws {
        var parsed = try data.withUnsafeBytes { raw in try TokenizerFile.parse(raw) }

        let vocab = parsed.vocab
        vocabularySize = vocab.count
        var bmp = [Int32](repeating: -1, count: 0x10000)
        var astral: [UInt32: Int32] = [:]
        let maxID = (0..<vocab.count).map { Int(vocab.value(at: $0)) }.max() ?? 0
        guard maxID < 16 * vocab.count + 1024 else { throw TokenizerError.malformed("vocab ids are too sparse") }
        var bytesByID = [[UInt8]](repeating: [], count: maxID + 1)
        for entry in 0..<vocab.count {
            let key = Array(vocab.key(at: entry))
            let id = vocab.value(at: entry)
            bytesByID[Int(id)] = key
            if let scalar = singleScalar(key) {
                if scalar < 0x10000 { bmp[Int(scalar)] = id } else { astral[scalar] = id }
            }
        }
        bmpIDs = bmp
        astralIDs = astral
        tokenBytes = bytesByID
        mergeBoundaries = MergeBoundaries(vocabulary: bytesByID)

        var bytes = [Int32](repeating: -1, count: 256)
        let hex = Array("0123456789ABCDEF".utf8)
        for value in 0..<256 {
            let name: [UInt8] = Array("<0x".utf8) + [hex[value >> 4], hex[value & 15], UInt8(ascii: ">")]
            bytes[value] = vocab.lookup(name) ?? -1
        }
        byteIDs = bytes
        var values = [Int16](repeating: -1, count: bytesByID.count)
        for (value, id) in bytes.enumerated() where id >= 0 { values[Int(id)] = Int16(value) }
        byteValues = values
        unknownID = parsed.unknownToken.flatMap { vocab.lookup(Array($0.utf8)) }

        // Merges: rank = position in the list; later duplicates overwrite earlier ones.
        let mergeTotal = parsed.mergeCount
        var table = PairTable(minimumCapacity: max(mergeTotal, 1))
        var joined: [UInt8] = []
        try parsed.mergeStorage.withUnsafeBytes { storage in
            for rank in 0..<mergeTotal {
                let left = UnsafeRawBufferPointer(rebasing: storage[parsed.mergeRange(2 * rank)])
                let right = UnsafeRawBufferPointer(rebasing: storage[parsed.mergeRange(2 * rank + 1)])
                guard let leftID = vocab.lookup(left), let rightID = vocab.lookup(right) else {
                    throw TokenizerError.malformed("merge \(rank) refers to a token missing from the vocab")
                }
                joined.removeAll(keepingCapacity: true)
                joined.append(contentsOf: left)
                joined.append(contentsOf: right)
                guard let mergedID = joined.withUnsafeBytes({ vocab.lookup($0) }) else {
                    throw TokenizerError.malformed("merge \(rank) produces a token missing from the vocab")
                }
                let key = UInt64(UInt32(bitPattern: leftID)) << 32 | UInt64(UInt32(bitPattern: rightID))
                table.insert(key, UInt64(UInt32(rank)) << 32 | UInt64(UInt32(bitPattern: mergedID)))
            }
        }
        merges = table
        mergeCount = mergeTotal
        parsed.mergeStorage = []

        let added = parsed.addedTokens.sorted { $0.content.count > $1.content.count }
        addedTokens = added.map(\.content)
        addedTokenIDs = added.map(\.id)
        addedIsUserDefined = added.map { Self.userDefinedTokens.contains($0.content) }
        var first = [Bool](repeating: false, count: 256)
        for token in added where !token.content.isEmpty { first[Int(token.content[0])] = true }
        addedFirstBytes = first
        prefixIDs = parsed.prefixIDs
        suffixIDs = parsed.suffixIDs
        let longest = max(bytesByID.map(\.count).max() ?? 0, added.map(\.content.count).max() ?? 0)
        maxBytesPerToken = bytes.allSatisfy({ $0 >= 0 }) && longest > 0 ? longest : nil
    }

    /// A merge cannot cross two directly encoded scalars unless the resulting
    /// vocabulary entry contains them next to each other.
    private struct MergeBoundaries {
        private static let asciiBits = 128 * 128
        private static let unicodeBits = 1 << 16
        private var words = [UInt64](repeating: 0, count: (asciiBits + unicodeBits) / 64)

        init(vocabulary: [[UInt8]]) {
            for key in vocabulary where key.count > 1 {
                var previous: UInt32?
                for scalar in String(decoding: key, as: UTF8.self).unicodeScalars {
                    let current = scalar.value == 0x2581 ? 0x20 : scalar.value
                    if let previous {
                        let bit = Self.index(previous, current)
                        words[bit >> 6] |= 1 << (bit & 63)
                    }
                    previous = current
                }
            }
        }

        // ASCII has an exact table. Other pairs share a fixed-size bitmap: a
        // collision can only skip a possible cut, never permit an unsafe one.
        private static func index(_ left: UInt32, _ right: UInt32) -> Int {
            if left < 128 && right < 128 { return Int(left * 128 + right) }
            let hash = (left &* 0x9E3779B1) ^ (right &* 0x85EBCA77)
            return asciiBits + Int((hash ^ (hash >> 16)) & UInt32(unicodeBits - 1))
        }

        func contains(_ left: UInt32, _ right: UInt32) -> Bool {
            let bit = Self.index(left == 0x2581 ? 0x20 : left, right == 0x2581 ? 0x20 : right)
            return words[bit >> 6] & (1 << (bit & 63)) != 0
        }
    }

    private func canSplit(_ utf8: UnsafeBufferPointer<UInt8>, at index: Int) -> Bool {
        guard utf8[index] & 0xC0 != 0x80 else { return false }
        var start = index - 1
        while utf8[start] & 0xC0 == 0x80 { start -= 1 }
        let left = decodeScalar(utf8, at: start).0
        let right = decodeScalar(utf8, at: index).0
        guard !mergeBoundaries.contains(left, right) else { return false }

        func directlyEncoded(_ scalar: UInt32) -> Bool {
            let normalized = scalar == 0x20 ? 0x2581 : scalar
            if normalized < 0x10000 { return bmpIDs[Int(normalized)] >= 0 }
            return astralIDs[normalized] != nil
        }
        return directlyEncoded(left) && directlyEncoded(right)
    }

    // MARK: - Encoding

    /// Added tokens that llama.cpp keeps cutting out of the text with `parse_special` false: the EmbeddingGemma 2 GGUF
    /// types these six USER_DEFINED and the other added tokens CONTROL. tokenizer.json marks all of them special, so
    /// the file does not tell them apart; the list was checked against llama-server b12428.
    static let userDefinedTokens: Set<[UInt8]> = Set(
        ["<|tool_call>", "<tool_call|>", "<tool_response|>", #"<|"|>"#, "<|channel>", "<channel|>"].map { Array($0.utf8) })

    /// Token ids of `text`, with <bos>/<eos> unless `addSpecialTokens` is false. With `parseSpecial` false (llama.cpp's
    /// parse_special) the text of a control token such as "<bos>" is not cut out first but encoded like any other
    /// text; the user-defined tokens above still are.
    public func encode(_ text: String, addSpecialTokens: Bool = true, parseSpecial: Bool = true) -> [Int32] {
        var output: [Int32] = []
        var scratch = BPEScratch()
        encode(text, addSpecialTokens: addSpecialTokens, parseSpecial: parseSpecial, into: &output, scratch: &scratch)
        return output
    }

    /// Appends the ids of `text` to `output`, reusing `scratch` between calls.
    public func encode(_ text: String, addSpecialTokens: Bool = true, parseSpecial: Bool = true,
                       into output: inout [Int32], scratch: inout BPEScratch)
    {
        output.reserveCapacity(output.count + text.utf8.count / 3 + 2)
        if addSpecialTokens { output.append(contentsOf: prefixIDs) }
        encodeBody(text, parseSpecial: parseSpecial, into: &output, scratch: &scratch, stopAt: Int.max)
        if addSpecialTokens { output.append(contentsOf: suffixIDs) }
    }

    public func encodeForEmbedding(_ text: String, limit: Int) -> BoundedTokenization {
        encodeForEmbedding(text, limit: limit, truncate: false)
    }

    public func encodeForEmbedding(_ text: String, limit: Int, truncate: Bool) -> BoundedTokenization {
        var scratch = BPEScratch()
        return encodeForEmbedding(text, limit: limit, truncate: truncate, scratch: &scratch)
    }

    func encodeForEmbedding(_ text: String, limit: Int, truncate: Bool, scratch: inout BPEScratch) -> BoundedTokenization {
        let framing = prefixIDs.count + suffixIDs.count
        guard limit >= framing else { return .tooLong(atLeast: framing) }
        let bodyLimit = limit - suffixIDs.count
        var output = prefixIDs
        output.reserveCapacity(min(limit, text.utf8.count / 3 + framing))
        encodeBody(text, parseSpecial: true, into: &output, scratch: &scratch,
                   stopAt: bodyLimit == Int.max ? Int.max : bodyLimit + 1)
        let clipped = output.count > bodyLimit
        if clipped {
            guard truncate, limit > 0 else { return .tooLong(atLeast: limit + 1) }
            output.removeLast()
        }
        output.append(contentsOf: suffixIDs)
        return clipped ? .clipped(output) : .ids(output)
    }

    private func encodeBody(_ text: String, parseSpecial: Bool, into output: inout [Int32],
                            scratch: inout BPEScratch, stopAt: Int)
    {
        // Start looking for a safe cut sooner when the caller will keep only a
        // short prefix. Larger budgets amortize BPE across a wider chunk.
        let minPieceBytes = max(64, min(stopAt, 1024)) * 4
        var text = text
        text.withUTF8 { utf8 in
            var pieceStart = 0
            var index = 0
            while index < utf8.count {
                if addedFirstBytes[Int(utf8[index])],
                   let (tokenIndex, length) = matchAddedToken(utf8, at: index, userDefinedOnly: !parseSpecial)
                {
                    if index > pieceStart {
                        encodePiece(UnsafeBufferPointer(rebasing: utf8[pieceStart..<index]), into: &output,
                                    scratch: &scratch, stopAt: stopAt)
                        if output.count == stopAt { return }
                    }
                    output.append(addedTokenIDs[tokenIndex])
                    if output.count == stopAt { return }
                    index += length
                    pieceStart = index
                } else {
                    // Earlier positions have already been checked for added tokens,
                    // so no added token can start before and extend across this cut.
                    if stopAt != Int.max, index - pieceStart >= minPieceBytes, canSplit(utf8, at: index) {
                        encodePiece(UnsafeBufferPointer(rebasing: utf8[pieceStart..<index]), into: &output,
                                    scratch: &scratch, stopAt: stopAt)
                        if output.count == stopAt { return }
                        pieceStart = index
                    }
                    index += 1
                }
            }
            if utf8.count > pieceStart {
                encodePiece(UnsafeBufferPointer(rebasing: utf8[pieceStart..<utf8.count]), into: &output,
                            scratch: &scratch, stopAt: stopAt)
            }
        }
    }

    /// The number of tokens `encode` would return.
    public func count(_ text: String, addSpecialTokens: Bool = true) -> Int {
        encode(text, addSpecialTokens: addSpecialTokens).count
    }

    /// The vocabulary entry of each id as text (byte tokens stay "<0xXX>").
    public func tokens(_ ids: [Int32]) -> [String] {
        ids.map { id in
            guard id >= 0, Int(id) < tokenBytes.count else { return "<invalid \(id)>" }
            return String(decoding: tokenBytes[Int(id)], as: UTF8.self)
        }
    }

    // MARK: - Decoding

    /// The bytes of text one token stands for, as tokenizer.json's decoder produces them: "▁" becomes " ", a <0xXX>
    /// token is its byte, and any other token, special tokens included, is its vocabulary entry. nil for an id outside
    /// the vocabulary. One token's bytes need not be valid UTF-8 (a byte token is part of a character).
    public func pieceBytes(_ id: Int32) -> [UInt8]? {
        guard id >= 0, Int(id) < tokenBytes.count else { return nil }
        if byteValues[Int(id)] >= 0 { return [UInt8(byteValues[Int(id)])] }
        let entry = tokenBytes[Int(id)]
        // An empty entry is a gap between the ids of a sparse vocabulary, not a token.
        guard !entry.isEmpty else { return nil }
        guard entry.count >= 3 else { return entry }
        var bytes: [UInt8] = []
        bytes.reserveCapacity(entry.count)
        var index = 0
        while index < entry.count {
            if index + 2 < entry.count, entry[index] == 0xE2, entry[index + 1] == 0x96, entry[index + 2] == 0x81 {
                bytes.append(0x20)
                index += 3
            } else {
                bytes.append(entry[index])
                index += 1
            }
        }
        return bytes
    }

    /// The id of a vocabulary entry, compared byte for byte.
    public func id(of token: String) -> Int32? {
        let bytes = Array(token.utf8)
        // Linear only for the rare debug use; encoding never calls this.
        return tokenBytes.firstIndex(of: bytes).map { Int32($0) }
    }

    /// Leftmost-longest match of an added token at `index` (tokens are sorted longest first).
    @inline(__always)
    private func matchAddedToken(_ utf8: UnsafeBufferPointer<UInt8>, at index: Int, userDefinedOnly: Bool) -> (Int, Int)? {
        for (tokenIndex, token) in addedTokens.enumerated() {
            let length = token.count
            guard length > 0, index + length <= utf8.count, !userDefinedOnly || addedIsUserDefined[tokenIndex] else {
                continue
            }
            var equal = true
            for offset in 0..<length where utf8[index + offset] != token[offset] {
                equal = false
                break
            }
            if equal { return (tokenIndex, length) }
        }
        return nil
    }

    /// Normalizes (" " -> "▁"), splits into initial symbols and runs BPE on one piece.
    private func encodePiece(_ utf8: UnsafeBufferPointer<UInt8>, into output: inout [Int32],
                             scratch: inout BPEScratch, stopAt: Int)
    {
        scratch.reset()
        // For small pieces avoid growing four separate arrays during a single
        // encode. Long pieces can contain many multibyte scalars; size those on demand.
        if utf8.count <= 256 {
            scratch.ids.reserveCapacity(utf8.count)
            scratch.previous.reserveCapacity(utf8.count)
            scratch.next.reserveCapacity(utf8.count)
            scratch.heap.reserveCapacity(utf8.count)
        }
        var index = 0
        var pendingUnknown = false
        while index < utf8.count {
            let (scalar, length) = decodeScalar(utf8, at: index)
            let normalized: UInt32 = scalar == 0x20 ? 0x2581 : scalar
            let id: Int32
            if normalized < 0x10000 {
                id = bmpIDs[Int(normalized)]
            } else {
                id = astralIDs[normalized] ?? -1
            }
            if id >= 0 {
                pendingUnknown = false
                scratch.append(id)
            } else {
                // byte_fallback: <0xXX> per UTF-8 byte of the (normalized) scalar
                var encoded: [UInt8] = []
                appendUTF8(normalized, to: &encoded)
                if encoded.allSatisfy({ byteIDs[Int($0)] >= 0 }) {
                    pendingUnknown = false
                    for byte in encoded { scratch.append(byteIDs[Int(byte)]) }
                } else if let unknownID {
                    // fuse_unk: consecutive unknowns become one <unk>
                    if !pendingUnknown { scratch.append(unknownID) }
                    pendingUnknown = true
                }
            }
            index += length
        }
        scratch.mergeAll(merges)
        scratch.collect(into: &output, stopAt: stopAt)
    }
}

/// Reusable working memory for BPE: a doubly linked list of symbols and a binary min-heap of
/// candidate merges ordered by (rank, position).
public struct BPEScratch: Sendable {
    var ids: [Int32] = []
    var previous: [Int32] = []
    var next: [Int32] = []
    var heap: [HeapEntry] = []

    struct HeapEntry: Sendable {
        /// rank << 32 | position: the heap order
        var order: UInt64
        var mergedID: Int32
    }

    public init() {}

    mutating func reset() {
        ids.removeAll(keepingCapacity: true)
        previous.removeAll(keepingCapacity: true)
        next.removeAll(keepingCapacity: true)
        heap.removeAll(keepingCapacity: true)
    }

    @inline(__always)
    mutating func append(_ id: Int32) {
        let index = Int32(ids.count)
        ids.append(id)
        previous.append(index - 1)
        next.append(index + 1)
    }

    @inline(__always)
    private func pairKey(_ left: Int32, _ right: Int32) -> UInt64 {
        UInt64(UInt32(bitPattern: left)) << 32 | UInt64(UInt32(bitPattern: right))
    }

    @inline(__always)
    private mutating func push(position: Int32, merges: PairTable) {
        let right = next[Int(position)]
        guard right < Int32(ids.count), let found = merges.lookup(pairKey(ids[Int(position)], ids[Int(right)])) else {
            return
        }
        let entry = HeapEntry(order: (found >> 32) << 32 | UInt64(UInt32(position)),
                              mergedID: Int32(bitPattern: UInt32(truncatingIfNeeded: found)))
        heap.append(entry)
        var child = heap.count - 1
        while child > 0 {
            let parent = (child - 1) / 2
            if heap[parent].order <= heap[child].order { break }
            heap.swapAt(parent, child)
            child = parent
        }
    }

    private mutating func pop() -> HeapEntry? {
        guard let top = heap.first else { return nil }
        let last = heap.removeLast()
        if !heap.isEmpty {
            heap[0] = last
            var parent = 0
            while true {
                let left = 2 * parent + 1, right = left + 1
                var smallest = parent
                if left < heap.count, heap[left].order < heap[smallest].order { smallest = left }
                if right < heap.count, heap[right].order < heap[smallest].order { smallest = right }
                if smallest == parent { break }
                heap.swapAt(parent, smallest)
                parent = smallest
            }
        }
        return top
    }

    /// Applies merges until none is left (tokenizers' Word::merge_all without dropout).
    mutating func mergeAll(_ merges: PairTable) {
        let count = Int32(ids.count)
        guard count > 1 else { return }
        for position in 0..<(count - 1) { push(position: position, merges: merges) }
        while let top = pop() {
            let position = Int32(truncatingIfNeeded: top.order & 0xFFFF_FFFF)
            // Skip stale entries: the symbol was merged away, or its pair no longer produces the same
            // token. Like tokenizers, only the produced id is compared, not the rank.
            guard ids[Int(position)] >= 0 else { continue }
            let right = next[Int(position)]
            guard right < count,
                let found = merges.lookup(pairKey(ids[Int(position)], ids[Int(right)])),
                Int32(bitPattern: UInt32(truncatingIfNeeded: found)) == top.mergedID
            else { continue }
            // Merge right into position.
            ids[Int(position)] = top.mergedID
            ids[Int(right)] = -1
            let after = next[Int(right)]
            next[Int(position)] = after
            if after < count { previous[Int(after)] = position }
            let before = previous[Int(position)]
            if before >= 0 { push(position: before, merges: merges) }
            push(position: position, merges: merges)
        }
    }

    func collect(into output: inout [Int32], stopAt: Int = Int.max) {
        guard !ids.isEmpty else { return }
        var index: Int32 = 0
        while index < Int32(ids.count), output.count < stopAt {
            output.append(ids[Int(index)])
            index = next[Int(index)]
        }
    }
}

// MARK: - Helpers

@inline(__always)
func decodeScalar(_ utf8: UnsafeBufferPointer<UInt8>, at index: Int) -> (UInt32, Int) {
    let lead = UInt32(utf8[index])
    if lead < 0x80 { return (lead, 1) }
    if lead < 0xE0 { return ((lead & 0x1F) << 6 | UInt32(utf8[index + 1] & 0x3F), 2) }
    if lead < 0xF0 {
        return ((lead & 0x0F) << 12 | UInt32(utf8[index + 1] & 0x3F) << 6 | UInt32(utf8[index + 2] & 0x3F), 3)
    }
    return ((lead & 0x07) << 18 | UInt32(utf8[index + 1] & 0x3F) << 12 | UInt32(utf8[index + 2] & 0x3F) << 6
        | UInt32(utf8[index + 3] & 0x3F), 4)
}

/// The scalar value when `bytes` is exactly one well-formed UTF-8 scalar.
func singleScalar(_ bytes: [UInt8]) -> UInt32? {
    guard (1...4).contains(bytes.count) else { return nil }
    var iterator = bytes.makeIterator()
    var decoder = UTF8()
    guard case .scalarValue(let scalar) = decoder.decode(&iterator) else { return nil }
    guard case .emptyInput = decoder.decode(&iterator) else { return nil }
    return scalar.value
}
