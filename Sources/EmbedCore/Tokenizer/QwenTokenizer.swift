import Foundation

/// The NFC → Split → ByteLevel BPE tokenizer of pplx-embed-v1-0.6b and Qwen3-Embedding-0.6B.
public final class QwenTokenizer: TextTokenizer {
    private enum Span {
        case ordinary(String)
        case added(Int32)
    }

    private let splitter: NSRegularExpression
    private let byteIDs: [Int32]
    private let merges: PairTable
    private let addedByFirstByte: [[AddedToken]]
    private let addedBytes: [Int32: [UInt8]]
    private let vocabBytes: [[UInt8]]
    private let maxOrdinaryBytes: Int
    public let suffixIDs: [Int32]

    func checked(for model: ModelID) throws -> Self {
        guard suffixIDs == (model == .qwen3Embedding ? [151643] : []) else {
            throw TokenizerError.unsupported("tokenizer post-processor does not match " + model.rawValue)
        }
        return self
    }

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
        let parsed = try data.withUnsafeBytes { try TokenizerFile.parse($0, format: .qwen) }
        suffixIDs = parsed.suffixIDs
        // ICU allows nested sets, including inside a complemented character class.
        let pattern = TokenizerFile.qwenSplitPattern
            .replacingOccurrences(of: #"\p{L}"#, with: "[\(HFClasses.letter)]")
            .replacingOccurrences(of: #"\p{N}"#, with: "[\(HFClasses.number)]")
            .replacingOccurrences(of: #"\s"#, with: "[\(HFClasses.space)]")
            .replacingOccurrences(of: #"\S"#, with: "[^\(HFClasses.space)]")
        splitter = try NSRegularExpression(pattern: pattern)

        var groups = [[AddedToken]](repeating: [], count: 256)
        var addedPieces: [Int32: [UInt8]] = [:]
        var addedContents: Set<[UInt8]> = []
        for token in parsed.addedTokens.sorted(by: { $0.content.count > $1.content.count }) {
            guard addedPieces[token.id] == nil, addedContents.insert(token.content).inserted else {
                throw TokenizerError.malformed("duplicate Qwen added token")
            }
            groups[Int(token.content[0])].append(token)
            addedPieces[token.id] = token.content
        }
        addedByFirstByte = groups
        addedBytes = addedPieces

        // GPT-2 bytes_to_unicode keeps printable bytes and assigns the rest consecutive
        // scalars from U+0100 in byte order. The inverse maps vocabulary pieces back to bytes.
        let vocab = parsed.vocab
        var inverse = [Int16](repeating: -1, count: 512)
        var extra = 256
        byteIDs = try (0..<256).map { byte in
            let scalar: Int
            switch byte {
            case 33...126, 161...172, 174...255: scalar = byte
            default:
                scalar = extra
                extra += 1
            }
            inverse[scalar] = Int16(byte)
            guard let id = vocab.lookup(Array(String(Unicode.Scalar(scalar)!).utf8)) else {
                throw TokenizerError.malformed("missing alphabet byte \(byte)")
            }
            return id
        }
        let maxID = (0..<vocab.count).map { Int(vocab.value(at: $0)) }.max() ?? 0
        guard maxID < 16 * vocab.count + 1024 else { throw TokenizerError.malformed("vocab ids are too sparse") }
        var bytesByID = [[UInt8]](repeating: [], count: maxID + 1)
        var maxBytes = 0
        for entry in 0..<vocab.count {
            let id = Int(vocab.value(at: entry))
            guard bytesByID[id].isEmpty else { throw TokenizerError.malformed("duplicate vocab id \(id)") }
            let key = vocab.key(at: entry)
            guard !key.isEmpty else { throw TokenizerError.malformed("empty vocab entry") }
            var bytes: [UInt8] = []
            for scalar in String(decoding: key, as: UTF8.self).unicodeScalars {
                let value = Int(scalar.value)
                guard value < inverse.count, inverse[value] >= 0 else {
                    throw TokenizerError.malformed("non-ByteLevel vocab entry at id \(id)")
                }
                bytes.append(UInt8(inverse[value]))
            }
            bytesByID[id] = bytes
            maxBytes = max(maxBytes, bytes.count)
        }
        vocabBytes = bytesByID
        maxOrdinaryBytes = maxBytes

        var table = PairTable(minimumCapacity: parsed.mergeCount)
        var joined: [UInt8] = []
        try parsed.mergeStorage.withUnsafeBytes { raw in
            for rank in 0..<parsed.mergeCount {
                let left = UnsafeRawBufferPointer(rebasing: raw[parsed.mergeRange(2 * rank)])
                let right = UnsafeRawBufferPointer(rebasing: raw[parsed.mergeRange(2 * rank + 1)])
                joined.removeAll(keepingCapacity: true)
                joined.append(contentsOf: left)
                joined.append(contentsOf: right)
                guard let l = vocab.lookup(left), let r = vocab.lookup(right), let id = vocab.lookup(joined) else {
                    throw TokenizerError.malformed("missing vocab entry at merge \(rank)")
                }
                table.insert(UInt64(l) << 32 | UInt64(r), UInt64(rank) << 32 | UInt64(id))
            }
        }
        merges = table
    }

    public func encode(_ text: String) -> [Int32] {
        encode(text, addSpecial: true, parseSpecial: true)
    }

    public func encode(_ text: String, addSpecial: Bool, parseSpecial: Bool) -> [Int32] {
        var scratch = BPEScratch()
        var ids: [Int32] = []
        forEachSpan(in: text, parseSpecial: parseSpecial) { span in
            append(span, to: &ids, scratch: &scratch)
            return true
        }
        if addSpecial { ids += suffixIDs }
        return ids
    }

    public func pieceBytes(_ id: Int32) -> [UInt8]? {
        // The pinned model has one added ID that is also a ByteLevel vocabulary entry.
        if id >= 0, Int(id) < vocabBytes.count, !vocabBytes[Int(id)].isEmpty {
            return vocabBytes[Int(id)]
        }
        return addedBytes[id]
    }

    public func encodeForEmbedding(_ text: String, limit: Int) -> BoundedTokenization {
        encodeForEmbedding(text, limit: limit, truncate: false)
    }

    public func encodeForEmbedding(_ text: String, limit: Int, truncate: Bool) -> BoundedTokenization {
        guard limit >= suffixIDs.count else { return .tooLong(atLeast: suffixIDs.count) }
        let bodyLimit = limit - suffixIDs.count
        let stopAt = bodyLimit == Int.max ? Int.max : bodyLimit + 1
        var scratch = BPEScratch()
        var ids: [Int32] = []
        var exceedsLimit = false
        forEachSpan(in: text, parseSpecial: true) { span in
            if !truncate, case .ordinary(let normalized) = span {
                let remaining = bodyLimit - ids.count
                let (whole, tail) = normalized.utf8.count.quotientAndRemainder(dividingBy: maxOrdinaryBytes)
                if whole > remaining || (whole == remaining && tail > 0) {
                    exceedsLimit = true
                    return false
                }
            }
            append(span, to: &ids, scratch: &scratch, stopAt: stopAt)
            exceedsLimit = ids.count > bodyLimit
            return !exceedsLimit
        }
        guard exceedsLimit else { return .ids(ids + suffixIDs) }
        guard truncate, limit > 0 else { return .tooLong(atLeast: limit + 1) }
        ids.removeLast()
        return .clipped(ids + suffixIDs)
    }

    /// Visits spans until the input ends or the consumer returns false.
    private func forEachSpan(in text: String, parseSpecial: Bool, _ consume: (Span) -> Bool) {
        var input = text
        input.withUTF8 { bytes in
            func ordinary(_ range: Range<Int>) -> Bool {
                guard !range.isEmpty else { return true }
                let raw = String(decoding: bytes[range], as: UTF8.self)
                if bytes[range].allSatisfy({ $0 < 128 }) {
                    return consume(.ordinary(raw))
                }
                // HF unicode-normalization-alignments 0.1.12 uses Unicode 9. Later
                // scalars are inert boundaries, even when newer NFC would compose them.
                guard let normalized = raw.applyingTransform(StringTransform("[:age=9.0:] NFC"), reverse: false) else {
                    preconditionFailure("ICU NFC transform unavailable")
                }
                return consume(.ordinary(normalized))
            }
            var start = 0
            var index = 0
            while index < bytes.count {
                let candidates = addedByFirstByte[Int(bytes[index])]
                guard let token = candidates.first(where: { bytes[index...].starts(with: $0.content) }) else {
                    index += 1
                    continue
                }
                let end = index + token.content.count
                // HF selects the longest raw match before deciding whether to emit it.
                if parseSpecial || !token.special {
                    guard ordinary(start..<index), consume(.added(token.id)) else { return }
                    start = end
                }
                index = end
            }
            _ = ordinary(start..<bytes.count)
        }
    }

    private func append(_ span: Span, to ids: inout [Int32], scratch: inout BPEScratch, stopAt: Int = Int.max) {
        switch span {
        case .added(let id):
            ids.append(id)
        case .ordinary(let text):
            visitSplits(in: text) { piece in
                scratch.reset()
                for byte in piece.utf8 { scratch.append(byteIDs[Int(byte)]) }
                scratch.mergeAll(merges)
                scratch.collect(into: &ids, stopAt: stopAt)
                return ids.count < stopAt
            }
        }
    }

    func forEachSplit(in text: String, _ consume: (String) -> Void) {
        visitSplits(in: text) { piece in
            consume(piece)
            return true
        }
    }

    private func visitSplits(in text: String, _ consume: (String) -> Bool) {
        // NSString slices can bisect a grapheme at the regex's scalar boundary.
        let source = text as NSString
        var offset = 0
        var stopped = false
        withoutActuallyEscaping(consume) { consume in
            splitter.enumerateMatches(in: text, range: NSRange(location: 0, length: source.length)) { match, _, stop in
                guard let match else { return }
                precondition(match.range.location == offset && match.range.length > 0)
                offset = NSMaxRange(match.range)
                if !consume(source.substring(with: match.range)) {
                    stopped = true
                    stop.pointee = true
                }
            }
        }
        precondition(stopped || offset == source.length)
    }
}
