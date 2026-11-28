import Foundation

struct Token {
  let id: Int
  let range: Range<Int>  // UTF-8 bytes in the NFC input
}

struct BPETokenizer {
  private struct Info: Decodable {
    let special: [String: Int]
    let pattern: String
  }
  private struct Pair: Hashable {
    let left: Int
    let right: Int
  }
  private struct Merge {
    let rank: Int
    let id: Int
  }
  private let bytes: [Int], merges: [Pair: Merge], pattern: NSRegularExpression
  private let maxTokenBytes: Int
  let special: [String: Int]
  private let controls: NSRegularExpression

  init(_ root: URL) throws {
    let info = try root.decode("tokens.json", as: Info.self)
    special = info.special
    pattern = try NSRegularExpression(pattern: info.pattern)
    controls = try NSRegularExpression(
      pattern: info.special.keys.sorted { $0.count > $1.count }
        .map(NSRegularExpression.escapedPattern(for:)).joined(separator: "|"))
    let data = try Data(contentsOf: root.appendingPathComponent("bpe.bin"), options: .mappedIfSafe)
    guard data.count >= 1024, (data.count - 1024) % 12 == 0, !info.special.isEmpty else {
      throw DecisionError("Malformed BPE table")
    }
    let words = data.withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
    bytes = words.prefix(256).map(Int.init)
    var widths = Dictionary(bytes.map { ($0, 1) }, uniquingKeysWith: max)
    var maxWidth = 1
    var rules = [Pair: Merge]()
    rules.reserveCapacity((words.count - 256) / 3)
    for i in stride(from: 256, to: words.count, by: 3) {
      let pair = Pair(left: Int(words[i]), right: Int(words[i + 1]))
      let id = Int(words[i + 2])
      rules[pair] = Merge(rank: i, id: id)
      if maxWidth != .max, let left = widths[pair.left], let right = widths[pair.right] {
        let (width, overflow) = left.addingReportingOverflow(right)
        if !overflow, widths[id, default: width] == width {
          widths[id] = width
          maxWidth = max(maxWidth, width)
          continue
        }
      }
      // Unordered or ambiguous tables can still use the exact token count.
      maxWidth = .max
    }
    merges = rules
    maxTokenBytes = maxWidth
  }

  private func piece(_ piece: String, offset: Int) -> [Token] {
    var tokens = piece.utf8.enumerated().map {
      Token(id: bytes[Int($0.element)], range: (offset + $0.offset)..<(offset + $0.offset + 1))
    }
    while tokens.count > 1 {
      var best: (Int, Merge)?
      for i in 1..<tokens.count {
        if let m = merges[Pair(left: tokens[i - 1].id, right: tokens[i].id)],
          m.rank < best?.1.rank ?? Int.max
        {
          best = (i, m)
        }
      }
      guard let (i, m) = best else { break }
      tokens.replaceSubrange(
        (i - 1)...i,
        with: [Token(id: m.id, range: tokens[i - 1].range.lowerBound..<tokens[i].range.upperBound)])
    }
    return tokens
  }

  func encode(_ input: String) -> [Token] { encode(input, upTo: .max)! }

  /// Returns nil as soon as this exact encoding is known to exceed the limit.
  func encode(_ input: String, upTo limit: Int) -> [Token]? {
    guard limit >= 0 else { return nil }
    let normalized = input.precomposedStringWithCanonicalMapping
    let string = normalized as NSString
    var tokens = [Token]()
    var offset = 0
    var cursor = 0
    func ordinary(_ range: NSRange) -> Bool {
      var fits = true
      pattern.enumerateMatches(in: normalized, range: range) { match, _, stop in
        guard let m = match else { return }
        let part = string.substring(with: m.range)
        let byteCount = part.utf8.count
        guard byteCount > 0 else { return }
        let remaining = limit - tokens.count
        if (byteCount - 1) / maxTokenBytes < remaining {
          let encoded = piece(part, offset: offset)
          if encoded.count <= remaining {
            tokens += encoded
            offset += byteCount
            return
          }
        }
        fits = false
        stop.pointee = true
      }
      return fits
    }
    for hit in controls.matches(
      in: normalized, range: NSRange(location: 0, length: string.length))
    {
      guard ordinary(NSRange(location: cursor, length: hit.range.location - cursor)),
        tokens.count < limit
      else { return nil }
      let key = string.substring(with: hit.range)
      let end = offset + key.utf8.count
      tokens.append(Token(id: special[key]!, range: offset..<end))
      offset = end
      cursor = NSMaxRange(hit.range)
    }
    return ordinary(NSRange(location: cursor, length: string.length - cursor)) ? tokens : nil
  }
}
