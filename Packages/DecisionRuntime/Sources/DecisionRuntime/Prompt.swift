import Foundation

struct Branch {
  let question: DecisionQuestion, candidates: [DecisionCandidate]
  let roles: [Range<Int>]
}
struct Prompt {
  let ids: [Int], prefix: Int, segments: [Int], branches: [Branch]
}

struct PromptBudgetExceeded: Error {
  var message = "Prompt exceeds bundle context"
  var inputError: DecisionError { DecisionError(message, input: true) }
}

struct PreparedDecision {
  let prompts: [Prompt]
  let truncated: Bool

  func tokenCounts(sharingPrefix: Bool) -> (input: Int, compute: Int) {
    let compute = prompts.reduce(0) { $0 + $1.ids.count }
    guard sharingPrefix else { return (compute, compute) }
    let repeated = prompts.dropFirst().reduce(0) { $0 + $1.prefix }
    return (compute - repeated, compute)
  }

  static func fitting(
    _ state: String, truncate: Bool, render: (String) throws -> [Prompt]
  ) throws -> Self {
    do {
      return try Self(prompts: render(state), truncated: false)
    } catch let error as PromptBudgetExceeded {
      guard truncate, !state.isEmpty else { throw error.inputError }
    }

    var best: [Prompt]
    do {
      best = try render("")
    } catch let error as PromptBudgetExceeded {
      throw error.inputError
    }

    // Keep whole characters and re-encode complete prompts at each cut. BPE
    // can merge across template boundaries, so standalone state counts aren't sufficient.
    var lower = state.startIndex
    var distance = state.count
    while distance > 1 {
      let step = distance / 2
      let end = state.index(lower, offsetBy: step)
      do {
        best = try render(String(state[..<end]))
        lower = end
        distance -= step
      } catch is PromptBudgetExceeded {
        distance = step
      }
    }
    return Self(prompts: best, truncated: true)
  }
}

struct PromptBuilder {
  let tokenizer: BPETokenizer, valen: Bool, codes: [String], context: Int

  private func checked(_ text: String) throws -> String {
    let normalized = text.precomposedStringWithCanonicalMapping
    guard !tokenizer.special.keys.contains(where: { normalized.range(of: $0, options: .literal) != nil }) else {
      throw DecisionError("Reserved control token in request", input: true)
    }
    return normalized
  }

  func build(_ request: DecisionInput, visualTokens: Int) throws -> PreparedDecision {
    guard !request.questions.isEmpty,
      Set(request.questions.map(\.name)).count == request.questions.count,
      valen || visualTokens == 0
    else { throw DecisionError("Invalid questions or unsupported image", input: true) }
    let state = try checked(request.state)
    var compiled = [(DecisionQuestion, [DecisionCandidate])]()
    for q in request.questions {
      guard !q.name.isEmpty, !q.instructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      else { throw DecisionError("Missing question name or instructions", input: true) }
      _ = try checked(q.instructions)
      let candidates: [DecisionCandidate]
      if q.kind == .noul {
        guard q.candidates.isEmpty else { throw DecisionError("Unexpected predicate candidates", input: true) }
        candidates = [
          .init(key: "true", description: valen ? "True / 是：满足问题中的条件。" : "Yes"),
          .init(key: "false", description: valen ? "False / 否：不满足问题中的条件。" : "No"),
        ]
      } else {
        candidates = q.candidates
      }
      guard (1...255).contains(candidates.count),
        Set(candidates.map(\.key)).count == candidates.count,
        q.kind != .score
          || (candidates.count >= 2 && candidates.map(\.key) == candidates.indices.map(String.init))
      else { throw DecisionError("Invalid ordered candidates", input: true) }
      for c in candidates {
        guard !c.key.isEmpty, !c.description.isEmpty else { throw DecisionError("Empty candidate", input: true) }
        _ = try checked(c.key)
        _ = try checked(c.description)
      }
      compiled.append((q, candidates))
    }
    if !valen {
      let layouts = try compiled.map { q, candidates in
        guard candidates.count <= codes.count else { throw DecisionError("Missing option codes", input: true) }
        let lines = zip(codes, candidates).map { code, c in
          "(\(code)) \(c.key)" + (c.description == c.key ? "" : ": \(c.description)")
        }
        let suffix =
          "\n\nQuestion: " + q.instructions
          + "\n\nOptions:\n" + lines.joined(separator: "\n")
          + "\n\nThink it through briefly, then answer with the option letter only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: ("
        return (suffix, Branch(question: q, candidates: candidates, roles: []))
      }
      return try PreparedDecision.fitting(state, truncate: request.truncate) { state in
        try layouts.map { suffix, branch in
          let text = "<|im_start|>user\nContext:\n" + state + suffix
          guard let tokens = tokenizer.encode(text, upTo: context - 8) else {
            throw PromptBudgetExceeded()
          }
          return Prompt(ids: tokens.map(\.id), prefix: 0, segments: [], branches: [branch])
        }
      }
    }
    let visual =
      visualTokens == 0
      ? ""
      : "<|vision_start|>" + String(repeating: "<|image_pad|>", count: visualTokens)
        + "<|vision_end|>"
    return try PreparedDecision.fitting(state, truncate: request.truncate) { state in
      [try valenPrompt(state: state, visual: visual, compiled: compiled)]
    }
  }

  private func valenPrompt(
    state: String, visual: String, compiled: [(DecisionQuestion, [DecisionCandidate])]
  ) throws -> Prompt {
    let content = (visual + state).trimmingCharacters(in: .whitespacesAndNewlines)
    guard let prefixTokens = tokenizer.encode("<|im_start|>user\n" + content + "<|im_end|>\n",
                                              upTo: context) else {
      throw PromptBudgetExceeded()
    }
    var ids = prefixTokens.map(\.id)
    let prefix = ids.count
    var segments = Array(repeating: -1, count: prefix)
    var branchRoles = [[Range<Int>]]()
    @discardableResult func append(_ text: String, block: Int = 0, role: Range<Int>? = nil) throws
      -> Range<Int>
    {
      let start = ids.count
      guard let tokens = tokenizer.encode(text, upTo: context - start) else {
        throw PromptBudgetExceeded()
      }
      ids += tokens.map(\.id)
      // In bbox_blocks mode all non-localization questions share one attention block.
      segments += Array(repeating: block == 0 ? 0 : 1, count: tokens.count)
      guard let role else { return start..<ids.count }
      guard let first = tokens.firstIndex(where: { $0.range.overlaps(role) }),
        let last = tokens.lastIndex(where: { $0.range.overlaps(role) })
      else { throw DecisionError("Empty role span", input: true) }
      return (start + first)..<(start + last + 1)
    }
    try append("<|im_start|>user\n")
    for (i, entry) in compiled.enumerated() {
      let (q, candidates) = entry
      let block = i + 1
      let instruction = try checked(q.instructions)
      let head = "Question \(block)\nTask: \(q.kind.rawValue)\nQuestion: "
      var roles = [
        0..<prefix,
        try append(
          head + instruction + "\nCandidates:\n", block: block,
          role: head.utf8.count..<(head.utf8.count + instruction.utf8.count)),
      ]
      for c in candidates {
        let key = q.kind == .score ? "" : try checked(c.key) + ": "
        let description = try checked(c.description)
        roles.append(
          try append(
            key + description, block: block,
            role: key.utf8.count..<(key.utf8.count + description.utf8.count)))
        try append("\n", block: block)
      }
      branchRoles.append(roles)
    }
    try append("<|im_end|>\n<|im_start|>assistant\n")
    var branches = [Branch]()
    for (i, entry) in compiled.enumerated() {
      try append("Question \(i+1) Decision:", block: i + 1)
      branches.append(
        Branch(
          question: entry.0, candidates: entry.1,
          roles: branchRoles[i] + [(ids.count - 1)..<ids.count]))
      try append("\n", block: i + 1)
    }
    return Prompt(ids: ids, prefix: prefix, segments: segments, branches: branches)
  }
}
