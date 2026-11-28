import Foundation

/// OneJev renders each question as an independent branch of the same state.
struct ElephantPrompt {
  let tokenizer: BPETokenizer
  let context: Int
  let labels: [String]

  func build(_ input: DecisionInput, visualTokens: Int) throws -> PreparedDecision {
    func checked(_ text: String) throws -> String {
      guard !tokenizer.special.keys.contains(where: { text.range(of: $0, options: .literal) != nil }) else {
        throw DecisionError("Reserved control token in request", input: true)
      }
      return text.precomposedStringWithCanonicalMapping
    }
    guard !input.questions.isEmpty,
      Set(input.questions.map(\.name)).count == input.questions.count
    else { throw DecisionError("Invalid question names", input: true) }
    let state = try checked(input.state)
    let image = visualTokens > 0
      ? "<|vision_start|>" + String(repeating: "<|image_pad|>", count: visualTokens)
        + "<|vision_end|>\n" : ""
    let head = "<|im_start|>system\nApply the question to the state. "
      + "Choose exactly one of the listed options. Respond with only its uppercase letter, "
      + "with no explanation or reasoning.<|im_end|>\n<|im_start|>user\n<state>\n"
      + image
    let layouts = try input.questions.map { question -> (suffix: [Int], branch: Branch) in
      var candidates = question.candidates
      let fallback: String
      switch question.kind {
      case .noul:
        guard candidates.isEmpty else { throw DecisionError("Unexpected predicate candidates", input: true) }
        candidates = [.init(key: "yes", description: "the statement is true / the answer is yes"),
                      .init(key: "no", description: "the statement is false / the answer is no")]
        fallback = "Is the statement true, or is the answer to the question yes?"
      case .choice:
        fallback = "Which option applies to the state?"
      case .score:
        guard candidates.count <= 10, candidates.enumerated().allSatisfy({ String($0.offset) == $0.element.key })
        else { throw DecisionError("Invalid score levels", input: true) }
        fallback = "Which level describes the state?"
      }
      guard !question.name.isEmpty, (1...255).contains(candidates.count),
        candidates.count <= labels.count, Set(candidates.map(\.key)).count == candidates.count
      else { throw DecisionError("Invalid candidates", input: true) }
      let instruction = try checked(question.instructions).trimmingCharacters(in: .whitespacesAndNewlines)
      let options = try zip(labels, candidates).map { label, candidate in
        let key = try checked(candidate.key)
        guard !key.isEmpty else { throw DecisionError("Empty candidate key", input: true) }
        let name = question.kind == .score ? "level " + key : key
        let description = try checked(candidate.description).trimmingCharacters(in: .whitespacesAndNewlines)
        return label + ". " + name + (description.isEmpty ? "" : ": " + description)
      }
      let unit = candidates.count <= 26 ? "letter" : "label"
      let suffix = (question.kind == .score ? "Rate the state: " : "Question: ")
        + (instruction.isEmpty ? fallback : instruction) + "\n\nOptions:\n"
        + options.joined(separator: "\n") + "\n\nAnswer with one " + unit + ": "
        + labels.prefix(candidates.count).joined(separator: ", ") + ".<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
      return (tokenizer.encode(suffix).map(\.id), Branch(question: question, candidates: candidates, roles: []))
    }
    return try PreparedDecision.fitting(state, truncate: input.truncate) { state in
      // A newline ends the prefix and a letter starts every suffix, a BPE split boundary.
      let prefix = tokenizer.encode(head + state + "\n</state>\n\n").map(\.id)
      guard prefix.count < context else { throw PromptBudgetExceeded(message: "State exceeds context") }
      var requestTokens = prefix.count
      for row in layouts {
        guard prefix.count + row.suffix.count <= context else {
          throw PromptBudgetExceeded(message: "Question exceeds context")
        }
        requestTokens += row.suffix.count
        guard requestTokens <= 65536 else { throw PromptBudgetExceeded(message: "Request exceeds token limit") }
      }
      return layouts.map { row in
        let ids = prefix + row.suffix
        return Prompt(ids: ids, prefix: prefix.count, segments: Array(repeating: 1, count: ids.count),
                      branches: [row.branch])
      }
    }
  }
}

struct ElephantLabels: Decodable { let codes: [String] }
