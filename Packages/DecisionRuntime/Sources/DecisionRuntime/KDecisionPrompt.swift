import Foundation

/// The causal model predicts the symbol immediately before each square in one JSON skeleton.
enum KDecisionPrompt {
  static let alphabet = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZ").map(String.init)
  static let instruction =
    "당신은 한국어 의사결정 엔진이다. [상태]와 [질문]들을 읽고, 각 질문의 정답 심볼을 예측하라. 질문 유형: choice(단일선택), noul(예/아니오). 설명 없이 각 필드의 심볼을 결정하며, 모든 필드는 한 번에 판단한다."

  static func build(_ request: DecisionInput, tokenizer: BPETokenizer, context: Int) throws
    -> PreparedDecision
  {
    func validText(_ text: String) -> Bool {
      !tokenizer.special.keys.contains(where: { text.range(of: $0, options: .literal) != nil })
    }
    let state = request.state.replacingOccurrences(of: "□", with: "▪")
      .replacingOccurrences(of: "@", with: "＠")
    guard request.image == nil, validText(request.state), validText(state),
      !request.questions.isEmpty,
      Set(request.questions.map(\.name)).count == request.questions.count
    else { throw DecisionError("Invalid K-Decision request", input: true) }

    let head = "<|im_start|>user\n" + [instruction, "", "[상태]", ""].joined(separator: "\n")
    var lines = ["", "", "[질문]"]
    var options = [[DecisionCandidate]]()
    for q in request.questions {
      let id = q.name
      guard !id.isEmpty, validText(id), validText(q.instructions),
        !id.contains(where: { "\"\\□".contains($0) || $0.isNewline }),
        !q.instructions.contains("□")
      else { throw DecisionError("Invalid K-Decision question", input: true) }
      let display: String
      switch q.kind {
      case .choice:
        guard (2...alphabet.count).contains(q.candidates.count),
          Set(q.candidates.map(\.key)).count == q.candidates.count,
          q.candidates.allSatisfy({ !$0.key.isEmpty })
        else { throw DecisionError("K-Decision requires 2 to 26 distinct choices", input: true) }
        let labels = q.candidates.map { $0.description.isEmpty ? $0.key : $0.description }
        guard labels.allSatisfy({ validText($0) && !$0.contains("□") }) else {
          throw DecisionError("Reserved token in K-Decision option", input: true)
        }
        display = zip(alphabet, labels).map { "\($0)=\($1)" }.joined(separator: ", ")
        options.append(q.candidates)
      case .noul:
        guard q.candidates.isEmpty else {
          throw DecisionError("Unexpected predicate options", input: true)
        }
        display = "Y=예, N=아니오"
        options.append([
          .init(key: "true", description: "예"), .init(key: "false", description: "아니오"),
        ])
      case .score:
        throw DecisionError("K-Decision supports choice and noul", input: true)
      }
      lines.append("- \(id) (\(q.kind.rawValue)) \(q.instructions) 옵션: \(display)")
    }
    lines += ["", "[결정] 각 필드 심볼을 JSON으로 답하라."]
    let skeleton = request.questions.map { "\"\($0.name)\":□" }.joined(separator: ",")
    let tail =
      lines.joined(separator: "\n")
      + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{" + skeleton + "}<|im_end|>\n"
    return try PreparedDecision.fitting(state, truncate: request.truncate) { state in
      guard state.unicodeScalars.count <= 6000 else {
        throw PromptBudgetExceeded(message: "K-Decision state exceeds 6000 Unicode scalars")
      }
      let ids = tokenizer.encode(head + state + tail).map(\.id)
      let positions = ids.indices.filter { ids[$0] == 169260 }
      guard positions.count == options.count, positions.first != 0 else {
        throw DecisionError("K-Decision context or marker mismatch", input: true)
      }
      guard ids.count <= context else {
        throw PromptBudgetExceeded(message: "K-Decision context or marker mismatch")
      }
      let branches = positions.enumerated().map { i, marker in
        Branch(question: request.questions[i], candidates: options[i], roles: [(marker - 1)..<marker])
      }
      return [Prompt(ids: ids, prefix: 0, segments: [], branches: branches)]
    }
  }
}
