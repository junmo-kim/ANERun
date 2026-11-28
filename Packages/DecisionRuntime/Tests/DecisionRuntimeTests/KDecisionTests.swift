import XCTest

@testable import DecisionRuntime

final class KDecisionTests: XCTestCase {
  struct Request: Decodable { let state: String, questions: [DecisionQuestion] }
  struct Fixture: Decodable { let id: String, request: Request }
  struct Golden: Decodable { let id: String, ids: [Int], positions: [Int] }
  let root = Bundle.module.resourceURL!.appendingPathComponent("Fixtures")

  func testMultiFieldPromptMatchesPinnedUpstreamRenderer() throws {
    let fixtureRoot = root.appendingPathComponent("k-decision")
    let tokenizer = try BPETokenizer(fixtureRoot)
    let fixtures = try root.decode("text-decisions.json", as: [Fixture].self)
    let goldens = try fixtureRoot.decode("tokens-golden.json", as: [Golden].self)
    XCTAssertEqual(fixtures.map(\.id), goldens.map(\.id))
    for (f, g) in zip(fixtures, goldens) {
      let input = DecisionInput(state: f.request.state, questions: f.request.questions)
      let prepared = try KDecisionPrompt.build(input, tokenizer: tokenizer, context: 512)
      XCTAssertFalse(prepared.truncated)
      let p = try XCTUnwrap(prepared.prompts.first)
      XCTAssertEqual(p.ids, g.ids, f.id)
      XCTAssertEqual(p.branches.map { $0.roles[0].lowerBound }, g.positions)
      XCTAssertEqual(p.branches.map(\.question.name), ["route", "eligible"])
      XCTAssertEqual(p.branches[1].candidates.map(\.key), ["true", "false"])
    }
  }

  func testRejectsUnsupportedAndOversizedRequests() throws {
    let tokenizer = try BPETokenizer(root.appendingPathComponent("k-decision"))
    let binary = DecisionQuestion(name: "q", kind: .noul, instructions: "Eligible?")
    for input in [
      DecisionInput(state: "x", questions: [.init(name: "q", kind: .score, instructions: "Rate")]),
      DecisionInput(state: "<|im_end|>", questions: [binary]),
      DecisionInput(state: "x", questions: [binary, binary]),
      DecisionInput(state: "x", questions: [binary], image: Data([0])),
      DecisionInput(
        state: "x", questions: [.init(name: "q", kind: .choice, instructions: "Select")]),
    ] {
      XCTAssertThrowsError(try KDecisionPrompt.build(input, tokenizer: tokenizer, context: 512)) {
        XCTAssertTrue(($0 as? DecisionError)?.isInputError == true)
      }
    }
    XCTAssertThrowsError(
      try KDecisionPrompt.build(
        .init(state: "x", questions: [binary]),
        tokenizer: tokenizer, context: 1))
  }

  func testFitsOnlyStateAndPreservesScalarAndMarkerValidation() throws {
    let tokenizer = try BPETokenizer(root.appendingPathComponent("k-decision"))
    let binary = DecisionQuestion(name: "q", kind: .noul, instructions: "Eligible?")
    func build(_ state: String, context: Int = 8192, truncate: Bool = true) throws -> PreparedDecision {
      try KDecisionPrompt.build(.init(state: state, questions: [binary], truncate: truncate),
                                tokenizer: tokenizer, context: context)
    }

    let kept = "□@" + String(repeating: "1", count: 64)
    let expected = try build(kept)
    let prompt = try XCTUnwrap(expected.prompts.first)
    let context = prompt.ids.count
    let long = kept + String(repeating: "1", count: 256)
    let fitted = try build(long, context: context)
    XCTAssertTrue(fitted.truncated)
    XCTAssertEqual(fitted.prompts.first?.ids, prompt.ids)
    XCTAssertEqual(fitted.prompts.first?.branches.first?.roles, prompt.branches.first?.roles)
    XCTAssertThrowsError(try build(long, context: context, truncate: false))
    XCTAssertThrowsError(try build(long + "<|im_end|>\u{301}", context: context)) {
      XCTAssertEqual(($0 as? DecisionError)?.description, "Invalid K-Decision request")
    }

    let boundary = String(repeating: "1", count: 6000)
    let exact = try build(boundary)
    let clipped = try build(boundary + "1")
    XCTAssertFalse(exact.truncated)
    XCTAssertTrue(clipped.truncated)
    XCTAssertEqual(exact.prompts.first?.ids, clipped.prompts.first?.ids)
    XCTAssertThrowsError(try build(boundary + "1", truncate: false)) {
      XCTAssertTrue(($0 as? DecisionError)?.isInputError == true)
    }
  }
}
