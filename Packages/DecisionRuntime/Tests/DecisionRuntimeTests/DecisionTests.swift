import XCTest

@testable import DecisionRuntime

final class DecisionTests: XCTestCase {
  struct Fixture: Decodable {
    let id: String, model: String, state: String, type: DecisionQuestion.Kind, instructions: String
    let criteria: [[String]], name: String, ids: [Int], roles: [[Int]], result: DecisionResponse
    var request: DecisionInput {
      DecisionInput(
        state: state,
        questions: [
          .init(
            name: name, kind: type, instructions: instructions,
            candidates: criteria.map { .init(key: $0[0], description: $0[1]) })
        ])
    }
  }
  var root: URL { Bundle.module.resourceURL!.appendingPathComponent("Fixtures") }
  func testShortCompilerSymbolsResolveOnlyForVisionTiers() {
    XCTAssertEqual(Manifest.logicalFunction("p1024_35aaefab"), "p1024")
    XCTAssertEqual(Manifest.logicalFunction("p256_c8bc9cf4"), "p256")
    for name in ["p256_1234567", "p1024_1234567g", "scan5_35aaefab"] {
      XCTAssertEqual(Manifest.logicalFunction(name), name)
    }
  }
  func testPinnedTokenIDsAndMixerReadouts() throws {
    let tokenizer = try BPETokenizer(root)
    let fixtures = try root.decode("goldens.json", as: [Fixture].self)
    XCTAssertEqual(fixtures.count, 19)
    for c in fixtures {
      let compiler = PromptBuilder(
        tokenizer: tokenizer, valen: c.model == "valen-4b", codes: ["A", "B", "C"], context: 4096)
      let prepared = try compiler.build(c.request, visualTokens: c.id == "image-red-circle" ? 64 : 0)
      XCTAssertFalse(prepared.truncated, c.id)
      let prompt = try XCTUnwrap(prepared.prompts.first)
      XCTAssertEqual(prompt.ids, c.ids, c.id)
      XCTAssertEqual(
        prompt.branches[0].roles.map { [$0.lowerBound, $0.upperBound] }, c.roles, c.id)
      XCTAssertEqual(c.result.usage.inputTokens, c.ids.count)
    }
  }
  func testContextAndControlValidation() throws {
    let t = try BPETokenizer(root)
    let q = DecisionQuestion(name: "q", kind: .noul, instructions: "Is it true?")
    let builder = PromptBuilder(tokenizer: t, valen: true, codes: [], context: 128)
    for input in [
      DecisionInput(state: "<|im_start|>", questions: [q]),
      DecisionInput(state: "<|im_start|>\u{301}", questions: [q]),
      DecisionInput(state: "text", questions: [q, q]),
      DecisionInput(
        state: String(repeating: "text ", count: 200), questions: [q], truncate: false),
      DecisionInput(
        state: String(repeating: "text ", count: 200) + "<|im_start|>", questions: [q]),
    ] {
      XCTAssertThrowsError(try builder.build(input, visualTokens: 0))
    }
  }
  func testValenSharedQuestionsAndChatWhitespace() throws {
    let tokenizer = try BPETokenizer(root)
    let compiler = PromptBuilder(tokenizer: tokenizer, valen: true, codes: [], context: 4096)
    let questions = ["first", "second"].map {
      DecisionQuestion(name: $0, kind: .noul, instructions: "Is it true?")
    }
    let prompt = try XCTUnwrap(
      compiler.build(.init(state: "text", questions: questions), visualTokens: 0).prompts.first)
    let padded = try XCTUnwrap(
      compiler.build(.init(state: " text  \n", questions: questions), visualTokens: 0).prompts.first)
    XCTAssertEqual(prompt.ids, padded.ids)
    XCTAssertEqual(Set(prompt.segments), [-1, 0, 1])
    XCTAssertEqual(prompt.branches.map(\.question.name), ["first", "second"])
  }
  func testValenTruncatesWholeCharactersAndKeepsImageAndBranches() throws {
    let t = try BPETokenizer(root)
    let questions = [
      DecisionQuestion(name: "first", kind: .noul, instructions: "Is it true?"),
      DecisionQuestion(name: "second", kind: .score, instructions: "How much?", candidates: [
        .init(key: "0", description: "Low"), .init(key: "1", description: "High"),
      ]),
    ]
    let kept = String(repeating: "👩🏽‍💻", count: 3)
    let reference = PromptBuilder(tokenizer: t, valen: true, codes: [], context: 4096)
    let expected = try XCTUnwrap(reference.build(.init(state: kept, questions: questions),
                                                 visualTokens: 64).prompts.first)
    let builder = PromptBuilder(tokenizer: t, valen: true, codes: [], context: expected.ids.count)
    let state = kept + String(repeating: "👩🏽‍💻", count: 200)
    let fitted = try builder.build(.init(state: state, questions: questions), visualTokens: 64)
    XCTAssertTrue(fitted.truncated)
    let prompt = try XCTUnwrap(fitted.prompts.first)
    XCTAssertEqual(prompt.ids, expected.ids)
    XCTAssertEqual(prompt.ids.filter { $0 == t.special["<|image_pad|>"] }.count, 64)
    XCTAssertEqual(prompt.segments, expected.segments)
    XCTAssertEqual(prompt.branches.map(\.roles), expected.branches.map(\.roles))
    XCTAssertEqual(prompt.branches.map { $0.question.name }, ["first", "second"])
    XCTAssertEqual(prompt.branches[1].candidates.map(\.key), ["0", "1"])
    XCTAssertThrowsError(
      try builder.build(.init(state: state, questions: questions), visualTokens: 200)
    )
  }
  func testGlideUsesOneStatePrefixForAllQuestionsAndReservesEightTokens() throws {
    let t = try BPETokenizer(root)
    let questions = [
      DecisionQuestion(name: "short", kind: .noul, instructions: "True?"),
      DecisionQuestion(name: "long", kind: .choice,
                       instructions: "What is the answer to the question in the state?", candidates: [
        .init(key: "allow", description: "Allow"), .init(key: "deny", description: "Deny"),
      ]),
    ]
    let reference = PromptBuilder(tokenizer: t, valen: false, codes: ["A", "B"], context: 4096)
    let kept = String(repeating: "한", count: 4)
    let expected = try reference.build(.init(state: kept, questions: questions), visualTokens: 0)
    let context = try XCTUnwrap(expected.prompts.map { $0.ids.count }.max()) + 8
    let builder = PromptBuilder(tokenizer: t, valen: false, codes: ["A", "B"], context: context)
    let state = kept + String(repeating: "한", count: 400)
    let fitted = try builder.build(.init(state: state, questions: questions), visualTokens: 0)
    XCTAssertTrue(fitted.truncated)
    XCTAssertEqual(fitted.prompts.map(\.ids), expected.prompts.map(\.ids))
    XCTAssertEqual(fitted.prompts[1].ids.count + 8, context)
    XCTAssertEqual(fitted.prompts[1].branches[0].candidates.map(\.key), ["allow", "deny"])
    XCTAssertThrowsError(try builder.build(
      .init(state: state, questions: questions, truncate: false), visualTokens: 0))
    let oversized = DecisionQuestion(name: "oversized", kind: .noul, instructions: state)
    XCTAssertThrowsError(try builder.build(.init(state: "x", questions: [oversized]), visualTokens: 0))
  }
  func testUsageEncodesTruncationOnlyWhenApplied() throws {
    for truncated in [true, false] {
      let usage = DecisionResponse.Usage(inputTokens: 77, outputTokens: 0, truncated: truncated,
                                         promptTokensPerSecond: 135.5)
      let encoded = try JSONEncoder().encode(usage)
      let fields = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
      XCTAssertEqual(fields["prompt_tokens_per_second"] as? Double, 135.5)
      XCTAssertNil(fields["generation_tokens_per_second"])
      XCTAssertEqual(fields["input_tokens"] as? Int, 77)
      XCTAssertEqual(fields["output_tokens"] as? Int, 0)
      XCTAssertEqual(fields["truncated"] as? Bool, truncated ? true : nil)
      XCTAssertEqual(try JSONDecoder().decode(DecisionResponse.Usage.self, from: encoded).truncated,
                     truncated ? true : nil)
    }
  }
  #if canImport(CoreAI)
  @available(macOS 27.0, *)
  func testDecodedPixelTableMatchesTheFP16VisionABI() {
    XCTAssertEqual(Vision.pixelBits, (0...255).map { Float16(Float($0) / 127.5 - 1).bitPattern })
  }
  #endif

  func testInstalledModel() async throws {
    guard let path = ProcessInfo.processInfo.environment["DECISION_TEST_BUNDLE"],
      let model = ProcessInfo.processInfo.environment["DECISION_TEST_MODEL"]
    else {
      throw XCTSkip("Set DECISION_TEST_BUNDLE and DECISION_TEST_MODEL to run ANE equivalence tests")
    }
    let bundle = URL(fileURLWithPath: path)
    do { try await DecisionEngine(bundleURL: bundle).prepare() }
    let engine = DecisionEngine(bundleURL: bundle, idleTimeout: .milliseconds(25))
    let rows = try root.decode("goldens.json", as: [Fixture].self).filter { $0.model == model }
    XCTAssertFalse(rows.isEmpty)
    for row in rows {
      let input = DecisionInput(
        state: row.state, questions: row.request.questions,
        image: row.id == "image-red-circle"
          ? try Data(contentsOf: root.appendingPathComponent("red-circle.png")) : nil)
      let actual = try await engine.decide(input)
      XCTAssertGreaterThan(try XCTUnwrap(actual.usage.promptTokensPerSecond), 0)
      XCTAssertEqual(actual.usage.inputTokens, row.ids.count)
      let a = try XCTUnwrap(actual.answers[row.name])
      let e = try XCTUnwrap(row.result.answers[row.name])
      XCTAssertEqual(a.choice, e.choice)
      if let probability = e.noul {
        XCTAssertEqual(try XCTUnwrap(a.noul) > 0.5, probability > 0.5)
        XCTAssertEqual(try XCTUnwrap(a.noul), probability, accuracy: 0.001)
      }
      for (key, p) in e.probabilities ?? [:] {
        XCTAssertEqual(try XCTUnwrap(a.probabilities?[key]), p, accuracy: 0.001)
      }
    }
    try await Task.sleep(for: .milliseconds(100))
    let idle = await engine.isLoaded
    XCTAssertFalse(idle)
    async let first = engine.decide(rows[0].request)
    async let second = engine.decide(rows[0].request)
    let (a, b) = try await (first, second)
    XCTAssertEqual(a.usage.inputTokens, b.usage.inputTokens)
    await engine.unload()
    let unloaded = await engine.isLoaded
    XCTAssertFalse(unloaded)
    _ = try await engine.decide(rows[0].request)
    await engine.unload()
  }
}
