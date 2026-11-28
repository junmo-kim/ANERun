import XCTest

@testable import DecisionRuntime

final class ElephantTests: XCTestCase {
  struct Layout: Decodable {
    let caseID: String
    let question: String
    let ids: [Int]
    let prefix: Int
  }
  struct Request: Decodable {
    let state: String
    let questions: [DecisionQuestion]
    let image: Data?
  }
  let root = Bundle.module.resourceURL!.appendingPathComponent("Fixtures/elephant")

  func testOneJevTokenLayout() throws {
    let renderer = ElephantPrompt(tokenizer: try BPETokenizer(root), context: 512, labels: ["A", "B", "C"])
    let goldens = try root.decode("token-layout.json", as: [Layout].self)
    for group in Dictionary(grouping: goldens, by: \.caseID) {
      let request = try root.decode(group.key + ".json", as: Request.self)
      let input = DecisionInput(state: request.state, questions: request.questions, image: request.image)
      let prepared = try renderer.build(input, visualTokens: input.image == nil ? 0 : 64)
      XCTAssertFalse(prepared.truncated)
      let branches = prepared.prompts
      XCTAssertEqual(branches.count, group.value.count)
      for (branch, golden) in zip(branches, group.value) {
        XCTAssertEqual(branch.branches.first?.question.name, golden.question)
        XCTAssertEqual(branch.ids, golden.ids, group.key)
        XCTAssertEqual(branch.prefix, golden.prefix, group.key)
      }
    }
  }

  func testUsageCountsSharedInputOnceAndEveryProcessedToken() {
    let prompts = [[1, 2, 3], [1, 2, 4, 5]].map {
      Prompt(ids: $0, prefix: 2, segments: [], branches: [])
    }
    let counts = PreparedDecision(prompts: prompts, truncated: false).tokenCounts(sharingPrefix: true)
    XCTAssertEqual(counts.input, 5)
    XCTAssertEqual(counts.compute, 7)
  }

  func testRejectsInvalidBranches() throws {
    let renderer = ElephantPrompt(tokenizer: try BPETokenizer(root), context: 256, labels: ["A", "B"])
    let question = DecisionQuestion(name: "q", kind: .noul, instructions: "")
    for input in [
      DecisionInput(state: "<|vision_start|>", questions: [question]),
      DecisionInput(state: "state", questions: [question, question]),
      DecisionInput(state: String(repeating: "x ", count: 300), questions: [question], truncate: false),
      DecisionInput(state: "state", questions: [.init(name: "q", kind: .score, instructions: "",
                                                     candidates: [.init(key: "2", description: "higher")])]),
    ] {
      XCTAssertThrowsError(try renderer.build(input, visualTokens: 0))
    }
  }

  func testFitsSharedImagePrefixAgainstAggregateCap() throws {
    let tokenizer = try BPETokenizer(root)
    let renderer = ElephantPrompt(tokenizer: tokenizer, context: 8192, labels: ["A", "B"])
    let stem = String(repeating: "1", count: 400)
    let first = DecisionQuestion(name: "0", kind: .noul, instructions: stem)
    let baseline = try XCTUnwrap(
      renderer.build(.init(state: "1", questions: [first]), visualTokens: 64).prompts.first)
    let suffixLength = baseline.ids.count - baseline.prefix
    let count = (65536 - baseline.prefix - 100) / suffixLength
    let questions = (0..<count).map { DecisionQuestion(name: String($0), kind: .noul, instructions: stem) }
    let long = String(repeating: "1", count: 2048)
    let prepared = try renderer.build(.init(state: long, questions: questions), visualTokens: 64)
    XCTAssertTrue(prepared.truncated)
    let prefix = try XCTUnwrap(prepared.prompts.first).prefix
    let kept = String(repeating: "1", count: prefix - baseline.prefix + 1)
    let exact = try renderer.build(.init(state: kept, questions: questions, truncate: false), visualTokens: 64)
    XCTAssertEqual(prepared.prompts.map(\.ids), exact.prompts.map(\.ids))
    XCTAssertEqual(prepared.prompts.map { $0.branches[0].question.name }, questions.map(\.name))
    XCTAssertEqual(prepared.prompts.reduce(prefix) { $0 + $1.ids.count - $1.prefix }, 65536)
    let imagePad = try XCTUnwrap(tokenizer.special["<|image_pad|>"])
    XCTAssertTrue(prepared.prompts.allSatisfy { $0.ids.prefix($0.prefix).filter { $0 == imagePad }.count == 64 })
    XCTAssertThrowsError(
      try renderer.build(.init(state: long, questions: questions, truncate: false), visualTokens: 64)
    ) {
      XCTAssertEqual(($0 as? DecisionError)?.description, "Request exceeds token limit")
    }
    XCTAssertThrowsError(
      try renderer.build(.init(state: long + "<|vision_start|>\u{301}", questions: questions), visualTokens: 64)
    ) {
      XCTAssertEqual(($0 as? DecisionError)?.description, "Reserved control token in request")
    }
  }

  func testInstalledElephantAnswers() async throws {
    guard let path = ProcessInfo.processInfo.environment["ELEPHANT_TEST_BUNDLE"] else {
      throw XCTSkip("Set ELEPHANT_TEST_BUNDLE to compare the packed model with the CPU references")
    }
    let engine = DecisionEngine(bundleURL: URL(fileURLWithPath: path))
    let refs = try root.decode("cpu-answers.json", as: [String: [String: DecisionAnswer]].self)
    for key in refs.keys.sorted() {
      let request = try root.decode(key + ".json", as: Request.self)
      let response = try await engine.decide(
        .init(state: request.state, questions: request.questions, image: request.image))
      let expected = try XCTUnwrap(refs[key])
      XCTAssertEqual(Set(response.answers.keys), Set(expected.keys), key)
      for (name, oracle) in expected {
        let actual = try XCTUnwrap(response.answers[name])
        XCTAssertEqual(actual.type, oracle.type)
        XCTAssertEqual(actual.choice, oracle.choice)
        XCTAssertEqual(actual.legend, oracle.legend)
        if let p = oracle.noul {
          XCTAssertEqual(try XCTUnwrap(actual.noul), p, accuracy: 0.001)
        }
        if let score = oracle.score {
          XCTAssertEqual(try XCTUnwrap(actual.score), score, accuracy: 0.02)
        }
        if let confidence = oracle.confidence {
          XCTAssertEqual(try XCTUnwrap(actual.confidence), confidence, accuracy: 0.02)
        }
        if let probabilities = oracle.probabilities {
          let values = try XCTUnwrap(actual.probabilities)
          XCTAssertEqual(Set(values.keys), Set(probabilities.keys))
          for (label, probability) in probabilities {
            XCTAssertEqual(try XCTUnwrap(values[label]), probability, accuracy: 0.001)
          }
        }
      }
    }
    await engine.unload()
  }
}
