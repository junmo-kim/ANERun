import DecisionRuntime
import Foundation

struct Case: Decodable {
  let state: String, type: DecisionQuestion.Kind, instructions: String, criteria: [[String]],
    name: String, id: String, model: String
}
@main struct Bench {
  static func main() async {
    do { try await run() } catch {
      FileHandle.standardError.write(Data("\(error)\n".utf8))
      exit(1)
    }
  }
  static func run() async throws {
    let a = CommandLine.arguments
    guard a.count >= 3 else { throw Usage() }
    let url = URL(fileURLWithPath: a[2])
    let engine = DecisionEngine(
      bundleURL: url,
      idleTimeout: a[1] == "worker" && a.count == 4 ? .milliseconds(Int(a[3]) ?? 0) : nil)
    if a[1] == "worker" {
      struct Command: Decodable {
        let op: String, state: String?, questions: [DecisionQuestion]?, image: Data?
      }
      while let line = readLine() {
        let command = try JSONDecoder().decode(Command.self, from: Data(line.utf8))
        switch command.op {
        case "status": print(await engine.isLoaded ? "true" : "false")
        case "unload":
          await engine.unload()
          print("true")
        case "decide":
          guard let state = command.state, let questions = command.questions else { throw Usage() }
          let reply = try await engine.decide(
            DecisionInput(state: state, questions: questions, image: command.image))
          print(String(decoding: try JSONEncoder().encode(reply), as: UTF8.self))
        default: throw Usage()
        }
        fflush(stdout)
      }
      await engine.unload()
      return
    }
    if a[1] == "prepare" {
      let start = Date().timeIntervalSince1970
      try await engine.prepare()
      print("prepare", start, Date().timeIntervalSince1970)
      return
    }
    guard a[1] == "golden", a.count >= 5 else { throw Usage() }
    let source = URL(fileURLWithPath: a[3])
    let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: source))
    for c in cases where c.model == a[4] && (a.count < 6 || c.id == a[5]) {
      let q = DecisionQuestion(
        name: c.name, kind: c.type, instructions: c.instructions,
        candidates: c.criteria.map { .init(key: $0[0], description: $0[1]) })
      let image =
        c.id == "image-red-circle"
        ? try Data(
          contentsOf: source.deletingLastPathComponent().appendingPathComponent("red-circle.png"))
        : nil
      let input = DecisionInput(state: c.state, questions: [q], image: image)
      for i in 0..<2 {
        let start = Date().timeIntervalSince1970
        let response = try await engine.decide(input)
        let end = Date().timeIntervalSince1970
        print(
          String(
            decoding: try JSONEncoder().encode(
              Row(id: c.id, cold: i == 0, start: start, end: end, result: response)), as: UTF8.self)
        )
      }
      await engine.unload()
    }
  }
  struct Row: Encodable {
    let id: String, cold: Bool, start: Double, end: Double, result: DecisionResponse
  }
  struct Usage: Error, CustomStringConvertible {
    var description: String { "decision-bench prepare BUNDLE | golden BUNDLE GOLDENS MODEL [CASE]" }
  }
}
