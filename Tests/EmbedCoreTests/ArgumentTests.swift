import Foundation
import Testing

@testable import EmbedCore

@Suite("Arguments")
struct ArgumentTests {
    let spec = ArgumentSpec(values: ["image", "prompt", "format"], flags: ["stdin"])

    @Test func parsesPositionalsOptionsAndFlags() throws {
        let parsed = try spec.parse(["a", "--image", "x.png", "--image=y.png", "--stdin", "--port", "9", "--", "--not-an-option", "b"])
        #expect(parsed.positionals == ["a", "--not-an-option", "b"])
        #expect(parsed.values("image") == ["x.png", "y.png"])
        #expect(parsed.has("stdin"))
        #expect(parsed.configFlags == ["port": "9"])
    }

    @Test func rejectsUnknownAndIncompleteOptions() {
        #expect(throws: ServiceError.self) { try spec.parse(["--nope"]) }
        #expect(throws: ServiceError.self) { try spec.parse(["--image"]) }
        #expect(throws: ServiceError.self) { try spec.parse(["--stdin=1"]) }
        #expect(throws: ServiceError.self) { try spec.parse(["-x"]) }
        do {
            _ = try spec.parse(["--nope"])
        } catch {
            #expect(error.exitCode == .usage)
        }
    }

    @Test func helpForms() throws {
        #expect(try spec.parse(["-h"]).has("help"))
        #expect(try spec.parse(["--help"]).has("help"))
        #expect(try spec.parse(["-"]).positionals == ["-"])
    }
}
