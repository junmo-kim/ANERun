import Foundation
import Testing

@testable import EmbedCore

@Suite("OpenAI request parsing")
struct OpenAIRequestTests {
    func parse(_ json: String) throws(ServiceError) -> EmbeddingsRequest {
        try EmbeddingsRequest.parse(Data(json.utf8))
    }

    @Test func acceptsStringsListsAndOptions() throws {
        #expect(try parse(#"{"input": "a"}"#) == EmbeddingsRequest(input: .texts(["a"])))
        let request = try parse(#"{"input": ["a", "b"], "model": "x", "encoding_format": "base64", "dimensions": 256, "input_type": "document", "user": "u"}"#)
        #expect(request == EmbeddingsRequest(input: .texts(["a", "b"]), encodingFormat: .base64, dimensions: 256, inputType: .document))
    }

    @Test func pplxDefaultsToFullWidthAndAcceptsEveryIntegerPrefix() throws {
        let request = try EmbeddingsRequest.parse(Data(#"{"input":"x","model":"embeddinggemma-2","input_type":"query"}"#.utf8),
                                                  modelID: .pplx)
        #expect(request.dimensions == 1024 && request.inputType == .query)
        for width in [1, 300, 1024] {
            let body = Data(#"{"input":"x","dimensions":\#(width)}"#.utf8)
            #expect(try EmbeddingsRequest.parse(body, modelID: .pplx).dimensions == width)
        }
        for width in ["0", "1025", "1.5", #""300""#] {
            let body = Data(#"{"input":"x","dimensions":\#(width)}"#.utf8)
            do {
                _ = try EmbeddingsRequest.parse(body, modelID: .pplx)
                Issue.record("accepted dimensions \(width)")
            } catch {
                #expect(error.reason == "invalid_dimensions" && error.param == "dimensions")
            }
        }
    }

    @Test func keepsALeadingByteOrderMark() throws {
        #expect(try parse("{\"input\": \"\u{FEFF}x\"}").input == .texts(["\u{FEFF}x"]))
    }

    /// The number of items is checked before any image is base64-decoded (a body full of tiny image items
    /// must not be decoded only to be refused afterwards).
    @Test func countsItemsBeforeDecodingImages() throws {
        var limits = ServiceLimits()
        limits.maxImages = 2
        limits.maxInputs = 3
        let undecodable = #"{"content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!!"}}]}"#
        let images = "{\"input\": [\(Array(repeating: undecodable, count: 3).joined(separator: ","))]}"
        #expect(throws: ServiceError.badRequest("too_many_images", "at most 2 image items per request", param: "input")) {
            try EmbeddingsRequest.parse(Data(images.utf8), limits: limits)
        }
        #expect(throws: ServiceError.badRequest("too_many_inputs", "at most 3 inputs per request", param: "input")) {
            try EmbeddingsRequest.parse(Data(#"{"input": ["a", "b", "c", "d"]}"#.utf8), limits: limits)
        }
        let two = "{\"input\": [\(Array(repeating: undecodable, count: 2).joined(separator: ","))]}"
        #expect((try? EmbeddingsRequest.parse(Data(two.utf8), limits: limits)) == nil)  // within the limit: decoded, and fails there
        do {
            _ = try EmbeddingsRequest.parse(Data(two.utf8), limits: limits)
        } catch {
            #expect(error.reason == "image_decode_failed")
        }
    }

    @Test func acceptsLlamaServerImageItems() throws {
        let url = "data:image/png;base64," + Data([1, 2, 3]).base64EncodedString()
        let request = try parse(#"{"input": [{"content": [{"type": "image_url", "image_url": {"url": "\#(url)"}}]}]}"#)
        #expect(request.input == .images([Data([1, 2, 3])]))
    }

    @Test(arguments: [
        (#"not json"#, "invalid_json"), (#"[1]"#, "invalid_json"), (#"{}"#, "invalid_input"),
        (#"{"input": []}"#, "empty_input"), (#"{"input": [1, 2, 3]}"#, "invalid_input"),
        (#"{"input": ["a", {"content": []}]}"#, "invalid_input"),
        (#"{"input": "a", "dimensions": 300}"#, "invalid_dimensions"), (#"{"input": "a", "dimensions": "256"}"#, "invalid_dimensions"),
        (#"{"input": "a", "encoding_format": "int8"}"#, "invalid_encoding_format"),
        (#"{"input": "a", "input_type": "passage"}"#, "invalid_input_type"),
        (#"{"input": [{"content": [{"type": "image_url", "image_url": {"url": "http://x/y.png"}}]}]}"#, "invalid_image_url"),
        (#"{"input": [{"content": [{"type": "text", "text": "x"}]}]}"#, "invalid_image_item"),
        (#"{"input": [{"content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}], "input_type": "query"}"#, "invalid_input_type"),
    ])
    func rejectsWithOpenAIErrors(_ json: String, _ reason: String) {
        do {
            _ = try parse(json)
            Issue.record("accepted \(json)")
        } catch {
            #expect(error.reason == reason, "\(json)")
            #expect(error.httpStatus == 400)
            let body = error.openAIBody
            #expect(body["error"]?["type"]?.stringValue == "invalid_request_error")
            #expect(body["error"]?["message"]?.stringValue?.isEmpty == false)
            #expect(body["error"]?["code"]?.stringValue == reason)
        }
    }

    @Test func encodesFloatAndBase64Responses() throws {
        let result = EmbeddingResult(vectors: [Vectors.l2Normalized((0..<768).map { Float($0 % 7) - 3 })], tokenCounts: [5])
        let floats = try JSONValue.parse(EmbeddingsResponse.body(result, request: EmbeddingsRequest(input: .texts(["x"])), model: "m"))
        #expect(floats["object"] == "list")
        #expect(floats["model"] == "m")
        #expect(floats["usage"] == ["prompt_tokens": 5, "total_tokens": 5])
        let item = try #require(floats["data"]?.arrayValue?.first)
        #expect(item["object"] == "embedding" && item["index"] == 0)
        let vector = try #require(item["embedding"]?.arrayValue).map { Float($0.doubleValue!) }
        #expect(vector == result.vectors[0])

        let request = EmbeddingsRequest(input: .texts(["x"]), encodingFormat: .base64, dimensions: 128)
        let encoded = try JSONValue.parse(EmbeddingsResponse.body(result, request: request, model: "m"))
        let decoded = try #require(Vectors.decodeBase64(encoded["data"]!.arrayValue![0]["embedding"]!.stringValue!))
        #expect(decoded == Vectors.truncated(result.vectors[0], to: 128))
    }
}
