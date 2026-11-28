// llama-server-compatible POST /tokenize and POST /detokenize, so clients that measure or split their inputs with
// llama-server's tokenizer API work unchanged:
//
//   /tokenize    content        string (required), at most 1 MiB of UTF-8
//                add_special    use the loaded tokenizer’s automatic special tokens (default false)
//                parse_special  control-token text such as "<bos>" becomes its id (default true); false encodes it as
//                               ordinary text (the user-defined added tokens are still cut out, as in llama.cpp)
//                with_pieces    each token as {"id", "piece"}: the text it stands for, or a list of its bytes when
//                               they are not valid UTF-8 alone (default false)
//                -> {"tokens": [ids]} or {"tokens": [{"id", "piece"}]}
//   /detokenize  tokens         [ids] -> {"content": text}; special tokens appear as their text, and bytes that do
//                               not form valid UTF-8 become U+FFFD
//
// The ids are the loaded model’s own. /tokenize
// counts every token, beyond the embedding limit too: a client measures texts longer than that limit to split them.
// Where llama-server is lenient this is strict, like the rest of the API: a missing or non-string `content`, missing
// `tokens`, a flag that is not a boolean and an id outside the vocabulary are 400s.
import Foundation

public struct TokenizeRequest: Sendable, Equatable {
    /// The longest `content` /tokenize takes, in UTF-8 bytes. Tokenizing costs time and memory in proportion to the
    /// text (and more than that for one long word), so a body up to `--max-body` is not tokenized whole.
    public static let maxContentBytes = 1 << 20

    public var content: String
    public var addSpecial = false
    public var parseSpecial = true
    public var withPieces = false

    public init(content: String, addSpecial: Bool = false, parseSpecial: Bool = true, withPieces: Bool = false) {
        self.content = content
        self.addSpecial = addSpecial
        self.parseSpecial = parseSpecial
        self.withPieces = withPieces
    }

    public static func parse(_ body: Data) throws(ServiceError) -> TokenizeRequest {
        let object = try jsonObject(body)
        guard let content = object["content"]?.stringValue else {
            throw .badRequest("invalid_content", "content must be a string", param: "content")
        }
        guard content.utf8.count <= maxContentBytes else {
            throw ServiceError(.payloadTooLarge, "content_too_long",
                               "content is \(content.utf8.count) bytes; /tokenize takes at most \(maxContentBytes)",
                               param: "content")
        }
        var request = TokenizeRequest(content: content)
        request.addSpecial = try flag(object, "add_special", default: request.addSpecial)
        request.parseSpecial = try flag(object, "parse_special", default: request.parseSpecial)
        request.withPieces = try flag(object, "with_pieces", default: request.withPieces)
        return request
    }

    static func flag(_ object: JSONObject, _ name: String, default value: Bool) throws(ServiceError) -> Bool {
        switch object[name] {
        case nil, .null?: return value
        case .bool(let flag)?: return flag
        default: throw .badRequest("invalid_\(name)", "\(name) must be true or false", param: name)
        }
    }
}

public enum DetokenizeRequest {
    /// The most bytes the ids of one /detokenize request may stand for: the longest `content` /tokenize takes plus room
    /// for the <bos>/<eos> text `add_special` puts around it, so the ids of any text /tokenize accepted decode. Counted
    /// before invalid UTF-8 becomes U+FFFD (3 bytes) and before JSON escaping, which bound the response at a few times this.
    public static let maxContentBytes = TokenizeRequest.maxContentBytes + 64

    /// The ids of the body. Whether each is a token of the model is checked while decoding.
    public static func parse(_ body: Data) throws(ServiceError) -> [Int32] {
        let object = try jsonObject(body)
        guard let items = object["tokens"]?.arrayValue else {
            throw .badRequest("invalid_tokens", "tokens must be a list of token ids", param: "tokens")
        }
        var ids: [Int32] = []
        ids.reserveCapacity(items.count)
        for (index, item) in items.enumerated() {
            guard let id = item.intValue, id >= 0, id <= Int(Int32.max) else { throw invalidToken(index) }
            ids.append(Int32(id))
        }
        return ids
    }

    static func invalidToken(_ index: Int) -> ServiceError {
        .badRequest("invalid_token", "tokens[\(index)] is not a token id of this model", param: "tokens[\(index)]")
    }
}

private func jsonObject(_ body: Data) throws(ServiceError) -> JSONObject {
    let root: JSONValue
    do {
        root = try JSONValue.parse(body)
    } catch {
        throw .badRequest("invalid_json", "request body is not valid JSON (\(error.description))")
    }
    guard let object = root.objectValue else { throw .badRequest("invalid_json", "expected a JSON object") }
    return object
}

public enum TokenizerResponse {
    /// `{"tokens": [...]}` for `ids`, written directly so a long text does not build a JSON tree.
    public static func tokens(_ ids: [Int32], pieces tokenizer: (any TextTokenizer)?) -> Data {
        var output: [UInt8] = []
        output.reserveCapacity(ids.count * (tokenizer == nil ? 7 : 24) + 16)
        output.append(contentsOf: #"{"tokens":["#.utf8)
        for (index, id) in ids.enumerated() {
            if index > 0 { output.append(UInt8(ascii: ",")) }
            guard let tokenizer else {
                output.append(contentsOf: String(id).utf8)
                continue
            }
            output.append(contentsOf: #"{"id":\#(id),"piece":"#.utf8)
            let bytes = tokenizer.pieceBytes(id) ?? []
            if let text = String(validating: bytes, as: UTF8.self) {
                writeJSONString(text, to: &output)
            } else {
                output.append(UInt8(ascii: "["))
                output.append(contentsOf: bytes.map(String.init).joined(separator: ",").utf8)
                output.append(UInt8(ascii: "]"))
            }
            output.append(UInt8(ascii: "}"))
        }
        output.append(contentsOf: "]}".utf8)
        return Data(output)
    }

    /// The text of `ids`: their bytes joined, invalid UTF-8 replaced with U+FFFD. Stops at the first id that is not a
    /// token and once the text outgrows `DetokenizeRequest.maxContentBytes`.
    public static func content(_ ids: [Int32], tokenizer: any TextTokenizer) throws(ServiceError) -> String {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(min(ids.count * 4, DetokenizeRequest.maxContentBytes))
        for (index, id) in ids.enumerated() {
            guard let piece = tokenizer.pieceBytes(id) else { throw DetokenizeRequest.invalidToken(index) }
            bytes.append(contentsOf: piece)
            guard bytes.count <= DetokenizeRequest.maxContentBytes else {
                throw ServiceError(.payloadTooLarge, "content_too_long",
                                   "the text of tokens is over \(DetokenizeRequest.maxContentBytes) bytes",
                                   param: "tokens")
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }
}
