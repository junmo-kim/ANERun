// One error model for every surface (same as macvis): a stable `error` class and `reason` code, a
// human message and a hint. It maps to a CLI exit code, an HTTP status with an OpenAI-style body, and
// an MCP tool result with isError.
import Foundation

/// Process exit codes. 64 (EX_USAGE) means the request itself is wrong; retrying will not help.
public enum ExitCode: Int32, Sendable {
    case success = 0
    case runtimeError = 1
    case usage = 64
    /// EX_DATAERR: the input could not be processed (undecodable image, too many tokens).
    case dataError = 65
    /// EX_UNAVAILABLE: no model installed, no Neural Engine, model failed to load.
    case unavailable = 69
    /// EX_TEMPFAIL: still loading or compiling; retry later.
    case temporaryFailure = 75
}

public struct ServiceError: Error, Sendable, Equatable, CustomStringConvertible {
    public enum Kind: String, Sendable {
        /// Malformed or invalid request (bad JSON, wrong types, unknown option).
        case badRequest = "bad_request"
        /// Valid request the model cannot take (too many tokens, undecodable image).
        case invalidInput = "invalid_input"
        case notFound = "not_found"
        case payloadTooLarge = "payload_too_large"
        /// The model is loading or warming; retry.
        case loading = "loading"
        /// No usable model (not installed, failed to load, no Neural Engine).
        case unavailable = "unavailable"
        case `internal` = "internal_error"
    }

    public let kind: Kind
    /// A stable machine-readable code, e.g. `context_length_exceeded`, `invalid_dimensions`.
    public let reason: String
    public let message: String
    /// The request field at fault (OpenAI `param`).
    public let param: String?
    public let hint: String?

    public init(_ kind: Kind, _ reason: String, _ message: String, param: String? = nil, hint: String? = nil) {
        self.kind = kind
        self.reason = reason
        self.message = message
        self.param = param
        self.hint = hint
    }

    public static func badRequest(_ reason: String, _ message: String, param: String? = nil) -> ServiceError {
        ServiceError(.badRequest, reason, message, param: param)
    }

    public var description: String { "\(kind.rawValue): \(reason): \(message)" }

    public var httpStatus: Int {
        switch kind {
        case .badRequest, .invalidInput: return 400
        case .notFound: return 404
        case .payloadTooLarge: return 413
        case .loading, .unavailable: return 503
        case .internal: return 500
        }
    }

    public var exitCode: ExitCode {
        switch kind {
        case .badRequest, .notFound: return .usage
        case .invalidInput, .payloadTooLarge: return .dataError
        case .loading: return .temporaryFailure
        case .unavailable: return .unavailable
        case .internal: return .runtimeError
        }
    }

    /// What an HTTP client sees. Server-side failures (500 internal, 503 unavailable) keep their kind, reason, param
    /// and hint, with a generic message: the detail (paths, Core ML and ImageIO errors) is for the server log.
    /// Client errors and `loading` describe the request or a documented state and stay as they are.
    public var publicFacing: ServiceError {
        switch kind {
        case .internal:
            return ServiceError(kind, reason, "internal server error (\(reason)); details are in the server log",
                                param: param, hint: hint)
        case .unavailable:
            return ServiceError(kind, reason, "the model is not available (\(reason)); details are in the server log",
                                param: param, hint: hint)
        default:
            return self
        }
    }

    /// OpenAI's error object: {"error": {"message", "type", "param", "code"}}.
    public var openAIBody: JSONValue {
        let type: String
        switch kind {
        case .badRequest, .invalidInput, .notFound, .payloadTooLarge: type = "invalid_request_error"
        case .loading, .unavailable, .internal: type = "server_error"
        }
        return ["error": ["message": .string(message), "type": .string(type), "param": param.map(JSONValue.string) ?? .null,
                          "code": .string(reason)]]
    }

    /// The structured form for stderr (CLI) and MCP tool errors.
    public var structured: JSONValue {
        var object: JSONObject = ["error": .string(kind.rawValue), "reason": .string(reason), "detail": .string(message)]
        if let param { object["param"] = .string(param) }
        if let hint { object["hint"] = .string(hint) }
        return .object(object)
    }
}
