// MCP over JSON-RPC 2.0: initialize, ping, tools/list, tools/call. Used by `anerun mcp` (stdio,
// newline-delimited) and `POST /mcp`. Tool failures are results with isError and a structured text
// (error/reason/detail/hint) the agent can act on; protocol errors are JSON-RPC errors.
import Foundation

public struct MCPHandler: Sendable {
    public static let protocolVersions = ["2025-06-18", "2025-03-26", "2024-11-05"]

    public let state: ServerState
    public let allowPaths: Bool
    /// stdio waits for the model on tools/call; HTTP answers `loading` right away.
    public let waitForModel: Bool
    /// HTTP: server-side failures are reported with a generic message (ServiceError.publicFacing) and their detail
    /// goes to `log`; stdio, a local client, gets the detail.
    public let publicErrors: Bool
    public let modelID: ModelID
    private let configuredMaxTokens: Int
    public let log: @Sendable (String) -> Void

    public init(state: ServerState, allowPaths: Bool, waitForModel: Bool, publicErrors: Bool = false,
                modelID: ModelID = .embeddingGemma2, configuredMaxTokens: Int = ServiceLimits().maxTokens,
                log: @escaping @Sendable (String) -> Void = { _ in })
    {
        self.state = state
        self.allowPaths = allowPaths
        self.waitForModel = waitForModel
        self.publicErrors = publicErrors
        self.modelID = modelID
        self.configuredMaxTokens = configuredMaxTokens
        self.log = log
    }

    /// Handles one JSON-RPC message; nil for notifications (no response).
    public func handle(_ message: JSONValue) async -> JSONValue? {
        guard let object = message.objectValue, object["jsonrpc"]?.stringValue == "2.0",
            let method = object["method"]?.stringValue
        else {
            return Self.error(id: message["id"] ?? .null, code: -32600, message: "invalid JSON-RPC request")
        }
        guard let id = object["id"], !id.isNull else { return nil }  // notification
        let params = object["params"]?.objectValue ?? JSONObject()
        switch method {
        case "initialize":
            let requested = params["protocolVersion"]?.stringValue ?? Self.protocolVersions[0]
            let version = Self.protocolVersions.contains(requested) ? requested : Self.protocolVersions[0]
            return Self.result(id: id, [
                "protocolVersion": .string(version), "capabilities": ["tools": [:]],
                "serverInfo": ["name": "aneembed", "version": .string(aneembedVersion)],
            ])
        case "ping":
            return Self.result(id: id, [:])
        case "tools/list":
            let maxContextLength: Int
            if case .ready(let service) = state.status {
                maxContextLength = service.maxTokens
            } else {
                maxContextLength = configuredMaxTokens
            }
            return Self.result(id: id, ["tools": .array(Tools.definitions(for: modelID, maxContextLength: maxContextLength))])
        case "tools/call":
            guard let name = params["name"]?.stringValue else {
                return Self.error(id: id, code: -32602, message: "tools/call needs params.name")
            }
            guard Tools.names.contains(name) else {
                return Self.error(id: id, code: -32602, message: "unknown tool \(name)")
            }
            let arguments = params["arguments"]?.objectValue ?? JSONObject()
            let format = arguments["format"]?.stringValue.flatMap(OutputFormat.init(rawValue:)) ?? .json
            do throws(ServiceError) {
                let service = waitForModel ? try await state.waitForService() : try state.service()
                let output = try await Tools.call(name, arguments: arguments,
                                                  context: ToolContext(service: service, allowPaths: allowPaths))
                return Self.result(id: id, ["content": [["type": "text", "text": .string(output.render(as: format))]]])
            } catch {
                var shown = error
                if publicErrors {
                    if error.kind == .internal { log("POST /mcp tools/call \(name): \(error.description)") }
                    shown = error.publicFacing
                }
                return Self.result(id: id, [
                    "content": [["type": "text", "text": .string(shown.structured.render(as: format))]], "isError": true,
                ])
            }
        default:
            return Self.error(id: id, code: -32601, message: "method not found: \(method)")
        }
    }

    /// Parses and handles one line/body; invalid JSON is a -32700 parse error.
    public func handle(raw: Data) async -> JSONValue? {
        do {
            return await handle(try JSONValue.parse(raw))
        } catch {
            return Self.error(id: .null, code: -32700, message: "parse error: \(error.description)")
        }
    }

    /// The answer to a stdio message longer than the line limit (it is discarded unread, so its id is unknown).
    public static func messageTooLarge(bytes: Int, limit: Int) -> JSONValue {
        error(id: .null, code: -32600, message: "message of \(bytes) bytes exceeds the \(limit)-byte limit (--max-body)")
    }

    static func result(id: JSONValue, _ result: JSONValue) -> JSONValue {
        ["jsonrpc": "2.0", "id": id, "result": result]
    }

    static func error(id: JSONValue, code: Int, message: String) -> JSONValue {
        ["jsonrpc": "2.0", "id": id, "error": ["code": .int(code), "message": .string(message)]]
    }
}
