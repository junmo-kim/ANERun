/// Runs `transform` on every element, at most `width` at a time, and returns the results in the order of
/// `elements`. A new element starts as soon as one finishes. The first error is thrown; the elements still running
/// are cancelled and none is started after it.
func windowedMap<Element: Sendable, Output: Sendable>(
    _ elements: [Element], width: Int, _ transform: @escaping @Sendable (Element) async throws -> Output
) async throws -> [Output] {
    var results = [Output?](repeating: nil, count: elements.count)
    try await withThrowingTaskGroup(of: (Int, Output).self) { group in
        var started = 0
        func startNext() {
            guard started < elements.count else { return }
            let index = started, element = elements[index]
            started += 1
            group.addTask { (index, try await transform(element)) }
        }
        for _ in 0..<max(1, width) { startNext() }
        while let (index, output) = try await group.next() {
            results[index] = output
            startNext()
        }
    }
    return results.map { $0! }
}
