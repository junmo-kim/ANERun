/// Measures a single request with a monotonic clock, before response serialization.
package struct InputProcessingTimer {
  private let start = ContinuousClock.now

  package init() {}

  package func tokensPerSecond(_ inputTokens: Int) -> Double? {
    let elapsed = start.duration(to: .now).components
    let seconds = Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
    guard inputTokens > 0, seconds > 0 else { return nil }
    return Double(inputTokens) / seconds
  }
}
