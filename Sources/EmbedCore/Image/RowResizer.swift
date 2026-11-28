// The torch resize of ImagePrep, fed one source row at a time. The source arrives top to bottom in its stored
// orientation; the EXIF orientation (PIL's exif_transpose, applied before the resize in the official recipe) is
// folded into the two passes, so neither the full source nor an oriented copy of it is ever held. Each output
// value is computed from the same inputs, with the same int16 weights and the same uint8 rounding between the
// passes, as resizing the oriented image would: the result is bit-exact.
//
// With O the oriented image, torch runs the horizontal pass over O's rows, then the vertical pass over O's
// columns. For orientations 1-4, O's rows are source rows (flipped for 2 and 3, in reverse order for 3 and 4): the
// horizontal pass runs on each source row as it arrives, and the vertical pass over the arriving results. For
// 5-8, O is transposed: its horizontal pass combines source rows (a LinePass over the arriving rows), and its
// vertical pass then runs along each line that pass emits, which is one column of the output.
import Foundation

final class RowResizer {
    let sourceWidth: Int
    let sourceHeight: Int
    let width: Int
    let height: Int
    private let transposed: Bool
    /// Orientations 1-4: reverse each source row first. 5-8: reverse each line `lines` emits.
    private let reverse: Bool
    /// The pass applied to each source row (1-4) or to each emitted line (5-8); nil: torch skips it.
    private let lineAxis: ImagePrep.AxisWeights?
    private let lines: LinePass
    private var output: [UInt8]
    private let reversed: UnsafeMutablePointer<UInt8>
    private let resampled: UnsafeMutablePointer<UInt8>
    private var pushed = 0

    init(sourceWidth: Int, sourceHeight: Int, orientation: Int, toWidth: Int, toHeight: Int) {
        let orientation = (2...8).contains(orientation) ? orientation : 1
        self.sourceWidth = sourceWidth
        self.sourceHeight = sourceHeight
        width = toWidth
        height = toHeight
        transposed = orientation >= 5
        let orientedWidth = transposed ? sourceHeight : sourceWidth
        let orientedHeight = transposed ? sourceWidth : sourceHeight
        let horizontal = toWidth != orientedWidth ? ImagePrep.axisWeights(input: orientedWidth, output: toWidth) : nil
        let vertical = toHeight != orientedHeight ? ImagePrep.axisWeights(input: orientedHeight, output: toHeight) : nil
        if transposed {
            // O column x is source row x (5, 8) or sourceHeight - 1 - x (6, 7); O row y is source column y (5, 6)
            // or sourceWidth - 1 - y (7, 8).
            lines = LinePass(lineBytes: sourceWidth * 3, inputs: sourceHeight, axis: horizontal,
                             flip: orientation == 6 || orientation == 7)
            reverse = orientation == 7 || orientation == 8
            lineAxis = vertical
        } else {
            lines = LinePass(lineBytes: toWidth * 3, inputs: sourceHeight, axis: vertical,
                             flip: orientation == 3 || orientation == 4)
            reverse = orientation == 2 || orientation == 3
            lineAxis = horizontal
        }
        output = [UInt8](repeating: 0, count: toWidth * toHeight * 3)
        let longest = max(sourceWidth, sourceHeight, toWidth, toHeight) * 3
        reversed = .allocate(capacity: longest)
        resampled = .allocate(capacity: longest)
    }

    deinit {
        reversed.deallocate()
        resampled.deallocate()
    }

    /// The next source row: `sourceWidth` RGB pixels.
    func push(_ row: UnsafePointer<UInt8>) {
        pushed += 1
        if transposed {
            lines.push(row) { index, line in
                let column = oriented(line, pixels: sourceWidth)
                output.withUnsafeMutableBufferPointer { out in
                    for y in 0..<height {
                        let o = (y * width + index) * 3
                        out[o] = column[y * 3]
                        out[o + 1] = column[y * 3 + 1]
                        out[o + 2] = column[y * 3 + 2]
                    }
                }
            }
        } else {
            let line = oriented(row, pixels: sourceWidth)
            lines.push(line) { index, line in
                output.withUnsafeMutableBufferPointer { out in
                    (out.baseAddress! + index * width * 3).update(from: line, count: width * 3)
                }
            }
        }
    }

    /// The resized image, once every source row has been pushed.
    func finish() -> [UInt8] {
        precondition(pushed == sourceHeight && lines.isDone, "RowResizer: \(pushed) of \(sourceHeight) rows")
        return output
    }

    /// `line` reversed if the orientation flips it, then through `lineAxis` if there is one.
    private func oriented(_ line: UnsafePointer<UInt8>, pixels: Int) -> UnsafePointer<UInt8> {
        var source = line
        if reverse {
            for x in 0..<pixels {
                let s = (pixels - 1 - x) * 3
                reversed[x * 3] = line[s]
                reversed[x * 3 + 1] = line[s + 1]
                reversed[x * 3 + 2] = line[s + 2]
            }
            source = UnsafePointer(reversed)
        }
        guard let axis = lineAxis else { return source }
        ImagePrep.resample(source, axis, into: resampled)
        return UnsafePointer(resampled)
    }
}

/// A resize pass across the lines as they arrive (the vertical pass of torch's kernel, run on a stream): output
/// line i is the weighted sum of input lines start[i] ..< start[i] + count[i]. Only the last `max(count)` lines are
/// kept, which is all a pending output can still need, because the input range of the outputs only moves forward.
final class LinePass {
    let lineBytes: Int
    let inputs: Int
    private let axis: ImagePrep.AxisWeights?
    /// Arrival k is input position inputs - 1 - k, so the outputs complete from the last to the first.
    private let flip: Bool
    private let capacity: Int
    private let ring: UnsafeMutablePointer<UInt8>
    private let sums: UnsafeMutablePointer<Int32>
    private let line: UnsafeMutablePointer<UInt8>
    private var arrived = 0
    private var emitted = 0

    /// `axis` nil: no resize along this axis, every line is passed through.
    init(lineBytes: Int, inputs: Int, axis: ImagePrep.AxisWeights?, flip: Bool) {
        self.lineBytes = lineBytes
        self.inputs = inputs
        self.axis = axis
        self.flip = flip
        capacity = max(axis?.count.max() ?? 0, 1)
        ring = .allocate(capacity: axis == nil ? 1 : capacity * lineBytes)
        sums = .allocate(capacity: axis == nil ? 1 : lineBytes)
        line = .allocate(capacity: axis == nil ? 1 : lineBytes)
    }

    deinit {
        ring.deallocate()
        sums.deallocate()
        line.deallocate()
    }

    var outputs: Int { axis?.start.count ?? inputs }
    var isDone: Bool { emitted == outputs }

    /// Takes the next line and calls `emit(index, line)` for each output line it completes.
    func push(_ input: UnsafePointer<UInt8>, emit: (Int, UnsafePointer<UInt8>) -> Void) {
        let k = arrived
        arrived += 1
        guard let axis else {
            emit(flip ? inputs - 1 - k : k, input)
            emitted += 1
            return
        }
        (ring + (k % capacity) * lineBytes).update(from: input, count: lineBytes)
        let half = Int32(1) << (axis.precision - 1), shift = axis.precision
        axis.weights.withUnsafeBufferPointer { w in
            while emitted < outputs {
                let i = flip ? outputs - 1 - emitted : emitted
                let lo = axis.start[i], size = axis.count[i]
                let last = size == 0 ? -1 : (flip ? inputs - 1 - lo : lo + size - 1)
                guard last <= k else { break }
                for b in 0..<lineBytes { sums[b] = half }
                for j in 0..<size {
                    let position = lo + j, source = flip ? inputs - 1 - position : position
                    let weight = Int32(w[i * axis.stride + j]), row = ring + (source % capacity) * lineBytes
                    for b in 0..<lineBytes { sums[b] += Int32(row[b]) * weight }
                }
                for b in 0..<lineBytes { line[b] = UInt8(clamping: min(max(sums[b] >> shift, 0), 255)) }
                emit(i, line)
                emitted += 1
            }
        }
    }
}
