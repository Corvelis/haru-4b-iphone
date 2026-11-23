#if HARU_ENABLED
import MLX
import MLXNN

// Causal VibeVoice codec adapted from higurashi-ai/Haru (Apache-2.0).
final class HaruCodec {
    let encoder: Stream
    let decoder: Stream

    init(_ w: HaruWeights) throws {
        let strides = [8, 5, 5, 4, 2, 2]
        var encode = [try Layer(w, "codec.encoder.stem", input: 1, output: 32, stride: 1)]
        for (index, depth) in [3, 3, 3, 3, 3, 3, 8].enumerated() {
            let channels = 32 << index
            if index > 0 {
                encode.append(try Layer(w, "codec.encoder.down.\(index - 1)", input: channels / 2,
                                        output: channels, stride: strides[6 - index]))
            }
            for block in 0..<depth {
                encode.append(try Layer(w, "codec.encoder.stages.\(index).\(block)", channels: channels))
            }
        }
        encode.append(try Layer(w, "codec.encoder.head", input: 2048, output: 64, stride: 1))
        encoder = Stream(encode)

        var decode = [try Layer(w, "codec.decoder.stem", input: 64, output: 2048, stride: 1)]
        for (index, depth) in [8, 3, 3, 3, 3, 3, 3].enumerated() {
            let channels = 2048 >> index
            if index > 0 {
                decode.append(try Layer(w, "codec.decoder.up.\(index - 1)", input: channels * 2,
                                        output: channels, stride: strides[index - 1], transposed: true))
            }
            for block in 0..<depth {
                decode.append(try Layer(w, "codec.decoder.stages.\(index).\(block)", channels: channels))
            }
        }
        decode.append(try Layer(w, "codec.decoder.head", input: 32, output: 1, stride: 1))
        decoder = Stream(decode)
    }

    final class Stream {
        private let layers: [Layer]
        private(set) var state: [MLXArray]
        private lazy var forward = compile { [layers] (inputs: [MLXArray]) -> [MLXArray] in
            var x = inputs[0]
            var states: [MLXArray] = []
            for (layer, state) in zip(layers, inputs.dropFirst()) {
                let (output, next) = layer(x, state)
                x = output
                states.append(next)
            }
            return [x] + states
        }

        fileprivate init(_ layers: [Layer]) {
            self.layers = layers
            state = layers.map { MLXArray.zeros($0.stateShape, dtype: .bfloat16) }
        }

        func reset() {
            state = layers.map { MLXArray.zeros($0.stateShape, dtype: .bfloat16) }
        }

        func saveState() -> () -> Void {
            let state = self.state
            return { self.state = state }
        }

        func run(_ input: MLXArray) -> MLXArray {
            let output = forward([input.asType(.bfloat16)] + state)
            state = Array(output.dropFirst())
            return output[0].asType(.float32)
        }
    }

    fileprivate struct Layer {
        let stateShape: [Int]
        let run: (MLXArray, MLXArray) -> (MLXArray, MLXArray)

        func callAsFunction(_ x: MLXArray, _ state: MLXArray) -> (MLXArray, MLXArray) { run(x, state) }

        init(_ w: HaruWeights, _ name: String, input: Int, output: Int, stride: Int, transposed: Bool = false) throws {
            let kernel = stride == 1 ? 7 : 2 * stride
            if transposed {
                let matrix = try w.dense(name, input: input, output: output * kernel)
                let bias = try w.take(name + ".bias", shape: [output])
                stateShape = [stride, output]
                run = { x, state in
                    let n = x.dim(0)
                    let windows = matrix(x).reshaped(n, output, 2, stride).transposed(0, 2, 3, 1)
                    let first = windows[0..., 0]
                    let second = windows[0..., 1]
                    let overlap = n == 1 ? state.expandedDimensions(axis: 0)
                        : concatenated([state.expandedDimensions(axis: 0), second[0..<(n - 1)]], axis: 0)
                    return ((first + overlap).reshaped(n * stride, output) + bias, second[n - 1])
                }
            } else {
                let matrix = try w.dense(name, input: input * kernel, output: output, bias: true)
                let contextSize = kernel - stride
                stateShape = [contextSize, input]
                run = { x, state in
                    let remainder = x.dim(0) % stride
                    let audio = remainder == 0 ? x : concatenated([x, MLXArray.zeros([stride - remainder, input], dtype: x.dtype)], axis: 0)
                    let n = audio.dim(0)
                    let context = concatenated([state, audio], axis: 0)
                    let windows: MLXArray
                    if stride == 1 {
                        windows = concatenated((0..<kernel).map { context[$0..<($0 + n)] }, axis: 1)
                    } else {
                        let hops = context.reshaped(-1, stride * input)
                        windows = concatenated([hops[0..<(hops.dim(0) - 1)], hops[1...]], axis: 1)
                    }
                    return (matrix(windows), context[n...])
                }
            }
        }

        init(_ w: HaruWeights, _ name: String, channels: Int) throws {
            let norm = try w.take(name + ".norm.weight", shape: [channels])
            let conv = try w.take(name + ".conv.weight", shape: [channels, 7])
            let bias = try w.take(name + ".conv.bias", shape: [channels])
            let gamma = try w.take(name + ".gamma", shape: [channels])
            let ffnNorm = try w.take(name + ".ffn_norm.weight", shape: [channels])
            let fc1 = try w.dense(name + ".fc1", input: channels, output: 4 * channels, bias: true)
            let fc2 = try w.dense(name + ".fc2", input: 4 * channels, output: channels, bias: true)
            let ffnGamma = try w.take(name + ".ffn_gamma", shape: [channels])
            stateShape = [6, channels]
            run = { x, state in
                let n = x.dim(0)
                let normalized = MLXFast.rmsNorm(x, weight: norm, eps: 1e-5)
                let mixed = Self.mixer([x, normalized, state, conv, bias, gamma],
                    template: [("OutT", x.dtype), ("T", n), ("C", channels)],
                    grid: (channels, n + 6, 1), threadGroup: (min(256, channels), 1, 1),
                    outputShapes: [[n, channels], [6, channels]], outputDTypes: [x.dtype, state.dtype])
                let h = MLXFast.rmsNorm(mixed[0], weight: ffnNorm, eps: 1e-5)
                return (mixed[0] + ffnGamma * fc2(gelu(fc1(h))), mixed[1])
            }
        }

        private static let mixer = MLXFast.metalKernel(name: "haru_causal_mixer",
            inputNames: ["x", "n", "state", "w", "bias", "gamma"], outputNames: ["out", "new_state"], source: """
            uint c = thread_position_in_grid.x;
            uint r = thread_position_in_grid.y;
            if (c >= C || r >= T + 6) return;
            if (r < T) {
                float acc = float(bias[c]);
                for (int k = 0; k < 7; ++k) {
                    int i = int(r) + k - 6;
                    float v = i < 0 ? float(state[(i + 6) * C + c]) : float(n[i * C + c]);
                    acc += float(w[c * 7 + k]) * v;
                }
                out[r * C + c] = static_cast<OutT>(float(x[r * C + c]) + float(gamma[c]) * acc);
            } else {
                new_state[(r - T) * C + c] = r < 6 ? state[r * C + c] : n[(r - 6) * C + c];
            }
            """)
    }
}
#endif
