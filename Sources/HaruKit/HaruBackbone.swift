#if HARU_ENABLED
import Foundation
import MLX
import MLXNN

// Qwen3.5 layers and recurrent kernel adapted from higurashi-ai/Haru (Apache-2.0).
final class HaruBackbone {
    let embedding: HaruMatrix
    let layers: [Layer]
    let norm: MLXArray
    private(set) var position = 0

    init(_ weights: HaruWeights) throws {
        embedding = try weights.embedding()
        layers = try (0..<32).map { try Layer(weights, index: $0) }
        norm = try weights.take("lm.norm.weight", shape: [2560], dtype: .float32)
    }

    func forward(_ embeddings: MLXArray) -> MLXArray {
        var hidden = embeddings
        let growingCache = embeddings.dim(1) == 1 && position.isMultiple(of: Self.cacheBlock)
        for (index, layer) in layers.enumerated() {
            hidden = layer.forward(hidden, offset: position)
            if growingCache && index % 4 == 3 {
                // Materialize cache growth by groups of layers so the old and
                // new buffers for all eight attention layers do not overlap.
                eval([hidden] + layers[(index - 3)...index].flatMap(\.state))
            }
        }
        position += embeddings.dim(1)
        return haruNorm(hidden, norm)
    }

    var state: [MLXArray] { layers.flatMap(\.state) }
    private static let cacheBlock = 64

    func withTemporaryState(_ body: () -> Void) {
        let savedPosition = position
        let restore = layers.map { $0.saveState() }
        defer {
            restore.forEach { $0() }
            position = savedPosition
        }
        body()
    }

    final class Layer {
        private let inputNorm: MLXArray
        private let postNorm: MLXArray
        private let gateUp: HaruMatrix
        private let down: HaruMatrix
        private let delta: Delta?
        private let attention: Attention?
        private var convolution: MLXArray?
        private var memory: MLXArray?
        private var keys: MLXArray?
        private var values: MLXArray?
        private lazy var deltaStep = compile { [unowned self] (args: [MLXArray]) -> [MLXArray] in
            let (projected, convolution, memory) = self.delta!(haruNorm(args[0], self.inputNorm), args[1], args[2])
            return [self.feedForward(args[0] + projected), convolution, memory]
        }
        private lazy var attentionProjection = compile { [unowned self] x in
            self.attention!.project(haruNorm(x, self.inputNorm))
        }
        private lazy var attentionFinish = compile { [unowned self] (args: [MLXArray]) -> [MLXArray] in
            [self.finishAttention(args[0], mix: args[1], gate: args[2], attention: self.attention!)]
        }
        var state: [MLXArray] { [convolution, memory, keys, values].compactMap { $0 } }

        func saveState() -> () -> Void {
            let convolution = self.convolution
            let memory = self.memory
            // Subscript writes replace the array's MLX handle. Use a separate
            // handle so the saved attention cache keeps its original contents.
            let keys = self.keys.map { $0.reshaped($0.shape) }
            let values = self.values.map { $0.reshaped($0.shape) }
            return {
                self.convolution = convolution
                self.memory = memory
                self.keys = keys
                self.values = values
            }
        }

        init(_ w: HaruWeights, index: Int) throws {
            let p = "lm.layers.\(index)"
            inputNorm = try w.take(p + ".input_layernorm.weight", shape: [2560], dtype: .float32)
            postNorm = try w.take(p + ".post_attention_layernorm.weight", shape: [2560], dtype: .float32)
            gateUp = try w.quantized(p + ".mlp.gate_up.weight", input: 2560, output: 18432)
            down = try w.quantized(p + ".mlp.down_proj.weight", input: 9216, output: 2560)
            delta = index % 4 == 3 ? nil : try Delta(w, p + ".linear_attn")
            attention = index % 4 == 3 ? try Attention(w, p + ".self_attn") : nil
        }

        func forward(_ x: MLXArray, offset: Int) -> MLXArray {
            if let delta {
                let conv = convolution ?? MLXArray.zeros([1, 3, 8192], dtype: x.dtype)
                let mem = memory ?? MLXArray.zeros([1, 32, 128, 128], dtype: .float32)
                if x.dim(1) == 1 {
                    let output = deltaStep([x, conv, mem])
                    convolution = output[1]
                    memory = output[2]
                    return output[0]
                }
                let (output, newConv, newMem) = delta(haruNorm(x, inputNorm), conv, mem)
                convolution = newConv
                memory = newMem
                return feedForward(x + output)
            }
            guard let attention else { preconditionFailure("Missing attention layer") }
            let single = x.dim(1) == 1
            let (query, key, value, gate) = single ? attentionProjection(x) : attention.project(haruNorm(x, inputNorm))
            let end = offset + key.dim(2)
            let capacity = keys?.dim(2) ?? 0
            if end > capacity {
                let block = HaruBackbone.cacheBlock
                let extra = ((end - capacity + block - 1) / block) * block
                let padding = MLXArray.zeros([1, 4, extra, 256], dtype: x.dtype)
                keys = keys.map { concatenated([$0, padding], axis: 2) } ?? padding
                values = values.map { concatenated([$0, padding], axis: 2) }
                    ?? MLXArray.zeros([1, 4, extra, 256], dtype: x.dtype)
            }
            keys![0..., 0..., offset..<end, 0...] = attention.rotate(key, offset: offset)
            values![0..., 0..., offset..<end, 0...] = value
            let mix = MLXFast.scaledDotProductAttention(
                queries: attention.rotate(query, offset: offset),
                keys: keys![0..., 0..., 0..<end, 0...], values: values![0..., 0..., 0..<end, 0...],
                scale: 1 / 16, mask: single ? .none : .causal)
            return single ? attentionFinish([x, mix, gate])[0] : finishAttention(x, mix: mix, gate: gate, attention: attention)
        }

        private func finishAttention(_ x: MLXArray, mix: MLXArray, gate: MLXArray, attention: Attention) -> MLXArray {
            let output = attention.output(mix.transposed(0, 2, 1, 3).reshaped(1, x.dim(1), 4096) * sigmoid(gate))
            return feedForward(x + output)
        }

        private func feedForward(_ x: MLXArray) -> MLXArray {
            let (gate, up) = gateUp(haruNorm(x, postNorm)).split(axis: -1)
            return x + down(silu(gate) * up)
        }
    }

    private struct Attention {
        let qkv: HaruMatrix
        let output: HaruMatrix
        let queryNorm: MLXArray
        let keyNorm: MLXArray

        init(_ w: HaruWeights, _ p: String) throws {
            qkv = try w.quantized(p + ".qkv.weight", input: 2560, output: 10240)
            output = try w.quantized(p + ".o_proj.weight", input: 4096, output: 2560)
            queryNorm = try w.take(p + ".q_norm.weight", shape: [256], dtype: .float32)
            keyNorm = try w.take(p + ".k_norm.weight", shape: [256], dtype: .float32)
        }

        func project(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
            let n = x.dim(1)
            let parts = qkv(x).split(indices: [8192, 9216], axis: -1)
            let (query, gate) = parts[0].reshaped(1, n, 16, 512).split(axis: -1)
            return (haruNorm(query, queryNorm).transposed(0, 2, 1, 3),
                    haruNorm(parts[1].reshaped(1, n, 4, 256), keyNorm).transposed(0, 2, 1, 3),
                    parts[2].reshaped(1, n, 4, 256).transposed(0, 2, 1, 3), gate.reshaped(1, n, 4096))
        }

        func rotate(_ input: MLXArray, offset: Int) -> MLXArray {
            MLXFast.RoPE(input, dimensions: 64, traditional: false, base: 10_000_000, scale: 1, offset: offset)
        }
    }

    private struct Delta {
        let input: HaruMatrix
        let output: HaruMatrix
        let conv: MLXArray
        let decayLog: MLXArray
        let decayBias: MLXArray
        let norm: MLXArray

        init(_ w: HaruWeights, _ p: String) throws {
            input = try w.quantized(p + ".in_proj.weight", input: 2560, output: 12352)
            output = try w.quantized(p + ".out_proj.weight", input: 4096, output: 2560)
            conv = try w.take(p + ".conv1d.weight", shape: [8192, 4, 1])
            decayLog = try w.take(p + ".A_log", shape: [32])
            decayBias = try w.take(p + ".dt_bias", shape: [32])
            norm = try w.take(p + ".norm.weight", shape: [128])
        }

        func callAsFunction(_ x: MLXArray, _ convolution: MLXArray, _ memory: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
            let n = x.dim(1)
            let proj = input(x).split(indices: [8192, 12288, 12320], axis: -1)
            let context = concatenated([convolution, proj[0]], axis: 1)
            let qkv = silu(conv1d(context, conv, groups: 8192)).split(indices: [2048, 4096], axis: -1)
            let epsilon: Float = 1e-6 / 128
            let q = (1.0 / 128) * MLXFast.rmsNorm(qkv[0].reshaped(1, n, 16, 128), weight: .mlxNone, eps: epsilon)
            let k = (1.0 / sqrt(128.0)) * MLXFast.rmsNorm(qkv[1].reshaped(1, n, 16, 128), weight: .mlxNone, eps: epsilon)
            let v = qkv[2].reshaped(1, n, 32, 128)
            let g = exp(-exp(decayLog.asType(.float32)) * softplus(proj[3].asType(.float32) + decayBias))
            let beta = sigmoid(proj[2])
            let recurrence = Self.kernel([q, k, v, g, beta, memory, n],
                template: [("InT", q.dtype)], grid: (32, 128, 32), threadGroup: (32, 4, 1),
                outputShapes: [[1, n, 32, 128], memory.shape], outputDTypes: [q.dtype, .float32])
            let y = MLXFast.rmsNorm(recurrence[0], weight: norm, eps: 1e-6).asType(.float32)
            let gate = silu(proj[1].reshaped(1, n, 32, 128).asType(.float32))
            return (output((y * gate).asType(x.dtype).reshaped(1, n, 4096)), context[0..., n..., 0...], recurrence[1])
        }

        private static let kernel = MLXFast.metalKernel(name: "haru_gated_delta",
            inputNames: ["q", "k", "v", "decay", "beta", "old_memory", "T"], outputNames: ["y", "memory"], source: """
            uint head = thread_position_in_grid.z;
            uint key_head = head / 2;
            uint row = thread_position_in_grid.y;
            uint column = thread_position_in_threadgroup.x * 4;
            uint offset = (head * 128 + row) * 128 + column;
            float state[4];
            for (uint j = 0; j < 4; ++j) state[j] = old_memory[offset + j];
            for (uint t = 0; t < uint(T); ++t) {
                uint key_offset = (t * 16 + key_head) * 128 + column;
                uint value_offset = (t * 32 + head) * 128 + row;
                float prediction = 0.0f;
                for (uint j = 0; j < 4; ++j) {
                    state[j] *= decay[t * 32 + head];
                    prediction += state[j] * float(k[key_offset + j]);
                }
                prediction = simd_sum(prediction);
                float delta = (float(v[value_offset]) - prediction) * float(beta[t * 32 + head]);
                float output = 0.0f;
                for (uint j = 0; j < 4; ++j) {
                    state[j] += float(k[key_offset + j]) * delta;
                    output += state[j] * float(q[key_offset + j]);
                }
                output = simd_sum(output);
                if (thread_index_in_simdgroup == 0) y[value_offset] = static_cast<InT>(output);
            }
            for (uint j = 0; j < 4; ++j) memory[offset + j] = state[j];
            """)
    }
}
#endif
