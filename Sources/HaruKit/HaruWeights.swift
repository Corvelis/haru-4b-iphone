#if HARU_ENABLED
import Foundation
import MLX
import MLXNN

// Model layout adapted from higurashi-ai/Haru, Apache-2.0. See NOTICE.
public enum HaruError: LocalizedError {
    case invalidPack(String)
    case microphone
    case audioFormat
    case interrupted
    case backgrounded
    case noInput
    case referenceAudio
    case modelDownloadInProgress
    case conversationInProgress

    public var errorDescription: String? {
        switch self {
        case .invalidPack(let detail): return "Haru-4Bのモデルを読み込めません（\(detail)）"
        case .microphone: return "マイクへのアクセスを許可してください。"
        case .audioFormat: return "音声入出力を開始できませんでした。"
        case .interrupted: return "iPhoneの音声接続が中断されました。マイクをオンにして再開できます。"
        case .backgrounded: return "アプリがバックグラウンドに移動したため、マイクをオフにしました。"
        case .noInput: return "マイクから音声を受け取れませんでした。接続を確認してもう一度オンにしてください。"
        case .referenceAudio: return "話し声が入った、数秒の音声を選んでください。"
        case .modelDownloadInProgress: return "モデルのダウンロードがすでに進行中です。"
        case .conversationInProgress: return "先に現在のHaru-4Bの会話を終了してください。"
        }
    }
}

struct HaruPack: Codable {
    let format: Int
    let bits: Int
    let groupSize: Int
    let precisionOverrides: [String: Int]?
    let embeddingTokenCount: Int?
    let prompt: [Int32]
    let shards: [String]
    let revision: String

    init(at directory: URL) throws {
        self = try JSONDecoder().decode(Self.self, from: Data(contentsOf: directory.appendingPathComponent("pack.json")))
        let tokenCount = embeddingTokenCount ?? 248_320
        guard format == 1, [4, 5, 6, 8].contains(bits), groupSize == 64,
              (precisionOverrides ?? [:]).values.allSatisfy({ [4, 5, 6, 8].contains($0) }),
              revision == "8d6e83bcd3701a76fc27a764965665da1df7da78",
              !prompt.isEmpty, prompt.count <= 512,
              prompt.allSatisfy({ (0..<248_320).contains($0) }),
              (25...248_320).contains(tokenCount),
              !shards.isEmpty, shards.count <= 128, Set(shards).count == shards.count,
              shards.allSatisfy({ $0.range(of: #"^weights-[0-9]{3}\.safetensors$"#, options: .regularExpression) != nil })
        else { throw HaruError.invalidPack("pack.json") }
    }
}

final class HaruWeights {
    private var tensors: [String: MLXArray] = [:]
    private var compacted: Set<String> = []
    let pack: HaruPack

    init(directory: URL, compactAcoustics: Bool) throws {
        pack = try HaruPack(at: directory)
        for name in pack.shards {
            try autoreleasepool {
                let shard = try loadArrays(url: directory.appendingPathComponent(name))
                for (name, tensor) in shard {
                    guard tensors[name] == nil else { throw HaruError.invalidPack(name) }
                    if compactAcoustics && pack.precisionOverrides?[name] == nil && Self.canCompact(name, tensor) {
                        guard tensors[name + ".scales"] == nil, tensors[name + ".biases"] == nil else {
                            throw HaruError.invalidPack(name)
                        }
                        let (values, scales, biases) = MLX.quantized(tensor, groupSize: pack.groupSize, bits: 8)
                        guard let biases else { throw HaruError.invalidPack(name) }
                        eval(values, scales, biases)
                        tensors[name] = values
                        tensors[name + ".scales"] = scales
                        tensors[name + ".biases"] = biases
                        compacted.insert(name)
                    } else {
                        tensors[name] = tensor
                    }
                }
            }
        }
    }

    private static func canCompact(_ name: String, _ tensor: MLXArray) -> Bool {
        (name.hasPrefix("head.") || name.hasPrefix("codec.decoder.")) && name.hasSuffix(".weight")
            && tensor.dtype == .bfloat16 && tensor.ndim == 2 && tensor.dim(-1) >= 1024
            && tensor.dim(-1).isMultiple(of: 64) && tensor.size >= 2_000_000
    }

    func take(_ name: String, shape: [Int], dtype: DType = .bfloat16) throws -> MLXArray {
        guard let tensor = tensors.removeValue(forKey: name), tensor.shape == shape, tensor.dtype == dtype else {
            throw HaruError.invalidPack(name)
        }
        return tensor
    }

    func embedding() throws -> HaruMatrix {
        try quantized("lm.embed_tokens.weight", input: 2560, output: pack.embeddingTokenCount ?? 248_320)
    }

    func dense(_ name: String, input: Int, output: Int, bias: Bool = false) throws -> HaruMatrix {
        let offset = bias ? try take(name + ".bias", shape: [output]) : nil
        let key = name + ".weight"
        if pack.precisionOverrides?[key] != nil || compacted.contains(key) {
            var matrix = try quantized(key, input: input, output: output)
            matrix.bias = offset
            return matrix
        }
        return try HaruMatrix(weight: take(key, shape: [output, input]), bias: offset)
    }

    func quantized(_ name: String, input: Int, output: Int) throws -> HaruMatrix {
        let bits = compacted.contains(name) ? 8 : pack.precisionOverrides?[name] ?? pack.bits
        let scaleShape = [output, input / pack.groupSize]
        return try HaruMatrix(
            weight: take(name, shape: [output, input * bits / 32], dtype: .uint32),
            scales: take(name + ".scales", shape: scaleShape),
            offsets: take(name + ".biases", shape: scaleShape), bits: bits, groupSize: pack.groupSize)
    }

    func validateComplete() throws {
        if let extra = tensors.keys.first { throw HaruError.invalidPack(extra) }
    }
}

struct HaruMatrix {
    let weight: MLXArray
    var bias: MLXArray? = nil
    var scales: MLXArray? = nil
    var offsets: MLXArray? = nil
    var bits = 8
    var groupSize = 64

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let scales {
            let output = quantizedMM(x, weight, scales: scales, biases: offsets, groupSize: groupSize, bits: bits)
            return bias.map { output + $0 } ?? output
        }
        if let bias { return addMM(bias, x, weight.T) }
        return matmul(x, weight.T)
    }

    func rows(_ ids: MLXArray) -> MLXArray {
        let rows = weight.take(ids, axis: 0)
        guard let scales else { return rows }
        return dequantized(rows, scales: scales.take(ids, axis: 0), biases: offsets?.take(ids, axis: 0),
                           groupSize: groupSize, bits: bits)
    }
}

func haruNorm(_ x: MLXArray, _ weight: MLXArray, eps: Float = 1e-6) -> MLXArray {
    MLXFast.rmsNorm(x.asType(.float32), weight: weight.asType(.float32), eps: eps).asType(x.dtype)
}
#endif
