#if HARU_ENABLED
import Foundation
import MLX
import MLXNN

// MeanFlow and the duplex frame loop are adapted from higurashi-ai/Haru (Apache-2.0).
final class HaruSession {
    static let frameSamples = 3200
    static let sampleRate: Double = 24_000
    static let maxFrames = 27_000
    static let conversationTemperature: Float = 0.65
    static let conversationEndOfPauseBias: Float = 0.85
    // Official Japanese card template. 自分はハル。相手の名前は未設定。
    // 相手への呼びかけは省いて自然に話す。
    // 質問されたら、まずいま聞かれたことに一言で答え、それから必要なら説明する。
    // 話題が変わったら、相手がいま話したことに合わせて返す。
    private static let japaneseConversation: [Int32] = [
        248045, 8678, 198, 234251, 178046, 108544, 151393, 5205, 247214, 18602,
        247693, 15685, 247395, 74688, 1710, 198, 10586, 183779, 108544, 112308,
        10638, 198, 227603, 149257, 31532, 1710, 148921, 28743, 45220, 15685,
        191871, 114463, 1710, 247359, 15303, 247303, 15303, 184296, 167701, 15685,
        212776, 1710, 198, 10586, 183779, 247252, 10638, 198, 149257, 31532,
        198, 10586, 183779, 96011, 150004, 41004, 10638, 198, 235063, 58852,
        198, 10586, 204684, 247252, 10638, 198, 96449, 112308, 198, 10586,
        168839, 95911, 10638, 198, 206670, 98199, 155200, 120094, 247524, 1710,
        198, 10586, 228260, 10638, 198, 247214, 170872, 200003, 151794, 14876,
        96446, 148814, 96963, 19054, 212776, 1710, 247222, 170521, 31333, 5205,
        159481, 188487, 100246, 190521, 172534, 118333, 15685, 187760, 5205, 222341,
        98174, 150066, 247206, 52020, 1710, 122365, 189404, 160441, 5205, 247214,
        178698, 16223, 98199, 148344, 172534, 161202, 97650, 16711, 1710, 248046,
        198,
    ]
    private static func usesOfficialJapaneseCard(_ ids: [Int32]) -> Bool {
        // Match the persona tokens in the official Japanese model card.
        let persona: [Int32] = [17, 17, 200837, 15685, 96684, 19, 137094, 1710,
                               104764, 14876, 89196, 155727, 1710, 247951]
        return ids.starts(with: japaneseConversation.prefix(22) + persona)
    }

    private let codec: HaruCodec
    private let backbone: HaruBackbone
    private let head: MeanFlow
    private let modelProject: AudioProjector
    private let userProject: AudioProjector
    private let latentScale: MLXArray
    private let latentBias: MLXArray
    private let prompt: [Int32]
    private let temperature: Float
    private let endOfPauseBias: Float
    private let playSpeechOnly: Bool
    private var turnTaking = HaruTurnTaking()
    private var speechOutput = HaruSpeechOutput()
    private let tokenIDs: MLXArray?
    private let pad: Int32 = 248076
    private let eop: Int32 = 248071
    private let eos: Int32 = 248044
    private var tokenRows: [Int32: Int32] = [:]
    private var token: MLXArray?
    private(set) var lastToken: Int32 = 248076
    private(set) var lastSuppressedAudio = false
    private var latent: MLXArray?
    private var quietLatent: MLXArray?
    private var history: [Int32] = []
    private var randomKey: MLXArray
    private lazy var controlRows = MLXArray([row(for: pad), row(for: eop)])
    private lazy var controlEmbeddings = backbone.embedding.rows(controlRows).asType(.float32)
    private lazy var controlLogitBias = MLXArray([Float.zero, endOfPauseBias])
    private lazy var listeningLogitBias = MLXArray([Float(1.75), -Float(1.0)])
    private lazy var pauseRows = MLXArray([row(for: pad), row(for: eos)])
    private lazy var textOnlyRows = MLXArray([row(for: pad), row(for: eop), row(for: eos)])

    init(directory: URL, seed: UInt64 = UInt64.random(in: .min ... .max), promptOverride: [Int32]? = nil,
         instructions: String? = nil,
         compactAcoustics: Bool = false, temperature: Float = 0.7, endOfPauseBias: Float = 0,
         playSpeechOnly: Bool = false) throws {
        // Build the BPE table before allocating the model weights, and keep only the prompt tokens.
        let editedPrompt = try instructions.map { try HaruConversationInstructions.tokenize($0) }
        let weights = try HaruWeights(directory: directory, compactAcoustics: compactAcoustics)
        codec = try HaruCodec(weights)
        backbone = try HaruBackbone(weights)
        head = try MeanFlow(weights)
        modelProject = try AudioProjector(weights, "model_audio_proj")
        userProject = try AudioProjector(weights, "user_audio_proj")
        latentScale = try weights.take("latent_scale", shape: [], dtype: .float32)
        latentBias = try weights.take("latent_bias", shape: [], dtype: .float32)
        if let count = weights.pack.embeddingTokenCount {
            let ids = try weights.take("lm.embed_token_ids", shape: [count], dtype: .int32)
            let values = ids.asArray(Int32.self)
            guard values.allSatisfy({ (0..<248_320).contains($0) }),
                  zip(values, values.dropFirst()).allSatisfy({ $0 < $1 }) else {
                throw HaruError.invalidPack("lm.embed_token_ids")
            }
            tokenIDs = ids
            for (index, id) in values.enumerated() { tokenRows[id] = Int32(index) }
        } else {
            tokenIDs = nil
        }
        let rows = tokenRows
        let useConversationCard = Self.usesOfficialJapaneseCard(weights.pack.prompt)
            && (tokenIDs == nil || Self.japaneseConversation.allSatisfy { rows[$0] != nil })
        if let promptOverride {
            prompt = promptOverride
        } else if let editedPrompt {
            let keep: (Int32) -> Bool = tokenIDs == nil ? { _ in true } : { rows[$0] != nil }
            var ids: [Int32] = []
            for token in editedPrompt { token.append(to: &ids, keeping: keep) }
            prompt = ids
        } else {
            prompt = useConversationCard ? Self.japaneseConversation : weights.pack.prompt
        }
        self.temperature = temperature
        self.endOfPauseBias = endOfPauseBias
        self.playSpeechOnly = playSpeechOnly
        randomKey = MLXRandom.key(seed)
        guard (prompt + [pad, eop, eos]).allSatisfy({ tokenIDs == nil || tokenRows[$0] != nil }) else {
            throw HaruError.invalidPack("prompt")
        }
        try weights.validateComplete()
    }

    func start(reference: [Float], isCancelled: () -> Bool = { false }) throws -> [Float] {
        precondition(!reference.isEmpty && token == nil)
        var hidden: MLXArray = MLXArray.zeros([1, 1, 2560], dtype: .bfloat16)
        // Bound the peak size of the prefill activations.
        let prefillChunk = 32
        for start in stride(from: 0, to: prompt.count, by: prefillChunk) {
            if isCancelled() { throw CancellationError() }
            let ids = MLXArray(prompt[start..<min(start + prefillChunk, prompt.count)].map(row(for:))).reshaped(1, -1)
            hidden = backbone.forward(backbone.embedding.rows(ids))
            eval([hidden] + backbone.state)
        }
        let frames = (reference.count + Self.frameSamples - 1) / Self.frameSamples
        var means: [MLXArray] = []
        for offset in stride(from: 0, to: reference.count, by: Self.frameSamples) {
            if isCancelled() { throw CancellationError() }
            let end = min(offset + Self.frameSamples, reference.count)
            let samples = MLXArray(Array(reference[offset..<end])).reshaped(-1, 1)
            let mean = codec.encoder.run(samples)
            eval([mean] + codec.encoder.state)
            means.append(mean)
        }
        if isCancelled() { throw CancellationError() }
        let voice = sampleLatent(concatenated(means, axis: 0))
        let ids = MLXArray(Array(repeating: row(for: pad), count: frames - 1) + [row(for: eop)]).reshaped(1, -1)
        let embeddings = backbone.embedding.rows(ids) + modelProject(scale(voice)).expandedDimensions(axis: 0)
        for start in stride(from: 0, to: frames, by: prefillChunk) {
            if isCancelled() { throw CancellationError() }
            hidden = backbone.forward(embeddings[0..., start..<min(start + prefillChunk, frames), 0...])
            eval([hidden] + backbone.state)
        }
        codec.encoder.reset()
        Memory.clearCache()
        return emit(hidden[0..., -1, 0...])
    }

    // Compile the real-time shapes before capture without consuming a user frame.
    func warmUp() {
        let savedToken = token
        let savedID = lastToken
        let savedLatent = latent
        let savedQuietLatent = quietLatent
        let savedHistory = history
        let savedKey = randomKey
        let savedTurnTaking = turnTaking
        let savedSpeechOutput = speechOutput
        let savedSuppressedAudio = lastSuppressedAudio
        let restoreEncoder = codec.encoder.saveState()
        let restoreDecoder = codec.decoder.saveState()
        defer {
            token = savedToken
            lastToken = savedID
            latent = savedLatent
            quietLatent = savedQuietLatent
            history = savedHistory
            randomKey = savedKey
            turnTaking = savedTurnTaking
            speechOutput = savedSpeechOutput
            lastSuppressedAudio = savedSuppressedAudio
            restoreEncoder()
            restoreDecoder()
        }
        backbone.withTemporaryState {
            _ = step(Array(repeating: 0, count: Self.frameSamples))
        }
    }

    func resetTurnTaking() {
        turnTaking = HaruTurnTaking()
    }

    // Unit boundaries are synchronized only in the standalone profiler.
    func step(_ pcm: [Float], measure: ((String) -> Void)? = nil) -> [Float] {
        precondition(pcm.count == Self.frameSamples && token != nil && latent != nil)
        let turn = endOfPauseBias > 0 ? turnTaking.observe(pcm) : .normal
        let mean = codec.encoder.run(MLXArray(pcm).reshaped(-1, 1))
        if let measure {
            eval([mean] + codec.encoder.state)
            measure("encoder")
        }
        let user = sampleLatent(mean)
        // During overlapping speech, give the current input more influence
        // than the model's own previous audio.
        let previousText = backbone.embedding.rows(token!)
        let previousVoice = modelProject(scale(latent!))
        let currentVoice = userProject(scale(user))
        let embedding = turnTaking.isUserSpeaking
            ? previousText + previousVoice * 0.55 + currentVoice * 1.20
            : previousText + previousVoice + currentVoice
        let hidden = backbone.forward(embedding.reshaped(1, 1, 2560))
        if let measure {
            eval([hidden] + backbone.state)
            measure("backbone")
        }
        return emit(hidden[0..., 0, 0...], turn: turn, measure: measure)
    }

    private func row(for id: Int32) -> Int32 { tokenRows[id] ?? id }

    private func scale(_ x: MLXArray) -> MLXArray { ((x + latentBias) * latentScale).asType(.bfloat16) }

    private func nextKey() -> MLXArray {
        let (next, key) = MLXRandom.split(key: randomKey)
        randomKey = next
        return key
    }

    private func sampleLatent(_ mean: MLXArray) -> MLXArray {
        let (scaleKey, noiseKey) = MLXRandom.split(key: nextKey())
        let scale = MLXRandom.normal([1, 1, 1], key: scaleKey)
        let noise = MLXRandom.normal([1] + mean.shape, key: noiseKey)
        return (mean.expandedDimensions(axis: 0) + scale * 0.625 * noise)[0]
    }

    private func emit(_ hidden: MLXArray, turn: HaruTurnTaking.Cue = .normal, measure: ((String) -> Void)? = nil) -> [Float] {
        let logits = backbone.embedding(hidden).asType(.float32).reshaped(-1)
        let penalties = repetitionPenalties()
        if !penalties.isEmpty {
            let ids = MLXArray(penalties.map { row(for: $0.0) })
            logits[ids] = logits[ids] - MLXArray(penalties.map(\.1))
        }
        let controlLogits = matmul(hidden.asType(.float32), controlEmbeddings.T).reshaped(-1)
        logits[controlRows] = controlLogits + (turnTaking.isUserSpeaking ? listeningLogitBias : controlLogitBias)
        switch turn {
        case .normal: break
        case .start: logits[pauseRows] = MLXArray(-Float.infinity)
        case .continueReply: logits[textOnlyRows] = MLXArray(-Float.infinity)
        }
        let scores = logits / temperature
        let top = argPartition(-scores, kth: 24)[0..<25]
        let pick = MLXRandom.categorical(scores[top].reshaped(1, 25), key: nextKey())
        token = top[pick].asType(.int32).reshaped(1)
        if let measure {
            eval(token!)
            measure("text")
        }
        let noise = MLXRandom.normal([1, 64], key: nextKey()).asType(.bfloat16)
        let audioLatent = head.sample(hidden, noise).asType(.float32) / latentScale - latentBias
        latent = audioLatent
        if let measure {
            eval(audioLatent)
            measure("head")
        }
        let pcm = clip(codec.decoder.run(audioLatent).reshaped(-1), min: -1, max: 1)
        let sampledID = tokenIDs?.take(token!, axis: 0) ?? token!
        eval([sampledID, pcm, audioLatent] + codec.encoder.state + codec.decoder.state + backbone.state)
        measure?("decoder")
        let id = sampledID.item(Int32.self)
        lastToken = id
        if id != pad, id != eop, id != eos {
            history.append(id)
            if history.count > 1057 { history.removeFirst() }
        }
        var samples = pcm.asArray(Float.self)
        lastSuppressedAudio = false
        let isText = id < eos
        if playSpeechOnly {
            switch speechOutput.process(&samples, hasText: isText) {
            case .speech: break
            case .quiet: quietLatent = audioLatent
            case .suppressed:
                lastSuppressedAudio = true
                // Avoid conditioning the following frame on the sound we just rejected.
                if let quietLatent { latent = quietLatent }
            }
        }
        if endOfPauseBias > 0 {
            turnTaking.didGenerate(samples, isText: isText, isEndOfPause: id == eop)
        }
        return samples
    }

    private func repetitionPenalties() -> [(Int32, Float)] {
        guard let last = history.last, history.count > 1 else { return [] }
        let end = history.count - 1
        var longest: [Int32: Int] = [:]
        for index in max(0, end - 1024)..<end where history[index] == last {
            var length = 1
            while length < 32 && index >= length && history[index - length] == history[end - length] { length += 1 }
            let id = history[index + 1]
            longest[id] = max(longest[id] ?? 0, length)
        }
        return longest.map { ($0.key, min(32, 0.8 * pow(1.75, Float($0.value - 1)))) }
    }
}

private struct AudioProjector {
    let fc1: HaruMatrix
    let norm: MLXArray
    let fc2: HaruMatrix

    init(_ weights: HaruWeights, _ name: String) throws {
        fc1 = try weights.dense(name + ".fc1", input: 64, output: 2560, bias: true)
        norm = try weights.take(name + ".norm.weight", shape: [2560])
        fc2 = try weights.dense(name + ".fc2", input: 2560, output: 2560, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(MLXFast.rmsNorm(fc1(x), weight: norm, eps: 1e-6))
    }
}

private final class MeanFlow {
    let latent: HaruMatrix
    let condition: HaruMatrix
    let modulation: HaruMatrix
    let final: HaruMatrix
    let layers: [Layer]
    let scalarConditions: [MLXArray]
    let unitGain = MLXArray.ones([2560], dtype: .float32)
    private lazy var sampler = compile { [unowned self] (hidden: MLXArray, noise: MLXArray) in
        let conditioned = self.scalarConditions.reduce(self.condition(hidden), +)
        let mods = self.modulation(silu(conditioned)).split(indices: [7680, 15360, 23040, 30720], axis: -1)
        var x = self.latent(noise)
        for (layer, parameters) in zip(self.layers, mods) {
            let parts = parameters.split(parts: 3, axis: -1)
            let normalized = haruNorm(x, self.unitGain, eps: 1e-5) * layer.norm
            let h = normalized * (1 + parts[1]) + parts[0]
            x = x + parts[2] * layer.down(silu(layer.gate(h)) * layer.up(h))
        }
        let (shift, scale) = mods[4].split(axis: -1)
        return noise + self.final(haruNorm(x, self.unitGain, eps: 1e-5) * (1 + scale) + shift)
    }

    init(_ w: HaruWeights) throws {
        latent = try w.dense("head.latent_proj", input: 64, output: 2560)
        condition = try w.dense("head.cond_proj", input: 2560, output: 2560)
        modulation = try w.dense("head.ada", input: 2560, output: 35840)
        final = try w.dense("head.final.proj", input: 2560, output: 64)
        layers = try (0..<4).map { try Layer(w, "head.layers.\($0)") }
        scalarConditions = try [("t_embed", Float(0)), ("dt_embed", 7), ("h_embed", 1), ("omega_embed", 1 - 1 / 1.5)].map { name, value in
            let fc1 = try w.dense("head.\(name).fc1", input: 256, output: 2560)
            let fc2 = try w.dense("head.\(name).fc2", input: 2560, output: 2560)
            let freqs = MLXArray((0..<128).map { exp(-log(Float(10_000)) * Float($0) / 128) })
            let args = MLXArray(value).asType(.bfloat16).asType(.float32) * freqs
            let features = concatenated([cos(args), sin(args)], axis: 0).asType(.bfloat16).reshaped(1, 256)
            let embedded = fc2(silu(fc1(features)))
            eval(embedded)
            return embedded
        }
    }

    func sample(_ hidden: MLXArray, _ noise: MLXArray) -> MLXArray { sampler(hidden, noise) }

    struct Layer {
        let norm: MLXArray
        let gate: HaruMatrix
        let up: HaruMatrix
        let down: HaruMatrix

        init(_ w: HaruWeights, _ p: String) throws {
            norm = try w.take(p + ".norm.weight", shape: [2560])
            gate = try w.dense(p + ".ffn.gate_proj", input: 2560, output: 7680)
            up = try w.dense(p + ".ffn.up_proj", input: 2560, output: 7680)
            down = try w.dense(p + ".ffn.down_proj", input: 7680, output: 2560)
        }
    }
}
#endif
