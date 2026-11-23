#if HARU_ENABLED
struct HaruSpeechOutput {
    enum Frame { case speech, quiet, suppressed }

    // Text leads the spoken audio. One token can cover several syllables and
    // the tail of an utterance can continue through many PAD frames.
    private static let tailFrames = 16
    private static let fadeSamples = 192
    private var remaining = 0
    private var wasOpen = false

    mutating func process(_ pcm: inout [Float], hasText: Bool) -> Frame {
        if hasText { remaining = Self.tailFrames + 1 }
        let isOpen = remaining > 0
        remaining = max(0, remaining - 1)
        let fade = min(Self.fadeSamples, pcm.count)
        guard pcm.allSatisfy(\.isFinite) else {
            pcm = Array(repeating: 0, count: pcm.count)
            wasOpen = false
            return .suppressed
        }

        if isOpen {
            if !wasOpen {
                for i in 0..<fade { pcm[i] *= Float(i + 1) / Float(fade) }
            }
            wasOpen = true
            return .speech
        }

        let audible = pcm.contains { abs($0) > 0.003 }
        if wasOpen {
            for i in 0..<fade { pcm[i] *= Float(fade - i - 1) / Float(fade) }
            for i in fade..<pcm.count { pcm[i] = 0 }
        } else {
            pcm = Array(repeating: 0, count: pcm.count)
        }
        wasOpen = false
        return audible ? .suppressed : .quiet
    }
}
#endif
