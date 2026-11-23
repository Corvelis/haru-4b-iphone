#if HARU_ENABLED
struct HaruTurnTaking {
    enum Cue { case normal, start, continueReply }

    // Counts are in 133 ms model frames. All nudges are bounded to the first
    // two seconds of a pause, so room silence cannot keep restarting replies.
    private static let initialWaitFrames = 5
    private static let retryWaitFrames = 12
    private static let replyWindowFrames = 16

    private var userVoice = 0
    private var userQuiet = 0
    private var replyVoice = 0
    private var sustainedText = 0
    private var replyQuiet = 0
    private var startAttempts = 0
    private var awaitingText = false
    private var continuationAttempts = 0

    var isUserSpeaking: Bool { userVoice >= 2 && userQuiet < 2 }

    mutating func observe(_ pcm: [Float]) -> Cue {
        let voiced = pcm.reduce(Float.zero) { $0 + $1 * $1 } >= 0.000064 * Float(pcm.count)
        if voiced {
            if userQuiet >= 4 { userVoice = 0 }
            userQuiet = 0
            userVoice = min(6, userVoice + 1)
            replyVoice = 0
            sustainedText = 0
            replyQuiet = 0
            startAttempts = 0
            awaitingText = false
            continuationAttempts = 0
            return .normal
        }
        userQuiet = min(Self.replyWindowFrames, userQuiet + 1)
        guard userVoice >= 3, userQuiet < Self.replyWindowFrames else { return .normal }
        if awaitingText, continuationAttempts < 2 {
            awaitingText = false
            continuationAttempts += 1
            return .continueReply
        }
        let due = startAttempts == 0 ? Self.initialWaitFrames : Self.retryWaitFrames
        if userQuiet >= due, startAttempts < 2,
           replyQuiet >= 3, (replyVoice < 6 || sustainedText < 6) {
            startAttempts += 1
            return .start
        }
        return .normal
    }

    mutating func didGenerate(_ pcm: [Float], isText: Bool, isEndOfPause: Bool) {
        guard userVoice >= 3, userQuiet > 0, userQuiet < Self.replyWindowFrames else { return }
        let voiced = pcm.reduce(Float.zero) { $0 + $1 * $1 } >= 0.000025 * Float(pcm.count)
        if voiced {
            replyVoice = min(8, replyVoice + 1)
            replyQuiet = 0
        } else {
            replyQuiet = min(6, replyQuiet + 1)
        }
        if isText {
            sustainedText = min(6, sustainedText + 1)
        }
        // A single backchannel can have text and a spoken frame. Keep the
        // reply eligible until some speech and text have actually followed.
        if isEndOfPause, userQuiet >= Self.initialWaitFrames, sustainedText < 6, replyQuiet > 0 {
            awaitingText = true
        }
    }
}
#endif
