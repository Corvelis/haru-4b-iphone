#if HARU_ENABLED
import Accelerate
import Foundation
import Synchronization

// Single producer on the inference worker, single consumer in the Core Audio
// callback. The producer can lengthen speech without holding up the callback.
final class HaruPlayback {
    private struct Frame {
        var length = 0
        var quiet = true
        var capturedEnd = 0.0
        var prebufferUntil = 0.0
    }

    private struct Timing {
        var time = 0.0
        var capturedEnd = 0.0
    }

    private static let slots = 12
    private static let frame = HaruSession.frameSamples
    private static let rate = HaruSession.sampleRate
    private static let fadeSamples = 192
    private static let maxSamples = 5600
    private static let startDelay = 0.27
    private static let maxFrameAge = 0.64

    private let writeSequence = Atomic<Int>(0)
    private let readSequence = Atomic<Int>(0)
    private let startTime = Atomic<UInt64>(0)
    private let latestLatency = Atomic<Int>(.min)
    private let consumed = Atomic<Int>(0)
    private let samples = UnsafeMutablePointer<Float>.allocate(capacity: slots * maxSamples)
    private let frames = UnsafeMutablePointer<Frame>.allocate(capacity: slots)
    private var timing: [Timing]
    private var timingCount = 0

    // Producer only. The median reacts to sustained slowdowns without making
    // one unusually slow model step change the cadence of the voice.
    private var intervals: [Double] = []
    private var lastArrival: Double?
    private var outputSamples = 0
    private let stretchedSamples = Atomic<Int>(0)
    private let rejectedFrames = Atomic<Int>(0)

    // Render thread only until the engine has stopped.
    private var started = false
    private var playing = false
    private var inSpeech = false
    private var offset = 0
    private var fadeIn = fadeSamples
    private var fadeOut = 0
    private var fadeStart: Float = 0
    private var fadeOutStart: Float = 0
    private var previousSample: Float = 0
    private var lastVoiceTime = -Double.infinity
    private var gaps = 0
    private var audibleGaps = 0
    private var missingSamples = 0
    private var recoveries = 0
    private var expiredFrames = 0
    private var renderedSamples = 0

    init(measure: Bool) {
        timing = measure ? Array(repeating: Timing(), count: 2400) : []
        samples.initialize(repeating: 0, count: Self.slots * Self.maxSamples)
        frames.initialize(repeating: Frame(), count: Self.slots)
        intervals.reserveCapacity(9)
    }

    deinit {
        samples.deallocate()
        frames.deallocate()
    }

    var lastLatencyMs: Int? {
        let value = latestLatency.load(ordering: .relaxed)
        return value == .min ? nil : value
    }

    var metrics: (gaps: Int, audibleGaps: Int, missingSamples: Int, timing: [[String: Double]],
                  recoveries: Int, droppedFrames: Int, stretchedSamples: Int) {
        let firstCapture = timing.first?.capturedEnd ?? 0
        let values = timing.prefix(timingCount).map {
            ["time": $0.capturedEnd - firstCapture, "latencyMs": ($0.time - $0.capturedEnd) * 1000]
        }
        return (gaps, audibleGaps, missingSamples, values, recoveries,
                expiredFrames + rejectedFrames.load(ordering: .relaxed),
                stretchedSamples.load(ordering: .relaxed))
    }

    func begin(firstCapture: Double) {
        if startTime.load(ordering: .relaxed) == 0 {
            startTime.store((firstCapture + Self.startDelay).bitPattern, ordering: .releasing)
        }
    }

    func enqueue(_ pcm: [Float], capturedEnd: Double, at now: Double = ProcessInfo.processInfo.systemUptime) {
        guard pcm.count == Self.frame else { return }
        let seq = writeSequence.load(ordering: .relaxed)
        let read = readSequence.load(ordering: .acquiring)
        guard seq - read < Self.slots else {
            rejectedFrames.wrappingAdd(1, ordering: .relaxed)
            return
        }
        let index = seq % Self.slots
        if let lastArrival {
            let interval = min(0.26, now - lastArrival)
            if intervals.count == 9 { intervals.removeFirst() }
            intervals.append(interval)
        }
        lastArrival = capturedEnd > 0 ? now : nil

        let silent = !pcm.contains { abs($0) > 0.003 }
        let queued = Double(outputSamples - consumed.load(ordering: .relaxed)) / Self.rate
        let frameSeconds = Double(Self.frame) / Self.rate
        let recent = intervals.sorted()
        let cadence = recent.isEmpty ? frameSeconds : recent[recent.count / 2]
        // Maintain a small reserve during speech. Spend it in pauses to
        // return to the current input clock.
        let reserve = Self.startDelay + slowSpeechAllowance(cadence) - max(0, now - capturedEnd)
        let desiredLength = cadence + min(0.023, max(-0.045, (reserve - queued) * 0.12))
        let extra = Int((desiredLength * Self.rate).rounded()) - Self.frame
        let available = Self.maxSamples - Self.frame
        let lengthened = !silent && extra >= 168 && queued < 0.36
        let output = lengthened ? HaruTimeScale.process(pcm, length: pcm.count + min(available, extra)) : pcm
        output.withUnsafeBufferPointer { values in
            samples.advanced(by: index * Self.maxSamples).update(from: values.baseAddress!, count: values.count)
        }
        // At the start of speech, wait for another block up to the recent
        // generation interval plus a little jitter margin. Cap the total age
        // so an isolated long step can't delay the start of a reply indefinitely.
        let prebufferUntil = min(capturedEnd + 0.49, now + cadence + 0.025)
        frames[index] = Frame(length: output.count, quiet: silent, capturedEnd: capturedEnd,
                              prebufferUntil: prebufferUntil)
        outputSamples += output.count
        stretchedSamples.wrappingAdd(output.count - pcm.count, ordering: .relaxed)
        writeSequence.store(seq + 1, ordering: .releasing)
    }

    private func slowSpeechAllowance(_ cadence: Double) -> Double {
        min(0.1, max(0, 2.5 * (cadence - Double(Self.frame) / Self.rate)))
    }

    func render(_ target: UnsafeMutablePointer<Float>, count: Int, at time: Double) -> Bool {
        var written = 0
        if !started {
            let start = Double(bitPattern: startTime.load(ordering: .acquiring))
            let wait = (start - time) * Self.rate
            if start == 0 || wait >= Double(count) {
                target.update(repeating: 0, count: count)
                return true
            }
            written = Int(max(0, wait.rounded(.up)))
            target.update(repeating: 0, count: written)
            started = true
        }

        var hadAudio = false
        var seq = readSequence.load(ordering: .relaxed)
        let end = writeSequence.load(ordering: .acquiring)
        while written < count && seq < end {
            let index = seq % Self.slots
            let at = time + Double(written) / Self.rate
            let frame = frames[index]
            let capturedEnd = frame.capturedEnd
            let length = frame.length
            let silent = frame.quiet
            if offset == 0, capturedEnd > 0 {
                let age = at - capturedEnd
                // Resume at a fresh block after an isolated GPU stall.
                // Silence can be shortened while voiced audio keeps its pitch.
                if (age > Self.maxFrameAge && seq + 1 < end) ||
                    (silent && age > Self.startDelay + 0.04 && seq + 1 < end) {
                    if silent { recoveries += 1 } else {
                        expiredFrames += 1
                        inSpeech = false
                    }
                    renderedSamples += length
                    seq += 1
                    readSequence.store(seq, ordering: .releasing)
                    continue
                }
                if !silent {
                    if !inSpeech {
                        if end - seq < 2 && at < frame.prebufferUntil { break }
                        beginFade()
                    }
                    latestLatency.store(Int((max(0, age) * 1000).rounded()), ordering: .relaxed)
                }
                if timingCount < timing.count {
                    timing[timingCount] = Timing(time: at, capturedEnd: capturedEnd)
                    timingCount += 1
                }
            }
            let take = min(count - written, length - offset)
            let output = target.advanced(by: written)
            output.update(from: samples.advanced(by: index * Self.maxSamples + offset), count: take)
            for i in 0..<min(take, fadeIn) {
                let gain = Float(Self.fadeSamples - fadeIn + 1) / Float(Self.fadeSamples)
                output[i] = gain * output[i] + (1 - gain) * fadeStart
                fadeIn -= 1
            }
            previousSample = output[take - 1]
            if !silent { lastVoiceTime = at + Double(take) / Self.rate }
            hadAudio = true
            fadeOut = 0
            playing = true
            inSpeech = !silent
            written += take
            renderedSamples += take
            offset += take
            if offset == length {
                offset = 0
                seq += 1
                readSequence.store(seq, ordering: .releasing)
                if silent && seq == end { playing = false }
            }
        }
        consumed.store(renderedSamples, ordering: .relaxed)

        if written == count { return !hadAudio }
        if playing {
            playing = false
            inSpeech = false
            gaps += 1
            let gapStart = time + Double(written) / Self.rate
            if gapStart - lastVoiceTime < 0.015 {
                audibleGaps += 1
            }
            fadeOut = Self.fadeSamples
            fadeOutStart = previousSample
        }
        missingSamples += count - written
        let fade = min(fadeOut, count - written)
        for i in 0..<fade {
            target[written + i] = fadeOutStart * Float(fadeOut - i - 1) / Float(Self.fadeSamples)
        }
        if fade > 0 { previousSample = target[written + fade - 1] }
        fadeOut -= fade
        target.advanced(by: written + fade).update(repeating: 0, count: count - written - fade)
        return fade == 0 && !hadAudio
    }

    private func beginFade() {
        fadeIn = Self.fadeSamples
        fadeStart = previousSample
    }
}

enum HaruTimeScale {
    // WSOLA keeps the pitch and both ends of each block. The search
    // matches a short overlap before adding the next segment.
    static func process(_ pcm: [Float], length: Int) -> [Float] {
        let hop = 240
        let window = 2 * hop
        guard pcm.count > window, length > window, pcm.count != length else { return pcm }
        let speed = Double(pcm.count - window) / Double(length - window)
        var result = Array(repeating: Float.zero, count: length)
        pcm.withUnsafeBufferPointer { input in
            result.withUnsafeMutableBufferPointer { output in
                let inBase = input.baseAddress!
                let outBase = output.baseAddress!
                outBase.update(from: inBase, count: window)
                var target = hop
                while target + window < length {
                    let proposed = Int((Double(target) * speed).rounded())
                    let low = max(0, proposed - 120)
                    let high = min(pcm.count - window, proposed + 120)
                    let overlap = outBase.advanced(by: target)
                    var best = proposed
                    var peak: Float = -.infinity
                    for start in stride(from: low, through: high, by: 4) {
                        let candidate = inBase.advanced(by: start)
                        var cross: Float = 0
                        var energy: Float = 0
                        vDSP_dotpr(overlap, 2, candidate, 2, &cross, vDSP_Length(hop / 2))
                        vDSP_svesq(candidate, 2, &energy, vDSP_Length(hop / 2))
                        let score = cross / sqrt(max(1e-8, energy))
                        if score > peak {
                            best = start
                            peak = score
                        }
                    }
                    for i in 0..<hop {
                        let mix = Float(i + 1) / Float(hop)
                        overlap[i] += (inBase[best + i] - overlap[i]) * mix
                    }
                    overlap.advanced(by: hop).update(from: inBase.advanced(by: best + hop), count: hop)
                    target += hop
                }
                let tail = inBase.advanced(by: pcm.count - (length - target))
                let overlap = outBase.advanced(by: target)
                for i in 0..<hop {
                    let mix = Float(i + 1) / Float(hop)
                    overlap[i] += (tail[i] - overlap[i]) * mix
                }
                overlap.advanced(by: hop).update(from: tail.advanced(by: hop), count: length - target - hop)
            }
        }
        return result
    }
}

// Accessed under the inference worker's capture lock. Take a little more
// source audio per inference step when capture is ahead, retaining contiguous
// words until the worker has caught up to the microphone.
struct HaruCaptureBuffer {
    private static let frame = HaruSession.frameSamples
    private static let rate = HaruSession.sampleRate
    private var pending: [Float] = []
    private var pendingEnd = 0.0
    private(set) var discardedSamples = 0
    private(set) var compressedSamples = 0

    var hasFrame: Bool { pending.count >= Self.frame }

    mutating func append(_ samples: [Float], at hostTime: Double) {
        pending += samples
        pendingEnd = hostTime + Double(samples.count) / Self.rate
        // Bound the queue even if the GPU stalls completely. Normally the
        // worker removes the surplus gradually in pop().
        let excess = pending.count - 4 * Self.frame
        if excess > 0 {
            pending.removeFirst(excess)
            discardedSamples += excess
        }
    }

    mutating func pop() -> ([Float], Double)? {
        guard hasFrame else { return nil }
        let extra = min(Self.frame / 2, max(0, pending.count - Self.frame - Self.frame / 8))
        let length = Self.frame + extra
        let samples = Array(pending.prefix(length))
        pending.removeFirst(length)
        compressedSamples += extra
        return (samples, pendingEnd - Double(pending.count) / Self.rate)
    }

    mutating func clear() {
        pending.removeAll(keepingCapacity: true)
    }
}
#endif
