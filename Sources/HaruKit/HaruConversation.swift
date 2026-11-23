import AVFoundation
import Combine
import Metal
import MLX
import UIKit
import os

@MainActor
public final class HaruConversation: NSObject, ObservableObject {
    public enum Status: Sendable {
        case idle, loading, connecting, running, paused, closing
    }

    @Published public private(set) var status: Status = .idle
    @Published public private(set) var failure: Error?
    @Published public private(set) var inputLevel: Float = 0
    @Published public private(set) var inferenceMs: Double?
    @Published public private(set) var playbackMs: Int?

    private static var activeRun: HaruRun?
    private let worker = DispatchQueue(label: "haru.inference", qos: .userInitiated)
    private var run: HaruRun?
    private var audio: HaruAudio?
    private var preparedAudio: [Float]?
    private var hasReceivedAudio = false
    private var audioRetry: DispatchWorkItem?
    private var routeRestartAttempted = false

    public override init() {
        super.init()
        NotificationCenter.default.addObserver(self, selector: #selector(backgrounded),
            name: UIApplication.didEnterBackgroundNotification, object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(openPreparedAudio),
            name: UIApplication.didBecomeActiveNotification, object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(interrupted(_:)),
            name: AVAudioSession.interruptionNotification, object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(audioConfigurationChanged(_:)),
            name: .AVAudioEngineConfigurationChange, object: nil)
    }

    isolated deinit {
        if let run {
            run.cancel()
            Self.release(run, after: worker)
        }
        audioRetry?.cancel()
        audio?.stop()
    }

    public func start(engine: HaruEngine, model: URL, reference: URL, instructions: String? = nil) throws {
        guard status == .idle, UIApplication.shared.applicationState != .background else { return }
        #if targetEnvironment(simulator)
        throw CocoaError(.featureUnsupported, userInfo: [NSLocalizedDescriptionKey: "Haru-4Bの会話はiPhone実機で試してください。"])
        #else
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw CocoaError(.featureUnsupported, userInfo: [NSLocalizedDescriptionKey: "Metal GPUを利用できません。"])
        }
        guard Self.activeRun == nil else { throw HaruError.conversationInProgress }
        let job = HaruRun()
        Self.activeRun = job
        run = job
        failure = nil
        status = .loading
        AVAudioApplication.requestRecordPermission { [weak self, weak job] allowed in
            DispatchQueue.main.async {
                guard let self, let job, self.run === job else { return }
                guard allowed else { return self.finish(job, error: HaruError.microphone) }
                self.prepare(job, engine: engine, model: model, reference: reference, instructions: instructions)
            }
        }
        #endif
    }

    public func mute() {
        guard let run, status == .running || status == .connecting else { return }
        pause(run)
    }

    public func resume() {
        guard status == .paused else { return }
        failure = nil
        status = .connecting
        preparedAudio = Array(repeating: 0, count: HaruSession.frameSamples)
        openPreparedAudio()
    }

    public func stop() {
        if let run { finish(run) }
    }

    private func prepare(_ job: HaruRun, engine: HaruEngine, model: URL, reference: URL, instructions: String?) {
        worker.async { [weak self, weak job] in
            guard let job, job.isActive else { return }
            let previousCacheLimit = engine == .mlx ? Memory.cacheLimit : nil
            if previousCacheLimit != nil { Memory.cacheLimit = 128 * 1024 * 1024 }
            defer {
                if let previousCacheLimit {
                    Memory.clearCache()
                    Memory.cacheLimit = previousCacheLimit
                }
            }
            do {
                let runSession = {
                    let (session, first) = try readHaruFile(at: model) { file in
                        try engine.load(model: file, reference: reference, instructions: instructions,
                                        seed: job.seed, isCancelled: { !job.isActive })
                    }
                    DispatchQueue.main.async { [weak self] in
                        guard let self, self.run === job else { return }
                        self.preparedAudio = first
                        self.status = .connecting
                        self.openPreparedAudio()
                    }
                    try job.consume(session)
                }
                if engine == .mlx {
                    try Device.withDefaultDevice(.gpu, runSession)
                } else {
                    try runSession()
                }
                DispatchQueue.main.async { [weak self, weak job] in
                    guard let self, let job else { return }
                    self.finish(job)
                }
            } catch {
                DispatchQueue.main.async { [weak self, weak job] in
                    guard let self, let job else { return }
                    self.finish(job, error: error)
                }
            }
        }
    }

    @objc private func openPreparedAudio() {
        guard let run, let first = preparedAudio, UIApplication.shared.applicationState == .active else { return }
        preparedAudio = nil
        routeRestartAttempted = false
        do { try openAudio(run, first: first, voiceProcessing: true) }
        catch { retryAudioWithoutVoiceProcessing(run, first: first) }
    }

    private func openAudio(_ job: HaruRun, first: [Float], voiceProcessing: Bool) throws {
        guard UIApplication.shared.applicationState == .active else { throw HaruError.audioFormat }
        let audio = HaruAudio(voiceProcessing: voiceProcessing)
        let capture = job.beginCapture { [weak self, weak audio] output, capturedEnd, medianMs in
            guard let audio else { return }
            audio.playback.enqueue(output, capturedEnd: capturedEnd)
            if let medianMs {
                DispatchQueue.main.async { [weak self, weak audio] in
                    guard let self, let audio, self.audio === audio else { return }
                    self.inferenceMs = medianMs
                    self.playbackMs = audio.playback.lastLatencyMs.map { $0 + audio.routeLatencyMs }
                }
            }
        }
        hasReceivedAudio = false
        try audio.start(first: first, capture: { [weak job] samples, time in
            job?.feed(samples, at: time, capture: capture)
        }, level: { [weak self, weak audio] level in
            guard let self, let audio, self.audio === audio else { return }
            if !self.hasReceivedAudio {
                self.hasReceivedAudio = true
                self.status = .running
            }
            self.inputLevel = level
        })
        self.audio = audio
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) { [weak self, weak job, weak audio] in
            guard let self, let job, let audio, self.audio === audio, !self.hasReceivedAudio else { return }
            if audio.usesVoiceProcessing { self.retryAudioWithoutVoiceProcessing(job, first: first) }
            else { self.pause(job, error: HaruError.noInput) }
        }
    }

    private func retryAudioWithoutVoiceProcessing(_ job: HaruRun, first: [Float]? = nil) {
        job.endCapture()
        closeAudio()
        let retry = DispatchWorkItem { [weak self, weak job] in
            guard let self, let job, self.run === job else { return }
            self.audioRetry = nil
            do {
                try self.openAudio(job, first: first ?? Array(repeating: 0, count: HaruSession.frameSamples),
                                   voiceProcessing: false)
            } catch { self.pause(job, error: error) }
        }
        audioRetry = retry
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2, execute: retry)
    }

    private func pause(_ job: HaruRun, error: Error? = nil) {
        guard run === job else { return }
        job.endCapture()
        closeAudio()
        if let error { failure = error }
        status = .paused
    }

    private func finish(_ job: HaruRun, error: Error? = nil) {
        guard run === job else { return }
        run = nil
        job.cancel()
        closeAudio()
        if let error { failure = error }
        status = .closing
        // The next conversation can start once pending Metal work releases its weights.
        Self.release(job, after: worker) { [weak self] in
            self?.status = .idle
        }
    }

    private static func release(_ job: HaruRun, after worker: DispatchQueue,
                                completion: @escaping @MainActor () -> Void = {}) {
        worker.async {
            DispatchQueue.main.async {
                if Self.activeRun === job { Self.activeRun = nil }
                completion()
            }
        }
    }

    private func closeAudio() {
        preparedAudio = nil
        audioRetry?.cancel()
        audioRetry = nil
        audio?.stop()
        audio = nil
        inputLevel = 0
        inferenceMs = nil
        playbackMs = nil
    }

    @objc private func backgrounded() {
        if let run { finish(run, error: HaruError.backgrounded) }
    }

    @objc nonisolated private func interrupted(_ note: Notification) {
        guard note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt == AVAudioSession.InterruptionType.began.rawValue else { return }
        DispatchQueue.main.async { [weak self] in
            guard let self, let run = self.run, self.audio != nil else { return }
            if !self.hasReceivedAudio, self.audio?.usesVoiceProcessing == true,
               UIApplication.shared.applicationState == .active {
                self.retryAudioWithoutVoiceProcessing(run)
                return
            }
            self.pause(run, error: HaruError.interrupted)
        }
    }

    @objc nonisolated private func audioConfigurationChanged(_ note: Notification) {
        guard let changed = note.object as? AVAudioEngine else { return }
        DispatchQueue.main.async { [weak self] in
            guard let self, let run = self.run, let audio = self.audio,
                  audio.engine === changed, !changed.isRunning else { return }
            let starting = ProcessInfo.processInfo.systemUptime - audio.startedAt < 1
            if (!self.hasReceivedAudio || starting), audio.usesVoiceProcessing,
               UIApplication.shared.applicationState == .active {
                if !self.routeRestartAttempted {
                    self.routeRestartAttempted = true
                    do {
                        try audio.restart()
                        return
                    } catch { }
                }
                if !self.hasReceivedAudio {
                    self.retryAudioWithoutVoiceProcessing(run)
                    return
                }
            }
            self.pause(run, error: HaruError.interrupted)
        }
    }
}

// The audio and inference queues share this state through the condition lock.
private final class HaruRun: @unchecked Sendable {
    typealias Output = ([Float], Double, Double?) -> Void
    let seed = UInt64.random(in: .min ... .max)
    private let condition = NSCondition()
    private var input = HaruCaptureBuffer()
    private var active = true
    private var capture: (id: UUID, emit: Output)?

    var isActive: Bool { condition.withLock { active } }

    func beginCapture(emit: @escaping Output) -> UUID {
        condition.withLock {
            let id = UUID()
            capture = (id, emit)
            return id
        }
    }

    func endCapture() {
        condition.withLock {
            capture = nil
            input.clear()
        }
    }

    func feed(_ samples: [Float], at hostTime: Double, capture id: UUID) {
        condition.withLock {
            guard active, capture?.id == id else { return }
            input.append(samples, at: hostTime)
            if input.hasFrame { condition.signal() }
        }
    }

    // Keep the session and all Metal inference on the same worker across mute and resume.
    func consume(_ model: any HaruFrameSession) throws {
        var times: [Double] = []
        var frames = 0
        var previousCapture: UUID?
        while frames < model.maxInputFrames, let (pcm, capturedEnd, capture) = nextFrame() {
            let start = ProcessInfo.processInfo.systemUptime
            if previousCapture != capture.id {
                previousCapture = capture.id
                times.removeAll(keepingCapacity: true)
                model.resetTurnTaking()
            }
            let output = try autoreleasepool {
                try model.infer(HaruTimeScale.process(pcm, length: HaruSession.frameSamples))
            }
            times.append((ProcessInfo.processInfo.systemUptime - start) * 1000)
            frames += 1
            var medianMs: Double?
            if times.count == 15 {
                medianMs = times.sorted()[times.count / 2]
                times.removeAll(keepingCapacity: true)
            }
            capture.emit(output, capturedEnd, medianMs)
        }
    }

    private func nextFrame() -> ([Float], Double, (id: UUID, emit: Output))? {
        condition.lock()
        defer { condition.unlock() }
        while active {
            if let capture, let (pcm, capturedEnd) = input.pop() {
                return (pcm, capturedEnd, capture)
            }
            condition.wait()
        }
        return nil
    }

    func cancel() {
        condition.withLock {
            active = false
            capture = nil
            input.clear()
            condition.signal()
        }
    }
}
