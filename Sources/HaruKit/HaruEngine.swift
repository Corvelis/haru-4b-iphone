import Foundation
import MLX
import os

protocol HaruFrameSession: AnyObject {
    var maxInputFrames: Int { get }
    func infer(_ samples: [Float]) throws -> [Float]
    func resetTurnTaking()
}

extension HaruSession: HaruFrameSession {
    var maxInputFrames: Int { Self.maxFrames }

    func infer(_ samples: [Float]) throws -> [Float] {
        try withError { step(samples) }
    }
}

public enum HaruEngine: String, CaseIterable, Sendable {
    case mlx, speechCpp

    @MainActor private static var modelDownloadInProgress = false

    public var label: String {
        switch self {
        case .mlx: "MLX"
        case .speechCpp: "speech.cpp"
        }
    }

    public var downloadedModelURL: URL {
        self == .mlx ? HaruModelInstaller.installed : HaruGGUFInstaller.installed
    }

    public var sharedModelURL: URL {
        self == .mlx ? HaruModelFolder.local : HaruGGUFInstaller.local
    }

    public var downloadBytes: Int {
        self == .mlx ? HaruModelInstaller.sourceBytes : HaruGGUFInstaller.sourceBytes
    }

    public var canResumeDownload: Bool {
        self == .mlx ? HaruModelInstaller.canResume : HaruGGUFInstaller.canResume
    }

    public var hasPartialDownload: Bool {
        self == .mlx ? HaruModelInstaller.hasPartialDownload : HaruGGUFInstaller.hasPartialDownload
    }

    /// Downloads the model into the calling app's container, resuming from saved data when available.
    @MainActor
    public func prepareModel(
        report: @Sendable (HaruModelDownloadProgress) async -> Void = { _ in }
    ) async throws -> URL {
        #if targetEnvironment(simulator)
        throw CocoaError(.featureUnsupported, userInfo: [NSLocalizedDescriptionKey: "モデルの準備はiPhone実機で試してください。"])
        #else
        try Task.checkCancellation()
        guard !Self.modelDownloadInProgress else { throw HaruError.modelDownloadInProgress }
        Self.modelDownloadInProgress = true
        defer { Self.modelDownloadInProgress = false }
        switch self {
        case .mlx: return try await HaruModelInstaller.install(in: HaruModelInstaller.directory, report: report)
        case .speechCpp: return try await HaruGGUFInstaller.install(in: HaruGGUFInstaller.directory, report: report)
        }
        #endif
    }

    @MainActor
    public func discardPartialDownload() throws {
        guard !Self.modelDownloadInProgress else { throw HaruError.modelDownloadInProgress }
        switch self {
        case .mlx: try HaruModelInstaller.deletePartialDownload()
        case .speechCpp: try HaruGGUFInstaller.deletePartialDownload()
        }
    }

    public func modelBytes(at url: URL) throws -> Int {
        try readHaruFile(at: url, localModelBytes)
    }

    private func localModelBytes(at url: URL) throws -> Int {
        switch self {
        case .mlx: try HaruModelFolder.size(of: url)
        case .speechCpp: try HaruGGUFInstaller.size(of: url)
        }
    }

    func load(model: URL, reference: URL, instructions: String? = nil, seed: UInt64, isCancelled: () -> Bool)
        throws -> (session: any HaruFrameSession, first: [Float]) {
        guard !isCancelled() else { throw CancellationError() }
        let instructions = instructions.map(HaruConversationInstructions.normalize)
        if let instructions, !HaruConversationInstructions.isValid(instructions) {
            throw HaruError.invalidPack("会話の指示を確認してください")
        }
        let size = try localModelBytes(at: model)
        let available = os_proc_available_memory()
        let workingMemory = 560 * 1024 * 1024
        if available > 0 && available < size + workingMemory {
            throw HaruError.invalidPack("空きメモリ約\(available / 1_048_576) MiB、重み約\(size / 1_048_576) MiB")
        }

        switch self {
        case .mlx:
            let clip = try readHaruFile(at: reference, HaruAudio.readReference)
            return try autoreleasepool { try withError {
                let session = try HaruSession(directory: model, seed: seed, instructions: instructions,
                                              compactAcoustics: true,
                                              temperature: HaruSession.conversationTemperature,
                                              endOfPauseBias: HaruSession.conversationEndOfPauseBias,
                                              playSpeechOnly: true)
                let first = try session.start(reference: clip, isCancelled: isCancelled)
                guard !isCancelled() else { throw CancellationError() }
                Memory.clearCache()
                session.warmUp()
                return (session, first)
            } }
        case .speechCpp:
            let clip = FileManager.default.temporaryDirectory.appending(path: "haru-reference-\(UUID()).wav")
            defer { try? FileManager.default.removeItem(at: clip) }
            try HaruReferenceVoice.importFile(from: reference, to: clip)
            let session = try HaruSpeechCppSession(model: model, reference: clip, instructions: instructions,
                                                   seed: seed, isCancelled: isCancelled)
            return (session, session.first)
        }
    }
}
