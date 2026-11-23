import CommonCrypto
import CryptoKit
import Foundation
import MLX

public struct HaruModelDownloadProgress: Sendable {
    public enum Phase: Sendable { case downloading, converting, checking }
    public let sourceBytesRead: Int
    public let phase: Phase
}

// Implements the Japanese preset of Tools/prepare.py with the same upstream tensor layouts.
enum HaruModelInstaller {
    static let sourceBytes = 10_540_113_510
    private static let installedBytes = 5_500_000_000
    private static let checkpointSHA256 = "ad2c3534ef6bd88525e22e5da3ac4fb739368406bf3b1c852cbafbade647c0fa"
    private static let embeddingName = "lm.embed_tokens.weight"
    private static let stagingName = "Download"
    static let directory = URL.applicationSupportDirectory.appending(path: "HaruModels", directoryHint: .isDirectory)
    static let installed = directory.appending(path: "Haru-JA", directoryHint: .isDirectory)

    typealias Progress = HaruModelDownloadProgress

    private struct Plan: Decodable {
        struct Input: Decodable {
            let name: String
            let start: Int
            let end: Int
            let shape: [Int]
        }

        struct Output: Decodable {
            enum Layout: String, Decodable {
                case copy, gamma, scalar
                case channelsLast = "channels_last"
                case convMatrix = "conv_matrix"
                case convTransposeMatrix = "conv_transpose_matrix"
            }
            let name: String
            let sources: [String]
            let layout: Layout
            let shape: [Int]
            let bits: Int
        }

        struct Batch: Decodable {
            let start: Int
            let end: Int
            let inputs: [Input]
            let outputs: [Output]
            let commit: Bool
        }

        let headerBytes: Int
        let headerSHA256: String
        let embeddingIDs: [Int32]
        let batches: [Batch]
        let pack: HaruPack
    }

    private struct Resume: Codable {
        let planSHA256: String
        var digest: Data
        var shardSizes: [Int]
    }

    static var canResume: Bool {
        FileManager.default.fileExists(atPath: directory.appending(path: stagingName).appending(path: "resume.json").path)
    }

    static var hasPartialDownload: Bool {
        FileManager.default.fileExists(atPath: directory.appending(path: stagingName).path)
    }

    static func deletePartialDownload() throws {
        try FileManager.default.removeItem(at: directory.appending(path: stagingName, directoryHint: .isDirectory))
    }

    static func install(in root: URL, report: @Sendable (Progress) async -> Void) async throws -> URL {
        let files = FileManager.default
        let final = root.appending(path: "Haru-JA", directoryHint: .isDirectory)
        if (try? HaruModelFolder.size(of: final)) != nil { return final }
        let stage = root.appending(path: stagingName, directoryHint: .isDirectory)
        let stateURL = stage.appending(path: "resume.json")
        guard let planURL = Bundle.module.url(forResource: "HaruDownloadPlan", withExtension: "json") else {
            throw HaruError.invalidPack("変換設定がアプリに入っていません")
        }
        let planData = try Data(contentsOf: planURL)
        let plan = try JSONDecoder().decode(Plan.self, from: planData)
        let planHash = hex(SHA256.hash(data: planData))
        let revision = plan.pack.revision
        let source = URL(string: "https://huggingface.co/higurashi-ai/Haru-4B/resolve/\(revision)/model.safetensors")!
        var context = CC_SHA256_CTX()
        CC_SHA256_Init(&context)
        var sizes: [Int] = []
        if let state = try? JSONDecoder().decode(Resume.self, from: Data(contentsOf: stateURL)),
           state.planSHA256 == planHash, state.digest.count == MemoryLayout<CC_SHA256_CTX>.size,
           !state.shardSizes.isEmpty, state.shardSizes.count <= plan.batches.count,
           plan.batches[state.shardSizes.count - 1].commit,
           zip(plan.pack.shards, state.shardSizes).allSatisfy({ name, size in
               size > 0 && (try? stage.appending(path: name).resourceValues(forKeys: [.fileSizeKey]).fileSize) == size
           }) {
            sizes = state.shardSizes
            // The saved SHA-256 state covers the original bytes through the last complete group.
            _ = withUnsafeMutableBytes(of: &context) { state.digest.copyBytes(to: $0) }
        } else if files.fileExists(atPath: stage.path) {
            try files.removeItem(at: stage)
        }

        try files.createDirectory(at: stage, withIntermediateDirectories: true)
        var downloadRoot = root
        var backup = URLResourceValues()
        backup.isExcludedFromBackup = true
        try downloadRoot.setResourceValues(backup)
        // A stopped conversion can leave output after the last resumable batch.
        for name in plan.pack.shards.dropFirst(sizes.count) + ["pending.safetensors"] {
            let file = stage.appending(path: name)
            if files.fileExists(atPath: file.path) { try files.removeItem(at: file) }
        }
        let remaining = max(0, installedBytes - sizes.reduce(0, +))
        if let free = try? root.resourceValues(forKeys: [.volumeAvailableCapacityKey]).volumeAvailableCapacity,
           free < remaining {
            let text = ByteCountFormatter.string(fromByteCount: Int64(remaining), countStyle: .file)
            throw HaruError.invalidPack("空き容量が足りません。あと約\(text)を保存できる容量を空けてください")
        }

        if sizes.isEmpty {
            let header = try await HaruCheckpointRange.fetch(source, bytes: 0..<plan.headerBytes, report: report)
            guard hex(SHA256.hash(data: header)) == plan.headerSHA256 else {
                throw HaruError.invalidPack("配布元のモデルの目次が一致しません")
            }
            update(&context, header)
        }
        let previousLimit = Memory.cacheLimit
        Memory.cacheLimit = 32 * 1024 * 1024
        defer {
            Memory.clearCache()
            Memory.cacheLimit = previousLimit
        }
        var waiting: [String: MLXArray] = [:]
        for index in sizes.count..<plan.batches.count {
            try Task.checkCancellation()
            let batch = plan.batches[index]
            let file = stage.appending(path: plan.pack.shards[index])
            let pending = stage.appending(path: "pending.safetensors")
            if let input = batch.inputs.first, input.name == embeddingName {
                var parts: [[MLXArray]] = []
                var cursor = 0
                let rows = input.shape[0]
                let width = input.shape[1]
                let rowsPerRequest = 16_384
                for first in stride(from: 0, to: rows, by: rowsPerRequest) {
                    let end = min(rows, first + rowsPerRequest)
                    let range = (batch.start + first * width * 2)..<(batch.start + end * width * 2)
                    let data = try await HaruCheckpointRange.fetch(source, bytes: range, report: report)
                    update(&context, data)
                    await report(.init(sourceBytesRead: range.upperBound, phase: .converting))
                    let previous = cursor
                    while cursor < plan.embeddingIDs.count, plan.embeddingIDs[cursor] < end { cursor += 1 }
                    if cursor > previous {
                        let indices = plan.embeddingIDs[previous..<cursor].map { $0 - Int32(first) }
                        let part = try Device.withDefaultDevice(.gpu) {
                            try withError {
                                let input = MLXArray(data, [end - first, width], dtype: .bfloat16)
                                let value = take(input, MLXArray(indices), axis: 0)
                                let part = try quantize(value, bits: 8, groupSize: plan.pack.groupSize)
                                eval(part)
                                return part
                            }
                        }
                        parts.append(part)
                        Memory.clearCache()
                    }
                }
                guard cursor == plan.embeddingIDs.count, !parts.isEmpty else {
                    throw HaruError.invalidPack("日本語用の語彙を準備できません")
                }
                try Device.withDefaultDevice(.gpu) {
                    let keys = [embeddingName, embeddingName + ".scales", embeddingName + ".biases"]
                    var tensors = ["lm.embed_token_ids": MLXArray(plan.embeddingIDs)]
                    for (index, key) in keys.enumerated() {
                        tensors[key] = concatenated(parts.map { $0[index] }, axis: 0)
                    }
                    try save(arrays: tensors, url: pending)
                }
            } else {
                let data = try await HaruCheckpointRange.fetch(source, bytes: batch.start..<batch.end, report: report)
                update(&context, data)
                await report(.init(sourceBytesRead: batch.end, phase: .converting))
                try Device.withDefaultDevice(.gpu) {
                    try autoreleasepool { try withError {
                        try convert(batch, data: data, groupSize: plan.pack.groupSize, waiting: &waiting, to: pending)
                    } }
                }
            }
            try files.moveItem(at: pending, to: file)
            let size = try file.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
            guard size > 0 else { throw HaruError.invalidPack("変換した重みを保存できません") }
            sizes.append(size)
            if batch.commit {
                let state = Resume(planSHA256: planHash, digest: withUnsafeBytes(of: &context) { Data($0) },
                                   shardSizes: sizes)
                try JSONEncoder().encode(state).write(to: stateURL, options: .atomic)
            }
            Memory.clearCache()
        }

        await report(.init(sourceBytesRead: sourceBytes, phase: .checking))
        try Task.checkCancellation()
        var digest = [UInt8](repeating: 0, count: Int(CC_SHA256_DIGEST_LENGTH))
        CC_SHA256_Final(&digest, &context)
        guard waiting.isEmpty, hex(digest) == checkpointSHA256 else {
            try files.removeItem(at: stage)
            throw HaruError.invalidPack("公式のチェックサムと一致しませんでした。ダウンロードし直してください")
        }
        try JSONEncoder().encode(plan.pack).write(to: stage.appending(path: "pack.json"), options: .atomic)
        for (resource, name) in [("Haru", "LICENSE-Haru"), ("Haru-NOTICE", "NOTICE")] {
            guard let from = Bundle.module.url(forResource: resource, withExtension: "txt", subdirectory: "Licenses") else {
                throw HaruError.invalidPack("モデルのライセンスを読み込めません")
            }
            try Data(contentsOf: from).write(to: stage.appending(path: name), options: .atomic)
        }
        _ = try HaruModelFolder.size(of: stage)
        if files.fileExists(atPath: final.path) { try files.removeItem(at: final) }
        try files.moveItem(at: stage, to: final)
        try? files.removeItem(at: final.appending(path: "resume.json"))
        return final
    }

    private static func convert(_ batch: Plan.Batch, data: Data, groupSize: Int,
                                waiting: inout [String: MLXArray], to file: URL) throws {
        for input in batch.inputs {
            waiting[input.name] = data.withUnsafeBytes { bytes in
                let range = UnsafeRawBufferPointer(rebasing: bytes[input.start..<input.end])
                return MLXArray(range, input.shape, type: UInt16.self).view(dtype: .bfloat16)
            }
        }
        var output: [String: MLXArray] = [:]
        for rule in batch.outputs {
            let parts = try rule.sources.map { name in
                guard let tensor = waiting.removeValue(forKey: name) else { throw HaruError.invalidPack(name) }
                return tensor
            }
            var value = parts.count == 1 ? parts[0] : concatenated(parts, axis: 0)
            switch rule.layout {
            case .copy: break
            case .gamma: value = value.asType(.float32) + Float(1)
            case .scalar: value = value.asType(.float32).reshaped([])
            case .channelsLast: value = value.transposed(0, 2, 1)
            case .convMatrix: value = value.transposed(0, 2, 1).reshaped(rule.shape)
            case .convTransposeMatrix: value = value.transposed(1, 2, 0).reshaped(rule.shape)
            }
            if rule.bits > 0 {
                let quantized = try quantize(value, bits: rule.bits, groupSize: groupSize)
                output[rule.name] = quantized[0]
                output[rule.name + ".scales"] = quantized[1]
                output[rule.name + ".biases"] = quantized[2]
            } else {
                output[rule.name] = contiguous(value)
            }
        }
        try save(arrays: output, url: file)
    }

    private static func quantize(_ value: MLXArray, bits: Int, groupSize: Int) throws -> [MLXArray] {
        let (weights, scales, offsets) = MLX.quantized(value, groupSize: groupSize, bits: bits)
        guard let offsets else { throw HaruError.invalidPack("重みを量子化できません") }
        return [weights, scales, offsets]
    }

    private static func update(_ digest: inout CC_SHA256_CTX, _ data: Data) {
        _ = data.withUnsafeBytes { CC_SHA256_Update(&digest, $0.baseAddress, CC_LONG($0.count)) }
    }

    private static func hex(_ bytes: some Sequence<UInt8>) -> String {
        bytes.map { String(format: "%02x", $0) }.joined()
    }
}

final class HaruCheckpointRange: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let bytes: Range<Int>
    private let totalBytes: Int
    private let stream: AsyncThrowingStream<Data, Error>.Continuation
    // URLSession calls the delegate on its serial delegate queue.
    private var received = 0

    private init(bytes: Range<Int>, totalBytes: Int, stream: AsyncThrowingStream<Data, Error>.Continuation) {
        self.bytes = bytes
        self.totalBytes = totalBytes
        self.stream = stream
    }

    static func fetch(_ url: URL, bytes: Range<Int>, totalBytes: Int = HaruModelInstaller.sourceBytes,
                      report: @Sendable (HaruModelInstaller.Progress) async -> Void) async throws -> Data {
        for attempt in 0..<3 {
            try Task.checkCancellation()
            do {
                return try await request(url, bytes: bytes, totalBytes: totalBytes, report: report)
            } catch let error as URLError where attempt < 2 && error.code != .cancelled {
                try await Task.sleep(for: .seconds(attempt + 1))
            }
        }
        throw URLError(.cannotLoadFromNetwork)
    }

    private static func request(_ url: URL, bytes: Range<Int>, totalBytes: Int,
                                report: @Sendable (HaruModelInstaller.Progress) async -> Void) async throws -> Data {
        let interval = "\(bytes.lowerBound)-\(bytes.upperBound - 1)"
        let (chunks, continuation) = AsyncThrowingStream<Data, Error>.makeStream()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 45
        let delegate = HaruCheckpointRange(bytes: bytes, totalBytes: totalBytes, stream: continuation)
        let session = URLSession(configuration: configuration, delegate: delegate, delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        var request = URLRequest(url: url.appending(queryItems: [URLQueryItem(name: "range", value: interval)]))
        request.setValue("bytes=\(interval)", forHTTPHeaderField: "Range")
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        await report(.init(sourceBytesRead: bytes.lowerBound, phase: .downloading))
        session.dataTask(with: request).resume()

        var result = Data(capacity: bytes.count)
        var lastReport = 0
        for try await data in chunks {
            result.append(data)
            if result.count - lastReport >= 4 * 1024 * 1024 {
                lastReport = result.count
                await report(.init(sourceBytesRead: bytes.lowerBound + result.count, phase: .downloading))
            }
        }
        try Task.checkCancellation()
        guard result.count == bytes.count else { throw URLError(.networkConnectionLost) }
        return result
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        let expected = "bytes \(bytes.lowerBound)-\(bytes.upperBound - 1)/\(totalBytes)"
        guard let http = response as? HTTPURLResponse, http.statusCode == 206,
              http.value(forHTTPHeaderField: "Content-Range") == expected else {
            stream.finish(throwing: URLError(.badServerResponse))
            completionHandler(.cancel)
            return
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        received += data.count
        guard received <= bytes.count else {
            stream.finish(throwing: URLError(.dataLengthExceedsMaximum))
            dataTask.cancel()
            return
        }
        stream.yield(data)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        stream.finish(throwing: error)
    }
}
