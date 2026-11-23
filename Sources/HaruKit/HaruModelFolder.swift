import Foundation

func readHaruFile<T>(at url: URL, _ read: (URL) throws -> T) throws -> T {
    guard url.startAccessingSecurityScopedResource() else { return try read(url) }
    defer { url.stopAccessingSecurityScopedResource() }

    var coordinationError: NSError?
    var result: Result<T, Error>?
    NSFileCoordinator().coordinate(readingItemAt: url, options: .withoutChanges, error: &coordinationError) {
        coordinatedURL in
        result = Result { try read(coordinatedURL) }
    }
    if let coordinationError { throw coordinationError }
    guard let result else { throw CocoaError(.fileReadUnknown) }
    return try result.get()
}

enum HaruModelFolder {
    static let local = URL.documentsDirectory.appending(path: "Haru-JA", directoryHint: .isDirectory)

    static func size(of folder: URL) throws -> Int {
        guard let pack = try? HaruPack(at: folder) else {
            throw HaruError.invalidPack("変換済みのHaru-JAフォルダー全体を選んでください")
        }
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .fileSizeKey]
        return try pack.shards.reduce(0) { total, name in
            let path = folder.appending(path: name)
            guard let values = try? path.resourceValues(forKeys: keys),
                  values.isRegularFile == true, let bytes = values.fileSize, bytes > 0 else {
                throw HaruError.invalidPack("\(name) が見つからないか、読み込めません")
            }
            return total + bytes
        }
    }
}
