import Foundation

public enum HaruConversationInstructions {
    public static let defaultText = "相手への呼びかけは省いて自然に話す。質問されたら、まずいま聞かれたことに一言で答え、それから必要なら説明する。話題が変わったら、相手がいま話したことに合わせて返す。"
    public static let characterLimit = 500

    private static let card = """
        system
        あなたは以下の人物として、相手と音声で会話します。
        【あなたの人物設定】
        自分はハル。気さくで聞き上手。日本語の普段の話し言葉で話す。
        【あなたの名前】
        ハル
        【あなたの相づち】
        少なめ
        【相手の名前】
        未設定
        【関係性】
        気軽に話せる友人同士。
        """

    public static func normalize(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines).precomposedStringWithCanonicalMapping
    }

    public static func isValid(_ text: String) -> Bool {
        text.unicodeScalars.count <= characterLimit && !text.contains("\0") && !text.contains("<|")
    }

    static func tokenize(_ instructions: String, table: URL? = Bundle.module.url(
        forResource: "HaruTokenizer", withExtension: "bin"
    )) throws -> [HaruPromptTokenizer.Token] {
        guard let table else { throw CocoaError(.fileNoSuchFile) }
        let tokenizer = try HaruPromptTokenizer(data: Data(contentsOf: table, options: .mappedIfSafe))
        let body = instructions.isEmpty ? card : card + "\n【状況】\n" + instructions
        // Frame the text with Haru's system-turn tokens. User text goes through the BPE alone.
        return [.byte(248045)] + tokenizer.encode(body) + [.byte(248046)] + tokenizer.encode("\n")
    }
}

struct HaruPromptTokenizer {
    indirect enum Token {
        case byte(Int32)
        case merge(Int32, Token, Token)

        var id: Int32 {
            switch self {
            case .byte(let id), .merge(let id, _, _): id
            }
        }

        // Japanese MLX packs omit some vocabulary rows but retain the byte alphabet.
        func append(to ids: inout [Int32], keeping isAvailable: (Int32) -> Bool) {
            if isAvailable(id) {
                ids.append(id)
            } else if case .merge(_, let left, let right) = self {
                left.append(to: &ids, keeping: isAvailable)
                right.append(to: &ids, keeping: isAvailable)
            } else {
                // Let the caller report an incompatible model if a required control or byte is absent.
                ids.append(id)
            }
        }
    }

    private struct Merge {
        let rank: Int
        let token: Int32
    }

    private let byteIDs: [Int32]
    private let merges: [UInt64: Merge]
    private let pieces: NSRegularExpression

    init(data: Data) throws {
        let headerBytes = 4 + 256 * 4
        guard data.count == headerBytes + 247_587 * 12, data.starts(with: "HBP1".utf8) else {
            throw CocoaError(.fileReadCorruptFile)
        }
        (byteIDs, merges) = data.withUnsafeBytes { bytes in
            func read(_ offset: Int) -> Int32 {
                Int32(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: Int32.self))
            }
            let byteIDs = (0..<256).map { read(4 + $0 * 4) }
            var merges: [UInt64: Merge] = [:]
            merges.reserveCapacity(247_587)
            for (rank, offset) in stride(from: headerBytes, to: bytes.count, by: 12).enumerated() {
                merges[Self.key(read(offset), read(offset + 4))] = Merge(rank: rank, token: read(offset + 8))
            }
            return (byteIDs, merges)
        }
        // Split used by the pinned Haru tokenizer.json. Apply NFC before byte-level BPE.
        pieces = try NSRegularExpression(pattern: #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#)
    }

    func encode(_ text: String) -> [Token] {
        let normalized = text.precomposedStringWithCanonicalMapping
        return pieces.matches(in: normalized, range: NSRange(normalized.startIndex..., in: normalized)).flatMap { match in
            let piece = (normalized as NSString).substring(with: match.range)
            return encodePiece(piece)
        }
    }

    private func encodePiece(_ text: String) -> [Token] {
        var tokens = text.utf8.map { Token.byte(byteIDs[Int($0)]) }
        while tokens.count > 1 {
            var best: (index: Int, merge: Merge)?
            for index in 0..<(tokens.count - 1) {
                if let merge = merges[Self.key(tokens[index].id, tokens[index + 1].id)],
                   merge.rank < (best?.merge.rank ?? .max) {
                    best = (index, merge)
                }
            }
            guard let (index, merge) = best else { break }
            tokens[index] = .merge(merge.token, tokens[index], tokens[index + 1])
            tokens.remove(at: index + 1)
        }
        return tokens
    }

    private static func key(_ left: Int32, _ right: Int32) -> UInt64 {
        UInt64(UInt32(bitPattern: left)) << 32 | UInt64(UInt32(bitPattern: right))
    }
}
