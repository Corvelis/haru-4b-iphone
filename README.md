# Haru-4B on iPhone

Haru-4Bと日本語で音声会話するためのiOS向けSwift SDKとサンプルアプリです。MLX Swiftとspeech.cppのMetal版に対応します。モデルを準備するとオフラインで会話できます。[Haru-4B公式リポジトリ](https://github.com/higurashi-ai/Haru)

**実機デモ** iPhoneでの日本語音声会話（約34秒）

https://github.com/user-attachments/assets/ef062366-3294-4b53-9d32-58ea46a0ef98

[MP4ファイルを開く](Demo/haru-4b-iphone.mp4)

- [iPhoneで試す](#iphoneで試す)
- [Swift SDKを組み込む](#swift-sdkを組み込む)
- [変換済みモデルを使う](#変換済みモデルを使う)

## 動作環境

このSDKとサンプルアプリはプレビュー版です。

- Apple Silicon Mac、Swift 6.3以降を含むXcode、Metal Toolchain。[Xcodeのシステム要件](https://developer.apple.com/xcode/system-requirements/) · [Metal Toolchainの追加](https://developer.apple.com/documentation/xcode/installing-additional-xcode-components)
- iOS 18以降のiPhone実機。モデルを読み込めるメモリと、次の表に示す保存容量が必要です。

モデルは初回にアプリから取得します。セットアップ用の空き容量にも余裕を持たせてください。

| エンジン | 初回の通信量 | 保存容量 |
| --- | ---: | ---: |
| MLX（日本語向け変換） | 約10.5 GB | 約5.5 GB |
| speech.cpp（Q6_K） | 約5.6 GB | 約5.6 GB |

[公式モデル](https://huggingface.co/higurashi-ai/Haru-4B) · [Q6_Kの配布元](https://huggingface.co/sakasegawa/Haru-4B-GGUF)

## iPhoneで試す

1. `Haru.xcodeproj` を開きます。Haruターゲットの **Signing & Capabilities** で自分のTeamとBundle Identifierを設定し、**Increased Memory Limit** を有効にします。
2. 実行先に接続したiPhoneを選び、**Run** を押します。
3. アプリで「モデルを追加」を開き、「モデル」でエンジンを選んで「モデルをダウンロード」を押します。準備中はアプリを開いたままにしてください。中断した場合は、保存済みのところから再開できます。
4. 「返答の声」で男性・女性の声を選べます。自分で用意した音声を使う場合は、使用する権利のあるファイルを指定してください。
5. 会話画面で「マイクをオン」を押し、マイクを許可して話します。マイクをオフにした後は同じ会話を再開できます。「会話を終了」でモデルのメモリを解放します。

設定の「会話の指示」で、次の会話から使う指示を編集したり、既定値に戻したりできます。「保存したモデルを管理」でモデルの容量を確認し、削除できます。speech.cppの会話は1回につき最大約10分です。

## Swift SDKを組み込む

Xcodeの **File → Add Package Dependencies** で、このリポジトリのURLと使うリリースバージョンを指定し、アプリターゲットに `HaruKit` を追加します。クローン済みの場合は **Add Local** でリポジトリのディレクトリを選びます。[Swift Packageの追加](https://developer.apple.com/documentation/xcode/adding-package-dependencies-to-your-app)

アプリ側で `NSMicrophoneUsageDescription` を設定し、実機用ターゲットに **Increased Memory Limit** を追加します。[Increased Memory Limit](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit)

```swift
import Foundation
import HaruKit

@MainActor
final class VoiceChat {
    let conversation = HaruConversation()

    func start() async throws {
        let engine = HaruEngine.mlx
        let model = try await engine.prepareModel()
        guard let voice = HaruReferenceVoice.female.url else {
            throw CocoaError(.fileNoSuchFile)
        }
        try conversation.start(
            engine: engine,
            model: model,
            reference: voice,
            instructions: "相手の質問に、普段の日本語で短く答える。"
        )
    }
}
```

- `HaruEngine` で `.mlx` または `.speechCpp` を選びます。`prepareModel(report:)` は保存済みのモデルを再利用し、取得中は進捗を渡します。用意済みのモデルのURLを `start` に渡すこともできます。
- 会話中は `HaruConversation` を保持し、`status` と `failure` を監視します。`mute()` と `resume()` でマイクを切り替え、`stop()` で会話を終了します。
- `reference` にプリセットまたは参照音声のURLを指定します。`instructions` を省略すると既定の指示が使われます。

## 変換済みモデルを使う

設定でエンジンを選び、MLXでは「Haru-JAフォルダーを選ぶ」、speech.cppでは「GGUFファイルを選ぶ」から取り込みます。Finderのファイル共有では `Haru-JA` フォルダーまたは `Haru-4B-Q6_K.gguf` を転送するとアプリが認識します。

Apple Silicon MacでMLX用の `Haru-JA` を作成する場合は、リポジトリのルートで次を実行します。取得済みの公式チェックポイントを使う場合は、`--checkpoint` にそのディレクトリを指定します。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install 'mlx==0.32.4' tokenizers numpy
.venv/bin/python Tools/prepare.py Haru-JA
```

## ライセンス

HaruKitとサンプルアプリはApache License 2.0です。Haru-4BとVibeVoiceのライセンス、speech.cppやMLXなどの依存ライブラリのライセンス、参照音声の出典は同梱のファイルを参照してください。[LICENSE](LICENSE) · [NOTICE](NOTICE) · [第三者ライセンスと音声の出典](Sources/HaruKit/Licenses)
