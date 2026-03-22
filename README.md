# movie2text

Windows でスピーカーに出ている音をループバック録音し、MP3 にします。任意でローカル文字起こし（**faster-whisper**、既定モデル **large-v3**、言語 **ja**）も実行できます。料金はかかりませんが、初回はモデルダウンロードと推論に CPU/GPU 負荷がかかります。

## 前提

- Python 3.10 以上
- **FFmpeg をインストールしていること**（ターミナルから `ffmpeg` コマンドが実行できること。PATH に `ffmpeg.exe` が含まれている状態）
- 仮想環境（venv）を使うこと

## セットアップ

### FFmpegのインストール

ループバック録音から MP3 へ変換する処理で [FFmpeg](https://ffmpeg.org/) を使います。次のいずれかで入れ、**新しい PowerShell またはターミナルを開き直して**から `ffmpeg -version` で確認してください。

1. **winget（推奨）**  
   管理者不要で使えることが多いです。

   ```powershell
   winget install --id Gyan.FFmpeg
   ```

   パッケージ名は環境により `ffmpeg` だけでも検索・インストールできる場合があります。

   ```powershell
   winget search ffmpeg
   winget install ffmpeg
   ```

2. **Chocolatey**（導入済みの場合）

   ```powershell
   choco install ffmpeg
   ```

3. **手動インストール**  
   [FFmpeg の公式ダウンロード](https://ffmpeg.org/download.html)の案内から、Windows 用ビルド（例: [gyan.dev のビルド](https://www.gyan.dev/ffmpeg/builds/)）を zip で取得し、展開先の `bin` フォルダ（`ffmpeg.exe` がある場所）を **環境変数 PATH** に追加します。

**動作確認**

```powershell
ffmpeg -version
```

バージョン情報が表示されれば、録音スクリプトから利用できます。

```powershell
cd c:\Users\Fantas03\Documents\cursor_project\movie2text
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## ループバック録音 → MP3

PowerShell では **コマンド全体を 1 行で** 入力してください。`record_loopback_to_mp3.py` の直後で改行し、次の行に `-d` だけを書くと、オプションが別コマンドとして解釈され、`out.mp3` が作られないことがあります（改行なしで `python record_loopback_to_mp3.py -d 20 -o out.mp3`）。

```powershell
# 一覧（Windows は「ループバック」として録音に使う出力デバイス名）
python record_loopback_to_mp3.py --list-devices

# 60 秒録音
python record_loopback_to_mp3.py -d 60 -o out.mp3

# Ctrl+C まで録音
python record_loopback_to_mp3.py -o out.mp3

# 録音後に文字起こし（large-v3 / 日本語）
python record_loopback_to_mp3.py -d 120 -o out.mp3 --transcribe
```

特定の出力デバイスを使う場合: `--speaker "一覧に出た名前の一部"`（Windows はスピーカー本体ではなく、ループバック用エントリにマッチします）。

**注意**: DRM 付きストリーミングなどではループバックが無音になることがあります。精度や負荷で合わない場合は、生成した MP3 を NotebookLM 等に渡してください。


## 既存の MP3 だけ文字起こし

```powershell
python transcribe_mp3.py path\to\audio.mp3
```

**実行直後のメッセージについて**: `HF_TOKEN` やシンボリックリンクに関する警告は **そのまま処理は続きます**（未ログインでもダウンロードは可能です）。そのあと画面に何も出ない時間が続くのは次のいずれかです。

1. **初回のみ** — `large-v3` 用のモデルが **Hugging Face から数GB** ダウンロードされている（回線・ディスクで **10分以上** かかることもあります）。`%USERPROFILE%\.cache\huggingface\hub\` の容量や、タスクマネージャーのネットワーク・ディスクで動きを確認できます。
2. **2回目以降** — ダウンロードは省略され、**CPU での推論**が主な待ち時間です。長い MP3 では **実時間の数倍** かかることも珍しくありません。GPU がある環境では自動で CUDA が使われ、だいぶ速くなります。

処理中は標準エラーに `[文字起こし]` で始まる進捗が出ます。

出力は同じフォルダに `audio.txt` と `audio.srt`（入力ファイル名の stem）。別フォルダへ出す場合:

```powershell
python transcribe_mp3.py path\to\audio.mp3 -o C:\work\out
```

出力パスを直接指定:

```powershell
python transcribe_mp3.py path\to\audio.mp3 --txt C:\work\script.txt --srt C:\work\subs.srt
```

モデルや言語を変える場合（通常は既定のままで可）:

```powershell
python transcribe_mp3.py audio.mp3 --model large-v3 --language ja
```

## テスト（任意）

FFmpeg とループバック周りの最低限の動作確認です。

```powershell
pip install -r requirements-dev.txt
pytest tests\test_smoke.py -v
```

`ffmpeg` が PATH にない環境では `test_wav_to_mp3` はスキップされます。
