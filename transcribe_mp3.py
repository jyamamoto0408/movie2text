"""Transcribe an existing MP3 (or other audio file supported by faster-whisper)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transcription import DEFAULT_LANGUAGE, DEFAULT_MODEL, transcribe_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="既存の MP3 などから文字起こし（既定モデル: large-v3、言語: ja）",
    )
    parser.add_argument(
        "audio",
        type=Path,
        help="入力音声ファイル（MP3/WAV など）",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="txt/srt の出力ディレクトリ（既定: 入力と同じフォルダ）",
    )
    parser.add_argument(
        "--txt",
        type=Path,
        default=None,
        help="出力テキストのパス（指定時は --output-dir より優先）",
    )
    parser.add_argument(
        "--srt",
        type=Path,
        default=None,
        help="出力 SRT のパス（指定時は --output-dir より優先）",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Whisper モデル（既定: {DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"言語コード（既定: {DEFAULT_LANGUAGE}）",
    )
    args = parser.parse_args()

    audio = args.audio.resolve()
    if not audio.is_file():
        print(f"ファイルが見つかりません: {audio}", file=sys.stderr)
        sys.exit(1)

    base_dir = args.output_dir.resolve() if args.output_dir else audio.parent
    stem = audio.stem
    txt_path = args.txt.resolve() if args.txt else (base_dir / f"{stem}.txt")
    srt_path = args.srt.resolve() if args.srt else (base_dir / f"{stem}.srt")

    transcribe_file(
        audio,
        txt_path=txt_path,
        srt_path=srt_path,
        model_size=args.model,
        language=args.language,
    )
    print(str(txt_path))
    print(str(srt_path))


if __name__ == "__main__":
    main()
