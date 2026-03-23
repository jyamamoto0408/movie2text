"""Transcribe an existing audio file with faster-whisper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transcription import DEFAULT_LANGUAGE, DEFAULT_MODEL, transcribe_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe existing audio")
    parser.add_argument("audio", type=Path, help="Input audio file (mp3/wav/...)")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for txt/srt (default: ./output)",
    )
    parser.add_argument("--txt", type=Path, default=None, help="Explicit output txt path")
    parser.add_argument("--srt", type=Path, default=None, help="Explicit output srt path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Whisper model (default: {DEFAULT_MODEL})")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help=f"Language code (default: {DEFAULT_LANGUAGE})")
    args = parser.parse_args()

    audio = args.audio.resolve()
    if not audio.is_file():
        print(f"Audio file not found: {audio}", file=sys.stderr)
        sys.exit(1)

    base_dir = args.output_dir.resolve() if args.output_dir else (Path.cwd() / "output").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

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

