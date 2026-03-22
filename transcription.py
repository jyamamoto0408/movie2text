"""Shared faster-whisper transcription (default: large-v3, Japanese)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

DEFAULT_MODEL = "large-v3"
DEFAULT_LANGUAGE = "ja"


def _pick_device_and_compute_type() -> tuple[str, str]:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def _format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(lines: Iterable[tuple[float, float, str]]) -> str:
    blocks: list[str] = []
    for i, (start, end, text) in enumerate(lines, start=1):
        t = text.strip()
        if not t:
            continue
        blocks.append(
            f"{i}\n{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}\n{t}\n"
        )
    return "\n".join(blocks) + ("\n" if blocks else "")


def transcribe_file(
    audio_path: Path,
    *,
    txt_path: Path,
    srt_path: Path,
    model_size: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
    device: str | None = None,
    compute_type: str | None = None,
) -> None:
    from faster_whisper import WhisperModel

    audio_path = audio_path.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    if device is None or compute_type is None:
        dev, ct = _pick_device_and_compute_type()
        device = device or dev
        compute_type = compute_type or ct

    print(
        f"[文字起こし] モデル '{model_size}' を準備中… "
        f"（初回は Hugging Face から数GBの取得があり、回線次第で10分以上かかることがあります）",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[文字起こし] デバイス: {device}, compute_type: {compute_type}",
        file=sys.stderr,
        flush=True,
    )
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print("[文字起こし] 音声の解析を開始しました…", file=sys.stderr, flush=True)

    segments, _info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
    )

    lines: list[tuple[float, float, str]] = []
    full_text_parts: list[str] = []
    last_log_end = -1.0
    for seg in segments:
        lines.append((seg.start, seg.end, seg.text))
        full_text_parts.append(seg.text.strip())
        if seg.end >= last_log_end + 30.0:
            print(
                f"[文字起こし] 進捗: 音声の約 {seg.end:.0f} 秒付近まで処理しました",
                file=sys.stderr,
                flush=True,
            )
            last_log_end = seg.end

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.parent.mkdir(parents=True, exist_ok=True)

    txt_path.write_text("\n".join(t for t in full_text_parts if t) + "\n", encoding="utf-8")
    srt_path.write_text(segments_to_srt(lines), encoding="utf-8")
    print("[文字起こし] 完了。txt / srt を書き出しました。", file=sys.stderr, flush=True)
