"""
Record system/loopback audio (WASAPI via pyaudiowpatch) and encode to MP3 with FFmpeg.
Optional local transcription with faster-whisper (default model: large-v3, Japanese).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pyaudiowpatch as pyaudio

from transcription import DEFAULT_LANGUAGE, DEFAULT_MODEL, transcribe_file


def _require_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg が PATH にありません。winget install ffmpeg などでインストールしてください。"
        )


def _get_wasapi_info(p: pyaudio.PyAudio) -> dict[str, Any]:
    return p.get_host_api_info_by_type(pyaudio.paWASAPI)


def _loopback_devices(p: pyaudio.PyAudio) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if d.get("isLoopbackDevice", False):
            devices.append(d)
    return devices


def _default_loopback(p: pyaudio.PyAudio) -> dict[str, Any]:
    wasapi = _get_wasapi_info(p)
    default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    default_name: str = default_out["name"]

    for lb in _loopback_devices(p):
        if lb["name"].startswith(default_name):
            return lb

    raise RuntimeError(
        f"デフォルト出力 '{default_name}' に対応するループバックが見つかりません。"
        " --list-devices の名前を --speaker に指定してください。"
    )


def _resolve_loopback(p: pyaudio.PyAudio, name: str | None) -> dict[str, Any]:
    if not name:
        return _default_loopback(p)
    name_norm = name.strip().lower()
    matches = [
        d for d in _loopback_devices(p) if name_norm in d["name"].lower()
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(repr(d["name"]) for d in matches)
        raise ValueError(f"複数マッチしました。名前を具体化してください: {names}")
    raise ValueError(f"ループバックが見つかりません: {name!r}（--list-devices で確認）")


def _list_recording_targets() -> None:
    p = pyaudio.PyAudio()
    try:
        print("ループバック録音に使える出力デバイス（--speaker に名前の一部を指定）:")
        for d in _loopback_devices(p):
            rate = int(d["defaultSampleRate"])
            ch = d["maxInputChannels"]
            print(f"  - {d['name']}  ({rate} Hz, {ch} ch)")
    finally:
        p.terminate()


def record_to_wav(
    wav_path: Path,
    duration_sec: float | None,
    speaker_name: str | None,
) -> None:
    p = pyaudio.PyAudio()
    try:
        device = _resolve_loopback(p, speaker_name)
        rate = int(device["defaultSampleRate"])
        channels = device["maxInputChannels"]
        sample_width = p.get_sample_size(pyaudio.paInt16)

        print(f"デバイス: {device['name']}", file=sys.stderr)
        if duration_sec is not None:
            print(f"録音中… {duration_sec:.0f} 秒", file=sys.stderr, flush=True)
        else:
            print("録音中… 終了は Ctrl+C", file=sys.stderr, flush=True)

        frames: list[bytes] = []

        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=device["index"],
            frames_per_buffer=1024,
        )

        try:
            if duration_sec is not None:
                total_frames = int(rate / 1024 * duration_sec)
                for _ in range(total_frames):
                    frames.append(stream.read(1024, exception_on_overflow=False))
            else:
                try:
                    while True:
                        frames.append(stream.read(1024, exception_on_overflow=False))
                except KeyboardInterrupt:
                    print("\n停止しました。", file=sys.stderr)
        finally:
            stream.stop_stream()
            stream.close()

        if not frames:
            raise RuntimeError("録音データがありません。")

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))

    finally:
        p.terminate()


def wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(wav_path),
        "-c:a", "libmp3lame",
        "-q:a", "2",
        str(mp3_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="ループバック録音 → MP3（任意で文字起こし）")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("recording.mp3"),
        help="出力 MP3 パス（既定: recording.mp3）",
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=None,
        help="録音秒数（省略時は Ctrl+C まで）",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        help="ループバック対象の出力デバイス名（部分一致）",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="ループバック録音に使えるデバイス一覧を表示して終了",
    )
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help=f"録音後に文字起こし（モデル既定: {DEFAULT_MODEL}、言語既定: {DEFAULT_LANGUAGE}）",
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

    if args.list_devices:
        _list_recording_targets()
        return

    _require_ffmpeg()
    out_mp3 = args.output.resolve()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    try:
        record_to_wav(wav_path, args.duration, args.speaker)
        wav_to_mp3(wav_path, out_mp3)
        size_kb = out_mp3.stat().st_size / 1024
        print(f"保存: {out_mp3}  ({size_kb:.1f} KB)", file=sys.stderr)
    except Exception as e:
        print(f"エラー: {e}", file=sys.stderr)
        raise
    finally:
        wav_path.unlink(missing_ok=True)

    if args.transcribe:
        stem = out_mp3.with_suffix("")
        transcribe_file(
            out_mp3,
            txt_path=Path(f"{stem}.txt"),
            srt_path=Path(f"{stem}.srt"),
            model_size=args.model,
            language=args.language,
        )
        print(f"文字起こし完了: {stem}.txt / {stem}.srt", file=sys.stderr)


if __name__ == "__main__":
    main()
