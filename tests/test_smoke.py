"""Smoke tests for ffmpeg pipeline and path defaults."""

from __future__ import annotations

import io
import re
import shutil
import struct
import sys
import wave
from contextlib import redirect_stderr
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_wav_to_mp3(tmp_path: Path) -> None:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg 縺・PATH 縺ｫ縺ゅｊ縺ｾ縺帙ｓ")

    import record_loopback_to_mp3 as rec

    wav = tmp_path / "tone.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<h", 100) * 800)

    mp3 = tmp_path / "tone.mp3"
    rec.wav_to_mp3(wav, mp3)
    assert mp3.is_file()
    assert mp3.stat().st_size > 0


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_list_devices_does_not_crash() -> None:
    import record_loopback_to_mp3 as rec

    buf = io.StringIO()
    with redirect_stderr(buf):
        rec._list_recording_targets()


def test_resolve_output_mp3_path(tmp_path: Path) -> None:
    import record_loopback_to_mp3 as rec

    p = rec.resolve_output_mp3_path(None, None)
    assert re.fullmatch(r"\d{12}\.mp3", p.name)
    assert p.parent.name == "output"

    q = rec.resolve_output_mp3_path(Path("clip.mp3"), tmp_path)
    assert q.name == "clip.mp3"
    assert q.parent.resolve() == tmp_path.resolve()


def test_transcription_helpers() -> None:
    from transcription import DEFAULT_LANGUAGE, DEFAULT_MODEL, segments_to_srt

    assert DEFAULT_MODEL == "large-v3"
    assert DEFAULT_LANGUAGE == "ja"
    srt = segments_to_srt([(0.0, 1.0, "hello")])
    assert "1" in srt
    assert "hello" in srt

