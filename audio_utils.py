"""Optional offline speech recognition and speech synthesis helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def transcribe_audio(audio: bytes, model_size: str = "base") -> str:
    """Transcribe browser-recorded audio locally with faster-whisper."""
    if not audio:
        raise ValueError("No audio was recorded.")
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Voice transcription is optional. Install faster-whisper to enable it.") from exc

    fd, filename = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(audio)
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(filename, beam_size=3)
        return " ".join(segment.text.strip() for segment in segments).strip()
    finally:
        Path(filename).unlink(missing_ok=True)


def synthesize_speech(text: str, allow_online_fallback: bool = False, language: str = "en") -> tuple[bytes, str]:
    """Prefer offline pyttsx3; use gTTS only after the user enables network fallback."""
    if not text.strip():
        raise ValueError("There is no reply text to speak.")
    try:
        import pyttsx3

        fd, filename = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            engine = pyttsx3.init()
            engine.save_to_file(text, filename)
            engine.runAndWait()
            data = Path(filename).read_bytes()
            if data:
                return data, "audio/wav"
        finally:
            Path(filename).unlink(missing_ok=True)
    except (ImportError, OSError, RuntimeError):
        pass

    if not allow_online_fallback:
        raise RuntimeError(
            "Offline speech synthesis is unavailable. Install pyttsx3 or explicitly enable gTTS "
            "(which sends text to Google's service)."
        )
    try:
        from gtts import gTTS

        import io

        buffer = io.BytesIO()
        gTTS(text=text, lang=language).write_to_fp(buffer)
        return buffer.getvalue(), "audio/mpeg"
    except (ImportError, OSError, ValueError) as exc:
        raise RuntimeError("Speech synthesis failed. Check the optional audio dependencies and network.") from exc
