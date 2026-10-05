import io
import wave
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import transcription_server as ts


class FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel without downloading weights."""

    def __init__(self, size, device, compute_type, fail_with=None):
        self.size = size
        self.device = device
        self.compute_type = compute_type
        self.fail_with = fail_with
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        if self.fail_with:
            err, self.fail_with = self.fail_with, None

            def failing():
                raise err
                yield  # pragma: no cover

            return failing(), None
        segments = (
            SimpleNamespace(start=0.0, end=1.0, text=" hello"),
            SimpleNamespace(start=1.0, end=2.0, text=" jarvis"),
        )
        info = SimpleNamespace(
            language=kwargs.get("language") or "en", language_probability=0.99, duration=2.0
        )
        # Lazy generator, like the real library.
        return (s for s in segments), info


def make_wav(seconds=0.5, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


@pytest.fixture
def wav_bytes():
    return make_wav()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Runs before test-specific fixtures so tests can opt back in to env vars."""
    monkeypatch.delenv("TRANSCRIPTION_API_KEY", raising=False)
    monkeypatch.delenv("TRANSCRIPTION_MODELS", raising=False)


@pytest.fixture
def created_models():
    return []


@pytest.fixture
def client(monkeypatch, created_models):
    def factory(size, device, compute_type):
        m = FakeWhisperModel(size, device, compute_type)
        created_models.append(m)
        return m

    monkeypatch.setattr(ts, "model_factory", factory)
    monkeypatch.setattr(ts, "cuda_device_count", lambda: 0)
    monkeypatch.setattr(ts, "gpu_name", lambda: "None")
    for key in ("requests_processed", "requests_failed"):
        ts.server_stats[key] = 0
    ts.server_stats["total_processing_time"] = 0.0
    ts.server_stats["total_audio_seconds"] = 0.0
    with TestClient(ts.app) as c:
        yield c
