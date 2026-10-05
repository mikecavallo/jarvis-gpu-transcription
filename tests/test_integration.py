"""End-to-end test against a real Whisper model. Downloads ~75 MB on first run.

Run with: pytest -m integration
"""

import pytest
from fastapi.testclient import TestClient

import transcription_server as ts
from conftest import make_wav

pytestmark = pytest.mark.integration


def test_real_tiny_model_transcribes_silence(monkeypatch):
    monkeypatch.setenv("TRANSCRIPTION_MODELS", "tiny")
    monkeypatch.setattr(ts, "model_factory", ts.default_model_factory)
    with TestClient(ts.app) as c:
        r = c.post("/transcribe/fast", files={"audio_file": ("silence.wav", make_wav(2.0), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["model_used"] == "tiny"
    assert body["duration"] == pytest.approx(2.0, abs=0.1)
    assert isinstance(body["text"], str)
