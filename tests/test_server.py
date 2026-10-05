import pytest
from fastapi.testclient import TestClient

import transcription_server as ts
from conftest import FakeWhisperModel


def upload(wav_bytes, name="clip.wav", content_type="audio/wav"):
    return {"audio_file": (name, wav_bytes, content_type)}


def test_health_reports_loaded_models(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["models_available"] == 4
    assert body["gpu_available"] is False


def test_dashboard_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Jarvis GPU Transcription Server" in r.text
    assert "tiny" in r.text


def test_transcribe_returns_text_and_segments(client, wav_bytes):
    r = client.post("/transcribe", files=upload(wav_bytes), data={"model": "small", "language": "en"})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "hello jarvis"
    assert body["model_used"] == "small"
    assert body["duration"] == 2.0
    # Segments come from a lazy generator; they must survive building the text.
    assert [s["text"] for s in body["segments"]] == [" hello", " jarvis"]


def test_form_fields_are_respected(client, wav_bytes, created_models):
    client.post(
        "/transcribe", files=upload(wav_bytes), data={"model": "tiny", "language": "es", "task": "translate"}
    )
    tiny = next(m for m in created_models if m.size == "tiny")
    _, kwargs = tiny.calls[-1]
    assert kwargs["language"] == "es"
    assert kwargs["task"] == "translate"
    assert kwargs["beam_size"] == 3


def test_auto_language_passes_none(client, wav_bytes, created_models):
    client.post("/transcribe", files=upload(wav_bytes), data={"language": "auto"})
    base = next(m for m in created_models if m.size == "base")
    assert base.calls[-1][1]["language"] is None


def test_fast_and_accurate_pick_smallest_and_largest(client, wav_bytes):
    assert client.post("/transcribe/fast", files=upload(wav_bytes)).json()["model_used"] == "tiny"
    assert client.post("/transcribe/accurate", files=upload(wav_bytes)).json()["model_used"] == "medium"


def test_unknown_model_rejected(client, wav_bytes):
    r = client.post("/transcribe", files=upload(wav_bytes), data={"model": "huge"})
    assert r.status_code == 400
    assert "not available" in r.json()["detail"]


def test_bad_task_rejected(client, wav_bytes):
    r = client.post("/transcribe", files=upload(wav_bytes), data={"task": "summarize"})
    assert r.status_code == 400


def test_non_audio_rejected(client):
    r = client.post("/transcribe", files=upload(b"hello", "notes.txt", "text/plain"))
    assert r.status_code == 400


def test_empty_file_rejected(client):
    r = client.post("/transcribe", files=upload(b""))
    assert r.status_code == 400


def test_temp_file_keeps_extension_and_is_deleted(client, wav_bytes, created_models):
    import os

    client.post("/transcribe", files=upload(wav_bytes, "clip.mp3", "audio/mpeg"))
    path = next(m for m in created_models if m.size == "base").calls[-1][0]
    assert path.endswith(".mp3")
    assert not os.path.exists(path)


def test_stats_and_metrics_track_requests(client, wav_bytes):
    client.post("/transcribe/fast", files=upload(wav_bytes))
    client.post("/transcribe", files=upload(wav_bytes), data={"model": "nope"})
    stats = client.get("/stats").json()
    assert stats["requests_processed"] == 1
    assert stats["total_audio_seconds"] == 2.0
    metrics = client.get("/metrics").text
    assert "transcription_requests_total 1" in metrics
    assert "transcription_models_loaded 4" in metrics


def test_models_configurable_via_env(monkeypatch, client):
    monkeypatch.setenv("TRANSCRIPTION_MODELS", "base")
    with TestClient(ts.app) as c:
        assert c.get("/health").json()["models_available"] == 1
        assert c.post("/transcribe/fast", files=upload(b"RIFF....")).json()["model_used"] == "base"


class TestApiKey:
    @pytest.fixture(autouse=True)
    def set_key(self, monkeypatch):
        monkeypatch.setenv("TRANSCRIPTION_API_KEY", "s3cret")

    def test_missing_key_rejected(self, client, wav_bytes):
        assert client.post("/transcribe", files=upload(wav_bytes)).status_code == 401
        assert client.get("/stats").status_code == 401
        assert client.get("/metrics").status_code == 401

    def test_wrong_key_rejected(self, client, wav_bytes):
        r = client.post("/transcribe", files=upload(wav_bytes), headers={"X-API-Key": "nope"})
        assert r.status_code == 401

    def test_correct_key_accepted(self, client, wav_bytes):
        r = client.post("/transcribe", files=upload(wav_bytes), headers={"X-API-Key": "s3cret"})
        assert r.status_code == 200

    def test_health_stays_open(self, client):
        assert client.get("/health").status_code == 200


def test_gpu_load_failure_falls_back_to_cpu(monkeypatch):
    devices = []

    def factory(size, device, compute_type):
        devices.append((device, compute_type))
        if device == "cuda":
            raise RuntimeError("Could not load library libcudnn_ops_infer.so.8")
        return FakeWhisperModel(size, device, compute_type)

    monkeypatch.setattr(ts, "model_factory", factory)
    monkeypatch.setattr(ts, "cuda_device_count", lambda: 1)
    monkeypatch.setattr(ts, "gpu_name", lambda: "Fake RTX")
    with TestClient(ts.app) as c:
        body = c.get("/health").json()
    assert body["models_available"] == 4
    assert body["gpu_available"] is False
    assert devices[0] == ("cuda", "float16")
    assert devices[-1] == ("cpu", "int8")


def test_gpu_runtime_error_reloads_on_cpu_and_retries(monkeypatch, wav_bytes):
    def factory(size, device, compute_type):
        fail = (
            RuntimeError("cuDNN failed with status CUDNN_STATUS_EXECUTION_FAILED")
            if device == "cuda"
            else None
        )
        return FakeWhisperModel(size, device, compute_type, fail_with=fail)

    monkeypatch.setattr(ts, "model_factory", factory)
    monkeypatch.setattr(ts, "cuda_device_count", lambda: 1)
    with TestClient(ts.app) as c:
        assert c.get("/health").json()["gpu_available"] is True
        r = c.post("/transcribe", files=upload(wav_bytes))
        assert r.status_code == 200
        assert r.json()["text"] == "hello jarvis"
        assert c.get("/health").json()["gpu_available"] is False


def test_non_gpu_error_returns_500(monkeypatch, wav_bytes):
    def factory(size, device, compute_type):
        return FakeWhisperModel(size, device, compute_type, fail_with=ValueError("corrupt audio"))

    monkeypatch.setattr(ts, "model_factory", factory)
    monkeypatch.setattr(ts, "cuda_device_count", lambda: 0)
    ts.server_stats["requests_failed"] = 0
    with TestClient(ts.app) as c:
        r = c.post("/transcribe", files=upload(wav_bytes))
        assert r.status_code == 500
        assert ts.server_stats["requests_failed"] == 1
