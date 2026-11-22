"""Exercise exported media and invalid studio inputs without a voice model."""
import io
import json
import zipfile

import numpy as np
import pytest
import soundfile as sf

from services import audio_service as A, profile_service as PS, synthesis_service as SY
from services import tts_service as T, indic_service as I


@pytest.fixture
def studio_run(env):
    slug = PS.create("Export check")["slug"]
    PS.update(slug, enrollment={"status": "enrolled"})
    run_id = "20261122-100000"
    directory = PS.profile_dir(slug) / "outputs" / run_id
    directory.mkdir(parents=True)
    fs = 24000
    audio = (.2 * np.sin(2 * np.pi * 220 * np.arange(fs) / fs)).astype(np.float32)
    sf.write(directory / "final.wav", audio, fs, subtype="PCM_16")
    sf.write(PS.profile_dir(slug) / "reference_clean.wav", audio, fs)
    np.save(PS.profile_dir(slug) / "speaker_embedding.npy", np.zeros(57, dtype=np.float32))
    assert PS.is_enrolled(slug)
    (directory / "metrics.json").write_text(json.dumps({"run_id": run_id, "profile": slug}))
    (directory / "private.pt").write_bytes(b"not-for-export")
    yield env.test_client(), slug, run_id
    PS.delete(slug)


@pytest.mark.parametrize("fmt", ["wav", "flac", "mp3", "ogg", "opus", "m4a", "aiff", "zip"])
def test_export_is_playable_or_contains_only_public_run_files(studio_run, tmp_path, fmt):
    client, slug, run_id = studio_run
    response = client.get(f"/profiles/{slug}/export/{run_id}/{fmt}")
    assert response.status_code == 200
    assert "attachment" in response.headers["Content-Disposition"]
    assert len(response.data) > 100
    if fmt == "zip":
        with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
            assert set(archive.namelist()) == {"final.wav", "metrics.json"}
            assert archive.read("final.wav").startswith(b"RIFF")
    else:
        exported = tmp_path / f"export.{fmt}"
        exported.write_bytes(response.data)
        decoded = tmp_path / "decoded.wav"
        A.transcode(exported, decoded, "pcm_s16le")
        info = sf.info(decoded)
        assert .9 < info.duration < 1.2
        assert info.channels == 1


def test_export_rejects_unknown_format_run_and_profile(studio_run):
    client, slug, run_id = studio_run
    assert client.get(f"/profiles/{slug}/export/{run_id}/exe").status_code == 400
    assert client.get(f"/profiles/{slug}/export/not-a-run/wav").status_code == 404
    assert client.get(f"/profiles/missing/export/{run_id}/wav").status_code == 404


@pytest.mark.parametrize("settings,message", [
    ({"language": "ta"}, "Unsupported language"),
    ({"backend": "missing"}, "Unsupported language"),
    ({"hp_hz": float("nan")}, "finite numbers"),
    ({"lp_hz": float("inf")}, "finite numbers"),
    ({"target_rms_dbfs": -100}, "allowed ranges"),
    ({"lp_hz": 12000}, "allowed ranges"),
])
def test_invalid_studio_settings_never_reach_tts(studio_run, monkeypatch, settings, message):
    _, slug, _ = studio_run

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid settings reached voice generation")

    monkeypatch.setattr(T, "synthesize", forbidden)
    with pytest.raises(SY.SynthesisError, match=message):
        SY.run(slug, "Hello from the voice studio.", use_gemini=False, **settings)


def test_unconfigured_indic_backend_reports_setup_before_loading_audio(monkeypatch):
    monkeypatch.setattr(I, "status", lambda: {"configured": False})
    with pytest.raises(T.TTSUnavailable, match="not installed locally"):
        I.synthesize("வணக்கம்", "ta", "missing.wav", "வணக்கம்", ".")
