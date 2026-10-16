import json, numpy as np
import config
from synth_voice import make_voice
from conftest import VOICES
from services import profile_service as PS, synthesis_service as SY, tts_service as T


def test_all_pages_and_404s(enrolled):
    c = enrolled
    for u in ("/", "/profiles", "/experiments", "/api/status", "/profiles/alice/enroll", "/profiles/alice/analysis",
              "/profiles/alice/synthesize", "/profiles/alice/results", "/profiles/bob/enroll"):
        assert c.get(u).status_code == 200, u
    assert c.get("/profiles/nobody/enroll").status_code == 404
    assert c.get("/profiles/..%2f..%2fetc/enroll").status_code == 404


def test_file_serving_is_restricted(enrolled):
    c = enrolled
    assert c.get("/files/alice/reference.wav").status_code == 200
    assert c.get("/files/alice/analysis/python/spectrogram.png").status_code == 200
    (PS.profile_dir("alice") / "xtts_latents.pt").write_bytes(b"x")
    for bad in ("/files/alice/xtts_latents.pt", "/files/alice/speaker_embedding.npy", "/files/alice/metadata.json.bak",
                "/files/alice/../bob/reference.wav", "/files/alice/%2e%2e/bob/reference.wav"):
        assert c.get(bad).status_code == 404, bad
    (PS.profile_dir("alice") / "xtts_latents.pt").unlink()


def test_analysis_page_python_and_matlab_unavailable(enrolled):
    r = enrolled.post("/profiles/bob/analysis", data={"backend": "python"}, follow_redirects=True)
    assert b"DSP analysis finished" in r.data and b"f0_mean_hz" in r.data
    r = enrolled.post("/profiles/bob/analysis", data={"backend": "matlab"}, follow_redirects=True)
    from services import matlab_service as M
    if not M.available()["any"]:
        assert b"Analysis failed" in r.data


def _stub_tts(calls, voices):
    def stub(slug, sentences, language="en", **kw):          # TEST DOUBLE: stands in for XTTS (not installed in CI)
        calls.append((slug, list(sentences), language))
        y, fs = make_voice(6, fs=24000, **{**voices[slug], 'seed': 11})
        return y, 24000, dict(model="TEST-DOUBLE", device="cpu", sample_rate=24000, inference_time_s=0.01,
                              audio_duration_s=6.0, real_time_factor=0.0, conditioning="stub", sentences=len(sentences), temperature=0.65)
    return stub


def test_full_synthesis_pipeline_per_profile_with_stub_tts(enrolled, monkeypatch):
    calls = []; monkeypatch.setattr(T, "synthesize", _stub_tts(calls, VOICES))
    c = enrolled
    ra = c.post("/profiles/alice/synthesize", data={"text": "Hello there. I have 3 cats.", "language": "en"}, follow_redirects=True)
    rb = c.post("/profiles/bob/synthesize", data={"text": "A completely different sentence.", "language": "en"}, follow_redirects=True)
    assert ra.status_code == rb.status_code == 200 and b"Generated audio" in ra.data and b"Generated audio" in rb.data
    assert [x[0] for x in calls] == ["alice", "bob"]                    # each request used only its own profile
    assert "three cats" in calls[0][1][1] and "different" in calls[1][1][0]
    for s in ("alice", "bob"):
        runs = PS.list_runs(s); assert len(runs) == 1
        m = SY.load_run(s, runs[0]); d = PS.profile_dir(s) / "outputs" / runs[0]
        assert m["profile"] == s and m["tts"]["model"] == "TEST-DOUBLE"
        for f in ("generated.wav", "final.wav", "final_compressed.wav", "compressed_stft_keep30.bin", "compression_rd.png", "spectrum_compare.png", "metrics.json"):
            assert (d / f).stat().st_size > 100, f
        assert m["verification"]["encoder"] == "mfcc-stats" and m["verification"]["cosine_to_claimed_profile"] is not None
        assert abs(m["postprocess"]["rms_after_dbfs"] + 20) < 1.0 and m["compression_default"]["ratio_zlib"] > 1
        assert len(m["compression"]) == 10 and {r["codec"] for r in m["compression"]} == {"STFT-threshold", "DCT-threshold"}
        assert m["file_bytes"]["final.wav"] > 0 and m["timings"]["tts_s"] == 0.01
        assert c.get(f"/files/{s}/outputs/{runs[0]}/final.wav").status_code == 200
    # The stub generated each run with that profile's own synthetic voice; the verifier must tell them apart.
    va = SY.load_run("alice", PS.list_runs("alice")[0])["verification"]; vb = SY.load_run("bob", PS.list_runs("bob")[0])["verification"]
    assert va["svm_majority_label"] == "alice" and vb["svm_majority_label"] == "bob"
    assert c.get(f"/profiles/alice/results/{PS.list_runs('alice')[0]}").status_code == 200


def test_synthesis_input_validation(enrolled, monkeypatch):
    monkeypatch.setattr(T, "synthesize", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")))
    c = enrolled
    assert b"Enter some text" in c.post("/profiles/alice/synthesize", data={"text": "   "}, follow_redirects=True).data
    assert b"too long" in c.post("/profiles/alice/synthesize", data={"text": "a " * (config.MAX_TEXT_CHARS)}, follow_redirects=True).data
    PS.create("Newbie")
    assert b"not enrolled" in c.post("/profiles/newbie/synthesize", data={"text": "hi"}, follow_redirects=True).data
    PS.delete("newbie")


def test_experiments_page_after_training(enrolled):
    c = enrolled
    assert b"Hold-out accuracy" in c.post("/experiments/classifier", follow_redirects=True).data
    assert b"model_bytes_int8" in c.post("/experiments/quantize", data={"which": "classifier"}, follow_redirects=True).data
    r = c.post("/experiments/compression", data={"slug": "alice"}, follow_redirects=True)
    assert b"STFT-threshold" in r.data and b"DCT-threshold" in r.data
    assert c.get("/exp-files/pca_mfcc-stats.png").status_code == 200 and c.get("/exp-files/x.txt").status_code == 404
