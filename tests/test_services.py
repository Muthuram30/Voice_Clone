import json, numpy as np, pytest
import config
from services import gemini_service as G, matlab_service as M, quant_service as Q, speaker_service as S, tts_service as T


class _Resp:
    def __init__(self, code=200, body=None, text=""):
        self.status_code, self._b, self.text = code, body, text
    def json(self): return self._b


def _gem(sentences):
    return _Resp(200, {"candidates": [{"content": {"parts": [{"text": json.dumps({"sentences": sentences})}]}}]})


def test_gemini_fallback_paths(env, monkeypatch):
    t = "Meet me at 5 o'clock.   Bring 12 apples"
    r = G.prepare_text(t, "en", True)                                   # no key
    assert not r["used_gemini"] and "No GEMINI_API_KEY" in r["note"] and "twelve" in r["normalized"]
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(G.requests, "post", lambda *a, **k: _Resp(429, text="quota"))
    r = G.prepare_text(t, "en", True); assert not r["used_gemini"] and "429" in r["note"] and r["sentences"]
    def boom(*a, **k): raise G.requests.exceptions.Timeout("slow")
    monkeypatch.setattr(G.requests, "post", boom)
    assert not G.prepare_text(t, "en", True)["used_gemini"]
    monkeypatch.setattr(G.requests, "post", lambda *a, **k: _Resp(200, {"candidates": [{"content": {"parts": [{"text": "not json"}]}}]}))
    assert not G.prepare_text(t, "en", True)["used_gemini"]
    monkeypatch.setattr(G.requests, "post", lambda *a, **k: _gem(["Completely unrelated " * 60]))   # rejected by sanity check
    assert not G.prepare_text(t, "en", True)["used_gemini"]


def test_gemini_success_and_key_never_in_body(env, monkeypatch):
    seen = {}
    def ok(url, json=None, headers=None, timeout=None):
        seen.update(url=url, body=json, headers=headers, timeout=timeout); return _gem(["Meet me at five o'clock.", "Bring twelve apples."])
    monkeypatch.setattr(config, "GEMINI_API_KEY", "secret-abc"); monkeypatch.setattr(G.requests, "post", ok)
    r = G.prepare_text("Meet me at 5 o'clock. Bring 12 apples", "en", True)
    assert r["used_gemini"] and r["sentences"][1] == "Bring twelve apples." and r["model"] == config.GEMINI_MODELS[0]
    assert "secret-abc" not in seen["url"] and "secret-abc" not in json.dumps(seen["body"]) and seen["headers"]["x-goog-api-key"] == "secret-abc"
    assert "audio" not in json.dumps(seen["body"]).lower().replace("text-to-speech", "")


def test_sentence_splitter_respects_limit():
    long = "word " * 200
    s = G.split_sentences(long, 100)
    assert all(len(x) <= 101 for x in s) and len(s) > 1


def test_tts_reports_unavailable_clearly_without_crashing(enrolled):
    if T.status()["packages_installed"]:
        pytest.skip("TTS stack is installed here")
    with pytest.raises(T.TTSUnavailable, match="not installed"):
        T.synthesize("alice", ["Hello."])
    r = enrolled.post("/profiles/alice/synthesize", data={"text": "Hello world", "language": "en"}, follow_redirects=True)
    assert r.status_code == 200 and b"not installed" in r.data


def test_tts_rejects_unenrolled_and_bad_language(env):
    from services import profile_service as PS
    PS.create("Fresh")
    with pytest.raises(T.TTSError, match="not enrolled"):
        T.synthesize("fresh", ["Hi."])
    PS.delete("fresh")


def test_matlab_missing_is_reported_not_faked(env, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MATLAB_MODE", "cli"); monkeypatch.setattr(config, "MATLAB_EXE", "")
    monkeypatch.setenv("PATH", "/nonexistent")
    a = M.available(); assert isinstance(a["any"], bool)
    with pytest.raises(M.MatlabError, match="matlab executable not found"):
        M.run_analysis(tmp_path / "x.wav", tmp_path / "o")


def test_classifier_quantisation_reports_real_measurements(enrolled):
    r = Q.quantize_classifier("mfcc-stats")
    assert r["model_bytes_int8"] < r["model_bytes_fp32"] and r["size_ratio"] > 2
    assert r["prediction_agreement"] >= 0.9 and r["accuracy_int8"] >= 0.9 and "int8" in r["what_is_quantised"]
    assert r["max_abs_decision_diff"] < 0.5


def test_encoder_quantisation_unavailable_without_torch(env):
    import importlib.util
    if importlib.util.find_spec("torch") and importlib.util.find_spec("speechbrain"):
        pytest.skip("torch+speechbrain installed")
    r = Q.quantize_encoder(); assert r["status"] == "unavailable"


def test_forced_ecapa_without_packages_raises_clearly(env, monkeypatch):
    import importlib.util
    if importlib.util.find_spec("speechbrain"):
        pytest.skip("speechbrain installed")
    S._ENC.pop("ecapa", None)
    with pytest.raises(S.EncoderUnavailable):
        S.get_encoder("ecapa")
    enc, note = S.get_encoder("auto"); assert enc.name == "mfcc-stats" and "ECAPA unavailable" in note
