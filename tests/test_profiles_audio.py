import io, numpy as np, pytest, soundfile as sf
import config
from services import profile_service as PS, audio_service as A
from conftest import upload


def test_profile_ids_and_path_safety(env):
    assert PS.slugify("User 1") == "user-1"
    for bad in ("../etc", "a/b", "", "A", "x" * 40, "..", "a b"):
        assert not PS.valid_slug(bad)
        with pytest.raises(PS.ProfileError):
            PS.profile_dir(bad)
    m = PS.create("Zed"); assert m["slug"] == "zed"
    with pytest.raises(PS.ProfileError):
        PS.create("zed")
    PS.delete("zed"); assert not PS.exists("zed")


def test_validation_rejects_short_silent_and_bad_files(env, wavs):
    c = env.test_client(); c.post("/profiles", data={"name": "Val"})
    def wav(y, fs=16000):
        b = io.BytesIO(); sf.write(b, y.astype(np.float32), fs, format="WAV"); return b.getvalue()
    r = upload(c, "val", wav(np.sin(np.arange(16000 * 20) / 20) * 0.5)); assert b"at least" in r.data          # 20 s
    r = upload(c, "val", wav(np.zeros(16000 * 90))); assert b"silent" in r.data
    r = upload(c, "val", wav(np.random.default_rng(0).standard_normal(16000 * 90) * 0.001)); assert (b"speech" in r.data or b"silent" in r.data)
    r = upload(c, "val", b"not audio", "x.wav"); assert b"Invalid or unreadable" in r.data or b"Could not decode" in r.data
    r = upload(c, "val", b"abc", "x.mp3"); assert b"Unsupported" in r.data
    assert not (PS.profile_dir("val") / "reference.wav").exists()      # nothing rejected was kept
    r = upload(c, "val", wavs["alice"]); assert b"Recording accepted" in r.data
    assert PS.get("val")["enrollment"]["status"] == "uploaded"
    PS.delete("val")
