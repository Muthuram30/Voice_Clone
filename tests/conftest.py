import io, sys, tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
import config
from synth_voice import make_voice

VOICES = {
    "alice": dict(f0=115, formants=((700, 90), (1220, 110), (2600, 160)), seed=1),
    "bob": dict(f0=210, formants=((450, 80), (1900, 130), (3100, 180)), seed=2),
}


@pytest.fixture(scope="module")
def env():
    """Isolated profiles/models/data dirs, MFCC encoder, Python DSP, no Gemini key."""
    mp = pytest.MonkeyPatch()
    tmp = Path(tempfile.mkdtemp())
    for k, v in dict(PROFILES_DIR=tmp / "profiles", MODELS_DIR=tmp / "models", DATA_DIR=tmp / "data",
                     ENCODER_BACKEND="mfcc", DSP_BACKEND="python", GEMINI_API_KEY="", SEED_DEFAULT_PROFILES=False).items():
        mp.setattr(config, k, v)
    from app import create_app
    app = create_app(); app.config["TESTING"] = True
    yield app
    mp.undo()


@pytest.fixture(scope="session")
def wavs():
    out = {}
    for name, kw in VOICES.items():
        y, fs = make_voice(100, **kw)
        b = io.BytesIO(); sf.write(b, y, fs, format="WAV", subtype="PCM_16"); out[name] = b.getvalue()
    return out


def upload(client, slug, data, name="ref.wav"):
    return client.post(f"/profiles/{slug}/upload", data={"audio": (io.BytesIO(data), name)}, follow_redirects=True)


@pytest.fixture(scope="module")
def enrolled(env, wavs):
    """Two profiles created, uploaded and enrolled through the real HTTP routes."""
    c = env.test_client()
    for name in ("Alice", "Bob"):
        c.post("/profiles", data={"name": name})
        assert b"Recording accepted" in upload(c, name.lower(), wavs[name.lower()]).data
        r = c.post(f"/profiles/{name.lower()}/process", follow_redirects=True)
        assert b"Enrollment SUCCESS" in r.data, r.data[-800:]
    return c
