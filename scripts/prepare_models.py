"""Download and verify the pretrained models (run once; works locally or on Colab).
Nothing is trained here: this only fetches
  * SpeechBrain ECAPA-TDNN speaker encoder (speechbrain/spkrec-ecapa-voxceleb)  -> models/ecapa
  * Coqui XTTS-v2 (needs COQUI_TOS_AGREED=1, non-commercial licence)             -> models/tts
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from services import speaker_service as S, tts_service as T


def size(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e6 if Path(p).exists() else 0


ok = True
try:
    t = time.time(); enc, _ = S.get_encoder("ecapa"); print(f"ECAPA encoder ready on {enc.device} ({time.time() - t:.0f}s, {size(config.MODELS_DIR / 'ecapa'):.0f} MB)")
except S.EncoderUnavailable as e:
    ok = False; print("ECAPA NOT available:", e)
try:
    t = time.time(); T.load(); print(f"XTTS-v2 ready on {T.status()['device']} ({time.time() - t:.0f}s, {size(config.MODELS_DIR / 'tts'):.0f} MB)")
except T.TTSUnavailable as e:
    ok = False; print("XTTS NOT available:", e)
sys.exit(0 if ok else 1)
