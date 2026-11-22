"""Optional IndicF5 runner isolated from XTTS's dependency versions.

Uses an already downloaded local model and a dedicated project venv. No hidden
downloads, external voice APIs or CPU fallback. Reference text must match the
unaltered short prompt, so the XTTS stitched reference is never reused here.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

import config

LANGUAGES = {"as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada",
             "ml": "Malayalam", "mr": "Marathi", "or": "Odia", "pa": "Punjabi", "ta": "Tamil", "te": "Telugu"}


def status():
    python = config.ROOT / ".venv-indic" / "Scripts" / "python.exe"
    model = config.ROOT / "models" / "indicf5"
    ready = python.is_file() and (model / "config.json").is_file() and (model / "model.safetensors").is_file()
    return dict(configured=ready, languages=LANGUAGES, model="ai4bharat/IndicF5",
                note="Separate setup required: see README. Model account access and CUDA are required." if not ready else
                     "Local model files found; inference readiness is checked when generating.")


def synthesize(text, language, reference, transcript, out_dir):
    from services.tts_service import TTSError, TTSUnavailable
    if language not in LANGUAGES:
        raise TTSError("Unsupported language for IndicF5.")
    if not status()["configured"]:
        raise TTSUnavailable("IndicF5 is not installed locally. Follow the separate .venv-indic setup in README; "
                             "its model requires Hugging Face account access. XTTS cannot synthesize Tamil.")
    transcript = (transcript or "").strip()
    if not transcript or len(transcript) > 1500:
        raise TTSError("Enter the exact transcript of the Indian-language prompt (1–1500 characters).")
    ref = sf.info(str(reference))
    if not 3 <= ref.duration <= 15:
        raise TTSError("IndicF5 needs an uncut 3–15 second prompt and its exact transcript.")
    output = Path(out_dir) / "indic-output.wav"
    args = dict(text=text, reference=str(Path(reference).resolve()), transcript=transcript, output=str(output.resolve()),
                model=str((config.ROOT / "models" / "indicf5").resolve()))
    try:
        result = subprocess.run([str(config.ROOT / ".venv-indic" / "Scripts" / "python.exe"),
                                 str(config.ROOT / "scripts" / "indic_worker.py")],
                                input=json.dumps(args), text=True, encoding="utf-8", capture_output=True,
                                timeout=600, cwd=config.ROOT)
    except subprocess.TimeoutExpired as exc:
        raise TTSError("IndicF5 exceeded ten minutes. Use a shorter passage.") from exc
    if result.returncode or not output.is_file():
        raise TTSError("IndicF5 failed on CUDA. Check its isolated environment, local model files and free VRAM. "
                       + result.stderr[-600:])
    wav, sr = sf.read(output, dtype="float32")
    if wav.ndim != 1 or not len(wav) or not np.isfinite(wav).all() or np.max(np.abs(wav)) < 1e-5:
        raise TTSError("IndicF5 returned invalid or silent audio.")
    return wav, sr
