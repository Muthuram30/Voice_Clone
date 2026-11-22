"""Speaker-conditioned TTS using the PRETRAINED Coqui XTTS-v2 model (zero-shot voice cloning).

What "enrollment" means here: XTTS-v2 is not trained or fine-tuned. From the enrolled profile's continuous speech
reference it computes *conditioning latents* (a GPT conditioning latent + a 512-d speaker embedding). These
are cached per profile (profiles/<slug>/xtts_latents.pt) and used to condition generation. A profile's
latents are only ever computed from that profile's own reference file.

LICENCE: XTTS-v2 weights are under the Coqui Public Model License (non-commercial). You must accept it
(set COQUI_TOS_AGREED=1) - see README. Use only voices you have permission to clone.

The synthesizer's model is moved to CUDA and its parameter device is checked.
Inference uses the model's default precision; no automatic CPU fallback is used.
"""
import importlib.util
import logging
import os
import threading
import time
import hashlib
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

import config
from services import profile_service as PS
from services import speech_quality as SQ

log = logging.getLogger(__name__)

LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]
LANGUAGE_NAMES = dict(zip(LANGUAGES, ["English", "Spanish", "French", "German", "Italian", "Portuguese",
    "Polish", "Turkish", "Russian", "Dutch", "Czech", "Arabic", "Mandarin Chinese", "Japanese", "Hungarian", "Korean", "Hindi"]))


class TTSUnavailable(Exception):
    """TTS stack/model cannot be used (not installed, licence not accepted, load failure)."""


class TTSError(Exception):
    """Generation failed."""


# One model instance is shared across all requests. A reentrant lock serialises concurrent
# Flask requests so that multiple threads never load separate model copies.
_lock = threading.RLock()
_tts = None          # the TTS wrapper (loaded once)
_device: Optional[str] = None  # the device string actually used by the loaded model
_dll_directory_handles = []


def _configure_ffmpeg_runtime():
    """Make a Windows full-shared FFmpeg install visible to TorchCodec.

    TorchCodec loads FFmpeg through native DLL dependencies, so adding only
    ffmpeg.exe to PATH is not sufficient on modern Windows. The handle from
    os.add_dll_directory must remain alive for the lifetime of the process.
    """
    ffmpeg_dir = str(getattr(config, "FFMPEG_BIN_DIR", "") or "").strip()
    if not ffmpeg_dir or not os.path.isdir(ffmpeg_dir):
        return
    if hasattr(os, "add_dll_directory"):
        if ffmpeg_dir not in [str(h) for h in _dll_directory_handles]:
            _dll_directory_handles.append(os.add_dll_directory(ffmpeg_dir))
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")


def _infer_gpu_flag(device: str) -> bool:
    """Translate config.TORCH_DEVICE to the boolean ``gpu`` kwarg that coqui-tts 0.27.x expects."""
    return device.startswith("cuda")


def _verify_model_device(tts_obj) -> str:
    """Return the device string where the XTTS model parameters actually live.

    Tries several attribute paths used across coqui-tts versions.  Falls back
    to reporting the device from the first found parameter tensor.
    """
    try:
        model = tts_obj.synthesizer.tts_model
        # Try to find a known parameter to read its .device
        for attr in ("gpt", "hifigan_decoder", "wavegrad_decoder", "vocoder"):
            sub = getattr(model, attr, None)
            if sub is not None:
                params = list(sub.parameters()) if hasattr(sub, "parameters") else []
                if params:
                    return str(params[0].device)
        # Fall back: any parameter of the model itself
        params = list(model.parameters())
        if params:
            return str(params[0].device)
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def status():
    config.initialize_runtime()
    inst = all(importlib.util.find_spec(m) for m in ("torch", "TTS"))
    actual_dev = _verify_model_device(_tts) if _tts is not None else None
    return dict(
        model=config.XTTS_MODEL,
        packages_installed=bool(inst),
        loaded=_tts is not None,
        device=_device or config.TORCH_DEVICE,
        actual_model_device=actual_dev,
        license_accepted=config.COQUI_TOS_AGREED == "1",
    )


def _load_model(use_gpu: bool) -> "TTS":  # noqa: F821
    """Internal: create and return a TTS instance placed on the correct device.

    coqui-tts 0.27.5 forward-compatible loading strategy
    -------------------------------------------------------
    1. Load with ``gpu=False`` (suppresses the deprecation warning).
    2. Call ``.to(device)`` on the synthesizer's tts_model (the nn.Module that
       holds the actual parameters) to move weights to the target device.
    3. Fall back to ``TTS(model_name, gpu=True)`` if step 2 raises so that
       older coqui-tts versions without ``.to()`` still work.

    The ``gpu=True`` path is kept as a fallback only.
    """
    from TTS.api import TTS  # type: ignore[import-untyped]
    log.info("[tts_service] Loading XTTS-v2 model '%s' (target_device=%s) ...", config.XTTS_MODEL, "cuda:0" if use_gpu else "cpu")
    t0 = time.time()

    # Coqui's name-based loader checks the remote XTTS config even when weights
    # are cached. Load a complete local snapshot directly so offline startup
    # cannot hang on that network check.
    tts_obj = TTS(gpu=False)
    cached = Path(tts_obj.manager.output_prefix) / config.XTTS_MODEL.replace("/", "--")
    local_xtts = "xtts" in config.XTTS_MODEL and all(
        (cached / name).is_file() for name in ("config.json", "model.pth", "vocab.json", "speakers_xtts.pth"))
    if local_xtts:
        from TTS.utils.synthesizer import Synthesizer
        log.info("[tts_service] Using local XTTS snapshot: %s", cached)
        tts_obj.model_name = config.XTTS_MODEL
        tts_obj.synthesizer = Synthesizer(model_dir=str(cached), use_cuda=False)
    else:
        tts_obj.load_tts_model_by_name(config.XTTS_MODEL, gpu=False)
    if use_gpu:
        try:
            # Move the synthesizer's model parameters to GPU.
            tts_obj.synthesizer.tts_model.to("cuda:0")
            if hasattr(tts_obj.synthesizer, "vocoder_model") and tts_obj.synthesizer.vocoder_model is not None:
                tts_obj.synthesizer.vocoder_model.to("cuda:0")
        except AttributeError:
            # Older coqui-tts without this attribute path — reload with gpu=True.
            import warnings
            log.warning("[tts_service] .to('cuda:0') via synthesizer attributes failed; retrying with gpu=True.")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                if local_xtts:
                    tts_obj.synthesizer = Synthesizer(model_dir=str(cached), use_cuda=True)
                else:
                    tts_obj = TTS(model_name=config.XTTS_MODEL, gpu=True)

    elapsed = time.time() - t0
    actual = _verify_model_device(tts_obj)
    log.info(
        "[tts_service] Model loaded in %.1f s | actual device: %s | requested: %s",
        elapsed, actual, "cuda:0" if use_gpu else "cpu",
    )
    if use_gpu and not actual.startswith("cuda"):
        raise TTSUnavailable(
            f"XTTS requested CUDA but its parameter device is '{actual}'. "
            "Run scripts/check_gpu.py to check the installed TTS and CUDA runtime."
        )
    return tts_obj


def _load():
    """Load (and download if needed) the XTTS model. Must be called with _lock held."""
    global _tts, _device
    if _tts is not None:
        return _tts

    for mod, pkg in (("torch", "torch"), ("TTS", "coqui-tts")):
        if importlib.util.find_spec(mod) is None:
            raise TTSUnavailable(
                f"'{pkg}' is not installed. Install the TTS requirements: "
                "pip install -r requirements-tts.txt (see README)."
            )
    if config.COQUI_TOS_AGREED != "1":
        raise TTSUnavailable(
            "XTTS-v2 is released under the Coqui Public Model License (non-commercial). "
            "Read https://coqui.ai/cpml and set COQUI_TOS_AGREED=1 in .env to download/use it."
        )

    os.environ.setdefault("TTS_HOME", str(config.MODELS_DIR / "tts"))
    os.environ["COQUI_TOS_AGREED"] = "1"
    _configure_ffmpeg_runtime()

    config.initialize_runtime()
    _device = config.TORCH_DEVICE
    use_gpu = _infer_gpu_flag(_device)

    if not use_gpu:
        raise TTSUnavailable("Voice synthesis requires CUDA; set TTS_DEVICE=cuda and run scripts/check_gpu.py.")
    try:
        _tts = _load_model(use_gpu=True)
    except Exception as exc:
        raise TTSUnavailable(f"Could not load {config.XTTS_MODEL} on CUDA: {exc}") from exc

    return _tts


def load():
    """Load (and download on first use) the XTTS model; raises TTSUnavailable with a clear reason."""
    with _lock:
        return _load()


def unload():
    """Release XTTS before launching a different CUDA voice model."""
    global _tts, _device
    import gc
    import torch
    with _lock:
        _tts = None
        _device = None
        gc.collect()
        torch.cuda.empty_cache()


def _latents(slug, tts):
    """Load cached conditioning latents for this profile, or compute them from its xtts_reference.wav."""
    import torch
    from services.speaker_service import ensure_continuous_tts_reference
    ensure_continuous_tts_reference(slug)
    ref = PS.profile_dir(slug) / "xtts_reference.wav"
    cache = PS.profile_dir(slug) / "xtts_latents.pt"
    if not ref.exists():
        raise TTSError("This profile has no TTS reference; enroll the voice first.")
    references = sorted((PS.profile_dir(slug) / "xtts_references").glob("reference_*.wav")) or [ref]
    model = tts.synthesizer.tts_model
    fingerprint = hashlib.sha256((config.XTTS_MODEL + ":conditioning-v3-continuous:" +
        str(getattr(model.config, "gpt_cond_len", "")) + ":" +
        str(getattr(model.config, "gpt_cond_chunk_len", "")) + ":" +
        str(getattr(model.config, "max_ref_len", ""))).encode() + b"".join(p.read_bytes() for p in references)).hexdigest()
    if cache.exists() and cache.stat().st_mtime >= ref.stat().st_mtime:
        try:
            d = torch.load(cache, map_location=_device, weights_only=True)
            if d.get("fingerprint") == fingerprint:
                return d["gpt_cond_latent"], d["speaker_embedding"], "cached"
        except (OSError, RuntimeError, ValueError, pickle.UnpicklingError, EOFError):
            log.warning("Ignoring invalid conditioning cache for %s", slug)
    cfg = model.config
    log.info("[tts_service] Computing XTTS conditioning latents for profile '%s' ...", slug)
    with torch.inference_mode():
        try:
            gpt, spk = model.get_conditioning_latents(
                audio_path=[str(path) for path in references],
                gpt_cond_len=cfg.gpt_cond_len,
                gpt_cond_chunk_len=cfg.gpt_cond_chunk_len,
                max_ref_length=cfg.max_ref_len,
            )
        except (TypeError, AttributeError):
            gpt, spk = model.get_conditioning_latents(audio_path=[str(path) for path in references])
    torch.save(
        dict(gpt_cond_latent=gpt.cpu(), speaker_embedding=spk.cpu(),
             reference=ref.name, created=PS.now_iso(), fingerprint=fingerprint),
        cache,
    )
    log.info("[tts_service] Latents computed and cached for '%s'.", slug)
    return gpt, spk, "computed"


def _run_inference(model, sentence, language, gpt, spk, settings):
    """Run one passage on CUDA with tokenizer-aware sentence splitting."""
    import torch
    try:
        with torch.inference_mode():
            out = model.inference(
                sentence, language,
                gpt.to(_device), spk.to(_device),
                **settings,
                enable_text_splitting=False,
            )
        return np.asarray(out["wav"], dtype=np.float32).ravel()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise TTSError(f"CUDA out of memory: {exc}") from exc
        raise


def _passages(model, sentences, language):
    tokenizer = model.tokenizer
    limit = min(config.SENTENCE_MAX_CHARS, tokenizer.char_limits.get(language.split('-')[0], 180))
    budget = model.args.gpt_max_text_tokens
    pending = SQ.pack_passages(sentences, limit)
    result = []
    while pending:
        text = pending.pop(0)
        if len(tokenizer.encode(text.strip().lower(), lang=language.split('-')[0])) < budget:
            result.append(text)
        elif len(text) > 1:
            pending[:0] = SQ.pack_passages([text], max(1, len(text) // 2))
        else:
            raise TTSError('Text cannot fit the model token budget in the selected language.')
    return result


def synthesize(slug, sentences, language="en", temperature=None, gap_s=0.18):
    """Generate profile-conditioned speech on CUDA, failing clearly on GPU OOM."""
    import torch

    if not PS.is_enrolled(slug):
        raise TTSError("This profile is not enrolled yet.")
    if language not in LANGUAGES:
        raise TTSError(f"Unsupported language '{language}'.")
    if not sentences:
        raise TTSError("No text to synthesise.")

    with _lock:
        tts = _load()
        t0 = time.time()
        gpt, spk, how = _latents(slug, tts)
        model = tts.synthesizer.tts_model
        sr = int(getattr(tts.synthesizer, "output_sample_rate", 24000))
        cfg = model.config
        settings = dict(temperature=float(temperature if temperature is not None else getattr(cfg, 'temperature', .75)),
                        top_k=int(getattr(cfg, 'top_k', 50)), top_p=float(getattr(cfg, 'top_p', .85)),
                        repetition_penalty=float(getattr(cfg, 'repetition_penalty', 5.)),
                        do_sample=True, speed=1.0)
        passages = _passages(model, sentences, language)

        # ---- attempt GPU synthesis ----------------------------------------
        oom_occurred = False
        parts = []
        try:
            for s in passages:
                wav_part = _run_inference(model, s, language, gpt, spk, settings)
                parts.append(wav_part)
        except TTSError as exc:
            if "out of memory" not in str(exc).lower() or not _device.startswith("cuda"):
                raise
            torch.cuda.empty_cache()
            raise TTSError("GPU memory is full. Close other GPU applications and generate a shorter passage. "
                           "No CPU fallback was used.") from exc
        except Exception as exc:  # noqa: BLE001
            raise TTSError(f"TTS generation failed: {exc}") from exc

        if not parts:
            raise TTSError("The model returned no audio.")
        try:
            wav, added_pauses = SQ.join_passages(parts, passages, sr, gap_s=gap_s)
        except ValueError as exc:
            raise TTSError(str(exc)) from exc
        if not len(wav) or not np.isfinite(wav).all() or np.max(np.abs(wav)) < 1e-5:
            raise TTSError("The model returned silent or invalid audio. Try a clean reference and shorter text.")
        # One global scale protects WAV headroom without flattening word stress
        # or normalizing every sentence to the same loudness.
        headroom_gain = min(1., .98 / float(np.abs(wav).max()))
        wav *= headroom_gain

        dt = time.time() - t0

    effective_device = _device
    info = dict(
        model=config.XTTS_MODEL,
        device=effective_device,
        sample_rate=sr,
        inference_time_s=round(dt, 2),
        audio_duration_s=round(len(wav) / sr, 2),
        real_time_factor=round(dt / max(len(wav) / sr, 1e-6), 2),
        conditioning=how,
        sentences=len(sentences),
        temperature=settings['temperature'], generation_settings=settings,
        passages=len(passages), added_pauses_s=added_pauses,
        headroom_gain_db=round(float(20 * np.log10(headroom_gain)), 3),
        oom_cpu_fallback=oom_occurred,
    )
    log.info(
        "[tts_service] Synthesis done for '%s': %.2f s audio in %.2f s "
        "(RTF=%.2f, device=%s, conditioning=%s)",
        slug, info["audio_duration_s"], dt, info["real_time_factor"], effective_device, how,
    )
    return wav, sr, info
