"""Speaker-conditioned TTS using the PRETRAINED Coqui XTTS-v2 model (zero-shot voice cloning).

What "enrollment" means here: XTTS-v2 is not trained or fine-tuned. From the enrolled profile's ~30 s speech
reference it computes *conditioning latents* (a GPT conditioning latent + a 512-d speaker embedding). These
are cached per profile (profiles/<slug>/xtts_latents.pt) and used to condition generation. A profile's
latents are only ever computed from that profile's own reference file.

LICENCE: XTTS-v2 weights are under the Coqui Public Model License (non-commercial). You must accept it
(set COQUI_TOS_AGREED=1) - see README. Use only voices you have permission to clone.

GPU notes (coqui-tts 0.27.5)
-----------------------------
The correct API to place the model on a device is the ``gpu`` keyword argument of ``TTS()``,
NOT ``.to(device)``. The ``TTS`` wrapper does not expose ``.to()``.  When ``gpu=True`` the
internal synthesizer and vocoder are moved to ``cuda:0``.

Mixed precision (float16) is NOT enabled by default because XTTS-v2's GPT sampling path can
produce NaN/silent outputs on some driver/cuDNN combinations at float16. It can be enabled via
``TTS_USE_FP16=1`` in the environment after verifying output quality.
"""
import importlib.util
import logging
import os
import threading
import time
from typing import Optional

import numpy as np

import config
from services import profile_service as PS

log = logging.getLogger(__name__)

LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]


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
    return _device or "unknown"


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

    # Load weights to CPU first (no deprecation warning), then move to device.
    tts_obj = TTS(model_name=config.XTTS_MODEL, gpu=False)
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
                tts_obj = TTS(model_name=config.XTTS_MODEL, gpu=True)

    elapsed = time.time() - t0
    actual = _verify_model_device(tts_obj)
    log.info(
        "[tts_service] Model loaded in %.1f s | actual device: %s | requested: %s",
        elapsed, actual, "cuda:0" if use_gpu else "cpu",
    )
    if use_gpu and not actual.startswith("cuda"):
        log.warning(
            "[tts_service] Model was loaded with gpu=True but parameter device is '%s'. "
            "CUDA inference may not be active. Check your CUDA/driver installation.",
            actual,
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

    try:
        _tts = _load_model(use_gpu)
    except Exception as exc:  # noqa: BLE001
        # If GPU load fails and we are in auto mode, fall back to CPU and warn clearly.
        if use_gpu and config.TTS_DEVICE.lower() == "auto":
            log.warning(
                "[tts_service] GPU model load failed (%s). "
                "Falling back to CPU (TTS_DEVICE=auto). "
                "Set TTS_DEVICE=cpu in .env to silence this warning.",
                exc,
            )
            _device = "cpu"
            try:
                _tts = _load_model(use_gpu=False)
            except Exception as cpu_exc:  # noqa: BLE001
                raise TTSUnavailable(f"Could not load {config.XTTS_MODEL}: {cpu_exc}") from cpu_exc
        else:
            raise TTSUnavailable(f"Could not load {config.XTTS_MODEL}: {exc}") from exc

    return _tts


def load():
    """Load (and download on first use) the XTTS model; raises TTSUnavailable with a clear reason."""
    with _lock:
        return _load()


def _latents(slug, tts):
    """Load cached conditioning latents for this profile, or compute them from its xtts_reference.wav."""
    import torch
    ref = PS.profile_dir(slug) / "xtts_reference.wav"
    cache = PS.profile_dir(slug) / "xtts_latents.pt"
    if not ref.exists():
        raise TTSError("This profile has no TTS reference; enroll the voice first.")
    model = tts.synthesizer.tts_model
    if cache.exists() and cache.stat().st_mtime >= ref.stat().st_mtime:
        d = torch.load(cache, map_location=_device, weights_only=False)
        return d["gpt_cond_latent"], d["speaker_embedding"], "cached"
    cfg = model.config
    log.info("[tts_service] Computing XTTS conditioning latents for profile '%s' ...", slug)
    with torch.inference_mode():
        try:
            gpt, spk = model.get_conditioning_latents(
                audio_path=[str(ref)],
                gpt_cond_len=cfg.gpt_cond_len,
                gpt_cond_chunk_len=cfg.gpt_cond_chunk_len,
                max_ref_length=cfg.max_ref_len,
            )
        except (TypeError, AttributeError):
            gpt, spk = model.get_conditioning_latents(audio_path=[str(ref)])
    torch.save(
        dict(gpt_cond_latent=gpt.cpu(), speaker_embedding=spk.cpu(),
             reference=ref.name, created=PS.now_iso()),
        cache,
    )
    log.info("[tts_service] Latents computed and cached for '%s'.", slug)
    return gpt, spk, "computed"


def _run_inference(model, sentence, language, gpt, spk, temperature):
    """Run a single sentence through XTTS inside inference_mode.

    Returns a float32 numpy array.  OOM errors are surfaced as TTSError so the
    caller can attempt CPU retry without reloading the model.
    """
    import torch
    try:
        with torch.inference_mode():
            out = model.inference(
                sentence, language,
                gpt.to(_device), spk.to(_device),
                temperature=temperature,
                enable_text_splitting=False,
            )
        return np.asarray(out["wav"], dtype=np.float32).ravel()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise TTSError(f"CUDA out of memory: {exc}") from exc
        raise


def synthesize(slug, sentences, language="en", temperature=0.65, gap_s=0.25):
    """Generate speech for `sentences` in the voice of profile `slug`.
    Returns (float32 waveform, sample_rate, info). Raises TTSUnavailable / TTSError.

    OOM recovery: if a CUDA out-of-memory error occurs, the GPU cache is cleared
    and the batch is retried on CPU using the same (already loaded) model weights
    moved temporarily to CPU.  The model is NOT reloaded; the shared _tts object
    is not modified.  The user profile and latents cache are preserved.
    """
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

        # ---- attempt GPU synthesis ----------------------------------------
        oom_occurred = False
        parts = []
        try:
            for s in sentences:
                wav_part = _run_inference(model, s, language, gpt, spk, temperature)
                parts.append(wav_part)
                parts.append(np.zeros(int(gap_s * sr), np.float32))
        except TTSError as exc:
            if "out of memory" not in str(exc).lower() or not _device.startswith("cuda"):
                raise
            oom_occurred = True
            log.warning(
                "[tts_service] CUDA OOM during synthesis for '%s'. "
                "Clearing GPU cache and retrying on CPU. "
                "Consider reducing MAX_TEXT_CHARS or upgrading VRAM. Error: %s",
                slug, exc,
            )
        except Exception as exc:  # noqa: BLE001
            raise TTSError(f"TTS generation failed: {exc}") from exc

        # ---- OOM retry on CPU (no model reload) ----------------------------
        if oom_occurred:
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            parts = []
            _fallback_device = "cpu"
            log.info("[tts_service] OOM retry: running inference on CPU.")
            try:
                for s in sentences:
                    with torch.inference_mode():
                        out = model.inference(
                            s, language,
                            gpt.to(_fallback_device), spk.to(_fallback_device),
                            temperature=temperature,
                            enable_text_splitting=False,
                        )
                    parts.append(np.asarray(out["wav"], dtype=np.float32).ravel())
                    parts.append(np.zeros(int(gap_s * sr), np.float32))
            except Exception as exc2:  # noqa: BLE001
                raise TTSError(f"TTS generation failed (CPU retry after OOM): {exc2}") from exc2

        try:
            wav = np.concatenate(parts[:-1])
        except ValueError:
            wav = np.zeros(sr, np.float32)  # silent fallback if parts is empty

        dt = time.time() - t0

    effective_device = "cpu" if oom_occurred else _device
    info = dict(
        model=config.XTTS_MODEL,
        device=effective_device,
        sample_rate=sr,
        inference_time_s=round(dt, 2),
        audio_duration_s=round(len(wav) / sr, 2),
        real_time_factor=round(dt / max(len(wav) / sr, 1e-6), 2),
        conditioning=how,
        sentences=len(sentences),
        temperature=temperature,
        oom_cpu_fallback=oom_occurred,
    )
    log.info(
        "[tts_service] Synthesis done for '%s': %.2f s audio in %.2f s "
        "(RTF=%.2f, device=%s, conditioning=%s)",
        slug, info["audio_duration_s"], dt, info["real_time_factor"], effective_device, how,
    )
    return wav, sr, info
