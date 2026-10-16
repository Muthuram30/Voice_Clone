"""GPU / XTTS-v2 diagnostic and benchmark script.

Run from the project root with the virtual environment activated:
    .venv\\Scripts\\python.exe scripts\\check_gpu.py

The script reports:
  - Python version
  - PyTorch version and CUDA runtime version
  - CUDA availability and GPU info (name, VRAM)
  - Coqui TTS version and any import errors
  - Configured TTS device (from .env / TTS_DEVICE)
  - Optional XTTS inference benchmark (skipped if model not downloaded or
    COQUI_TOS_AGREED != 1)

Benchmark metrics
-----------------
  model_load_s         : wall time to load XTTS-v2 (includes first-run download)
  warm_synth_s         : first synthesis (GPT sampling from scratch, no KV cache)
  steady_synth_s       : second synthesis on same text (KV cache may help)
  audio_duration_s     : duration of generated audio
  real_time_factor     : generation_time / audio_duration  (lower is better)
  peak_vram_mb         : peak GPU memory allocated since last reset (CUDA only)
  actual_model_device  : device where XTTS parameters actually live

No benchmark results are invented.  All numbers are measured on this machine.
"""
import sys
import time
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402  (side-effect: reads .env)


def _sep(title=""):
    w = 70
    if title:
        print(f"\n{'─' * 3} {title} {'─' * max(0, w - 5 - len(title))}")
    else:
        print("─" * w)


# ── 1. Python ─────────────────────────────────────────────────────────────────
_sep("Python")
print(f"  Version : {sys.version}")
print(f"  Executable: {sys.executable}")

# ── 2. PyTorch & CUDA ─────────────────────────────────────────────────────────
_sep("PyTorch / CUDA")
try:
    import torch

    print(f"  PyTorch version       : {torch.__version__}")
    print(f"  CUDA runtime (PyTorch): {torch.version.cuda}")
    print(f"  cuDNN version         : {torch.backends.cudnn.version()}")
    cuda_ok = torch.cuda.is_available()
    print(f"  torch.cuda.is_available(): {cuda_ok}")
    if cuda_ok:
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_mb = props.total_memory // (1024 ** 2)
        free_mb = (props.total_memory - torch.cuda.memory_allocated(0)) // (1024 ** 2)
        print(f"  GPU name  : {gpu_name}")
        print(f"  VRAM total: {total_mb} MB")
        print(f"  VRAM free : {free_mb} MB  (before model load)")
        cc = f"{props.major}.{props.minor}"
        print(f"  Compute capability: {cc}")
    else:
        print("  No CUDA GPU detected.")
        gpu_name = None
except ImportError as e:
    print(f"  PyTorch NOT installed: {e}")
    torch = None  # type: ignore[assignment]
    cuda_ok = False
    gpu_name = None

# ── 3. Coqui TTS ─────────────────────────────────────────────────────────────
_sep("Coqui TTS")
try:
    import TTS as _tts_pkg

    print(f"  Coqui TTS version: {_tts_pkg.__version__}")
    from TTS.api import TTS  # noqa: F401

    print("  TTS.api import: OK")
    tts_available = True
except ImportError as e:
    print(f"  Coqui TTS NOT installed: {e}")
    tts_available = False
except Exception as e:  # noqa: BLE001
    print(f"  Coqui TTS import error: {e}")
    tts_available = False

# ── 4. Configured TTS device ─────────────────────────────────────────────────
_sep("TTS device configuration")
print(f"  TTS_DEVICE env setting: {config.TTS_DEVICE!r}")
try:
    config._runtime_initialized = False  # reset so initialize_runtime re-runs fresh
    config.initialize_runtime()
    print(f"  Resolved TORCH_DEVICE : {config.TORCH_DEVICE}")
    print(f"  GPU name (config)     : {config.GPU_NAME}")
    print(f"  CUDA_AVAILABLE        : {config.CUDA_AVAILABLE}")
    print(f"  TORCH_AVAILABLE       : {config.TORCH_AVAILABLE}")
except config.CudaNotAvailableError as e:
    print(f"  ERROR: {e}")
except Exception as e:  # noqa: BLE001
    print(f"  Unexpected error in initialize_runtime: {e}")

# ── 5. XTTS inference benchmark ───────────────────────────────────────────────
_sep("XTTS-v2 benchmark")

if not tts_available:
    print("  SKIP — coqui-tts not installed.")
elif config.COQUI_TOS_AGREED != "1":
    print("  SKIP — COQUI_TOS_AGREED != 1.  Set it in .env after reading https://coqui.ai/cpml")
elif torch is None:
    print("  SKIP — PyTorch not available.")
else:
    print(f"  Model: {config.XTTS_MODEL}")
    print(f"  Target device: {config.TORCH_DEVICE}")
    use_gpu = config.TORCH_DEVICE.startswith("cuda")

    import os
    os.environ.setdefault("TTS_HOME", str(config.MODELS_DIR / "tts"))
    os.environ["COQUI_TOS_AGREED"] = "1"

    # Reset GPU memory stats so peak_allocated is meaningful.
    if cuda_ok:
        torch.cuda.reset_peak_memory_stats(0)

    print("\n  Loading model (may download on first run; this is expected)...")
    t_load = time.time()
    try:
        from TTS.api import _TTS  # attempt future import name
    except ImportError:
        from TTS.api import TTS as _TTS

    try:
        # Forward-compatible: load to CPU then move to device (avoids gpu= deprecation warning).
        tts_obj = _TTS(model_name=config.XTTS_MODEL, gpu=False)
        if use_gpu:
            try:
                tts_obj.synthesizer.tts_model.to("cuda:0")
                if hasattr(tts_obj.synthesizer, "vocoder_model") and tts_obj.synthesizer.vocoder_model is not None:
                    tts_obj.synthesizer.vocoder_model.to("cuda:0")
            except AttributeError:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    tts_obj = _TTS(model_name=config.XTTS_MODEL, gpu=True)
        model_load_s = round(time.time() - t_load, 2)
        print(f"  Model loaded in {model_load_s} s")

        # Verify actual device
        actual_device = "unknown"
        try:
            model_inner = tts_obj.synthesizer.tts_model
            for attr in ("gpt", "hifigan_decoder", "wavegrad_decoder"):
                sub = getattr(model_inner, attr, None)
                if sub is not None:
                    params = list(sub.parameters())
                    if params:
                        actual_device = str(params[0].device)
                        break
            if actual_device == "unknown":
                params = list(model_inner.parameters())
                if params:
                    actual_device = str(params[0].device)
        except Exception:  # noqa: BLE001
            pass
        print(f"  Actual model device : {actual_device}")

        if cuda_ok and use_gpu:
            vram_after_load_mb = torch.cuda.memory_allocated(0) // (1024 ** 2)
            print(f"  VRAM after load     : {vram_after_load_mb} MB")

        # Need a reference audio for conditioning latents.
        # Use the first enrolled profile's xtts_reference.wav if available.
        ref_wav = None
        from services import profile_service as PS_bench

        for p in PS_bench.list_profiles():
            cand = PS_bench.profile_dir(p["slug"]) / "xtts_reference.wav"
            if cand.exists():
                ref_wav = str(cand)
                print(f"  Using reference wav : {cand}")
                break

        if ref_wav is None:
            # Generate a tiny silent reference WAV to get *some* timing numbers.
            import soundfile as sf
            import numpy as np
            import tempfile

            _tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            _tmp.close()
            ref_wav = _tmp.name
            sf.write(ref_wav, np.zeros(int(0.5 * 22050), dtype=np.float32), 22050, subtype="PCM_16")
            print("  No enrolled profile found; using a 0.5 s silent reference (conditioning quality will be poor).")

        BENCH_TEXT = "The quick brown fox jumps over the lazy dog."
        LANGUAGE = "en"

        print(f"\n  Synthesis text: {BENCH_TEXT!r}")

        # Warm-up (first synthesis — GPT must allocate KV cache)
        if cuda_ok:
            torch.cuda.reset_peak_memory_stats(0)
        t_w = time.time()
        with torch.inference_mode():
            cfg_m = tts_obj.synthesizer.tts_model.config
            try:
                gpt, spk = tts_obj.synthesizer.tts_model.get_conditioning_latents(
                    audio_path=[ref_wav],
                    gpt_cond_len=cfg_m.gpt_cond_len,
                    gpt_cond_chunk_len=cfg_m.gpt_cond_chunk_len,
                    max_ref_length=cfg_m.max_ref_len,
                )
            except (TypeError, AttributeError):
                gpt, spk = tts_obj.synthesizer.tts_model.get_conditioning_latents(
                    audio_path=[ref_wav]
                )
            out1 = tts_obj.synthesizer.tts_model.inference(
                BENCH_TEXT, LANGUAGE, gpt, spk, temperature=0.65, enable_text_splitting=False
            )
        warm_s = round(time.time() - t_w, 2)
        wav1 = out1["wav"]
        sr = int(getattr(tts_obj.synthesizer, "output_sample_rate", 24000))
        dur1 = round(len(wav1) / sr, 2)
        rtf1 = round(warm_s / max(dur1, 1e-6), 2)
        print(f"  First synthesis : {warm_s} s | audio: {dur1} s | RTF: {rtf1}")

        # Steady-state (second call)
        t_s = time.time()
        with torch.inference_mode():
            out2 = tts_obj.synthesizer.tts_model.inference(
                BENCH_TEXT, LANGUAGE, gpt, spk, temperature=0.65, enable_text_splitting=False
            )
        steady_s = round(time.time() - t_s, 2)
        wav2 = out2["wav"]
        dur2 = round(len(wav2) / sr, 2)
        rtf2 = round(steady_s / max(dur2, 1e-6), 2)
        print(f"  Second synthesis: {steady_s} s | audio: {dur2} s | RTF: {rtf2}")

        if cuda_ok and actual_device.startswith("cuda"):
            peak_mb = torch.cuda.max_memory_allocated(0) // (1024 ** 2)
            print(f"  Peak VRAM used  : {peak_mb} MB")
        else:
            print("  Peak VRAM used  : N/A (CPU inference)")

        _sep()
        print("  Summary")
        print(f"    Model load           : {model_load_s} s  (includes download on first run)")
        print(f"    First synthesis RTF  : {rtf1}  (lower is faster than real-time)")
        print(f"    Second synthesis RTF : {rtf2}")
        print(f"    Actual device        : {actual_device}")
        if rtf2 < 1.0:
            print("    ✓ Faster than real-time!")
        else:
            print("    ✗ Slower than real-time — consider enabling GPU (TTS_DEVICE=cuda).")

    except Exception as exc:  # noqa: BLE001
        print(f"  Benchmark FAILED: {exc}")
        import traceback
        traceback.print_exc()

_sep()
print("Diagnostic complete.")
