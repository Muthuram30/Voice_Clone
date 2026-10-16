"""Central configuration. Everything tunable comes from environment variables / .env.
Services read these as `config.NAME` at call time (so tests can monkeypatch them)."""
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

log = logging.getLogger(__name__)

_runtime_initialized = False
TORCH_AVAILABLE = False
CUDA_AVAILABLE = False
TORCH_DEVICE = "cpu"
GPU_NAME = None


class CudaNotAvailableError(RuntimeError):
    """Raised when TTS_DEVICE=cuda but CUDA is not available."""


def initialize_runtime():
    """Detect the PyTorch device once and expose the application-wide choice.

    TTS_DEVICE semantics
    --------------------
    auto  - use CUDA if genuinely available, else CPU (never raises).
    cuda  - require GPU; raises CudaNotAvailableError with a diagnostic message
            if CUDA is not available so the operator can diagnose the issue.
    cpu   - force CPU regardless of GPU availability.
    """
    global _runtime_initialized, TORCH_AVAILABLE, CUDA_AVAILABLE, TORCH_DEVICE, GPU_NAME
    if _runtime_initialized:
        return
    _runtime_initialized = True

    _tts_device_setting = TTS_DEVICE.lower().strip()

    try:
        import torch
        TORCH_AVAILABLE = True
        CUDA_AVAILABLE = bool(torch.cuda.is_available())
        GPU_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else None
    except Exception as exc:  # noqa: BLE001 - CPU-only installs remain supported
        TORCH_AVAILABLE = False
        log.warning("PyTorch is not installed or failed to import (%s). Running CPU-only.", exc)

    if _tts_device_setting == "cpu":
        TORCH_DEVICE = "cpu"
    elif _tts_device_setting == "cuda":
        if not TORCH_AVAILABLE:
            raise CudaNotAvailableError(
                "TTS_DEVICE=cuda but PyTorch is not installed. "
                "Install a CUDA-enabled PyTorch wheel: "
                "https://pytorch.org/get-started/locally/"
            )
        if not CUDA_AVAILABLE:
            try:
                import torch as _t
                _pt_cuda = _t.version.cuda
            except Exception:  # noqa: BLE001
                _pt_cuda = "N/A"
            raise CudaNotAvailableError(
                "TTS_DEVICE=cuda but torch.cuda.is_available() returned False. "
                "Possible causes: CUDA-capable GPU not detected, driver/toolkit mismatch, "
                "or a CPU-only PyTorch build is installed. "
                f"PyTorch CUDA build tag: {_pt_cuda}. "
                "Run scripts/check_gpu.py for a full diagnostic."
            )
        TORCH_DEVICE = "cuda:0"
    else:  # auto
        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            TORCH_DEVICE = "cuda:0"
        else:
            TORCH_DEVICE = "cpu"

    # Log the selected device so it is always visible in the application log.
    if TORCH_DEVICE.startswith("cuda"):
        try:
            import torch
            props = torch.cuda.get_device_properties(0)
            free_mb = (props.total_memory - torch.cuda.memory_allocated(0)) // (1024 ** 2)
            total_mb = props.total_memory // (1024 ** 2)
            log.info(
                "[config] Device selected: %s | GPU: %s | VRAM: %d MB total, %d MB free",
                TORCH_DEVICE, GPU_NAME, total_mb, free_mb,
            )
        except Exception:  # noqa: BLE001
            log.info("[config] Device selected: %s | GPU: %s", TORCH_DEVICE, GPU_NAME)
    else:
        log.info(
            "[config] Device selected: cpu (TTS_DEVICE=%s, CUDA available=%s)",
            TTS_DEVICE, CUDA_AVAILABLE,
        )


def runtime_status():
    initialize_runtime()
    return dict(torch_available=TORCH_AVAILABLE, cuda_available=CUDA_AVAILABLE,
                device=TORCH_DEVICE, gpu=GPU_NAME,
                gpu_stages=["ECAPA-TDNN", "XTTS-v2"] if TORCH_DEVICE.startswith("cuda") else [],
                cpu_stages=["audio I/O", "librosa/SciPy DSP", "matplotlib", "scikit-learn", "MATLAB (optional)"])

def _f(name, default): return float(os.getenv(name, default))
def _i(name, default): return int(os.getenv(name, default))
def _p(name, default):
    p = Path(os.getenv(name) or default)
    return p if p.is_absolute() else ROOT / p

PROFILES_DIR = _p("PROFILES_DIR", "profiles")
MODELS_DIR = _p("MODELS_DIR", "models")
DATA_DIR = _p("DATA_DIR", "data")
MATLAB_DIR = ROOT / "matlab"

SECRET_KEY = os.getenv("SECRET_KEY", "dev-change-me")
SEED_DEFAULT_PROFILES = os.getenv("SEED_DEFAULT_PROFILES", "1") == "1"

# ---- DSP / analysis -------------------------------------------------------
TARGET_FS = _i("TARGET_FS", 16000)                 # analysis + speaker-encoder rate
DSP_BACKEND = os.getenv("DSP_BACKEND", "auto")     # auto | python | matlab
MATLAB_MODE = os.getenv("MATLAB_MODE", "auto")     # auto | engine | cli
MATLAB_DESKTOP = os.getenv("MATLAB_DESKTOP", "0")
MATLAB_EXE = os.getenv("MATLAB_EXE", "")

# ---- Enrollment validation ------------------------------------------------
ENROLL_TARGET_S = _f("ENROLL_TARGET_SECONDS", 120)       # what the script is written for
ENROLL_MIN_S = _f("ENROLL_MIN_SECONDS", 60)              # below -> rejected
ENROLL_MAX_S = _f("ENROLL_MAX_SECONDS", 300)             # above -> rejected
ENROLL_MIN_SPEECH_S = _f("ENROLL_MIN_SPEECH_SECONDS", 40)  # VAD speech needed
ENROLL_MIN_SPEECH_RATIO = _f("ENROLL_MIN_SPEECH_RATIO", 0.35)

# ---- Speaker encoder / classifier ----------------------------------------
ENCODER_BACKEND = os.getenv("ENCODER_BACKEND", "auto")   # auto | ecapa | mfcc
CHUNK_S = _f("EMBED_CHUNK_SECONDS", 3.0)
CHUNK_HOP_S = _f("EMBED_CHUNK_HOP_SECONDS", 1.5)

# ---- TTS ------------------------------------------------------------------
XTTS_MODEL = os.getenv("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
TTS_DEVICE = os.getenv("TTS_DEVICE", "auto")             # auto | cpu | cuda
FFMPEG_BIN_DIR = os.getenv("FFMPEG_BIN_DIR", "")         # full-shared FFmpeg bin directory (Windows)
COQUI_TOS_AGREED = os.getenv("COQUI_TOS_AGREED", "0")
MAX_TEXT_CHARS = _i("MAX_TEXT_CHARS", 1500)
SENTENCE_MAX_CHARS = _i("SENTENCE_MAX_CHARS", 200)

# TorchCodec resolves FFmpeg through native DLL dependencies. Configure this
# at project import time so direct TorchCodec use and the TTS service behave
# consistently in fresh Python processes.
_ffmpeg_dll_handle = None
if FFMPEG_BIN_DIR and os.path.isdir(FFMPEG_BIN_DIR):
    if hasattr(os, "add_dll_directory"):
        _ffmpeg_dll_handle = os.add_dll_directory(FFMPEG_BIN_DIR)
    os.environ["PATH"] = FFMPEG_BIN_DIR + os.pathsep + os.environ.get("PATH", "")

# ---- Gemini (optional text pre-processing only; never generates audio) ----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODELS = [m.strip() for m in os.getenv("GEMINI_MODELS", "gemini-2.5-flash").split(",") if m.strip()]
GEMINI_TIMEOUT_S = _f("GEMINI_TIMEOUT_SECONDS", 20)
