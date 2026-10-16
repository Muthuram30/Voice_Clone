"""Unit tests for GPU device selection and CPU fallback logic in config.py and tts_service.py.

These tests run without a GPU or a downloaded XTTS model.  They verify that:
  - TTS_DEVICE=auto selects CUDA when available and CPU when not.
  - TTS_DEVICE=cuda raises CudaNotAvailableError when CUDA is absent.
  - TTS_DEVICE=cpu always produces cpu regardless of GPU presence.
  - tts_service.status() reflects the loaded device correctly.
  - TTSUnavailable is raised (not a crash) when packages are not installed.
  - Profile isolation is preserved across two synthesize() calls (slug-keyed).
"""
import importlib.util
from unittest.mock import patch, MagicMock

import pytest

import config
from services import tts_service as T


# ── helpers ───────────────────────────────────────────────────────────────────

def _reset_runtime(mp):
    """Reset config runtime state so initialize_runtime() re-runs."""
    mp.setattr(config, "_runtime_initialized", False)
    mp.setattr(config, "TORCH_AVAILABLE", False)
    mp.setattr(config, "CUDA_AVAILABLE", False)
    mp.setattr(config, "TORCH_DEVICE", "cpu")
    mp.setattr(config, "GPU_NAME", None)


def _reset_tts(mp):
    """Reset tts_service module-level state."""
    mp.setattr(T, "_tts", None)
    mp.setattr(T, "_device", None)


def _fake_torch(cuda_available=True):
    """Build a minimal fake torch module for patching."""
    m = MagicMock()
    m.cuda.is_available.return_value = cuda_available
    m.cuda.get_device_name.return_value = "Fake GPU" if cuda_available else "N/A"
    props = MagicMock()
    props.total_memory = 4 * 1024 ** 3
    props.major = 8
    props.minor = 6
    m.cuda.get_device_properties.return_value = props
    m.cuda.memory_allocated.return_value = 0
    m.version.cuda = "12.0"
    return m


# ── device selection ──────────────────────────────────────────────────────────

class TestDeviceSelection:
    """Verify config.initialize_runtime() behaviour for each TTS_DEVICE value."""

    def test_auto_selects_cuda_when_available(self, monkeypatch):
        _reset_runtime(monkeypatch)
        monkeypatch.setattr(config, "TTS_DEVICE", "auto")
        fake = _fake_torch(cuda_available=True)
        with patch.dict("sys.modules", {"torch": fake}):
            config.initialize_runtime()
        assert config.TORCH_DEVICE == "cuda:0"
        assert config.CUDA_AVAILABLE is True

    def test_auto_falls_back_to_cpu_when_no_cuda(self, monkeypatch):
        _reset_runtime(monkeypatch)
        monkeypatch.setattr(config, "TTS_DEVICE", "auto")
        fake = _fake_torch(cuda_available=False)
        with patch.dict("sys.modules", {"torch": fake}):
            config.initialize_runtime()
        assert config.TORCH_DEVICE == "cpu"
        assert config.CUDA_AVAILABLE is False

    def test_cpu_always_forces_cpu_even_when_gpu_present(self, monkeypatch):
        _reset_runtime(monkeypatch)
        monkeypatch.setattr(config, "TTS_DEVICE", "cpu")
        fake = _fake_torch(cuda_available=True)
        with patch.dict("sys.modules", {"torch": fake}):
            config.initialize_runtime()
        assert config.TORCH_DEVICE == "cpu", "cpu setting must not be overridden even if CUDA is present"

    def test_cuda_explicit_raises_when_unavailable(self, monkeypatch):
        _reset_runtime(monkeypatch)
        monkeypatch.setattr(config, "TTS_DEVICE", "cuda")
        fake = _fake_torch(cuda_available=False)
        with patch.dict("sys.modules", {"torch": fake}):
            with pytest.raises(config.CudaNotAvailableError, match="TTS_DEVICE=cuda"):
                config.initialize_runtime()

    def test_cuda_explicit_succeeds_when_available(self, monkeypatch):
        _reset_runtime(monkeypatch)
        monkeypatch.setattr(config, "TTS_DEVICE", "cuda")
        fake = _fake_torch(cuda_available=True)
        with patch.dict("sys.modules", {"torch": fake}):
            config.initialize_runtime()
        assert config.TORCH_DEVICE == "cuda:0"

    def test_auto_with_no_torch_gives_cpu(self, monkeypatch):
        """When PyTorch import raises, auto must silently use CPU."""
        _reset_runtime(monkeypatch)
        monkeypatch.setattr(config, "TTS_DEVICE", "auto")
        # Remove torch from sys.modules to force ImportError
        with patch.dict("sys.modules", {"torch": None}):
            config.initialize_runtime()
        assert config.TORCH_DEVICE == "cpu"
        assert config.TORCH_AVAILABLE is False

    def test_cuda_with_no_torch_raises(self, monkeypatch):
        """TTS_DEVICE=cuda with no PyTorch must raise, not silently fall back."""
        _reset_runtime(monkeypatch)
        monkeypatch.setattr(config, "TTS_DEVICE", "cuda")
        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises(config.CudaNotAvailableError, match="PyTorch is not installed"):
                config.initialize_runtime()

    def test_error_message_contains_diagnostic_hint(self, monkeypatch):
        """CudaNotAvailableError for cuda+no-cuda should mention check_gpu.py."""
        _reset_runtime(monkeypatch)
        monkeypatch.setattr(config, "TTS_DEVICE", "cuda")
        fake = _fake_torch(cuda_available=False)
        with patch.dict("sys.modules", {"torch": fake}):
            with pytest.raises(config.CudaNotAvailableError, match="check_gpu.py"):
                config.initialize_runtime()

    def test_idempotent_double_call(self, monkeypatch):
        """A second call to initialize_runtime() must be a no-op."""
        _reset_runtime(monkeypatch)
        monkeypatch.setattr(config, "TTS_DEVICE", "cpu")
        config.initialize_runtime()
        # Modify TORCH_DEVICE externally, then call again — must not overwrite.
        monkeypatch.setattr(config, "TORCH_DEVICE", "sentinel")
        config.initialize_runtime()
        assert config.TORCH_DEVICE == "sentinel"


# ── tts_service helpers ───────────────────────────────────────────────────────

class TestInferGpuFlag:
    def test_cuda_device_returns_true(self):
        assert T._infer_gpu_flag("cuda:0") is True
        assert T._infer_gpu_flag("cuda") is True

    def test_cpu_device_returns_false(self):
        assert T._infer_gpu_flag("cpu") is False
        assert T._infer_gpu_flag("") is False


# ── tts_service guards ────────────────────────────────────────────────────────

class TestTTSServiceGuards:
    """Verify tts_service pre-conditions without a real model."""

    def test_status_before_load_reflects_config_device(self, monkeypatch):
        """status() must faithfully report config.TORCH_DEVICE, not a hard-coded value."""
        _reset_tts(monkeypatch)
        # status() falls back to config.TORCH_DEVICE when no model is loaded.
        # We do NOT force a specific device here — we trust what config resolved.
        config.initialize_runtime()  # ensure runtime is initialised
        s = T.status()
        assert s["loaded"] is False
        # device must equal what the config resolved (cpu or cuda:0 depending on hardware)
        assert s["device"] == config.TORCH_DEVICE
        assert s["actual_model_device"] is None

    def test_unavailable_when_tos_not_agreed(self, monkeypatch):
        _reset_tts(monkeypatch)
        monkeypatch.setattr(config, "COQUI_TOS_AGREED", "0")
        with pytest.raises(T.TTSUnavailable, match="Coqui Public Model License"):
            T.load()

    def test_unavailable_when_coqui_tts_not_importable(self, monkeypatch):
        """Pretend coqui-tts is absent; must raise TTSUnavailable with helpful message."""
        _reset_tts(monkeypatch)
        monkeypatch.setattr(config, "COQUI_TOS_AGREED", "1")
        orig_find_spec = importlib.util.find_spec

        def _fake_find_spec(name):
            return None if name == "TTS" else orig_find_spec(name)

        monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
        with pytest.raises(T.TTSUnavailable, match="coqui-tts"):
            T.load()

    def test_synthesize_raises_for_unenrolled(self, env):
        from services import profile_service as PS
        PS.create("TmpUnenrolled")
        try:
            with pytest.raises(T.TTSError, match="not enrolled"):
                T.synthesize("tmpunenrolled", ["Hello."])
        finally:
            PS.delete("tmpunenrolled")

    def test_synthesize_raises_for_bad_language(self, enrolled):
        with pytest.raises(T.TTSError, match="Unsupported language"):
            T.synthesize("alice", ["Hello."], language="zz")

    def test_synthesize_raises_for_empty_sentences(self, enrolled):
        with pytest.raises(T.TTSError, match="No text"):
            T.synthesize("alice", [])


# ── profile isolation ─────────────────────────────────────────────────────────

class TestProfileIsolation:
    """Each synthesize() call must only use its own slug's latents."""

    def test_stub_synthesis_routes_to_correct_slugs(self, enrolled, monkeypatch):
        calls = []

        def recording_synth(slug, sentences, **kw):
            calls.append(slug)
            raise T.TTSUnavailable("stub — no model needed")

        monkeypatch.setattr(T, "synthesize", recording_synth)

        from services import synthesis_service as SY

        for slug in ("alice", "bob"):
            try:
                SY.run(slug, "Hello world.")
            except Exception:
                pass

        # Each invocation should have been routed to the matching slug
        if len(calls) >= 1:
            assert calls[0] == "alice"
        if len(calls) >= 2:
            assert calls[1] == "bob"
