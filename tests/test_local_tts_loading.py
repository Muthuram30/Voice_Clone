"""Cached XTTS must load without a name-based remote configuration check."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

import config
from services import tts_service as T


def _local_stack(tmp_path, monkeypatch, move_works=True):
    cached = tmp_path / config.XTTS_MODEL.replace("/", "--")
    cached.mkdir(parents=True)
    for name in ("config.json", "model.pth", "vocab.json", "speakers_xtts.pth"):
        (cached / name).write_bytes(b"fixture")
    calls = []

    class Model:
        device = "cpu"

        def to(self, device):
            if move_works:
                self.device = device

        def parameters(self):
            return iter([SimpleNamespace(device=self.device)])

    class Wrapper:
        def __init__(self, **kwargs):
            assert "model_name" not in kwargs, "Name lookup can perform a remote config check"
            self.manager = SimpleNamespace(output_prefix=tmp_path)

        def load_tts_model_by_name(self, *args, **kwargs):
            pytest.fail("Cached model took the network-capable name loader")

    def synthesizer(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(tts_model=Model(), vocoder_model=None)

    for name in ("TTS", "TTS.api", "TTS.utils", "TTS.utils.synthesizer"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["TTS.api"].TTS = Wrapper
    sys.modules["TTS.utils.synthesizer"].Synthesizer = synthesizer
    return cached, calls


def test_cached_snapshot_skips_remote_model_manager_and_moves_to_cuda(tmp_path, monkeypatch):
    cached, calls = _local_stack(tmp_path, monkeypatch)
    model = T._load_model(use_gpu=True)
    assert calls == [{"model_dir": str(cached), "use_cuda": False}]
    assert T._verify_model_device(model) == "cuda:0"


def test_cuda_load_rejects_model_that_remains_on_cpu(tmp_path, monkeypatch):
    _local_stack(tmp_path, monkeypatch, move_works=False)
    with pytest.raises(T.TTSUnavailable, match="parameter device is 'cpu'"):
        T._load_model(use_gpu=True)
