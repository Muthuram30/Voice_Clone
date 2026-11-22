"""Run only with .venv-indic/Scripts/python.exe; protocol is one JSON request on stdin."""
import json
import os
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    if Path(sys.prefix).resolve() != (root / ".venv-indic").resolve():
        raise RuntimeError("IndicF5 must run in the project's .venv-indic environment.")
    # Custom model code may load a vocoder through huggingface_hub as well.
    # Require all assets to have been prepared explicitly before a web request.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoModel
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. No CPU fallback is allowed.")
    args = json.loads(sys.stdin.read())
    # Official IndicF5 custom model code must be present in this local snapshot.
    model = AutoModel.from_pretrained(args["model"], trust_remote_code=True, local_files_only=True)
    model.to("cuda:0").eval()
    with torch.inference_mode():
        audio = model(args["text"], ref_audio_path=args["reference"], ref_text=args["transcript"])
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.
    sf.write(args["output"], audio.astype(np.float32), 24000, subtype="PCM_16")


if __name__ == "__main__":
    main()
