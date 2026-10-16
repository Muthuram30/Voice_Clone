# Speaker-adaptive speech synthesis demo (Python + MATLAB DSP)

Each **profile** (the UI seeds "User 1" and "User 2", but you can create any number) goes through:

```
~2 min reading of the predefined script  ->  validation  ->  DSP preprocessing  ->  speaker embedding (+TTS reference)
  -> "Enrollment SUCCESS"  ->  any new text  ->  (Gemini/local text normalisation)  ->  speaker-conditioned TTS
  -> DSP post-processing -> spectral analysis -> compression experiment -> speaker verification -> play / download WAV
```

## What this is, and is not (read this)

* **No TTS network is trained.** Synthesis uses the *pretrained* Coqui **XTTS-v2** model with **zero-shot speaker
  conditioning**: from ~30 s of the profile's own reference speech it computes conditioning latents (cached per profile).
  The model is not fine-tuned. A short sample does not capture a whole voice; expect partial similarity.
* The speaker **embedding** comes from a *pretrained* SpeechBrain **ECAPA-TDNN** encoder. If PyTorch/SpeechBrain are not
  installed, an **MFCC-statistics baseline** (classical, not neural) is used and the profile metadata says so.
* The only models **fitted in this project** are the small **PCA + linear-SVM** speaker classifier (and its int8 version).
* Gemini only prepares text (punctuation, numbers, sentence splitting). It never produces audio. No key / quota / outage
  -> the local normaliser is used automatically. On the free tier Google may use prompts to improve products: don't send
  confidential text.
* Clone only voices you have permission to use. XTTS-v2 weights are **non-commercial** (Coqui Public Model License).
* MATLAB is **optional**. Everything MATLAB does is also implemented in Python (`services/dsp_service.py`).

## Project layout

```
app.py  config.py  requirements.txt  requirements-tts.txt  .env.example
routes/main.py            Flask routes (profile-scoped URLs: /profiles/<slug>/...)
services/
  profile_service.py      per-profile folders, metadata, wipe-on-new-upload, path safety
  audio_service.py        upload conversion, enrollment script, recording validation
  dsp_service.py          Python DSP (preprocess, VAD, pitch, V/UV, FFT, STFT, MFCC, LPC, features, post-process, compression)
  speaker_service.py      encoders (ECAPA | MFCC baseline), enrollment, PCA+SVM, verification
  tts_service.py          XTTS-v2 wrapper, per-profile cached conditioning latents
  gemini_service.py       optional text normalisation + guaranteed local fallback
  matlab_service.py       MATLAB Engine -> CLI bridge (optional)
  quant_service.py        int8 classifier (NumPy), dynamic int8 on ECAPA (PyTorch)
  synthesis_service.py    one synthesis run end-to-end + metrics.json
matlab/                   main_analysis, preprocess_audio, vad_analysis, vuv_analysis, fft_analysis, stft_analysis,
                          mfcc_analysis, pitch_analysis, lpc_formants, extract_features, compression_analysis,
                          postprocess_audio, main_postprocess (+ helpers frame_signal, smooth_mask, save_fig)
scripts/                  cli.py (headless), prepare_models.py (download pretrained models)
notebooks/colab_model_prep.ipynb
templates/ static/        UI: Home, Profiles, Enrollment, DSP analysis, Text & speech, Results, Experiments
profiles/<slug>/          reference.wav, reference_clean.wav, xtts_reference.wav, speaker_embedding.npy,
                          chunk_embeddings.npy, xtts_latents.pt, metadata.json, analysis/, outputs/<run>/
models/                   downloaded pretrained models + classifier_<encoder>.joblib (git-ignored)
tests/                    pytest suite + Octave cross-check script
```

User 1 and User 2 are separated by construction: every route is `/profiles/<slug>/...`, every file lives under
`profiles/<slug>/`, XTTS latents are computed only from that profile's `xtts_reference.wav`, and uploading a new
recording wipes everything derived from the old one for **that profile only**. The PCA/SVM classifier is the single object
trained from several profiles' embeddings (labelled by profile); it is used for verification, never for synthesis.

## Setup on Windows

1. Install **Python 3.11** (3.10-3.12 work). Open *Command Prompt* in the project folder:
   ```bat
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   copy .env.example .env
   python app.py
   ```
   Open http://127.0.0.1:5000. **Enrollment, DSP analysis, PCA/SVM, quantisation of the classifier and compression already
   work at this point** (MFCC-baseline encoder). Synthesis needs step 2.
2. **Speech synthesis (XTTS-v2 + ECAPA)**: install PyTorch for your machine from https://pytorch.org/get-started/locally/
   (CPU build is fine but slow; an NVIDIA GPU with CUDA is much faster), then
   ```bat
   pip install -r requirements-tts.txt
   ```
   Read https://coqui.ai/cpml, then set `COQUI_TOS_AGREED=1` in `.env`, and download the models once:
   ```bat
   python scripts\prepare_models.py
   ```
   (~2 GB). On Windows, SpeechBrain downloads use file copy instead of symlinks, so no developer mode is needed. If
   `pip install coqui-tts` fails to build, install the Microsoft C++ Build Tools or use Python 3.11 / Colab.
3. **GPU acceleration (NVIDIA RTX / GTX)** — see the "GPU setup on Windows" section below.
4. **Gemini (optional)**: create a key at https://aistudio.google.com, put it in `.env` as `GEMINI_API_KEY=...`. Model names
   change; adjust `GEMINI_MODELS` if a name is retired. Without a key everything still works.
5. **MATLAB (optional)** - see "How Python talks to MATLAB".

---

## GPU setup on Windows (NVIDIA RTX 3050 / any CUDA GPU)

### 1. Verify your NVIDIA driver
```powershell
nvidia-smi
```
Check the **Driver Version** (top-right) and the **CUDA Version** (top-right of the table). Both must be ≥ 12.0 for PyTorch cu121+ builds.

### 2. Check PyTorch CUDA availability
```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Expected output (CUDA build, GPU present): `2.14.1+cu130  True`.
If you see `False`, you have a CPU-only wheel — continue to step 3.

### 3. Install a CUDA-enabled PyTorch build
Go to https://pytorch.org/get-started/locally/ and select your OS / CUDA version.
For **CUDA 12.x** (RTX 3050 Laptop):
```powershell
.venv\Scripts\pip.exe install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```
> **Do not blindly overwrite existing compatible dependencies.**  
> Run `pip install --dry-run ...` first to see what would change.  
> The installed build `torch==2.14.1+cu130` is already CUDA-enabled; no reinstall is needed.

### 4. Run the GPU diagnostic script
```powershell
.venv\Scripts\python.exe scripts\check_gpu.py
```
This reports:
- Python / PyTorch / cuDNN / CUDA versions
- GPU name, total and free VRAM
- Coqui TTS version
- Configured `TTS_DEVICE` and the resolved device
- **XTTS benchmark**: model load time, first/second synthesis latency, RTF, peak VRAM

### 5. Configure `TTS_DEVICE` in `.env`
| Setting | Behaviour |
|---|---|
| `TTS_DEVICE=auto` | Use CUDA if available, else CPU silently |
| `TTS_DEVICE=cuda` | Require GPU — raises a diagnostic error if CUDA is absent |
| `TTS_DEVICE=cpu`  | Always use CPU |

The default is `auto`. No change needed if CUDA is available.

### 6. Start the Flask app and verify GPU use
```powershell
.venv\Scripts\python.exe app.py
```
You should see a startup log line such as:
```
16:51:01 INFO config: [config] Device selected: cuda:0 | GPU: NVIDIA GeForce RTX 3050 Laptop GPU | VRAM: 4096 MB total, 3924 MB free
```
While a synthesis request is running, open a second terminal and run:
```powershell
nvidia-smi
```
If GPU-Util shows > 0 % during synthesis, the model is running on GPU.

### 7. Troubleshoot CUDA errors
| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` returns `False` | Install CUDA-enabled PyTorch (step 3) |
| `CUDA out of memory` | Set `MAX_TEXT_CHARS=800` in `.env`; split long texts |
| `CudaNotAvailableError` at startup | Change `TTS_DEVICE=cpu` in `.env` to use CPU mode |
| Model loads but `actual_model_device` is `cpu` | Recheck your PyTorch build tag (`torch.__version__`) |
| `FutureWarning: torch.jit.script is deprecated` | Safe to ignore; comes from coqui-tts internals |

### 8. Use CPU mode when needed
Set `TTS_DEVICE=cpu` in `.env`. The application continues to work with no change to output quality or profile state.

### 9. VRAM budget for XTTS-v2 on 4 GB
- XTTS-v2 weights: ~1.8 GB
- Per-sentence KV cache during inference: ~200-400 MB
- Conditioning latents in cache: negligible
- Remaining headroom on a 4 GB card: ~1.5 GB → suitable for sentences up to ~`MAX_TEXT_CHARS=1000`

---


### Google Colab
Open `notebooks/colab_model_prep.ipynb` (T4 GPU runtime). It installs the TTS stack, downloads the pretrained models, enrolls
two profiles from your recordings, synthesises text for each, and runs the PCA/SVM + quantisation experiments through
`scripts/cli.py`. **Colab-only = model preparation and GPU inference.** Nothing is trained there either. Local inference =
the Flask app above (CPU or your own GPU); copy `models/` from Colab to reuse downloads.

## Using it

**User 1 enrollment** - Profiles -> `[ User 1 ]` -> *Enrollment*. Read the displayed script (~2 min), then either *Upload
recording* (WAV/FLAC/OGG) or *Record in browser*. The page shows duration, sampling rate, VAD speech time and a waveform. Press
**Process & Enroll Voice** -> status **SUCCESS** (DSP analysis, speech extraction, per-chunk embeddings, TTS reference).
Recordings < 60 s, > 300 s, almost silent, or with < 40 s of detected speech are rejected with the reason.

**User 2 enrollment** - exactly the same on `[ User 2 ]`; nothing from User 1 is read or written.

**Generate speech** - *Text & speech* for the chosen profile: type any text, pick the language, optionally enable Gemini
normalisation, *Generate Speech*. The Results page has players + downloads for the final WAV, the raw TTS output and the
compression-experiment reconstruction, plus all measured metrics. The first synthesis of a profile also computes and
caches its XTTS latents.

**Headless:** `python scripts/cli.py enroll --profile "User 1" --wav ref.wav`, `... synth --profile "User 1" --text "..."`.

## How Python talks to MATLAB

`services/matlab_service.py` is the only place MATLAB is touched. It calls the same `.m` entry points
(`main_analysis(wav, outDir, fs)`, `main_postprocess(wav, outDir, hp, lp, targetRmsDb)`) by one of two routes, each writing
PNGs + `results.json` into an output folder that Python then reads:

1. **MATLAB Engine API** (`MATLAB_MODE=auto|engine`): install once from your MATLAB folder,
   `cd "C:\Program Files\MATLAB\R2024b\extern\engines\python" && python -m pip install .`, then the engine is started lazily and kept alive.
2. **Command line** (`MATLAB_MODE=cli`, or automatic fallback if the Engine package is missing): runs
   `matlab -batch "addpath('matlab'); main_analysis('...','...',16000.0);"`. Set `MATLAB_EXE` in `.env` if `matlab` isn't on PATH.

A real MATLAB *error* is surfaced (not silently replaced by the CLI or by Python). With `DSP_BACKEND=python`, or when MATLAB
isn't found, only the Python implementation runs and the UI says so. Needs the **Signal Processing Toolbox** (no Audio Toolbox);
R2020a+ recommended (`stft/istft` need R2019a+).

## How the academic requirements are covered

| Requirement | Where | Notes |
|---|---|---|
| DSP chain (load, mono, resample, DC, normalise, pre-emphasis, denoise, VAD, V/UV/silence, FFT, STFT, MFCC, F0, LPC formants, feature vector, plots) | `matlab/*.m` and `services/dsp_service.py` | 53-feature vector, 7 plots, identical stage list in both implementations |
| ML / DL | pretrained ECAPA-TDNN embeddings, XTTS-v2 conditioning, **PCA + linear SVM** classifier, speaker verification (cosine + SVM) | pretrained models are labelled pretrained; only PCA/SVM is fitted here |
| Speaker verification metrics | Experiments page, `metrics.json` | hold-out accuracy, confusion matrix, cosine of generated speech vs own/other profiles |
| Post-processing | `dsp_service.postprocess`, `matlab/postprocess_audio.m` | DC removal, high-pass, optional low-pass, RMS normalisation + peak limit, fades |
| Compression | `dsp_service.compression_sweep`, `matlab/compression_analysis.m` | STFT-threshold and block-DCT codecs; top-k coefficients + log-magnitude (+phase/sign) quantisation; reconstruction decoded from the real compressed bytes; ratio, kbps, SNR, log-spectral distance |
| Quantisation | `services/quant_service.py` | (a) classifier -> int8 weights+activations, int32 accumulation: size, latency, agreement, accuracy. (b) ECAPA dynamic int8 via PyTorch - reports how many layers were really converted (ECAPA is convolutional, so possibly none). **XTTS-v2 is not quantised** (unverified for its inference path) |
| Metrics: SNR, RMS, spectral difference, compression ratio, file size, inference time | Results page | all measured per run (see `metrics.json`); none are hard-coded |

## Verification status (what was and was not tested)

Run `python -m pytest tests -q`: **26 tests pass** (environment used to write this: Python 3.12, CPU, no GPU, no MATLAB, no PyTorch).

Tested for real, with synthetic "speakers" (known F0/formants, not human speech): upload validation, profile isolation
and wipe-on-reupload, enrollment, PCA/SVM, int8 classifier, DSP accuracy (F0 within 5 %, formants within 6 %), post-processing,
compression round trip from bytes, Gemini success/timeout/HTTP-429/bad-JSON/rejected-output paths, TTS-unavailable and
MATLAB-unavailable error handling, every route, file-serving restrictions, the full synthesis pipeline.
Also run: a live `python app.py` + HTTP requests, and the CLI.

MATLAB: no MATLAB licence was available. The modules that do the numerical work (VAD, pitch, V/UV, LPC formants, MFCC,
feature vector, post-processing) were executed in **GNU Octave 8.4** on the same audio as Python, with test-only shims for
`spectrogram` and plotting (`tests/octave_shims`, `tests/octave_check.m`): VAD regions, F0 mean/std, feature names
and post-processing metrics matched Python closely (MFCC means differ by filterbank convention, correlation 0.99).

**NOT tested - you must verify:**
1. **XTTS-v2 synthesis** (`tts_service.py`) and the **ECAPA encoder** (`EcapaEncoder`): no model download/GPU here. The
   code follows the `coqui-tts` / SpeechBrain APIs, but exact signatures vary between versions. The synthesis tests use a
   clearly labelled test double (`TEST-DOUBLE`) in place of XTTS.
2. **MATLAB itself**: `stft/istft` calls, `exportgraphics` figure output and layout, `jsonencode`, the Engine/CLI bridge, and
   `compression_analysis.m` (uses `stft/istft`). Expect to fix small version-specific errors; send me the message.
3. The browser **recorder** (`static/js/record.js`): needs a browser and microphone.
4. **Gemini** against the live API (tested with mocked responses; endpoint/response shape follow the REST docs).
5. Torch **dynamic quantisation** of ECAPA (`quantize_encoder`).
6. Real-speech behaviour of VAD/pitch thresholds (tuned on synthetic signals; real voices may need threshold tweaks in `dsp_service.py`).

Classifier numbers need care: with one recording per speaker, SVM hold-out accuracy measures separability of those two
recordings (same session/microphone), not general speaker-verification accuracy. Report it that way.

## Troubleshooting

* "TTS packages are not installed" → step 2 above. "licence not accepted" → `COQUI_TOS_AGREED=1`.
* "ECAPA unavailable, using MFCC baseline" is shown in the enrollment status when SpeechBrain isn't usable; install
  `requirements-tts.txt` and re-enroll to switch encoders (profiles keep the encoder they were enrolled with).
* "Only N s of speech detected" → read the full script, avoid long pauses, reduce background noise.
* MATLAB: check `ver` for Signal Processing Toolbox; set `MATLAB_EXE`; use `DSP_BACKEND=python` to bypass MATLAB entirely.
* GPU / CUDA issues → run `scripts/check_gpu.py` and see the "GPU setup on Windows" section above.
* `CudaNotAvailableError` at startup → set `TTS_DEVICE=cpu` in `.env`.
* `CUDA out of memory` → set `MAX_TEXT_CHARS=800` in `.env` or use `TTS_DEVICE=cpu`.


## Final manual test checklist

- [ ] `pip install -r requirements.txt`, `python -m pytest tests -q` -> all pass; `python app.py` opens at :5000
- [ ] Home shows correct TTS / encoder / MATLAB / Gemini status
- [ ] Profiles page shows [User 1] and [User 2]; a third profile can be created and deleted
- [ ] Upload a < 60 s file -> rejected with reason; upload a silent file -> rejected; `.mp3` -> rejected
- [ ] User 1: read the script, upload/record ~2 min, see duration / sampling rate / waveform, *Process & Enroll Voice* -> SUCCESS
- [ ] User 2: same, with a different voice; User 1's page and files are unchanged
- [ ] Re-uploading for User 1 resets only User 1 to "uploaded"
- [ ] DSP analysis page: Python plots appear; with MATLAB installed, "Both" shows a Python-vs-MATLAB table
- [ ] After `requirements-tts.txt` + `prepare_models.py`: User 1 text -> audio plays, downloads; same text for User 2 sounds different
- [ ] Results: metrics tables, compression table + R-D plot, verification block (cosine, SVM label) are filled with measured values
- [ ] Gemini on/off and with a wrong key: generation still works (note says local normaliser used)
- [ ] Experiments: classifier trained (needs 2 enrolled profiles), int8 report, compression sweep
- [ ] Unplug internet / remove key: app still enrolls and (with models cached) synthesises
