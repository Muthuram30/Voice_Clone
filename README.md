# VoiceLab — speaker-adaptive speech synthesis

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-325B6B?logo=flask&logoColor=white) ![CUDA](https://img.shields.io/badge/CUDA-527F0B?logo=nvidia&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-B94133?logo=pytorch&logoColor=white) ![Audio DSP](https://img.shields.io/badge/Audio%20DSP-506274)

A local Flask voice studio with profile-separated enrollment, CUDA signal processing, pretrained speaker-conditioned speech generation, and measured DSP/ML experiments.

## Run the existing installation

From the project folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:5000>. Use the project virtual environment. The application binds to localhost; it is a personal demo without multi-user authentication.

CUDA is required for signal processing and speech generation. CPU operations include file conversion, plot/report generation, NumPy statistics, zlib compression, and scikit-learn classification. `TTS_DEVICE=cpu` can detect a CPU runtime but does not enable a CPU DSP or synthesis path. MATLAB source files remain for academic reference; the web studio runs CUDA DSP.

## Fresh installation on Windows

Use Python 3.11 for the main environment:

```powershell
py -3.11 -m venv .venv
```

Install matching CUDA-enabled PyTorch and torchaudio builds using the command selected for Windows/Pip/CUDA on the [official PyTorch installation page](https://pytorch.org/get-started/locally/), replacing `pip` with `.\.venv\Scripts\python.exe -m pip`. Keep a working CUDA installation rather than reinstalling it unnecessarily. Then:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-tts.txt
Copy-Item .env.example .env
```

Copy the template only for a new installation; preserve an existing `.env`. Set `TTS_DEVICE=cuda`. Read the [Coqui Public Model License](https://coqui.ai/cpml) and set `COQUI_TOS_AGREED=1` only after accepting it. Download the XTTS/ECAPA models explicitly:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_models.py
.\.venv\Scripts\python.exe -c "import config; import torch; config.initialize_runtime(); print(torch.__version__, torch.cuda.is_available(), config.TORCH_DEVICE)"
.\.venv\Scripts\python.exe app.py
```

A working CUDA setup prints `True` and `cuda:0`. The NVIDIA driver must support the CUDA runtime bundled with the installed PyTorch wheel. `nvidia-smi` reports the driver's supported CUDA version, not the version of a separately installed toolkit.

Install full-shared FFmpeg and set `FFMPEG_BIN_DIR` to its `bin` folder if needed. XTTS audio loading through TorchCodec may need FFmpeg DLLs as well as `ffmpeg.exe`. MP3/AAC/WebM/video imports and lossy exports also use FFmpeg.

## Use the studio

1. Create or choose a voice profile. Record a single consenting speaker in a quiet room for 60–300 seconds; aim for two minutes. Speak in the language/accent you want to preserve. The English reading passage is optional.
2. Import audio/video or record in the browser. Browser recording uses AudioWorklet PCM capture, an input meter, and a preview before upload. It requires microphone permission and localhost or HTTPS.
3. Upload and validate the recording, then choose **Process & Enroll Voice**. Enrollment requires at least 40 seconds of detected speech and a speech ratio of 0.35 by default. Review the analysis plots and enrollment status.
4. Open **Create speech**, choose a supported engine and language, then enter text or import TXT/SRT/VTT. Subtitle imports remove timing; they generate a continuous passage. Text over the configured limit must be split manually and is never silently truncated.
5. Generate speech, listen, review the measurements, and export WAV, FLAC, MP3, OGG, Opus, M4A, AIFF, or a ZIP containing the run's audio and reports.

Uploads are limited to 120 MB. Supported imports: WAV, FLAC, OGG, MP3, M4A, AAC, Opus, WebM, AIFF, WMA, and audio tracks from MP4/MOV/MKV/AVI. WAV/FLAC exports preserve the generated signal; lossy formats trade quality for size.

Each profile owns its recording, analysis, references, conditioning cache, embeddings and outputs. Uploading a replacement clears derived enrollment data for that profile. Existing runs remain available. Mutating web requests and synthesis jobs are serialized within the single application process to protect references and the shared GPU. Run one server process.

## Engines and voice quality

**XTTS-v2:** pretrained zero-shot speaker conditioning, not training or fine-tuning. It supports English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Mandarin Chinese, Japanese, Hungarian, Korean, and Hindi. Tamil is unsupported and is rejected rather than sent to XTTS.

Enrollment selects two uncut 12-second excerpts from the original recording, retaining breaths and natural pauses with only light filtering and tiny boundary fades. Separate excerpts guide XTTS conditioning, while a combined reference is retained for compatibility. Legacy stitched references are rebuilt once from the original recording on the next synthesis; the original recording is kept intact. Conditioning latents are cached per profile and fingerprinted against the reference audio and conditioning configuration.

**IndicF5:** optional separate backend for Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, and Telugu. It requires an uncut 3–15 second prompt from the intended speaker and the exact words spoken in that prompt. This prompt is supplied per run; the stitched XTTS reference is unsuitable because its transcript does not match a continuous recording.

Voice similarity and pronunciation vary by speaker, reference quality and language. There is no reliable American/British accent detector here. Speaker similarity scores do not establish native pronunciation or a perfect voice match.

## Optional IndicF5 setup

The main `.venv` must stay separate from IndicF5's older dependency stack. The [official IndicF5 repository](https://github.com/AI4Bharat/IndicF5) recommends Python 3.10. Create `.venv-indic` with that interpreter, install a compatible matching CUDA PyTorch/torchaudio pair inside it, then install:

```powershell
py -3.10 -m venv .venv-indic
.\.venv-indic\Scripts\python.exe -m pip install -r requirements-indic.txt
```

Do not install `requirements-indic.txt` into `.venv`. Model downloads require access granted through the [IndicF5 model page](https://huggingface.co/ai4bharat/IndicF5); review its access conditions and authenticate through Hugging Face before downloading. Save the full model snapshot, including its custom Python code, into `models/indicf5`. Prepare the model's additional assets, including its vocoder cache, in the isolated environment using the official instructions before web inference.

The service looks for `.venv-indic/Scripts/python.exe`, `models/indicf5/config.json` and `models/indicf5/model.safetensors`. Presence of these files means setup is configured, not that inference was verified. The worker runs with Hugging Face offline mode enabled, so missing assets fail instead of downloading during a web request. XTTS is unloaded before invoking the isolated CUDA worker. Generation has a ten-minute timeout.

This backend is optional and is not verified until its gated model and dependencies are available locally. On a 4 GB GPU, successful inference also depends on free VRAM and passage length. Do not assume it fits without testing it on the target machine.

## DSP and experiments

| Stage | Implementation / output |
|---|---|
| Preprocessing | CUDA resampling, DC removal, forward/backward high-pass biquads, spectral subtraction, normalization and pre-emphasis |
| Speech analysis | CUDA energy VAD, autocorrelation pitch, voiced/unvoiced/silence, FFT/Welch, STFT, MFCC and LPC formants |
| Features | 53 measured statistics, plots and JSON reports |
| Speaker encoder | Pretrained SpeechBrain ECAPA-TDNN; an explicitly labelled MFCC-statistics baseline if configured or ECAPA cannot load |
| Classification | StandardScaler → PCA → linear SVM, fitted to profile-labelled chunk embeddings |
| Verification | Cosine similarity and optional classifier prediction; never used to generate the voice |
| Post-processing | DC removal, high/optional low-pass, target RMS, peak limiting and fades |
| Compression | CUDA STFT/block-DCT transforms; host coefficient selection, quantization and zlib; metrics computed after decoding the actual compressed bytes |
| Quantization | Int8 classifier experiment and optional ECAPA dynamic quantization report; XTTS is not quantized |

Classifier evaluation uses a time-block split with a gap between overlapping training/test chunks. One recording per speaker measures separation of those sessions, not general speaker verification accuracy. ECAPA is largely convolutional, so dynamic quantization may convert few or no layers; inspect the actual report.

Gemini is optional and only normalizes text. It never generates audio. Without a key or after an API failure, local text preparation is used. Enable it in the form only when sending that text to Google is appropriate.

## Project layout

```text
app.py / config.py        Flask application, configuration and CUDA detection
routes/main.py            profile, enrollment, synthesis, report and export routes
services/gpu_dsp.py        CUDA signal kernels
services/dsp_service.py    analysis orchestration, plots, metrics and compression
services/audio_service.py conversion and recording validation
services/profile_service.py profile storage and isolation
services/speaker_service.py ECAPA/MFCC embeddings, references and PCA/SVM
services/tts_service.py    CUDA XTTS, conditioning and cache
services/indic_service.py  isolated optional IndicF5 service
scripts/indic_worker.py    isolated offline CUDA worker
services/synthesis_service.py end-to-end synthesis and metrics
services/gemini_service.py optional text preparation and local fallback
services/quant_service.py quantization experiments
templates/ / static/       voice studio, recorder and text import
profiles/<slug>/           private profile data and outputs (git-ignored)
models/                    model files and classifiers (git-ignored)
scripts/cli.py             headless enrollment/synthesis/experiments
scripts/check_gpu.py       CUDA diagnostic and optional XTTS benchmark
matlab/                    retained academic reference implementation
tests/                     synthetic-signal, service and route tests
```

## Validation

Run the suite in the CUDA-enabled project environment:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
node --check static/js/record.js
node --check static/js/pcm-worklet.js
node --check static/js/studio.js
```

The DSP tests check synthetic signals against known F0/formants, DC/peak/RMS behavior, and compression reconstruction from bytes. Enrollment tests check profile separation, replacement behavior, classification and verification. Route tests exercise pages, errors, generated reports and restricted file access. Synthesis pipeline tests use a labelled `TEST-DOUBLE`; passing them does not establish real XTTS/IndicF5 voice quality.

Live XTTS generation, browser microphone capture, the optional gated IndicF5 backend, and Gemini's live API require separate runtime checks. `scripts/check_gpu.py` may load/download XTTS and run a benchmark; inspect its output rather than treating package presence as proof of synthesis readiness.

## Troubleshooting

- **CUDA unavailable:** verify `nvidia-smi`, the matching torch/torchaudio wheels, the driver, and `torch.cuda.is_available()` in `.venv`. CPU mode does not provide a DSP fallback.
- **GPU out of memory:** close other GPU applications and generate a shorter passage. No automatic CPU synthesis fallback is used.
- **FFmpeg/TorchCodec loading error:** point `FFMPEG_BIN_DIR` at a full-shared FFmpeg `bin` directory with compatible DLLs and restart the server.
- **Too little speech:** record the full passage in a quiet room, avoid long pauses, and check the microphone level.
- **IndicF5 setup required:** complete the isolated environment, gated model snapshot and vocoder cache; use the exact prompt transcript.
- **Unclear pronunciation:** use native-script text for the chosen language and a clean reference in that language; reference conditioning cannot guarantee a particular accent.

### Real-model check on the target laptop

The local cached XTTS snapshot and pretrained ECAPA ran on the RTX 3050 with PyTorch `2.14.1+cu130` on 22 November 2026. The XTTS model loaded in 11.2 seconds after bypassing Coqui's remote configuration check for a complete cached snapshot.

Two existing enrolled profiles generated English speech through Flask's synthesis route, including DSP finishing, compression, speaker verification, result rendering, and all eight export options. Both samples were closest to their own enrolled profile. Measured generation results:

| Sample | Audio duration | XTTS inference including conditioning | Device |
|---|---:|---:|---|
| First profile | 4.03 s | 4.13 s | cuda:0 |
| Second profile | 3.01 s | 1.47 s | cuda:0 |

These are individual checks, not average latency or human listening scores. To repeat with an already enrolled profile (this creates a new output run):

```powershell
.\.venv\Scripts\python.exe scripts\verify_local_runtime.py --profile <slug>
```

Repeat `--profile` for additional voices. Browser microphone capture and IndicF5 inference remain unverified. Browser preview could not reach the sandboxed local server during this check; route verification used Flask's in-process request client.

Final automated validation on 22 November 2026: **68 passed, 3 skipped** (`python -m pytest tests -q`, 154.41 seconds). The skips cover package-unavailable scenarios that do not apply when the TTS/SpeechBrain packages are installed. Python compilation, JavaScript syntax checks, dependency consistency (`pip check`), and Git whitespace checks also passed.


## Speech naturalness safeguards

Text preparation no longer inserts full stops at artificial length cuts. Short adjacent sentences share context in a single bounded passage; longer text is split against both the language character limit and actual tokenizer budget. Inference uses the downloaded model configuration (temperature 0.75, top-p 0.85 and repetition penalty 5 for the installed checkpoint), stochastic sampling, and speed 1.0. These settings preserve the model's intended prosody; they do not guarantee human-sounding speech.

Assembly preserves the model's own pauses, adds only a missing minimum pause at a passage boundary, and applies 3 ms boundary fades. A single global headroom adjustment preserves relative word/sentence loudness. CUDA filters use unclamped linear biquads before the final peak limiter, avoiding intermediate hard clipping. The final WAV/FLAC is full quality; the separate STFT compression reconstruction is explicitly labelled lossy and hidden in an expandable experiment panel.

Each new run includes `audio_quality`: clipped-sample fraction, peak, RMS, longest internal pause, and artifact warnings. A clean report is an artifact screen, not a perceptual naturalness or pronunciation score. Human listening is still required for accent, rhythm, repeated/missing words and whether a voice sounds robotic.

New regressions cover punctuation preservation, tokenizer limits, continuous reference selection, pause handling, invalid passages, relative loudness preservation, and filter linearity. Private recordings, outputs, reference backups, environments, weights and local reports remain excluded from Git.

Validation note: one exploratory Windows CUDA process exited with native error `0xc0000008`; the immediate repeat passed all 12 focused checks, and the final full suite passed all 68 applicable tests. The native error was not reproduced or attributed to a specific code defect.

Latest real CUDA checks: conversational and long-sentence samples were generated for both existing profiles, with all eight export options checked for every run. All four final signals had 0% clipped samples and no artifact-screen warnings. Long-sentence outputs were 26.90 s and 18.39 s, used two context-bounded passages, and added only 20 ms / 50 ms where the existing model pause was shorter than the requested boundary pause. Longest internal pauses were 0.74 s / 0.78 s. Both long samples matched their own enrolled profile in speaker verification. These measurements do not certify pronunciation or perceptual naturalness; listen to Full-quality speech in each profile's Results page.

## License

The project source is available under the [MIT License](LICENSE). External models remain subject to their own terms.
