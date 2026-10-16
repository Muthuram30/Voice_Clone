"""Audio upload handling and enrollment-recording validation."""
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

import config
from services import dsp_service as D

ALLOWED = {".wav", ".flac", ".ogg"}

# ~280 words: read aloud at a natural pace this is roughly two minutes. The first paragraph is the
# fixed sentence from the project specification. Phonetically varied on purpose.
ENROLLMENT_SCRIPT = (
    "Hello, this is my voice sample for the speaker-adaptive speech synthesis demonstration. "
    "This recording will be used only to create my speaker reference representation.\n\n"
    "The quick brown fox jumps over the lazy dog, while seven quiet zebras watch from the hill. "
    "Please call Stella and ask her to bring these things with her from the store: six spoons of "
    "fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.\n\n"
    "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. "
    "The rainbow is a division of white light into many beautiful colors. These take the shape of "
    "a long, round arch, with its path high above, and its two ends apparently beyond the horizon.\n\n"
    "Technology changes the way we live, work, and communicate. A computer can listen to a voice, "
    "measure its pitch and its resonances, and describe the speaker with a handful of numbers. "
    "Yesterday I walked along the river at dawn, and the water was calm, cold, and perfectly clear. "
    "Today I am reading slowly and clearly, pausing at each full stop, so that every sound is captured.\n\n"
    "Numbers are useful too: one, two, three, four, five, six, seven, eight, nine, ten. "
    "On the twelfth of March, at a quarter past four, the train from Chennai to Coimbatore left "
    "exactly on time. Thank you for listening to this recording, and I give permission for it "
    "to be used in this demonstration."
)


class AudioError(Exception):
    pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def inspect(path) -> dict:
    try:
        i = sf.info(str(path))
    except Exception as e:  # noqa: BLE001
        raise AudioError(f"Invalid or unreadable audio file: {e}")
    return dict(duration_s=round(float(i.duration), 2), samplerate=int(i.samplerate), channels=int(i.channels),
                format=i.format, subtype=i.subtype)


def convert_to_wav(src: Path, dest: Path) -> dict:
    """Decode any supported file at `src` and write PCM16 WAV at `dest` (rate/channels kept)."""
    info = inspect(src)
    try:
        x, fs = sf.read(str(src), dtype="float32", always_2d=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest), x, fs, subtype="PCM_16")
    except Exception as e:  # noqa: BLE001
        raise AudioError(f"Could not decode audio: {e}")
    return info


def save_as_wav(file_storage, dest: Path) -> dict:
    """Save an uploaded file as PCM16 WAV at `dest`."""
    name = Path(file_storage.filename or "").name
    if Path(name).suffix.lower() not in ALLOWED:
        raise AudioError(f"Unsupported file type '{Path(name).suffix}'. Use: {', '.join(sorted(ALLOWED))}.")
    tmp = dest.with_suffix(".upload.tmp")
    dest.parent.mkdir(parents=True, exist_ok=True)
    file_storage.save(tmp)
    try:
        info = convert_to_wav(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    info["original_filename"] = name
    return info


def validate_reference(path) -> dict:
    """Check that a recording is usable for enrollment.
    Returns dict(ok, errors, warnings, duration_s, samplerate, channels, speech_seconds, speech_ratio, ...)."""
    errors, warnings = [], []
    info = inspect(path)
    dur = info["duration_s"]
    res = dict(info, errors=errors, warnings=warnings, ok=False)
    if dur < config.ENROLL_MIN_S:
        errors.append(f"Recording is {dur:.0f} s; at least {config.ENROLL_MIN_S:.0f} s is required "
                      f"(aim for about {config.ENROLL_TARGET_S:.0f} s: read the whole script).")
    if dur > config.ENROLL_MAX_S:
        errors.append(f"Recording is {dur:.0f} s; the maximum is {config.ENROLL_MAX_S:.0f} s.")
    if errors:
        return res
    x, fs = sf.read(str(path), dtype="float32", always_2d=True)
    y = x.mean(axis=1)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    res["peak"] = round(peak, 4)
    if peak < 0.01:
        errors.append("Recording is almost silent (peak below -40 dBFS). Check the microphone and levels.")
        return res
    clip = float(np.mean(np.abs(x) > 0.999))
    res["clipping_ratio"] = round(clip, 5)
    if clip > 0.01:
        warnings.append(f"{clip * 100:.1f}% of samples are clipped; lower the input gain and re-record if possible.")
    if fs < 16000:
        warnings.append(f"Sampling rate is {fs} Hz; 16 kHz or higher is recommended.")
    if abs(dur - config.ENROLL_TARGET_S) > 0.5 * config.ENROLL_TARGET_S:
        warnings.append(f"Duration {dur:.0f} s is far from the {config.ENROLL_TARGET_S:.0f} s target.")
    with D.LOCK:
        y16, fs16, _ = D.load_mono(path, 16000)
        P = D.preprocess(y16, fs16, denoise=False)
        V = D.energy_vad(P["processed"], fs16)
    speech_s = float(V["mask"].sum() * V["hop"] / fs16)
    res.update(speech_seconds=round(speech_s, 1), speech_ratio=round(V["speech_ratio"], 3))
    if speech_s < config.ENROLL_MIN_SPEECH_S:
        errors.append(f"Only {speech_s:.0f} s of speech detected; at least {config.ENROLL_MIN_SPEECH_S:.0f} s is needed. "
                      "Read the full script and avoid long pauses.")
    elif V["speech_ratio"] < config.ENROLL_MIN_SPEECH_RATIO:
        errors.append(f"Speech makes up only {V['speech_ratio'] * 100:.0f}% of the recording; "
                      "too much silence or noise.")
    res["ok"] = not errors
    return res
