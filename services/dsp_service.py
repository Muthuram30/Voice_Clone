"""Python implementation of the DSP chain (portable twin of matlab/*.m).

Stages: load -> mono -> resample -> DC removal -> high-pass -> spectral-subtraction noise reduction
-> peak normalisation -> pre-emphasis -> VAD -> pitch (autocorrelation) -> voiced/unvoiced/silence
-> FFT/Welch -> STFT -> MFCC -> LPC formants -> feature vector.  Post-processing and the STFT/DCT
compression experiment are also here.  Every number returned is computed from the signal.
"""
import json, struct, threading, time, zlib
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as sfft
import scipy.ndimage as ndi
import scipy.signal as ss
import soundfile as sf
from scipy.linalg import solve_toeplitz

LOCK = threading.RLock()          # matplotlib is not thread-safe
HOP_MS, FRAME_MS, PITCH_FRAME_MS = 10, 25, 40

FEATURE_NAMES = (
    [f"mfcc{i}_mean" for i in range(13)] + [f"mfcc{i}_std" for i in range(13)] +
    [f"dmfcc{i}_mean" for i in range(13)] +
    ["f0_mean_hz", "f0_std_hz", "f0_median_hz", "f0_p5_hz", "f0_p95_hz",
     "F1_hz", "F2_hz", "F3_hz", "centroid_mean_hz", "centroid_std_hz", "rolloff85_mean_hz",
     "zcr_mean", "speech_ratio", "voiced_ratio_of_speech"]
)


# ---------------------------------------------------------------- loading
def load_mono(path, target_fs=None):
    """Load any soundfile-readable file -> (mono float32, fs, info)."""
    x, fs0 = sf.read(str(path), always_2d=True, dtype="float32")
    info = dict(orig_fs=int(fs0), channels=int(x.shape[1]), duration=len(x) / fs0)
    y = x.mean(axis=1)
    if target_fs and fs0 != target_fs:
        y = librosa.resample(y, orig_sr=fs0, target_sr=target_fs, res_type="soxr_hq")
        fs0 = target_fs
    return y.astype(np.float32), int(fs0), info


def write_wav(path, y, fs):
    y = np.clip(np.asarray(y, dtype=np.float32), -1.0, 1.0)
    sf.write(str(path), y, int(fs), subtype="PCM_16")


def dbfs(x):
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12))


# ---------------------------------------------------------------- framing
def frames(y, L, H):
    """Centre-padded framing -> (n_frames, L); n_frames = 1 + len//H (same grid as librosa)."""
    assert L % 2 == 0
    yp = np.pad(y, (L // 2, L // 2))
    n = 1 + len(y) // H
    idx = np.arange(L)[None, :] + H * np.arange(n)[:, None]
    return yp[idx]


def _hop(fs):
    return int(round(fs * HOP_MS / 1000))


def _flen(fs, ms):
    return 2 * int(round(fs * ms / 2000))


# ---------------------------------------------------------------- preprocessing
def spectral_subtract(y, fs):
    """Spectral-subtraction noise reduction; noise = mean magnitude of the quietest 10 % frames."""
    S = librosa.stft(y, n_fft=512, hop_length=128, window="hann")
    mag = np.abs(S)
    e = (mag ** 2).sum(0)
    idx = np.argsort(e)[: max(1, int(0.1 * len(e)))]
    noise = mag[:, idx].mean(1, keepdims=True)
    mag2 = np.maximum(mag - 1.5 * noise, 0.05 * mag)
    out = librosa.istft(mag2 * np.exp(1j * np.angle(S)), hop_length=128, window="hann", length=len(y))
    return out.astype(np.float32)


def preprocess(y, fs, denoise=True):
    y = y - np.mean(y)                                           # DC removal
    b, a = ss.butter(4, 80 / (fs / 2), "high")                   # rumble filter
    hp = ss.filtfilt(b, a, y).astype(np.float32)
    yd = spectral_subtract(hp, fs) if denoise else hp
    yd = (yd / (np.max(np.abs(yd)) + 1e-12) * 0.95).astype(np.float32)   # peak normalisation
    emph = ss.lfilter([1, -0.97], [1], yd).astype(np.float32)    # pre-emphasis
    return dict(raw=y.astype(np.float32), hp=hp, processed=yd, emph=emph, fs=fs)


# ---------------------------------------------------------------- VAD
def energy_vad(y, fs):
    """Energy VAD with adaptive threshold, gap closing, short-burst removal and hangover."""
    L, H = _flen(fs, FRAME_MS), _hop(fs)
    e = 10 * np.log10(np.mean(frames(y, L, H) ** 2, axis=1) + 1e-12)
    floor, top = np.percentile(e, 10), np.percentile(e, 95)
    thr = max(floor + max(0.3 * (top - floor), 6.0), -55.0)
    m = e > thr
    m = ndi.binary_closing(m, np.ones(15))        # bridge gaps < 150 ms
    m = ndi.binary_opening(m, np.ones(10))        # drop bursts < 100 ms
    m = ndi.binary_dilation(m, np.ones(5))        # hangover
    d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    regions = [[float(a * H / fs), float(b * H / fs)] for a, b in zip(st, en)]
    return dict(mask=m, e_db=e, thr_db=float(thr), regions=regions,
                speech_ratio=float(m.mean()), hop=H, frame_len=L)


# ---------------------------------------------------------------- pitch / voicing
def pitch_track(y, fs, fmin=60.0, fmax=400.0, voicing_thr=0.50):
    """Autocorrelation pitch tracker. Returns f0 (NaN where unvoiced), voicing strength, times."""
    L, H = _flen(fs, PITCH_FRAME_MS), _hop(fs)
    w = np.hanning(L)
    fr = frames(y, L, H) * w
    n = 2 ** int(np.ceil(np.log2(2 * L)))
    r = np.fft.irfft(np.abs(np.fft.rfft(fr, n, axis=1)) ** 2, n, axis=1)[:, :L]
    rw = np.fft.irfft(np.abs(np.fft.rfft(w, n)) ** 2, n)[:L]       # undo window taper bias
    rn = r / (rw[None, :] + 1e-12)
    rn = rn / (rn[:, :1] + 1e-12)
    lo, hi = int(np.floor(fs / fmax)), int(np.ceil(fs / fmin))
    seg = rn[:, lo:hi + 1]
    interior = np.zeros_like(seg, dtype=bool)
    interior[:, 1:-1] = (seg[:, 1:-1] > seg[:, :-2]) & (seg[:, 1:-1] >= seg[:, 2:])
    mx = np.where(interior, seg, -np.inf).max(1)
    cand = interior & (seg >= 0.9 * mx[:, None])                  # first strong peak: avoids octave errors
    k = np.argmax(cand, axis=1)
    k = np.clip(k, 1, seg.shape[1] - 2)
    rows = np.arange(len(k))
    y1, y2, y3 = seg[rows, k - 1], seg[rows, k], seg[rows, k + 1]
    den = (y1 - 2 * y2 + y3)
    delta = np.where(np.abs(den) > 1e-12, 0.5 * (y1 - y3) / den, 0.0)
    lag = lo + k + np.clip(delta, -1, 1)
    strength = y2
    f0 = fs / lag
    rms = np.sqrt(np.mean(fr ** 2, axis=1))
    voiced = (strength > voicing_thr) & np.isfinite(mx) & (rms > 1e-4) & (f0 >= fmin) & (f0 <= fmax)
    f0m = ss.medfilt(np.where(voiced, f0, 0.0), 5)
    voiced = voiced & (f0m > 0)
    f0m = np.where(voiced, f0m, np.nan)
    t = np.arange(len(f0)) * H / fs
    return dict(f0=f0m, strength=strength, voiced=voiced, t=t)


def vuv_labels(vad_mask, voiced):
    """0 = silence, 1 = unvoiced speech, 2 = voiced speech."""
    lab = np.zeros(len(vad_mask), dtype=int)
    lab[vad_mask] = 1
    lab[vad_mask & voiced] = 2
    return lab


# ---------------------------------------------------------------- LPC formants
def _lpc(seg, order):
    r = np.correlate(seg, seg, "full")[len(seg) - 1: len(seg) + order]
    if r[0] <= 0:
        return None
    r = r.copy(); r[0] *= 1.000001
    try:
        a = solve_toeplitz(r[:order], r[1:order + 1])
    except Exception:
        return None
    return np.concatenate([[1.0], -a])


def lpc_formants(emph, fs, labels, hop):
    L, step, order = int(0.03 * fs), int(0.02 * fs), 2 + int(round(fs / 1000))
    win = np.hamming(L)
    out = []
    for s in range(0, len(emph) - L, step):
        k = min(int(round((s + L // 2) / hop)), len(labels) - 1)
        if labels[k] != 2:
            continue
        seg = emph[s:s + L] * win
        if np.sqrt(np.mean(seg ** 2)) < 1e-3:
            continue
        a = _lpc(seg, order)
        if a is None:
            continue
        rt = np.roots(a); rt = rt[np.imag(rt) > 0.01]
        fr = np.arctan2(rt.imag, rt.real) * fs / (2 * np.pi)
        bw = -np.log(np.abs(rt)) * fs / np.pi
        o = np.argsort(fr); fr, bw = fr[o], bw[o]
        fr = fr[(fr > 90) & (bw < 400)]
        if len(fr) >= 3:
            out.append(fr[:3])
    F = np.array(out) if out else np.zeros((0, 3))
    med = np.median(F, 0) if len(F) else np.array([np.nan] * 3)
    return dict(F=F, median=med, frames=int(len(F)))


# ---------------------------------------------------------------- features
def spectral_feats(y, fs):
    H = _hop(fs)
    cen = librosa.feature.spectral_centroid(y=y, sr=fs, n_fft=512, hop_length=H)[0]
    rol = librosa.feature.spectral_rolloff(y=y, sr=fs, n_fft=512, hop_length=H, roll_percent=0.85)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=_flen(fs, FRAME_MS), hop_length=H)[0]
    return cen, rol, zcr


def mfcc_feats(y, fs):
    H = _hop(fs)
    m = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=13, n_fft=512, hop_length=H, n_mels=40, fmin=0, fmax=fs / 2)
    return m, librosa.feature.delta(m, width=9)


def feature_vector(mfcc, dmfcc, f0, formants_median, cen, rol, zcr, vad_mask, vuv):
    n = min(mfcc.shape[1], len(vad_mask), len(cen), len(rol), len(zcr), len(f0))
    sp = vad_mask[:n]
    if sp.sum() < 3:
        sp = np.ones(n, dtype=bool)
    mf, dm = mfcc[:, :n][:, sp], dmfcc[:, :n][:, sp]
    f0v = f0[:n][~np.isnan(f0[:n])]
    p = (lambda q: float(np.percentile(f0v, q))) if len(f0v) else (lambda q: np.nan)
    sp_n = max(int(vad_mask[:n].sum()), 1)
    v = np.concatenate([
        mf.mean(1), mf.std(1), dm.mean(1),
        [np.mean(f0v) if len(f0v) else np.nan, np.std(f0v) if len(f0v) else np.nan,
         np.median(f0v) if len(f0v) else np.nan, p(5), p(95)],
        np.asarray(formants_median, float),
        [cen[:n][sp].mean(), cen[:n][sp].std(), rol[:n][sp].mean(), zcr[:n][sp].mean(),
         float(vad_mask[:n].mean()), float((vuv[:n] == 2).sum() / sp_n)],
    ])
    assert len(v) == len(FEATURE_NAMES)
    return v


def extract_speech(y, vad, fs):
    """Concatenate VAD speech regions (5 ms fades) -> cleaned speech-only signal."""
    H = vad["hop"]
    d = np.diff(np.concatenate([[0], vad["mask"].astype(int), [0]]))
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    fade = max(int(0.005 * fs), 1)
    parts = []
    for a, b in zip(st, en):
        seg = y[max(0, a * H): min(len(y), b * H)].copy()
        if len(seg) > 2 * fade:
            seg[:fade] *= np.linspace(0, 1, fade); seg[-fade:] *= np.linspace(1, 0, fade)
            parts.append(seg)
    return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)


def snr_estimate(y, vad):
    """Speech-vs-noise-floor SNR estimate from VAD frame energies (dB); None if no non-speech frames."""
    p = 10 ** (vad["e_db"] / 10)
    m = vad["mask"]
    if m.sum() == 0 or (~m).sum() < 5:
        return None
    return float(10 * np.log10(max(p[m].mean() - p[~m].mean(), 1e-12) / (p[~m].mean() + 1e-12)))


# ---------------------------------------------------------------- plotting
def _save(fig, path):
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _env(y, n=3000):
    """Min/max envelope so multi-million-sample waveforms plot quickly."""
    b = max(len(y) // n, 1)
    k = len(y) // b
    z = y[:k * b].reshape(k, b)
    return z.min(1), z.max(1), (np.arange(k) * b)


def plot_waveform(raw, proc, fs, vad_mask, hop, path, title2="Processed (denoised + normalised)"):
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for a, y, t in ((ax[0], raw, "Original (mono, resampled)"), (ax[1], proc, title2)):
        lo, hi, x = _env(y)
        a.fill_between(x / fs, lo, hi, color="#2b6cb0", lw=0.5); a.set_title(t, fontsize=10); a.set_ylabel("Amplitude")
    if vad_mask is not None:
        tt = np.arange(len(vad_mask)) * hop / fs
        ax[1].fill_between(tt, -1, 1, where=vad_mask, color="#38a169", alpha=0.15, label="speech (VAD)")
        ax[1].legend(loc="upper right", fontsize=8)
    ax[1].set_xlabel("Time (s)")
    _save(fig, path)


def plot_spectrogram(y, fs, path, title="Spectrogram (STFT, 512-pt Hann)"):
    S = librosa.stft(y, n_fft=512, hop_length=128, window="hann")
    D = librosa.amplitude_to_db(np.abs(S), ref=np.max, top_db=80)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(D, origin="lower", aspect="auto", extent=[0, len(y) / fs, 0, fs / 2], cmap="magma")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)"); ax.set_title(title)
    fig.colorbar(im, ax=ax, label="dB")
    _save(fig, path)


def fft_analysis(raw, proc, fs, labels, hop, path):
    """Single-frame FFT of the strongest voiced frame + Welch long-term spectrum (raw vs processed)."""
    N = 2048
    e = np.array([np.mean(proc[max(0, i * hop - N // 2): i * hop + N // 2] ** 2) if labels[i] == 2 else -1
                  for i in range(len(labels))])
    i0 = int(np.argmax(e)) if e.max() > 0 else len(proc) // (2 * hop)
    s = max(0, min(len(proc) - N, i0 * hop - N // 2))
    seg = proc[s:s + N] * np.hanning(N) if len(proc) >= N else np.pad(proc, (0, N - len(proc))) * np.hanning(N)
    Fm = 20 * np.log10(np.abs(np.fft.rfft(seg)) + 1e-9)
    fr = np.fft.rfftfreq(N, 1 / fs)
    f1, p1 = ss.welch(raw, fs, nperseg=2048); f2, p2 = ss.welch(proc, fs, nperseg=2048)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(fr, Fm, lw=0.8); ax[0].set_title("FFT of strongest voiced frame (128 ms)")
    ax[0].set_xlabel("Frequency (Hz)"); ax[0].set_ylabel("dB")
    ax[1].plot(f1, 10 * np.log10(p1 + 1e-14), label="original", lw=0.9)
    ax[1].plot(f2, 10 * np.log10(p2 + 1e-14), label="processed", lw=0.9)
    ax[1].set_title("Long-term average spectrum (Welch)"); ax[1].set_xlabel("Frequency (Hz)"); ax[1].legend()
    _save(fig, path)
    peak = float(fr[np.argmax(Fm[5:]) + 5])
    return dict(frame_peak_hz=peak)


def plot_vad(proc, fs, labels, hop, path):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    lo, hi, x = _env(proc)
    ax.fill_between(x / fs, lo, hi, color="#4a5568", lw=0.4)
    t = np.arange(len(labels)) * hop / fs
    ax.fill_between(t, -1, 1, where=labels == 2, color="#38a169", alpha=0.25, label="voiced")
    ax.fill_between(t, -1, 1, where=labels == 1, color="#dd6b20", alpha=0.30, label="unvoiced")
    ax.set_title("Voice activity: voiced / unvoiced / silence"); ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=8); ax.set_ylim(-1, 1)
    _save(fig, path)


def plot_mfcc(m, hop, fs, path):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(m, origin="lower", aspect="auto", extent=[0, m.shape[1] * hop / fs, 0, m.shape[0]], cmap="viridis")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("MFCC index"); ax.set_title("MFCC (13 coefficients)")
    fig.colorbar(im, ax=ax); _save(fig, path)


def plot_pitch(pt, path):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(pt["t"], pt["f0"], ".", ms=2); ax.set_ylim(50, 420)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("F0 (Hz)"); ax.set_title("Pitch (autocorrelation), voiced frames only")
    _save(fig, path)


def plot_formants(F, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if len(F):
        ax.scatter(F[:, 1], F[:, 0], s=6, alpha=0.3)
        ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_xlabel("F2 (Hz)"); ax.set_ylabel("F1 (Hz)"); ax.set_title("Formants F1-F2 (LPC)")
    _save(fig, path)


def plot_preview(y, fs, path):
    """Single waveform for the upload preview."""
    with LOCK:
        fig, ax = plt.subplots(figsize=(10, 2.6))
        lo, hi, x = _env(y)
        ax.fill_between(x / fs, lo, hi, color="#2b6cb0", lw=0.5)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude"); ax.set_title("Uploaded recording")
        _save(fig, path)


# ---------------------------------------------------------------- full analysis (contract = results.json)
PLOT_DEFS = [("waveform", "Original vs processed waveform"), ("spectrogram", "Spectrogram (STFT)"),
             ("fft", "FFT / long-term spectrum"), ("vad", "Voiced / unvoiced / silence"),
             ("mfcc", "MFCC"), ("pitch", "Pitch (F0)"), ("formants", "Formants (LPC)")]


def analyze(wav_path, out_dir, target_fs=16000, denoise=True):
    """Run the whole DSP chain on one recording. Writes plots, processed.wav, features.json,
    results.json into out_dir and returns the results dict (same contract as MATLAB's main_analysis)."""
    t0 = time.time()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with LOCK:
        y, fs, info = load_mono(wav_path, target_fs)
        P = preprocess(y, fs, denoise)
        V = energy_vad(P["processed"], fs)
        PT = pitch_track(P["processed"], fs)
        n = min(len(V["mask"]), len(PT["f0"]))
        lab = vuv_labels(V["mask"][:n], PT["voiced"][:n])
        R = lpc_formants(P["emph"], fs, lab, V["hop"])
        mf, dmf = mfcc_feats(P["processed"], fs)
        cen, rol, zcr = spectral_feats(P["processed"], fs)
        vec = feature_vector(mf, dmf, PT["f0"], R["median"], cen, rol, zcr, V["mask"], lab)
        write_wav(out / "processed.wav", P["processed"], fs)
        plot_waveform(P["raw"], P["processed"], fs, V["mask"], V["hop"], out / "waveform.png")
        plot_spectrogram(P["processed"], fs, out / "spectrogram.png")
        fa = fft_analysis(P["raw"], P["processed"], fs, lab, V["hop"], out / "fft.png")
        plot_vad(P["processed"], fs, lab, V["hop"], out / "vad.png")
        plot_mfcc(mf, V["hop"], fs, out / "mfcc.png")
        plot_pitch(PT, out / "pitch.png")
        plot_formants(R["F"], out / "formants.png")
    sp = V["mask"][:n]
    f0v = PT["f0"][~np.isnan(PT["f0"])]
    nz = lambda v: None if (v is None or not np.isfinite(v)) else round(float(v), 2)
    stats = dict(
        original_fs_hz=info["orig_fs"], channels=info["channels"], analysis_fs_hz=fs,
        duration_s=round(info["duration"], 2), speech_ratio=round(V["speech_ratio"], 3),
        speech_seconds=round(float(sp.sum() * V["hop"] / fs), 2),
        voiced_ratio_of_speech=round(float((lab == 2).sum() / max(sp.sum(), 1)), 3),
        unvoiced_ratio_of_speech=round(float((lab == 1).sum() / max(sp.sum(), 1)), 3),
        f0_mean_hz=nz(np.mean(f0v) if len(f0v) else None), f0_std_hz=nz(np.std(f0v) if len(f0v) else None),
        f0_median_hz=nz(np.median(f0v) if len(f0v) else None),
        spectral_centroid_hz=nz(cen[:len(sp)][sp].mean() if sp.sum() else None),
        formants_F1_F2_F3_hz=[nz(v) for v in R["median"]], formant_frames=R["frames"],
        rms_raw_dbfs=round(dbfs(P["raw"]), 2), rms_processed_dbfs=round(dbfs(P["processed"]), 2),
        snr_estimate_db=nz(snr_estimate(P["hp"], V)), fft_frame_peak_hz=nz(fa["frame_peak_hz"]),
        feature_vector_length=int(len(vec)), runtime_s=round(time.time() - t0, 2),
    )
    (out / "features.json").write_text(json.dumps(
        dict(names=FEATURE_NAMES, values=[None if not np.isfinite(v) else float(v) for v in vec])))
    res = dict(status="ok", backend="python", stats=stats,
               plots=[dict(key=k, title=t, file=f"{k}.png") for k, t in PLOT_DEFS], vad_regions=V["regions"][:200])
    (out / "results.json").write_text(json.dumps(res))
    # arrays other services need (kept in memory, returned to caller only)
    res["_internal"] = dict(processed=P["processed"], fs=fs, vad=V, features=vec)
    return res


def features_of(y, fs, target_fs=16000):
    """Feature vector (dict name -> value) for an arbitrary signal; used to compare generated vs reference."""
    if fs != target_fs:
        y = librosa.resample(y, orig_sr=fs, target_sr=target_fs, res_type="soxr_hq"); fs = target_fs
    P = preprocess(y, fs, denoise=False)
    V = energy_vad(P["processed"], fs); PT = pitch_track(P["processed"], fs)
    n = min(len(V["mask"]), len(PT["f0"]))
    lab = vuv_labels(V["mask"][:n], PT["voiced"][:n])
    R = lpc_formants(P["emph"], fs, lab, V["hop"])
    mf, dmf = mfcc_feats(P["processed"], fs); cen, rol, zcr = spectral_feats(P["processed"], fs)
    v = feature_vector(mf, dmf, PT["f0"], R["median"], cen, rol, zcr, V["mask"], lab)
    return dict(zip(FEATURE_NAMES, [None if not np.isfinite(x) else float(x) for x in v]))


# ---------------------------------------------------------------- quality metrics
def snr_db(ref, test):
    n = min(len(ref), len(test)); ref, test = ref[:n].astype(np.float64), test[:n].astype(np.float64)
    return float(10 * np.log10((np.sum(ref ** 2) + 1e-20) / (np.sum((ref - test) ** 2) + 1e-20)))


def lsd_db(a, b, n_fft=512, hop=128):
    """Log-spectral distance (dB), mean over frames (relative -80 dB floor)."""
    n = min(len(a), len(b))
    A = np.abs(librosa.stft(a[:n], n_fft=n_fft, hop_length=hop)) ** 2
    B = np.abs(librosa.stft(b[:n], n_fft=n_fft, hop_length=hop)) ** 2
    fl = 1e-8 * max(A.max(), B.max(), 1e-20)         # relative floor: -80 dB below the loudest bin
    d = 10 * np.log10(A + fl) - 10 * np.log10(B + fl)
    return float(np.mean(np.sqrt(np.mean(d ** 2, axis=0))))


def ltas_distance_db(a, fs_a, b, fs_b, fs=16000):
    """RMS dB difference between long-term average spectra (0-8 kHz) of two signals."""
    def ltas(x, f):
        if f != fs:
            x = librosa.resample(x, orig_sr=f, target_sr=fs, res_type="soxr_hq")
        _, p = ss.welch(x, fs, nperseg=1024)
        return 10 * np.log10(p + 1e-14)
    la, lb = ltas(a, fs_a), ltas(b, fs_b)
    la -= la.mean(); lb -= lb.mean()          # compare spectral shape, not level
    return float(np.sqrt(np.mean((la - lb) ** 2)))


# ---------------------------------------------------------------- post-processing
def postprocess(y, fs, hp_hz=70.0, lp_hz=None, target_rms_dbfs=-20.0, peak_limit=0.97):
    """DC removal -> high-pass -> optional low-pass -> RMS normalisation with peak limiting -> fades."""
    y = np.asarray(y, dtype=np.float64)
    out = y - y.mean()
    if hp_hz:
        b, a = ss.butter(2, hp_hz / (fs / 2), "high"); out = ss.filtfilt(b, a, out)
    if lp_hz and lp_hz < fs / 2:
        b, a = ss.butter(4, lp_hz / (fs / 2), "low"); out = ss.filtfilt(b, a, out)
    f = min(int(0.01 * fs), len(out) // 4)
    if f > 0:
        out[:f] *= np.linspace(0, 1, f); out[-f:] *= np.linspace(1, 0, f)
    gain = 10 ** (target_rms_dbfs / 20) / (np.sqrt(np.mean(out ** 2)) + 1e-12)
    out = out * gain
    pk = np.max(np.abs(out)) + 1e-12
    if pk > peak_limit:
        out *= peak_limit / pk
    # gain-matched SNR: measures distortion introduced by *filtering*, not by the level change
    g = np.dot(y, out) / (np.dot(y, y) + 1e-20)
    cen = lambda x: float(np.mean(librosa.feature.spectral_centroid(y=x.astype(np.float32), sr=fs, n_fft=512)))
    metrics = dict(
        rms_before_dbfs=round(dbfs(y), 2), rms_after_dbfs=round(dbfs(out), 2),
        peak_before=round(float(np.max(np.abs(y))), 4), peak_after=round(float(np.max(np.abs(out))), 4),
        dc_offset_before=round(float(y.mean()), 6), applied_gain_db=round(float(20 * np.log10(gain)), 2),
        snr_vs_input_gain_matched_db=round(snr_db(g * y, out), 2),
        lsd_vs_input_gain_matched_db=round(lsd_db((g * y).astype(np.float32), out.astype(np.float32)), 3),
        centroid_before_hz=round(cen(y), 1), centroid_after_hz=round(cen(out), 1),
        hp_hz=hp_hz, lp_hz=lp_hz, target_rms_dbfs=target_rms_dbfs)
    return out.astype(np.float32), metrics


# ---------------------------------------------------------------- compression experiment
_HDR = "<4sBIIIIff"   # magic, mode, n_samples, d1, d2, n_kept, lo_db, hi_db


def _entropy_bits(sym):
    if len(sym) == 0:
        return 0.0
    _, c = np.unique(sym, return_counts=True)
    p = c / c.sum()
    return float(-(c * np.log2(p)).sum())


def _qlog(mag, bits, lo, hi):
    db = 20 * np.log10(np.maximum(mag, 1e-9)); lv = 2 ** bits - 1
    return np.round((np.clip(db, lo, hi) - lo) / max(hi - lo, 1e-9) * lv).astype(np.uint8)


def _dqlog(q, bits, lo, hi):
    return 10 ** ((lo + q.astype(np.float64) / (2 ** bits - 1) * (hi - lo)) / 20)


def _topk_mask(mag, keep):
    k = max(1, int(round(keep * mag.size)))
    thr = np.partition(mag.ravel(), -k)[-k]
    return mag >= thr


def stft_encode(y, keep, mag_bits=6, phase_bits=5, n_fft=512, hop=128):
    """STFT -> keep the largest `keep` fraction of coefficients -> log-magnitude + phase quantisation
    -> bitmap + codes -> zlib. Returns (blob, stats)."""
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann")
    mag, ph = np.abs(S), np.angle(S)
    m = _topk_mask(mag, keep)
    mv = mag[m]
    lo, hi = float(20 * np.log10(max(mv.min(), 1e-9))), float(20 * np.log10(max(mv.max(), 1e-9)))
    qm = _qlog(mv, mag_bits, lo, hi)
    lv = 2 ** phase_bits
    qp = np.clip(np.floor((ph[m] + np.pi) / (2 * np.pi) * lv), 0, lv - 1).astype(np.uint8)
    payload = np.packbits(m.ravel()).tobytes() + qm.tobytes() + qp.tobytes()
    blob = struct.pack(_HDR, b"STFT", mag_bits * 16 + phase_bits, len(y), S.shape[0], S.shape[1], int(m.sum()), lo, hi) \
        + zlib.compress(payload, 9)
    fixed = m.size + int(m.sum()) * (mag_bits + phase_bits)
    ent = _entropy_bits(m.ravel().astype(np.uint8)) + _entropy_bits(qm) + _entropy_bits(qp)
    return blob, dict(fixed_bits=int(fixed), entropy_bits_est=float(ent), kept=int(m.sum()), total=int(m.size))


def stft_decode(blob, n_fft=512, hop=128):
    h = struct.calcsize(_HDR)
    magic, mode, n, d1, d2, nk, lo, hi = struct.unpack(_HDR, blob[:h])
    assert magic == b"STFT"
    mb, pb = mode // 16, mode % 16
    raw = zlib.decompress(blob[h:])
    nb = (d1 * d2 + 7) // 8
    m = np.unpackbits(np.frombuffer(raw[:nb], np.uint8))[: d1 * d2].astype(bool).reshape(d1, d2)
    qm = np.frombuffer(raw[nb:nb + nk], np.uint8); qp = np.frombuffer(raw[nb + nk:nb + 2 * nk], np.uint8)
    S = np.zeros((d1, d2), np.complex128)
    S[m] = _dqlog(qm, mb, lo, hi) * np.exp(1j * (-np.pi + (qp + 0.5) * 2 * np.pi / 2 ** pb))
    return librosa.istft(S, n_fft=n_fft, hop_length=hop, window="hann", length=n).astype(np.float32)


def dct_encode(y, keep, mag_bits=7, block=512):
    """Block DCT-II (orthonormal) -> keep largest `keep` fraction -> sign + log-magnitude codes -> zlib."""
    n = len(y); nb = int(np.ceil(n / block))
    X = sfft.dct(np.pad(y, (0, nb * block - n)).reshape(nb, block).astype(np.float64), type=2, norm="ortho", axis=1)
    mag = np.abs(X); m = _topk_mask(mag, keep); mv = mag[m]
    lo, hi = float(20 * np.log10(max(mv.min(), 1e-9))), float(20 * np.log10(max(mv.max(), 1e-9)))
    qm = _qlog(mv, mag_bits, lo, hi); sg = (X[m] < 0).astype(np.uint8)
    payload = np.packbits(m.ravel()).tobytes() + qm.tobytes() + np.packbits(sg).tobytes()
    blob = struct.pack(_HDR, b"DCT\x00", mag_bits, n, nb, block, int(m.sum()), lo, hi) + zlib.compress(payload, 9)
    fixed = m.size + int(m.sum()) * (mag_bits + 1)
    ent = _entropy_bits(m.ravel().astype(np.uint8)) + _entropy_bits(qm) + int(m.sum())
    return blob, dict(fixed_bits=int(fixed), entropy_bits_est=float(ent), kept=int(m.sum()), total=int(m.size))


def dct_decode(blob):
    h = struct.calcsize(_HDR)
    magic, mb, n, nb, block, nk, lo, hi = struct.unpack(_HDR, blob[:h])
    assert magic == b"DCT\x00"
    raw = zlib.decompress(blob[h:])
    nbm = (nb * block + 7) // 8
    m = np.unpackbits(np.frombuffer(raw[:nbm], np.uint8))[: nb * block].astype(bool).reshape(nb, block)
    qm = np.frombuffer(raw[nbm:nbm + nk], np.uint8)
    sg = np.unpackbits(np.frombuffer(raw[nbm + nk:], np.uint8))[:nk].astype(bool)
    X = np.zeros((nb, block)); v = _dqlog(qm, mb, lo, hi); v[sg] *= -1; X[m] = v
    return sfft.idct(X, type=2, norm="ortho", axis=1).ravel()[:n].astype(np.float32)


def compression_sweep(y, fs, keeps=(0.05, 0.1, 0.2, 0.3, 0.5), out_dir=None):
    """Rate-distortion experiment for both codecs. All numbers measured by actually encoding,
    decoding (from the compressed bytes) and comparing with the input."""
    orig_bytes = len(y) * 2                                    # 16-bit PCM reference
    rows = []
    for codec, enc, dec in (("STFT-threshold", lambda k: stft_encode(y, k), stft_decode),
                            ("DCT-threshold", lambda k: dct_encode(y, k), dct_decode)):
        for k in keeps:
            t0 = time.time(); blob, st = enc(k); t_enc = time.time() - t0
            rec = dec(blob)
            rows.append(dict(
                codec=codec, keep_fraction=k, compressed_bytes=len(blob),
                ratio_zlib=round(orig_bytes / len(blob), 2),
                ratio_fixed_length=round(orig_bytes * 8 / st["fixed_bits"], 2),
                ratio_entropy_est=round(orig_bytes * 8 / max(st["entropy_bits_est"], 1), 2),
                bitrate_kbps_zlib=round(len(blob) * 8 / (len(y) / fs) / 1000, 1),
                snr_db=round(snr_db(y, rec), 2), lsd_db=round(lsd_db(y, rec), 3),
                encode_time_s=round(t_enc, 3), _blob=blob, _rec=rec))
    if out_dir:
        with LOCK:
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            for c, mk in (("STFT-threshold", "o-"), ("DCT-threshold", "s--")):
                r = [x for x in rows if x["codec"] == c]
                ax[0].plot([x["ratio_zlib"] for x in r], [x["snr_db"] for x in r], mk, label=c)
                ax[1].plot([x["ratio_zlib"] for x in r], [x["lsd_db"] for x in r], mk, label=c)
            ax[0].set_xlabel("Compression ratio (vs 16-bit PCM, measured zlib bytes)"); ax[0].set_ylabel("SNR (dB)")
            ax[1].set_xlabel("Compression ratio"); ax[1].set_ylabel("Log-spectral distance (dB)")
            ax[0].legend(); ax[0].set_title("Rate-distortion: SNR"); ax[1].set_title("Rate-distortion: LSD")
            _save(fig, Path(out_dir) / "compression_rd.png")
    return rows


def public_rows(rows):
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]


def plot_spectrum_compare(a, b, fs, path, la="generated", lb="final (post-processed)"):
    with LOCK:
        f1, p1 = ss.welch(a, fs, nperseg=2048); f2, p2 = ss.welch(b, fs, nperseg=2048)
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(f1, 10 * np.log10(p1 + 1e-14), label=la, lw=0.9); ax.plot(f2, 10 * np.log10(p2 + 1e-14), label=lb, lw=0.9)
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("dB"); ax.set_title("Spectrum before / after post-processing"); ax.legend()
        _save(fig, path)
