"""CUDA signal kernels. File codecs, plots and JSON remain host operations.

No CPU fallback: a missing CUDA runtime is an actionable error. NumPy arrays
are used only at the boundaries of the existing analysis/report API.
"""
import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

import config


def tensor(x):
    config.initialize_runtime()
    if not config.TORCH_DEVICE.startswith("cuda") or not torch.cuda.is_available():
        raise config.CudaNotAvailableError("Signal processing requires CUDA. Run scripts/check_gpu.py in .venv.")
    if isinstance(x, torch.Tensor):
        return x.to(config.TORCH_DEVICE)
    return torch.as_tensor(np.array(x, copy=True), device=config.TORCH_DEVICE)


def host(x):
    return x.detach().cpu().numpy()


def resample(y, orig_sr, target_sr, **kwargs):
    return host(AF.resample(tensor(y).float(), orig_sr, target_sr))


def stft_tensor(y, n_fft=512, hop_length=128):
    x = tensor(y).float()
    return torch.stft(x, n_fft, hop_length, window=torch.hann_window(n_fft, device=x.device),
                      return_complex=True, pad_mode="constant")


def stft(y, n_fft=512, hop_length=128, **kwargs):
    return host(stft_tensor(y, n_fft, hop_length))


def istft(y, n_fft=None, hop_length=128, length=None, **kwargs):
    s = tensor(y)
    n_fft = n_fft or (s.shape[-2] - 1) * 2
    return host(torch.istft(s, n_fft, hop_length,
                           window=torch.hann_window(n_fft, device=s.device), length=length))


def frame_tensor(y, length, hop):
    return F.pad(tensor(y).float(), (length // 2, length // 2)).unfold(0, length, hop)


def filter_audio(y, fs, hp=0, lp=None):
    x = tensor(y).float()
    # Biquads operate on CUDA. Reversing twice gives zero phase without FFT wrap-around.
    for cutoff, highpass in ((hp, True), (lp, False)):
        if cutoff:
            if not 0 < cutoff < fs / 2:
                raise ValueError("Filter cutoff must lie between zero and Nyquist.")
            omega = 2 * math.pi * cutoff / fs
            cosine, alpha = math.cos(omega), math.sin(omega) / math.sqrt(2)
            a = x.new_tensor([1 + alpha, -2 * cosine, 1 - alpha])
            b = x.new_tensor([(1 + cosine) / 2, -(1 + cosine), (1 + cosine) / 2] if highpass else
                             [(1 - cosine) / 2, 1 - cosine, (1 - cosine) / 2])
            # torchaudio's convenience biquads clamp to [-1, 1] by default.
            # Filtering must remain linear; the final peak limiter sets headroom.
            x = AF.lfilter(x, a, b, clamp=False)
            x = AF.lfilter(x.flip(-1), a, b, clamp=False).flip(-1)
    return x


def spectral_subtract(y, fs):
    s = stft_tensor(y)
    mag = s.abs()
    idx = mag.square().sum(0).topk(max(1, int(.1 * mag.shape[1])), largest=False).indices
    noise = mag[:, idx].mean(1, keepdim=True)
    clean = torch.maximum(mag - 1.5 * noise, .05 * mag) * torch.exp(1j * s.angle())
    return istft(clean, length=len(y))


def preprocess(y, fs, denoise=True):
    x = tensor(y).float()
    x = x - x.mean()
    hp = filter_audio(x, fs, hp=80)
    clean = tensor(spectral_subtract(hp, fs)) if denoise else hp
    clean = clean / (clean.abs().max() + 1e-12) * .95
    emph = clean - .97 * F.pad(clean[:-1], (1, 0))
    return dict(raw=host(x), hp=host(hp), processed=host(clean), emph=host(emph), fs=fs)


def _morph(x, size, dilate):
    left, right = (size - 1) // 2, size // 2
    padded = F.pad(x.float()[None, None], (left, right), value=0 if dilate else 1)
    return (F.max_pool1d(padded if dilate else -padded, size, stride=1)[0, 0] * (1 if dilate else -1)) > .5


def energy_vad(y, fs):
    length, hop = 2 * round(fs * .025 / 2), round(fs * .01)
    e = 10 * torch.log10(frame_tensor(y, length, hop).square().mean(1) + 1e-12)
    floor, top = torch.quantile(e, tensor([.1, .95]).float())
    thr = torch.maximum(floor + torch.maximum(.3 * (top-floor), tensor(6.)), tensor(-55.))
    m = e > thr
    m = _morph(_morph(m, 15, True), 15, False)
    m = _morph(_morph(m, 10, False), 10, True)
    m = _morph(m, 5, True)
    d = torch.diff(F.pad(m.int(), (1, 1)))
    starts, ends = (d == 1).nonzero().flatten(), (d == -1).nonzero().flatten()
    regions = host(torch.stack((starts, ends), 1).float() * hop / fs).tolist()
    return dict(mask=host(m), e_db=host(e), thr_db=float(thr), regions=regions,
                speech_ratio=float(m.float().mean()), hop=hop, frame_len=length)


def pitch_track(y, fs, fmin=60., fmax=400., voicing_thr=.5):
    length, hop = 2 * round(fs * .04 / 2), round(fs * .01)
    w = torch.hann_window(length, periodic=False, device=tensor(0).device)
    fr = frame_tensor(y, length, hop) * w
    n = 2 ** math.ceil(math.log2(2 * length))
    r = torch.fft.irfft(torch.fft.rfft(fr, n).abs().square(), n)[:, :length]
    rw = torch.fft.irfft(torch.fft.rfft(w, n).abs().square(), n)[:length]
    rn = r / (rw + 1e-12)
    rn = rn / (rn[:, :1] + 1e-12)
    lo, hi = math.floor(fs/fmax), min(length-1, math.ceil(fs/fmin))
    seg = rn[:, lo:hi+1]
    peaks = torch.zeros_like(seg, dtype=torch.bool)
    peaks[:, 1:-1] = (seg[:, 1:-1] > seg[:, :-2]) & (seg[:, 1:-1] >= seg[:, 2:])
    mx = torch.where(peaks, seg, -torch.inf).max(1).values
    k = (peaks & (seg >= .9 * mx[:, None])).int().argmax(1).clamp(1, seg.shape[1]-2)
    rows = torch.arange(len(k), device=k.device)
    a, b, c = seg[rows, k-1], seg[rows, k], seg[rows, k+1]
    den = a-2*b+c
    delta = torch.where(den.abs() > 1e-12, .5*(a-c)/den, 0).clamp(-1, 1)
    f0 = fs/(lo+k+delta)
    voiced = (b > voicing_thr) & mx.isfinite() & (fr.square().mean(1).sqrt() > 1e-4) & (f0 >= fmin) & (f0 <= fmax)
    smooth = F.pad(torch.where(voiced, f0, 0), (2, 2)).unfold(0, 5, 1).median(1).values
    voiced &= smooth > 0
    return dict(f0=host(torch.where(voiced, smooth, torch.nan)), strength=host(b),
                voiced=host(voiced), t=host(torch.arange(len(k), device=k.device)*hop/fs))


def lpc_formants(emph, fs, labels, hop):
    length, step, order = int(.03*fs), int(.02*fs), 2+round(fs/1000)
    x = tensor(emph).float()
    if len(x) <= length:
        return dict(F=np.zeros((0, 3)), median=np.full(3, np.nan), frames=0)
    fr = x.unfold(0, length, step)
    ix = torch.round((torch.arange(len(fr), device=x.device)*step+length//2)/hop).long().clamp(max=len(labels)-1)
    fr = fr[tensor(labels)[ix] == 2] * torch.hamming_window(length, periodic=False, device=x.device)
    fr = fr[fr.square().mean(1).sqrt() >= 1e-3]
    if not len(fr):
        return dict(F=np.zeros((0, 3)), median=np.full(3, np.nan), frames=0)
    r = torch.fft.irfft(torch.fft.rfft(fr.double(), 2*length).abs().square(), 2*length)[:, :order+1]
    r[:, 0] *= 1.000001
    ix = torch.arange(order, device=x.device)
    toeplitz = r[:, (ix[:, None]-ix).abs()]
    a = torch.linalg.solve(toeplitz, r[:, 1:, None]).squeeze(-1)
    companion = torch.zeros((len(a), order, order), device=x.device, dtype=torch.float64)
    companion[:, 0, :] = a
    companion[:, 1:, :-1] = torch.eye(order-1, device=x.device)
    roots = torch.linalg.eigvals(companion)
    hz, bw = roots.angle()*fs/(2*math.pi), -roots.abs().log()*fs/math.pi
    ok = (roots.imag > .01) & (hz > 90) & (bw < 400)
    vals = torch.where(ok, hz, torch.inf).sort(1).values[:, :3]
    vals = vals[vals.isfinite().all(1)]
    med = torch.quantile(vals, .5, dim=0) if len(vals) else tensor([float('nan')]*3)
    return dict(F=host(vals), median=host(med), frames=len(vals))


def spectral_feats(y, fs):
    hop = round(fs*.01)
    mag = stft_tensor(y, hop_length=hop).abs()
    freq = torch.linspace(0, fs/2, mag.shape[0], device=mag.device)[:, None]
    cen = (mag*freq).sum(0)/(mag.sum(0)+1e-12)
    bins = (mag.cumsum(0) >= .85*mag.sum(0)).int().argmax(0)
    rol = freq[:, 0][bins]
    fr = frame_tensor(y, 2*round(fs*.025/2), hop)
    zcr = ((fr[:, 1:] >= 0) != (fr[:, :-1] >= 0)).float().sum(1)/fr.shape[1]
    return host(cen), host(rol), host(zcr)


def mfcc(y, sr=16000, n_mfcc=13, n_fft=512, hop_length=160, n_mels=40, **kwargs):
    import torchaudio.transforms as AT
    x = tensor(y).float()
    transform = AT.MFCC(sample_rate=sr, n_mfcc=n_mfcc,
        melkwargs=dict(n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
                       mel_scale="slaney", norm="slaney", pad_mode="constant")).to(x.device)
    return host(transform(x))


def delta(m, width=9):
    return host(AF.compute_deltas(tensor(m).float(), win_length=width))


def mfcc_feats(y, fs):
    m = mfcc(y, sr=fs, hop_length=round(fs*.01))
    return m, delta(m)


def welch(y, fs=1., nperseg=1024, **kwargs):
    x = tensor(y).float()
    n = min(nperseg, len(x)); hop = max(1, n//2)
    w = torch.hann_window(n, device=x.device)
    fr = x.unfold(0, n, hop); fr = fr-fr.mean(1, keepdim=True)
    power = torch.fft.rfft(fr*w).abs().square().mean(0)/(fs*w.square().sum())
    power[1:-1 if n%2 == 0 else None] *= 2
    return host(torch.fft.rfftfreq(n, 1/fs, device=x.device)), host(power)


@lru_cache(maxsize=4)
def _dct_matrix(n):
    x = tensor(torch.arange(n).float())
    mat = torch.cos(math.pi/n * x[:, None]*(x[None, :]+.5))*math.sqrt(2/n)
    mat[0] /= math.sqrt(2)
    return mat


def dct(x, **kwargs):
    a = tensor(x).float()
    return host(a @ _dct_matrix(a.shape[-1]).T)


def idct(x, **kwargs):
    a = tensor(x).float()
    return host(a @ _dct_matrix(a.shape[-1]))


def postprocess(y, fs, hp_hz=70., lp_hz=None, target_rms_dbfs=-20., peak_limit=.97):
    from services import dsp_service as D
    x = tensor(y).float()
    out = filter_audio(x-x.mean(), fs, hp_hz, lp_hz)
    fade = min(int(.01*fs), len(out)//4)
    if fade:
        ramp = torch.linspace(0, 1, fade, device=x.device)
        out[:fade] *= ramp; out[-fade:] *= ramp.flip(0)
    gain = 10**(target_rms_dbfs/20)/(out.square().mean().sqrt()+1e-12)
    out *= gain
    out *= torch.clamp(peak_limit/(out.abs().max()+1e-12), max=1.)
    g = torch.dot(x, out)/(torch.dot(x, x)+1e-20)
    a, b, matched = host(x), host(out), host(g*x)
    m = dict(rms_before_dbfs=round(D.dbfs(a), 2), rms_after_dbfs=round(D.dbfs(b), 2),
             peak_before=round(float(x.abs().max()), 4), peak_after=round(float(out.abs().max()), 4),
             dc_offset_before=round(float(x.mean()), 6), applied_gain_db=round(float(20*gain.log10()), 2),
             snr_vs_input_gain_matched_db=round(D.snr_db(matched, b), 2),
             lsd_vs_input_gain_matched_db=round(D.lsd_db(matched, b), 3),
             centroid_before_hz=round(float(spectral_feats(a, fs)[0].mean()), 1),
             centroid_after_hz=round(float(spectral_feats(b, fs)[0].mean()), 1),
             hp_hz=hp_hz, lp_hz=lp_hz, target_rms_dbfs=target_rms_dbfs, device=str(x.device))
    return b, m
