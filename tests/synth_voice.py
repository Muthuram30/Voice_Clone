"""Synthetic 'speakers' for tests: glottal pulse train -> formant resonators, syllable gating,
fricative noise bursts and pauses. NOT speech; just a signal with known F0/formants so DSP code
can be checked against ground truth."""
import numpy as np, scipy.signal as ss

def resonator(x, fs, f, bw):
    r = np.exp(-np.pi * bw / fs); th = 2 * np.pi * f / fs
    return ss.lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], x)

def make_voice(seconds=70.0, fs=16000, f0=120.0, formants=((700, 90), (1220, 110), (2600, 160)),
               seed=0, noise=0.002, vibrato=0.03):
    rng = np.random.default_rng(seed)
    n = int(seconds * fs); out = np.zeros(n); t = 0; pos = 0
    while pos < n:                                       # syllable-by-syllable
        dur = rng.uniform(0.15, 0.35); gap = rng.uniform(0.02, 0.5) if rng.random() < 0.6 else 0.02
        m = int(dur * fs)
        if pos + m >= n: break
        ff = f0 * (1 + vibrato * np.sin(2 * np.pi * rng.uniform(1, 3) * np.arange(m) / fs))
        phase = np.cumsum(ff / fs); src = np.zeros(m)
        src[np.where(np.diff(np.floor(phase)) > 0)[0]] = 1.0
        src = ss.lfilter([1], [1, -0.95], src)           # glottal-ish tilt
        y = src
        for f, bw in formants:
            y = resonator(y, fs, f * rng.uniform(0.9, 1.1), bw)
        y = y / (np.max(np.abs(y)) + 1e-9) * 0.5 * np.hanning(m) ** 0.5
        out[pos:pos + m] += y; pos += m
        if rng.random() < 0.3:                           # unvoiced fricative burst
            k = int(rng.uniform(0.05, 0.12) * fs)
            if pos + k < n:
                nz = ss.lfilter([1, -1], [1], rng.standard_normal(k)); out[pos:pos + k] += 0.12 * nz / np.max(np.abs(nz)) * np.hanning(k); pos += k
        pos += int(gap * fs)
    out += noise * rng.standard_normal(n)
    return (out / np.max(np.abs(out)) * 0.8).astype(np.float32), fs
