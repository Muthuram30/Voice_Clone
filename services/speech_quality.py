"""Conservative speech assembly and measurable artifact checks, not a naturalness score."""
import re

import numpy as np


def pack_passages(sentences, max_chars):
    """Keep neighboring sentences together and never invent punctuation at size cuts."""
    text = ' '.join(str(s).strip() for s in sentences if str(s).strip())
    max_chars = max(1, int(max_chars))
    passages = []
    while len(text) > max_chars:
        candidates = [m.start() + 1 for m in re.finditer(r'[.!?。！？;:,](?:\s|$)', text[:max_chars + 1])]
        cut = max((i for i in candidates if i >= .4 * max_chars), default=0)
        if not cut:
            cut = text.rfind(' ', 0, max_chars + 1)
        if cut <= 0:
            cut = max_chars
        passages.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        passages.append(text)
    return passages


def _edge_quiet(samples, fs, reverse=False):
    values = samples[::-1] if reverse else samples
    frame = max(1, round(.01 * fs))
    count = len(values) // frame
    if not count:
        return 0
    rms = np.sqrt(np.mean(values[:count * frame].reshape(count, frame) ** 2, axis=1))
    active = np.flatnonzero(rms > max(float(np.max(np.abs(samples))) * .003, 1e-5))
    return int(active[0] * frame) if len(active) else len(samples)


def join_passages(parts, texts, fs, gap_s=.18):
    """Fade tiny boundary edges, preserving model pauses and avoiding doubled silence."""
    if len(parts) != len(texts) or not parts:
        raise ValueError('Audio passages and their text must match.')
    cleaned = []
    for part in parts:
        x = np.asarray(part, dtype=np.float32).reshape(-1).copy()
        if not len(x) or not np.isfinite(x).all() or float(np.abs(x).max()) < 1e-5:
            raise ValueError('The model returned an empty, silent or invalid passage.')
        fade = min(round(.003 * fs), len(x) // 4)
        if fade:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            x[:fade] *= ramp
            x[-fade:] *= ramp[::-1]
        cleaned.append(x)
    assembled = [cleaned[0]]
    added = []
    for i, current in enumerate(cleaned[1:], 1):
        previous = cleaned[i - 1]
        end = texts[i - 1].rstrip()
        target = gap_s if re.search(r'[.!?。！？][\"\'’”)]*$', end) else min(gap_s, .08) if end.endswith((',', ';', ':')) else min(gap_s, .04)
        existing = _edge_quiet(previous, fs, reverse=True) + _edge_quiet(current, fs)
        extra = max(0, round(target * fs) - existing)
        if extra:
            assembled.append(np.zeros(extra, dtype=np.float32))
        added.append(round(extra / fs, 4))
        assembled.append(current)
    return np.concatenate(assembled), added


def inspect_audio(y, fs):
    """Report clipping, silence and level; passing cannot prove human-sounding speech."""
    samples = np.asarray(y, dtype=np.float32)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all() or fs <= 0:
        raise ValueError('Quality checks require finite, nonempty mono audio and a valid sample rate.')
    frame = max(1, round(.02 * fs))
    padded = np.pad(samples, (0, (-len(samples)) % frame))
    rms = np.sqrt(np.mean(padded.reshape(-1, frame) ** 2, axis=1))
    threshold = max(float(rms.max()) * .01, 1e-4)
    active = rms > threshold
    indices = np.flatnonzero(active)
    interior = active[indices[0]:indices[-1] + 1] if len(indices) else np.zeros(0, dtype=bool)
    transitions = np.diff(np.pad((~interior).astype(int), (1, 1)))
    runs = np.flatnonzero(transitions == -1) - np.flatnonzero(transitions == 1)
    longest = float(runs.max() * frame / fs) if len(runs) else 0.
    clipping = float(np.mean(np.abs(samples) >= .999))
    warnings = []
    if not len(indices):
        warnings.append('No audible signal detected.')
    if clipping > .001:
        warnings.append('Clipping detected; listen for harsh distortion.')
    if longest > 1.5:
        warnings.append('Long internal pause detected; review the spoken passage.')
    return dict(duration_s=round(len(samples) / fs, 3), peak=round(float(np.abs(samples).max()), 5),
                rms_dbfs=round(float(20 * np.log10(np.sqrt(np.mean(samples ** 2)) + 1e-12)), 2),
                clipped_fraction=round(clipping, 6), longest_internal_pause_s=round(longest, 3),
                warnings=warnings, artifact_checks_passed=not warnings,
                naturalness='Requires listening; signal metrics do not measure human-likeness.')
