"""One synthesis run for ONE profile:

text -> Gemini/local normalisation -> XTTS (conditioned on this profile only) -> generated.wav
     -> DSP post-processing -> final.wav -> spectral analysis -> compression experiment
     -> speaker verification vs this profile -> metrics.json

Every number in metrics.json is measured during the run.
"""
import json, re, time
from datetime import datetime
from pathlib import Path

import numpy as np

import config
from services import dsp_service as D, gemini_service as G, profile_service as PS
from services import speaker_service as S, tts_service as T

RUN_RE = re.compile(r"^\d{8}-\d{6}(-\d)?$")
COMPARE_KEYS = ["f0_mean_hz", "f0_median_hz", "F1_hz", "F2_hz", "F3_hz", "centroid_mean_hz", "rolloff85_mean_hz", "zcr_mean"]


class SynthesisError(Exception):
    pass


def valid_run(run_id) -> bool:
    return isinstance(run_id, str) and bool(RUN_RE.match(run_id))


def _new_run_dir(slug):
    base = datetime.now().strftime("%Y%m%d-%H%M%S")
    for i in range(10):
        rid = base if i == 0 else f"{base}-{i}"
        d = PS.profile_dir(slug) / "outputs" / rid
        if not d.exists():
            d.mkdir(parents=True)
            return rid, d
    raise SynthesisError("Could not allocate an output folder.")


def _reference_features(slug):
    p = PS.profile_dir(slug) / "analysis" / "python" / "features.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return dict(zip(d["names"], d["values"]))


def run(slug, text, language="en", use_gemini=True, hp_hz=70.0, lp_hz=None, target_rms_dbfs=-20.0):
    timings, t_all = {}, time.time()
    if not PS.exists(slug):
        raise SynthesisError(f"Profile '{slug}' does not exist.")
    if not PS.is_enrolled(slug):
        raise SynthesisError("This profile is not enrolled yet. Complete voice enrollment first.")
    text = (text or "").strip()
    if not text:
        raise SynthesisError("Enter some text first.")
    if len(text) > config.MAX_TEXT_CHARS:
        raise SynthesisError(f"Text is too long ({len(text)} characters; maximum {config.MAX_TEXT_CHARS}).")

    t = time.time(); prep = G.prepare_text(text, language, use_gemini); timings["text_preparation_s"] = round(time.time() - t, 2)
    if not prep["sentences"]:
        raise SynthesisError("No speakable text after normalisation.")

    wav, sr, tts_info = T.synthesize(slug, prep["sentences"], language)     # raises TTSUnavailable / TTSError
    timings["tts_s"] = tts_info["inference_time_s"]

    rid, rdir = _new_run_dir(slug)
    D.write_wav(rdir / "generated.wav", wav, sr)

    # ---- DSP post-processing + spectral analysis
    t = time.time()
    final, pm = D.postprocess(wav, sr, hp_hz=hp_hz, lp_hz=lp_hz, target_rms_dbfs=target_rms_dbfs)
    D.write_wav(rdir / "final.wav", final, sr)
    with D.LOCK:
        D.plot_spectrum_compare(wav, final, sr, rdir / "spectrum_compare.png")
        D.plot_spectrogram(final, sr, rdir / "spectrogram.png", "Spectrogram of final speech")
    pm["snr_estimate_db"] = _snr_est(final, sr)
    timings["postprocess_s"] = round(time.time() - t, 2)

    # ---- compression experiment on the final signal
    t = time.time()
    rows = D.compression_sweep(final, sr, out_dir=rdir)
    pick = next(r for r in rows if r["codec"] == "STFT-threshold" and r["keep_fraction"] == 0.3)
    (rdir / "compressed_stft_keep30.bin").write_bytes(pick["_blob"])
    D.write_wav(rdir / "final_compressed.wav", pick["_rec"], sr)
    timings["compression_s"] = round(time.time() - t, 2)

    # ---- speaker verification + feature comparison (this profile only)
    t = time.time()
    ver = S.verify(slug, final, sr)
    ref_f, gen_f = _reference_features(slug), D.features_of(final, sr)
    cmp = []
    if ref_f:
        for k in COMPARE_KEYS:
            a, b = ref_f.get(k), gen_f.get(k)
            cmp.append(dict(feature=k, reference=None if a is None else round(a, 2), generated=None if b is None else round(b, 2),
                            diff_pct=None if (a in (None, 0) or b is None) else round(100 * (b - a) / a, 1)))
    ref_clean = PS.profile_dir(slug) / "reference_clean.wav"
    yr, fr, _ = D.load_mono(ref_clean)
    ltas = round(D.ltas_distance_db(final, sr, yr, fr), 2)
    timings["verification_s"] = round(time.time() - t, 2)

    # ---- optional MATLAB post-processing (additional DSP report; the Python result above is authoritative)
    matlab = dict(used=False, note="skipped")
    if config.DSP_BACKEND != "python":
        from services import matlab_service as M
        if M.available()["any"]:
            try:
                r = M.run_postprocess(rdir / "generated.wav", rdir / "matlab", hp_hz, lp_hz or 0, target_rms_dbfs)
                matlab = dict(used=True, results=r)
            except M.MatlabError as e:
                matlab = dict(used=False, note=f"MATLAB post-processing failed: {str(e)[:300]}")
        else:
            matlab = dict(used=False, note="MATLAB not found; Python DSP used")

    files = {n: (rdir / n).stat().st_size for n in ("generated.wav", "final.wav", "final_compressed.wav", "compressed_stft_keep30.bin")}
    metrics = dict(
        run_id=rid, profile=slug, created=PS.now_iso(), language=language,
        text=dict(original=text, normalized=prep["normalized"], sentences=prep["sentences"],
                  used_gemini=prep["used_gemini"], gemini_model=prep["model"], note=prep["note"]),
        tts=tts_info, postprocess=pm, compression=D.public_rows(rows),
        compression_default=dict(codec=pick["codec"], keep_fraction=pick["keep_fraction"], ratio_zlib=pick["ratio_zlib"],
                                 snr_db=pick["snr_db"], lsd_db=pick["lsd_db"], bytes=pick["compressed_bytes"]),
        verification=ver, features_compare=cmp, ltas_distance_to_reference_db=ltas,
        file_bytes=files, timings=timings, total_time_s=round(time.time() - t_all, 2), matlab=matlab,
        reference_sha256=PS.get(slug)["enrollment"].get("reference_sha256"),
        disclaimer="Zero-shot cloning from a short reference with a pretrained model; similarity is limited and "
                   "not a perfect copy of the speaker. Use only voices you have permission to synthesise.")
    (rdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _snr_est(y, fs):
    P = D.preprocess(y, fs, denoise=False)
    V = D.energy_vad(P["processed"], fs)
    v = D.snr_estimate(P["hp"], V)
    return None if v is None else round(v, 2)


def load_run(slug, run_id):
    if not valid_run(run_id):
        raise SynthesisError("Invalid run id.")
    p = PS.profile_dir(slug) / "outputs" / run_id / "metrics.json"
    if not p.exists():
        raise SynthesisError("Run not found.")
    return json.loads(p.read_text())
