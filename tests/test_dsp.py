import numpy as np
from synth_voice import make_voice
from services import dsp_service as D
import soundfile as sf


def test_analysis_recovers_ground_truth(tmp_path):
    y, fs = make_voice(40, f0=120, seed=3)
    sf.write(tmp_path / "v.wav", y, fs)
    r = D.analyze(tmp_path / "v.wav", tmp_path / "out"); s = r["stats"]
    assert abs(s["f0_median_hz"] - 120) / 120 < 0.05                    # F0 within 5 %
    F1, F2, F3 = s["formants_F1_F2_F3_hz"]
    assert abs(F1 - 700) / 700 < 0.06 and abs(F2 - 1220) / 1220 < 0.06 and abs(F3 - 2600) / 2600 < 0.06
    assert 0.4 < s["speech_ratio"] < 0.95 and s["feature_vector_length"] == len(D.FEATURE_NAMES) == 53
    for k in ("waveform", "spectrogram", "fft", "vad", "mfcc", "pitch", "formants"):
        assert (tmp_path / "out" / f"{k}.png").stat().st_size > 1000
    assert (tmp_path / "out" / "results.json").exists() and (tmp_path / "out" / "processed.wav").exists()


def test_fricative_noise_is_not_labelled_voiced():
    rng = np.random.default_rng(0); fs = 16000
    noise = np.diff(rng.standard_normal(fs * 3)).astype(np.float32); noise /= np.abs(noise).max()
    pt = D.pitch_track(noise * 0.5, fs)
    assert pt["voiced"].mean() < 0.05


def test_postprocess_hits_target_limits_peak_and_removes_dc():
    y, fs = make_voice(6, seed=4)
    x = y * 3 + 0.1                                                    # loud + DC offset
    out, m = D.postprocess(x, fs, target_rms_dbfs=-20)
    assert m["peak_after"] <= 0.9701 and abs(np.mean(out)) < 1e-3 and abs(m["dc_offset_before"] - np.mean(x)) < 1e-3
    z = y - y.mean()
    _, mf = D.postprocess(z, fs, hp_hz=0, target_rms_dbfs=-30)         # no filter: only gain + 10 ms fades
    _, mh = D.postprocess(z, fs, hp_hz=70, target_rms_dbfs=-30)
    assert abs(mf["rms_after_dbfs"] + 30) < 0.1 and mf["snr_vs_input_gain_matched_db"] > 25    # metric ignores the level change
    assert mh["snr_vs_input_gain_matched_db"] < mf["snr_vs_input_gain_matched_db"]               # filtering adds measurable distortion


def test_compression_roundtrip_is_decoded_from_bytes_and_monotonic():
    y, fs = make_voice(6, seed=5)
    rows = D.compression_sweep(y, fs, keeps=(0.1, 0.5))
    for codec in ("STFT-threshold", "DCT-threshold"):
        lo, hi = [r for r in rows if r["codec"] == codec]
        assert hi["compressed_bytes"] > lo["compressed_bytes"] and hi["snr_db"] > lo["snr_db"] - 0.5
        assert len(hi["_rec"]) == len(y)
    b, _ = D.dct_encode(y, 0.3)
    assert np.allclose(D.dct_decode(b), D.dct_decode(b))                # deterministic
    assert D.snr_db(y, D.dct_decode(b)) > 20
