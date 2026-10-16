"""Speaker representation, enrollment, and the classical-ML experiment (PCA + SVM).

Encoders
    ecapa-tdnn   SpeechBrain ECAPA-TDNN, PRETRAINED on VoxCeleb (we do not train it). 192-d embeddings.
    mfcc-stats   Classical baseline: mean/std of MFCCs + delta std (57-d). Not a neural model. Always available;
                 used automatically when SpeechBrain/PyTorch is not installed. Metadata records which was used.

Enrollment = DSP analysis + speech extraction + per-chunk embeddings + centroid. Each profile keeps its own
files; the only cross-profile object is the PCA/SVM classifier, which is *trained from* each profile's chunk
embeddings (labelled by profile) and never used for synthesis.
"""
import importlib.util, json, time, threading
from pathlib import Path

import joblib
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import config
from services import audio_service as A, dsp_service as D, profile_service as PS


class EncoderUnavailable(Exception):
    pass


class EnrollmentError(Exception):
    pass


# ------------------------------------------------------------------ encoders
class MfccStatsEncoder:
    name, l2, pretrained = "mfcc-stats", False, False
    source = "librosa MFCC statistics (classical baseline, no neural network)"

    def embed(self, chunks):
        out = []
        for c in chunks:
            m = librosa.feature.mfcc(y=c, sr=16000, n_mfcc=20, n_fft=512, hop_length=160, n_mels=40)[1:]
            d = librosa.feature.delta(m, width=9)
            out.append(np.concatenate([m.mean(1), m.std(1), d.std(1)]))
        return np.asarray(out, dtype=np.float32)


class EcapaEncoder:
    name, l2, pretrained = "ecapa-tdnn", True, True
    source = "speechbrain/spkrec-ecapa-voxceleb (pretrained on VoxCeleb; not trained in this project)"

    def __init__(self):
        for mod in ("torch", "speechbrain"):
            if importlib.util.find_spec(mod) is None:
                raise EncoderUnavailable(f"'{mod}' is not installed (pip install -r requirements-tts.txt).")
        import torch
        config.initialize_runtime()
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:                                         # speechbrain < 1.0
            from speechbrain.pretrained import EncoderClassifier
        kw = {}
        try:                                                        # avoid Windows symlink permission errors
            from speechbrain.utils.fetching import LocalStrategy
            kw["local_strategy"] = LocalStrategy.COPY
        except Exception:                                           # noqa: BLE001
            pass
        # SpeechBrain expects an indexed CUDA device string; plain "cuda"
        # triggers its parser fallback even though inference still lands on GPU.
        self.device = config.TORCH_DEVICE
        try:
            self.clf = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(config.MODELS_DIR / "ecapa"),
                run_opts={"device": self.device}, **kw)
        except Exception as e:                                      # noqa: BLE001
            raise EncoderUnavailable(f"Could not load the ECAPA model (first run needs internet to download it): {e}")
        self.clf.eval()
        self.torch = torch

    @property
    def model(self):
        return self.clf.mods.embedding_model

    def embed(self, chunks):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(chunks), 16):
                x = self.torch.from_numpy(np.ascontiguousarray(chunks[i:i + 16])).float().to(self.device)
                out.append(self.clf.encode_batch(x).squeeze(1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


_ENC, _ENC_LOCK = {}, threading.Lock()


def get_encoder(name=None):
    """Return (encoder, note). name: auto | ecapa | mfcc (or the encoder .name strings)."""
    name = {"ecapa-tdnn": "ecapa", "mfcc-stats": "mfcc"}.get(name or config.ENCODER_BACKEND, name or config.ENCODER_BACKEND)
    with _ENC_LOCK:
        if name == "mfcc":
            return _ENC.setdefault("mfcc", MfccStatsEncoder()), None
        if name in ("ecapa", "auto"):
            if "ecapa" in _ENC:
                return _ENC["ecapa"], None
            try:
                _ENC["ecapa"] = EcapaEncoder()
                return _ENC["ecapa"], None
            except EncoderUnavailable as e:
                if name == "ecapa":
                    raise
                return _ENC.setdefault("mfcc", MfccStatsEncoder()), f"ECAPA unavailable, using MFCC-statistics baseline: {e}"
        raise EncoderUnavailable(f"Unknown encoder backend '{name}'.")


def encoder_status():
    ok = all(importlib.util.find_spec(m) for m in ("torch", "speechbrain"))
    return dict(configured=config.ENCODER_BACKEND, ecapa_packages_installed=ok,
                loaded=[k for k in _ENC])


# ------------------------------------------------------------------ embeddings
def chunk_signal(y, fs, seconds=None, hop_s=None):
    L, H = int((seconds or config.CHUNK_S) * fs), int((hop_s or config.CHUNK_HOP_S) * fs)
    if len(y) < L:
        return np.stack([np.pad(y, (0, L - len(y)))])
    return np.stack([y[s:s + L] for s in range(0, len(y) - L + 1, H)])


def embed_signal(y, fs, enc):
    if fs != 16000:
        y = librosa.resample(np.asarray(y, np.float32), orig_sr=fs, target_sr=16000, res_type="soxr_hq")
    y = np.asarray(y, np.float32)
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.9
    E = enc.embed(chunk_signal(y, 16000))
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12) if enc.l2 else E


def _centroid(E, l2):
    c = E.mean(0)
    return c / (np.linalg.norm(c) + 1e-12) if l2 else c


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ------------------------------------------------------------------ TTS reference (original sample rate)
def make_tts_reference(slug, vad, win_s=6.0, n_win=5):
    """~30 s of speech from the *original* recording for XTTS conditioning.
    The VAD speech regions are cut from the original-rate audio (light processing only: DC removal, 60 Hz
    high-pass, 5 ms fades, NO denoising so no artefacts reach the voice model), concatenated, and `n_win`
    evenly spaced windows of `win_s` seconds are taken so the whole recording contributes."""
    import scipy.signal as ss
    y, fs, _ = D.load_mono(PS.path(slug, "reference.wav"), None)
    if fs > 22050:
        y = librosa.resample(y, orig_sr=fs, target_sr=22050, res_type="soxr_hq"); fs = 22050
    y = y - y.mean()
    b, a = ss.butter(2, 60 / (fs / 2), "high"); y = ss.filtfilt(b, a, y).astype(np.float32)
    fade = max(int(0.005 * fs), 1); parts = []
    for t0, t1 in vad["regions"]:
        seg = y[int(t0 * fs): int(t1 * fs)].copy()
        if len(seg) > 2 * fade:
            seg[:fade] *= np.linspace(0, 1, fade); seg[-fade:] *= np.linspace(1, 0, fade); parts.append(seg)
    sp = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    W = int(win_s * fs)
    if len(sp) < W:
        raise EnrollmentError("Not enough speech to build the TTS reference.")
    k = min(n_win, len(sp) // W)
    starts = np.linspace(0, len(sp) - W, k).astype(int)
    ref_y = np.concatenate([sp[s:s + W] for s in starts])
    ref_y = ref_y / (np.max(np.abs(ref_y)) + 1e-9) * 0.9
    D.write_wav(PS.path(slug, "xtts_reference.wav"), ref_y, fs)
    return dict(sample_rate=int(fs), seconds=round(len(ref_y) / fs, 1), windows=int(k),
                window_start_in_speech_s=[round(s / fs, 1) for s in starts])


# ------------------------------------------------------------------ DSP analysis for a profile
def run_dsp_analysis(slug, backend="auto"):
    """backend: python | matlab | both | auto (python + MATLAB if installed). Returns dict of results by backend."""
    from services import matlab_service as M
    ref = PS.path(slug, "reference.wav")
    if not ref.exists():
        raise EnrollmentError("Upload a recording first.")
    out = {}
    if backend in ("python", "both", "auto", "matlab"):
        r = D.analyze(ref, PS.path(slug, "analysis", "python", "x").parent, config.TARGET_FS)
        out["python"] = r
    want_matlab = backend in ("matlab", "both") or (backend == "auto" and config.DSP_BACKEND != "python" and M.available()["any"])
    if want_matlab:
        try:
            out["matlab"] = M.run_analysis(ref, PS.path(slug, "analysis", "matlab", "x").parent, config.TARGET_FS)
        except M.MatlabError as e:
            if backend == "matlab":
                raise
            out["matlab"] = dict(status="error", message=str(e), note="Python DSP result retained")
    return out


def load_analysis(slug):
    res = {}
    for b in ("python", "matlab"):
        p = PS.profile_dir(slug) / "analysis" / b / "results.json"
        if p.exists():
            res[b] = json.loads(p.read_text())
    return res


# ------------------------------------------------------------------ reference upload
def commit_reference(slug, new_wav: Path, info: dict) -> dict:
    """Validate `new_wav` (already PCM16 WAV inside the profile folder). If usable it REPLACES the profile's
    reference and wipes everything derived from the old one; otherwise it is deleted and the old reference
    (if any) stays untouched. Raises AudioError with the reasons."""
    val = A.validate_reference(new_wav)
    if not val["ok"]:
        new_wav.unlink(missing_ok=True)
        raise A.AudioError(" ".join(val["errors"]))
    PS.clear_derived(slug)
    ref = PS.profile_dir(slug) / "reference.wav"
    new_wav.replace(ref)
    y, fs, _ = D.load_mono(ref)
    D.plot_preview(y, fs, PS.path(slug, "preview.png"))
    PS.update(slug, reference=dict(info, validation=val, uploaded_at=PS.now_iso()), enrollment=dict(status="uploaded"))
    return val


# ------------------------------------------------------------------ enrollment
def enroll(slug):
    t0 = time.time()
    if not PS.exists(slug):
        raise EnrollmentError(f"Profile '{slug}' does not exist.")
    ref = PS.path(slug, "reference.wav")
    if not ref.exists():
        raise EnrollmentError("Upload a recording first.")
    val = A.validate_reference(ref)
    if not val["ok"]:
        raise EnrollmentError(" ".join(val["errors"]))
    PS.update(slug, enrollment=dict(status="processing"))
    try:
        steps = {}
        # 1. DSP chain (Python path always runs: it produces the cleaned audio that feeds the ML stages)
        a_all = run_dsp_analysis(slug, "python")
        internal = a_all["python"].pop("_internal")
        fs = internal["fs"]
        steps["dsp_python"] = "ok"
        clean = D.extract_speech(internal["processed"], internal["vad"], fs)
        if len(clean) < 30 * fs:
            raise EnrollmentError(f"Only {len(clean) / fs:.0f} s of clean speech after VAD; need at least 30 s.")
        D.write_wav(PS.path(slug, "reference_clean.wav"), clean, fs)
        internal["vad"]["fs"] = fs
        ttsref = make_tts_reference(slug, internal["vad"])
        steps["tts_reference"] = "ok"
        # 2. optional MATLAB analysis (non-fatal here; explicit runs on the DSP page surface errors)
        if config.DSP_BACKEND != "python":
            from services import matlab_service as M
            if M.available()["any"]:
                try:
                    M.run_analysis(ref, PS.path(slug, "analysis", "matlab", "x").parent, config.TARGET_FS)
                    steps["dsp_matlab"] = "ok"
                except M.MatlabError as e:
                    steps["dsp_matlab"] = f"failed: {str(e)[:200]}"
            else:
                steps["dsp_matlab"] = "skipped (MATLAB not found)"
        else:
            steps["dsp_matlab"] = "skipped (DSP_BACKEND=python)"
        # 3. speaker embeddings
        enc, note = get_encoder()
        E = embed_signal(clean, fs, enc)
        cent = _centroid(E, enc.l2)
        np.save(PS.path(slug, "chunk_embeddings.npy"), E)
        np.save(PS.path(slug, "speaker_embedding.npy"), cent.astype(np.float32))
        steps["speaker_embedding"] = "ok"
        sims = [_cos(e, cent) for e in E]
        meta = PS.update(slug, enrollment=dict(
            status="enrolled", completed_at=PS.now_iso(), reference_sha256=A.sha256_file(ref),
            encoder=dict(name=enc.name, dim=int(E.shape[1]), pretrained=enc.pretrained, source=enc.source, note=note),
            n_chunks=int(len(E)), chunk_seconds=config.CHUNK_S, chunk_hop_seconds=config.CHUNK_HOP_S,
            clean_speech_seconds=round(len(clean) / fs, 1), tts_reference=ttsref,
            within_profile_cosine_mean=round(float(np.mean(sims)), 3), within_profile_cosine_std=round(float(np.std(sims)), 3),
            steps=steps, tts_conditioning="pending (computed on first synthesis)",
            runtime_s=round(time.time() - t0, 1),
            models=dict(speaker_encoder=enc.source,
                        tts=f"{config.XTTS_MODEL} (pretrained, zero-shot speaker conditioning; not trained or fine-tuned here)")))
        # 4. refresh classifier if >=2 profiles share this encoder (non-fatal)
        try:
            train_classifier(enc.name)
            steps["classifier"] = "retrained"
        except EnrollmentError as e:
            steps["classifier"] = f"skipped: {e}"
        PS.update(slug, enrollment=dict(steps=steps))
        return PS.get(slug)
    except Exception as e:
        PS.update(slug, enrollment=dict(status="failed", error=str(e)))
        raise


# ------------------------------------------------------------------ PCA + SVM speaker classifier
def _enrolled_by_encoder(enc_name):
    data = {}
    for m in PS.list_profiles():
        s = m["slug"]
        if PS.is_enrolled(s) and m["enrollment"].get("encoder", {}).get("name") == enc_name:
            f = PS.profile_dir(s) / "chunk_embeddings.npy"
            if f.exists():
                data[s] = np.load(f)
    return data


def _pipeline(n_train, dim):
    n_comp = max(2, min(20, n_train - 1, dim))
    return make_pipeline(StandardScaler(), PCA(n_components=n_comp, random_state=0),
                         LinearSVC(C=1.0, max_iter=20000, random_state=0)), n_comp


def split_blocks(data):
    """Time-block split per profile: first 70 % of chunks -> train, then a gap of `chunk/hop` chunks
    (overlapping chunks must not leak), remaining chunks -> test. Returns labels, Xtr, ytr, Xte, yte."""
    labels = sorted(data)
    gap = int(np.ceil(config.CHUNK_S / config.CHUNK_HOP_S))
    Xtr, ytr, Xte, yte = [], [], [], []
    for lab in labels:
        X = data[lab]; n_tr = int(0.7 * len(X))
        Xtr.append(X[:n_tr]); ytr += [lab] * n_tr
        Xte.append(X[n_tr + gap:]); yte += [lab] * len(X[n_tr + gap:])
    Xtr = np.concatenate(Xtr)
    Xte = np.concatenate(Xte) if any(len(x) for x in Xte) else np.zeros((0, Xtr.shape[1]))
    return labels, Xtr, ytr, Xte, yte


def train_classifier(enc_name):
    """Train StandardScaler -> PCA -> LinearSVC on per-chunk embeddings of all enrolled profiles that used
    `enc_name`. Evaluation = time-block hold-out (first 70 % of each profile's chunks train, last chunks test,
    with a 2-chunk gap so overlapping chunks do not leak). Final model is refit on all chunks."""
    data = _enrolled_by_encoder(enc_name)
    if len(data) < 2:
        raise EnrollmentError(f"need at least 2 enrolled profiles using '{enc_name}' (have {len(data)})")
    labels, Xtr, ytr, Xte, yte = split_blocks(data)
    pipe, n_comp = _pipeline(len(Xtr), Xtr.shape[1])
    pipe.fit(Xtr, ytr)
    rep = dict(encoder=enc_name, labels=labels, chunks_per_profile={k: int(len(v)) for k, v in data.items()},
               pca_components=n_comp, n_train=int(len(Xtr)), n_test=int(len(Xte)),
               explained_variance_ratio_sum=round(float(pipe[1].explained_variance_ratio_.sum()), 4),
               split="time-block hold-out (70% / gap / remaining); chunks overlap 50% inside a split",
               created=PS.now_iso())
    if len(Xte):
        pred = pipe.predict(Xte)
        yt = np.array(yte)
        rep["holdout_accuracy"] = round(float(np.mean(pred == yt)), 4)
        rep["confusion"] = {a: {b: int(np.sum((yt == a) & (pred == b))) for b in labels} for a in labels}
        rep["per_class_recall"] = {a: round(float(np.mean(pred[yt == a] == a)), 4) for a in labels}
    else:
        rep["holdout_accuracy"] = None
    rep["caveat"] = ("Chunks come from ONE recording per speaker, so hold-out accuracy measures separability of those "
                     "recordings (same session, same microphone), not general speaker-verification performance.")
    # final model on all data
    Xall = np.concatenate([data[l] for l in labels]); yall = sum(([l] * len(data[l]) for l in labels), [])
    final, _ = _pipeline(len(Xall), Xall.shape[1]); final.fit(Xall, yall)
    out = config.DATA_DIR / "experiments"; out.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(pipeline=final, labels=labels, report=rep), config.MODELS_DIR / f"classifier_{enc_name}.joblib")
    (out / f"classifier_{enc_name}.json").write_text(json.dumps(rep, indent=2))
    Z = final[1].transform(final[0].transform(Xall))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for lab in labels:
        sel = np.array(yall) == lab
        ax.scatter(Z[sel, 0], Z[sel, 1], s=10, alpha=0.6, label=lab)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(f"PCA of speaker embeddings ({enc_name})"); ax.legend(fontsize=8)
    with D.LOCK:
        fig.savefig(out / f"pca_{enc_name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return rep


def load_classifier(enc_name):
    p = config.MODELS_DIR / f"classifier_{enc_name}.joblib"
    return joblib.load(p) if p.exists() else None


def classifier_reports():
    out = {}
    d = config.DATA_DIR / "experiments"
    if d.exists():
        for p in sorted(d.glob("classifier_*.json")):
            out[p.stem.replace("classifier_", "")] = json.loads(p.read_text())
    return out


# ------------------------------------------------------------------ verification of generated speech
def verify(slug, y, fs):
    """Compare a signal (e.g. generated speech) with the enrolled profile `slug`; also against the other
    enrolled profiles that used the same encoder. Nothing is written to any profile."""
    meta = PS.get(slug)
    en = meta["enrollment"]
    if en.get("status") != "enrolled":
        raise EnrollmentError("Profile is not enrolled.")
    enc, _ = get_encoder(en["encoder"]["name"])
    E = embed_signal(y, fs, enc)
    g = _centroid(E, enc.l2)
    cent = np.load(PS.profile_dir(slug) / "speaker_embedding.npy")
    others = {}
    for s, X in _enrolled_by_encoder(enc.name).items():
        if s != slug:
            others[s] = round(_cos(g, _centroid(X, enc.l2)), 4)
    res = dict(encoder=enc.name, n_chunks=int(len(E)), cosine_to_claimed_profile=round(_cos(g, cent), 4),
               cosine_to_other_profiles=others,
               enrollment_chunk_cosine_mean=en.get("within_profile_cosine_mean"),
               enrollment_chunk_cosine_std=en.get("within_profile_cosine_std"))
    clf = load_classifier(enc.name)
    if clf and slug in clf["labels"]:
        pipe = clf["pipeline"]
        pred = pipe.predict(E)
        res["svm_fraction_chunks_as_claimed"] = round(float(np.mean(pred == slug)), 3)
        res["svm_majority_label"] = max(set(pred), key=list(pred).count)
        res["svm_labels"] = clf["labels"]
        if not enc.l2:
            sc = pipe[0]
            res["cosine_standardized"] = round(_cos(sc.transform(g[None])[0], sc.transform(cent[None])[0]), 4)
    else:
        res["svm_note"] = "No PCA/SVM classifier yet (needs >=2 enrolled profiles using the same encoder)."
    if res["cosine_to_other_profiles"]:
        res["closest_profile"] = max({slug: res["cosine_to_claimed_profile"], **others}, key=lambda k: {slug: res["cosine_to_claimed_profile"], **others}[k])
    res["note"] = ("Cosine similarity of the generated speech's centroid embedding to the enrolled centroid. "
                   "Compare with the enrollment self-similarity mean/std; absolute values are encoder-specific and "
                   "this is a proxy for speaker similarity, not a human listening test.")
    if not enc.l2:
        res["note"] += " The MFCC-statistics baseline is weak for cosine scoring; rely on the SVM numbers."
    return res
