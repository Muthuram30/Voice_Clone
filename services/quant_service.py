"""Quantisation experiments. Exactly what is quantised is stated in every report.

1. quantize_classifier  - the PCA + linear-SVM speaker classifier (scikit-learn, trained here).
   Weights (PCA components, SVM coefficients) are stored as int8 (symmetric, per-tensor scale) and activations
   are quantised to int8 with a calibrated scale; the two matrix products run in int32 integer arithmetic and
   are rescaled to float. Compared against the FP32 pipeline: size, latency, agreement, accuracy.
   NOTE: the models are tiny (a few KB), so latency differences are mostly overhead; the size reduction and the
   accuracy/score deviation are the meaningful results.

2. quantize_encoder     - PyTorch dynamic int8 quantisation of the pretrained ECAPA-TDNN speaker encoder.
   torch.ao.quantization.quantize_dynamic only converts nn.Linear (and RNN) layers. ECAPA-TDNN is almost entirely
   Conv1d, so very little (possibly nothing) is quantised; the report states the number of layers converted
   instead of claiming a speed-up. Requires torch + speechbrain.

3. XTTS-v2 itself is NOT quantised: its GPT/vocoder inference path is not documented to support
   dynamic int8, and doing so unverified would risk silently degrading speech.
"""
import copy, io, time

import numpy as np

import config
from services import speaker_service as S


def _q(w, s=None):
    s = s if s is not None else (np.max(np.abs(w)) / 127.0 + 1e-12)
    return np.clip(np.round(w / s), -127, 127).astype(np.int8), float(s)


class Int8Classifier:
    """Scaler -> PCA -> LinearSVC with int8 weights/activations and int32 accumulation."""

    def __init__(self, pipe, X_calib):
        sc, pca, svm = pipe[0], pipe[1], pipe[2]
        self.mean, self.scale = sc.mean_.astype(np.float32), sc.scale_.astype(np.float32)
        Z = (X_calib - self.mean) / self.scale
        self.Wq, self.sw1 = _q(pca.components_)
        self.sz = float(np.max(np.abs(Z)) / 127.0 + 1e-12)
        self.pca_bias = (-(pca.mean_ @ pca.components_.T)).astype(np.float32)
        P = (Z @ pca.components_.T + self.pca_bias)
        self.Cq, self.sc = _q(svm.coef_)
        self.sp = float(np.max(np.abs(P)) / 127.0 + 1e-12)
        self.b = svm.intercept_.astype(np.float32)
        self.classes = svm.classes_

    def decision(self, X):
        Z = (X - self.mean) / self.scale
        Zq = np.clip(np.round(Z / self.sz), -127, 127).astype(np.int8)
        P = (Zq.astype(np.int32) @ self.Wq.T.astype(np.int32)).astype(np.float32) * (self.sz * self.sw1) + self.pca_bias
        Pq = np.clip(np.round(P / self.sp), -127, 127).astype(np.int8)
        return (Pq.astype(np.int32) @ self.Cq.T.astype(np.int32)).astype(np.float32) * (self.sp * self.sc) + self.b

    def predict(self, X):
        S_ = self.decision(X)
        return self.classes[(S_[:, 0] > 0).astype(int)] if S_.shape[1] == 1 else self.classes[np.argmax(S_, 1)]

    def nbytes(self):
        return int(self.mean.nbytes + self.scale.nbytes + self.Wq.nbytes + self.pca_bias.nbytes + self.Cq.nbytes
                   + self.b.nbytes + 4 * 4)          # + four float scales


def _fp32_bytes(pipe):
    sc, pca, svm = pipe[0], pipe[1], pipe[2]
    return int(sum(a.astype(np.float32).nbytes for a in (sc.mean_, sc.scale_, pca.components_, pca.mean_, svm.coef_, svm.intercept_)))


def _time(fn, X, reps=30):
    fn(X)
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(X); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def quantize_classifier(enc_name):
    data = S._enrolled_by_encoder(enc_name)
    if len(data) < 2:
        raise S.EnrollmentError(f"need at least 2 enrolled profiles using '{enc_name}' to build a classifier")
    labels, Xtr, ytr, Xte, yte = S.split_blocks(data)
    if len(Xte) == 0:
        raise S.EnrollmentError("not enough chunks for a hold-out test set")
    pipe, ncomp = S._pipeline(len(Xtr), Xtr.shape[1]); pipe.fit(Xtr, ytr)
    qc = Int8Classifier(pipe, Xtr)
    yt = np.array(yte)
    p32, p8 = pipe.predict(Xte), qc.predict(Xte)
    s32 = pipe.decision_function(Xte).reshape(len(Xte), -1); s8 = qc.decision(Xte)
    t32, t8 = _time(pipe.predict, Xte), _time(qc.predict, Xte)
    b32, b8 = _fp32_bytes(pipe), qc.nbytes()
    return dict(
        what_is_quantised="PCA projection matrix and linear-SVM coefficients -> int8 (per-tensor symmetric); "
                          "input and PCA-output activations -> int8 (calibrated on training chunks); int32 accumulation.",
        not_quantised="StandardScaler statistics, biases (float32); the speaker encoder; XTTS-v2.",
        encoder=enc_name, labels=labels, n_test_chunks=int(len(Xte)), pca_components=ncomp, input_dim=int(Xtr.shape[1]),
        accuracy_fp32=round(float(np.mean(p32 == yt)), 4), accuracy_int8=round(float(np.mean(p8 == yt)), 4),
        prediction_agreement=round(float(np.mean(p32 == p8)), 4),
        max_abs_decision_diff=round(float(np.max(np.abs(s32 - s8))), 4),
        model_bytes_fp32=b32, model_bytes_int8=b8, size_ratio=round(b32 / b8, 2),
        latency_ms_fp32=round(t32 * 1e3, 3), latency_ms_int8=round(t8 * 1e3, 3),
        latency_note="batch of all test chunks, median of 30 runs. The matrices are tiny, so timings are dominated by call overhead and are indicative only.")


def quantize_encoder():
    """PyTorch dynamic int8 quantisation of the pretrained ECAPA encoder (needs torch + speechbrain)."""
    try:
        enc, _ = S.get_encoder("ecapa")
    except S.EncoderUnavailable as e:
        return dict(status="unavailable", reason=str(e))
    import torch
    fp = copy.deepcopy(enc.clf).to("cpu")      # copies: never move/modify the shared encoder in place
    q = copy.deepcopy(fp)
    n_lin = sum(isinstance(m, torch.nn.Linear) for m in q.mods.embedding_model.modules())
    n_conv = sum(isinstance(m, torch.nn.Conv1d) for m in q.mods.embedding_model.modules())
    q.mods.embedding_model = torch.ao.quantization.quantize_dynamic(q.mods.embedding_model, {torch.nn.Linear}, dtype=torch.qint8)
    n_q = sum("quantized" in type(m).__module__ for m in q.mods.embedding_model.modules())

    def nbytes(m):
        b = io.BytesIO(); torch.save(m.state_dict(), b); return b.getbuffer().nbytes
    rep = dict(status="ok", what_is_quantised="nn.Linear layers of the ECAPA-TDNN embedding model (dynamic int8)",
               linear_layers_found=n_lin, conv1d_layers_not_quantised=n_conv, layers_converted=n_q,
               embedding_model_bytes_fp32=nbytes(fp.mods.embedding_model), embedding_model_bytes_int8=nbytes(q.mods.embedding_model))
    if n_q == 0:
        rep["conclusion"] = ("No layers were converted: ECAPA-TDNN is convolutional, and PyTorch dynamic quantisation does "
                             "not cover Conv1d. No speed/size claim is made. Static (calibrated) quantisation would be required.")
        return rep
    x = torch.randn(8, 48000)
    with torch.no_grad():
        t = time.perf_counter(); a = fp.encode_batch(x).squeeze(1); t32 = time.perf_counter() - t
        t = time.perf_counter(); b = q.encode_batch(x).squeeze(1); t8 = time.perf_counter() - t
    rep.update(cosine_fp32_vs_int8_mean=round(float(torch.nn.functional.cosine_similarity(a, b).mean()), 5),
               latency_s_fp32=round(t32, 3), latency_s_int8=round(t8, 3))
    return rep
