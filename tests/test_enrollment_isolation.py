import numpy as np, json
import config
from services import profile_service as PS, speaker_service as S
from conftest import upload


def test_two_profiles_are_independent(enrolled, env):
    pd = {s: PS.profile_dir(s) for s in ("alice", "bob")}
    for s, d in pd.items():
        files = {p.name for p in d.iterdir()}
        assert {"reference.wav", "reference_clean.wav", "xtts_reference.wav", "speaker_embedding.npy", "chunk_embeddings.npy", "metadata.json"} <= files
        assert PS.is_enrolled(s)
    ea, eb = (np.load(pd[s] / "speaker_embedding.npy") for s in pd)
    assert np.linalg.norm(ea - eb) > 1.0                               # different speakers -> different embeddings
    assert PS.get("alice")["enrollment"]["reference_sha256"] != PS.get("bob")["enrollment"]["reference_sha256"]
    assert PS.get("alice")["enrollment"]["encoder"]["pretrained"] is False   # honest labelling of the MFCC baseline
    assert "pretrained" in PS.get("alice")["enrollment"]["models"]["tts"]


def test_classifier_separates_the_two_profiles(enrolled):
    rep = S.classifier_reports()["mfcc-stats"]
    assert sorted(rep["labels"]) == ["alice", "bob"] and rep["holdout_accuracy"] >= 0.9
    assert (config.MODELS_DIR / "classifier_mfcc-stats.joblib").exists()


def test_verify_scores_own_profile_higher(enrolled, wavs):
    import soundfile as sf, io
    from services import dsp_service as D
    ya, fs = sf.read(io.BytesIO(wavs["alice"]), dtype="float32"); seg = ya[16000 * 10: 16000 * 30]
    v = S.verify("alice", seg, fs)
    assert v["svm_majority_label"] == "alice" and v["svm_fraction_chunks_as_claimed"] > 0.8
    vb = S.verify("bob", seg, fs)                                       # Alice's voice claimed as Bob
    assert vb["svm_fraction_chunks_as_claimed"] < 0.2 and vb["svm_majority_label"] == "alice"


def test_new_upload_wipes_derived_data_of_that_profile_only(enrolled, wavs):
    c = enrolled
    bob_before = (PS.profile_dir("bob") / "speaker_embedding.npy").read_bytes()
    (PS.profile_dir("alice") / "xtts_latents.pt").write_bytes(b"stale")
    assert b"Recording accepted" in upload(c, "alice", wavs["alice"]).data
    d = PS.profile_dir("alice")
    assert not (d / "speaker_embedding.npy").exists() and not (d / "xtts_latents.pt").exists() and not (d / "analysis").exists()
    assert PS.get("alice")["enrollment"]["status"] == "uploaded"
    assert (PS.profile_dir("bob") / "speaker_embedding.npy").read_bytes() == bob_before      # Bob untouched
    assert b"Enrollment SUCCESS" in c.post("/profiles/alice/process", follow_redirects=True).data   # re-enroll for later tests


def test_training_creates_missing_models_dir(enrolled):
    """Regression: CLI runs never created models/, and joblib.dump then failed."""
    import shutil
    shutil.rmtree(config.MODELS_DIR, ignore_errors=True)
    S.train_classifier("mfcc-stats")
    assert (config.MODELS_DIR / "classifier_mfcc-stats.joblib").exists()
