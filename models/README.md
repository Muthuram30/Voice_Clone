Downloaded models land here (git-ignored):

* `ecapa/`   SpeechBrain ECAPA-TDNN speaker encoder (pretrained on VoxCeleb) - `python scripts/prepare_models.py`
* `tts/`     Coqui XTTS-v2 (pretrained; non-commercial CPML licence, set COQUI_TOS_AGREED=1)
* `classifier_<encoder>.joblib`  PCA+SVM classifier trained in THIS project from enrolled profiles' embeddings

Nothing in this folder is trained from scratch; the only model fitted by this project is the small PCA+SVM classifier.
