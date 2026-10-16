"""Command-line front end (no Flask). Useful on Google Colab / headless GPU machines.

  python scripts/cli.py enroll --profile "User 1" --wav my_reference.wav
  python scripts/cli.py synth  --profile "User 1" --text "Any new sentence." [--language en] [--no-gemini]
  python scripts/cli.py classifier          # PCA + SVM over all enrolled profiles
  python scripts/cli.py quantize            # int8 classifier + (if torch/speechbrain) encoder dynamic quantisation
"""
import argparse, json, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from services import audio_service as A, profile_service as PS, quant_service as Q, speaker_service as S
from services import synthesis_service as SY


def get_or_create(name):
    slug = PS.slugify(name)
    if not PS.exists(slug):
        PS.create(name)
    return slug


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enroll"); e.add_argument("--profile", required=True); e.add_argument("--wav", required=True)
    s = sub.add_parser("synth"); s.add_argument("--profile", required=True); s.add_argument("--text", required=True)
    s.add_argument("--language", default="en"); s.add_argument("--no-gemini", action="store_true")
    sub.add_parser("classifier"); sub.add_parser("quantize")
    a = ap.parse_args()
    config.PROFILES_DIR.mkdir(parents=True, exist_ok=True); (config.DATA_DIR / "experiments").mkdir(parents=True, exist_ok=True)
    try:
        if a.cmd == "enroll":
            slug = get_or_create(a.profile)
            new = PS.profile_dir(slug) / "reference.new.wav"
            info = A.convert_to_wav(Path(a.wav), new)
            val = S.commit_reference(slug, new, info)
            for w in val["warnings"]:
                print("warning:", w)
            m = S.enroll(slug)
            print(json.dumps(m["enrollment"], indent=2))
        elif a.cmd == "synth":
            m = SY.run(PS.slugify(a.profile), a.text, a.language, use_gemini=not a.no_gemini)
            print(json.dumps({k: m[k] for k in ("run_id", "tts", "postprocess", "compression_default", "verification", "timings")}, indent=2))
            print("Output folder:", PS.profile_dir(m["profile"]) / "outputs" / m["run_id"])
        elif a.cmd == "classifier":
            names = {x["enrollment"]["encoder"]["name"] for x in PS.list_profiles() if PS.is_enrolled(x["slug"])}
            for n in sorted(names):
                try:
                    print(json.dumps(S.train_classifier(n), indent=2))
                except S.EnrollmentError as ex:
                    print(f"{n}: {ex}")
        elif a.cmd == "quantize":
            names = {x["enrollment"]["encoder"]["name"] for x in PS.list_profiles() if PS.is_enrolled(x["slug"])}
            for n in sorted(names):
                try:
                    print(json.dumps(Q.quantize_classifier(n), indent=2)); break
                except S.EnrollmentError as ex:
                    print(f"classifier quantisation skipped: {ex}")
            print(json.dumps(Q.quantize_encoder(), indent=2))
    except Exception as ex:                                  # noqa: BLE001
        print(f"ERROR ({type(ex).__name__}): {ex}", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()
