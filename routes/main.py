import io, json, re, tempfile, zipfile
from pathlib import Path

from flask import (Blueprint, abort, flash, jsonify, redirect, render_template, request, send_file, send_from_directory,
                   session, url_for)

import config
from services import audio_service as A, dsp_service as D, gemini_service as G
from services import profile_service as PS, quant_service as Q, speaker_service as S
from services import synthesis_service as SY, tts_service as T
from services import indic_service as I

bp = Blueprint("main", __name__)
SERVE_EXT = {".png", ".wav", ".bin", ".json"}


def _profile(slug):
    if not PS.exists(slug):
        abort(404)
    session["slug"] = slug
    return PS.get(slug)


@bp.app_context_processor
def _ctx():
    return dict(all_profiles=PS.list_profiles(), current_slug=session.get("slug"),
                is_enrolled=PS.is_enrolled)


def _status():
    return dict(runtime=config.runtime_status(), tts=T.status(), indic=I.status(), gemini=G.configured(), encoder=S.encoder_status(),
                gemini_models=config.GEMINI_MODELS, dsp_backend=config.DSP_BACKEND)


# ------------------------------------------------------------------ home / profiles
@bp.get("/")
def index():
    return render_template("index.html", status=_status())


@bp.get("/profiles")
def profiles():
    return render_template("profiles.html")


@bp.post("/profiles")
def profile_create():
    try:
        m = PS.create(request.form.get("name", ""))
        flash(f"Profile '{m['name']}' created.")
        return redirect(url_for("main.enroll", slug=m["slug"]))
    except PS.ProfileError as e:
        flash(str(e))
        return redirect(url_for("main.profiles"))


@bp.post("/profiles/<slug>/delete")
def profile_delete(slug):
    _profile(slug)
    PS.delete(slug)
    session.pop("slug", None)
    flash(f"Profile '{slug}' and all its data were deleted.")
    return redirect(url_for("main.profiles"))


# ------------------------------------------------------------------ enrollment
@bp.get("/profiles/<slug>/enroll")
def enroll(slug):
    meta = _profile(slug)
    return render_template("enroll.html", p=meta, script=A.ENROLLMENT_SCRIPT, cfg=config,
                           has_ref=(PS.profile_dir(slug) / "reference.wav").exists())


@bp.post("/profiles/<slug>/upload")
def upload(slug):
    _profile(slug)
    ajax = request.headers.get("X-Requested-With") == "fetch"
    f = request.files.get("audio")

    def fail(msg):
        if ajax:
            return jsonify(ok=False, error=msg), 400
        flash(msg)
        return redirect(url_for("main.enroll", slug=slug))

    if not f or not f.filename:
        return fail("Choose or record an audio file first.")
    new = PS.profile_dir(slug) / "reference.new.wav"
    try:
        info = A.save_as_wav(f, new)
        S.commit_reference(slug, new, info)
    except A.AudioError as e:
        new.unlink(missing_ok=True)
        return fail(str(e))
    if ajax:
        return jsonify(ok=True, redirect=url_for("main.enroll", slug=slug))
    flash("Recording accepted. Review the details below, then press 'Process & Enroll Voice'.")
    return redirect(url_for("main.enroll", slug=slug))


@bp.post("/profiles/<slug>/process")
def process(slug):
    _profile(slug)
    try:
        S.enroll(slug)
        flash("Enrollment SUCCESS.")
    except (S.EnrollmentError, S.EncoderUnavailable, A.AudioError) as e:
        flash(f"Enrollment failed: {e}")
    except Exception as e:                       # noqa: BLE001
        flash(f"Enrollment failed unexpectedly: {type(e).__name__}: {e}")
    return redirect(url_for("main.enroll", slug=slug))


# ------------------------------------------------------------------ DSP analysis
@bp.route("/profiles/<slug>/analysis", methods=["GET", "POST"])
def analysis(slug):
    meta = _profile(slug)
    if request.method == "POST":
        backend = request.form.get("backend", "python")
        if backend not in ("python", "cuda", "auto"):
            backend = "python"
        try:
            S.run_dsp_analysis(slug, backend)
            flash(f"DSP analysis finished ({backend}).")
        except (S.EnrollmentError, ValueError) as e:
            flash(f"Analysis failed: {e}")
        return redirect(url_for("main.analysis", slug=slug))
    return render_template("analysis.html", p=meta, res=S.load_analysis(slug),
                           has_ref=(PS.profile_dir(slug) / "reference.wav").exists())


# ------------------------------------------------------------------ synthesis + results
@bp.route("/profiles/<slug>/synthesize", methods=["GET", "POST"])
def synthesize(slug):
    meta = _profile(slug)
    if request.method == "POST":
        def num(name, default):
            try:
                return float(request.form.get(name, default))
            except ValueError:
                return default
        lp = num("lp_hz", 0)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                backend = request.form.get("backend", "xtts")
                prompt = None
                if backend == "indicf5":
                    if not I.status()["configured"]:
                        raise T.TTSUnavailable(I.status()["note"])
                    file = request.files.get("prompt_audio")
                    if file and file.filename:
                        prompt = Path(tmp) / "prompt.wav"
                        A.save_as_wav(file, prompt)
                m = SY.run(slug, request.form.get("text", ""), language=request.form.get("language", "en"),
                           use_gemini=request.form.get("use_gemini") == "on", hp_hz=num("hp_hz", 70),
                           lp_hz=lp or None, target_rms_dbfs=num("target_rms", -20), backend=backend,
                           prompt_path=prompt, prompt_text=request.form.get("prompt_text", ""))
            return redirect(url_for("main.results", slug=slug, run_id=m["run_id"]))
        except (SY.SynthesisError, T.TTSUnavailable, T.TTSError, S.EnrollmentError, S.EncoderUnavailable, A.AudioError) as e:
            flash(f"{type(e).__name__.replace('Error', '')}: {e}")
        except Exception as e:                   # noqa: BLE001
            flash(f"Unexpected error: {type(e).__name__}: {e}")
    return render_template("synthesize.html", p=meta, langs=T.LANGUAGES, language_names=T.LANGUAGE_NAMES,
                           gemini=G.configured(), tts=T.status(), indic=I.status(),
                           max_chars=config.MAX_TEXT_CHARS)


@bp.get("/profiles/<slug>/results")
@bp.get("/profiles/<slug>/results/<run_id>")
def results(slug, run_id=None):
    meta = _profile(slug)
    runs = PS.list_runs(slug)
    run = None
    if run_id or runs:
        try:
            run = SY.load_run(slug, run_id or runs[0])
        except SY.SynthesisError:
            abort(404)
    return render_template("results.html", p=meta, run=run, runs=runs, analysis=S.load_analysis(slug))


# ------------------------------------------------------------------ experiments
@bp.get("/experiments")
def experiments():
    return render_template("experiments.html", clf=S.classifier_reports(), quant=_load_exp("quant_classifier"),
                           enc_quant=_load_exp("quant_encoder"), comp=_load_exp("compression"), status=_status())


def _load_exp(name):
    p = config.DATA_DIR / "experiments" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _save_exp(name, data):
    (config.DATA_DIR / "experiments").mkdir(parents=True, exist_ok=True)
    (config.DATA_DIR / "experiments" / f"{name}.json").write_text(json.dumps(data, indent=2))


@bp.post("/experiments/classifier")
def exp_classifier():
    enc = {}
    for m in PS.list_profiles():
        if PS.is_enrolled(m["slug"]):
            enc.setdefault(m["enrollment"]["encoder"]["name"], 0)
            enc[m["enrollment"]["encoder"]["name"]] += 1
    done = 0
    for name, n in enc.items():
        if n >= 2:
            S.train_classifier(name); done += 1
    flash(f"Trained {done} classifier(s)." if done else "Need at least 2 enrolled profiles that used the same encoder.")
    return redirect(url_for("main.experiments"))


@bp.post("/experiments/quantize")
def exp_quantize():
    which = request.form.get("which", "classifier")
    try:
        if which == "encoder":
            _save_exp("quant_encoder", Q.quantize_encoder())
        else:
            names = {m["enrollment"]["encoder"]["name"] for m in PS.list_profiles() if PS.is_enrolled(m["slug"])}
            for n in sorted(names):
                try:
                    _save_exp("quant_classifier", Q.quantize_classifier(n)); break
                except S.EnrollmentError as e:
                    err = str(e)
            else:
                flash(f"Quantisation not run: {err if names else 'enroll at least 2 profiles first'}")
    except Exception as e:                       # noqa: BLE001
        flash(f"Quantisation failed: {type(e).__name__}: {e}")
    return redirect(url_for("main.experiments"))


@bp.post("/experiments/compression")
def exp_compression():
    slug = request.form.get("slug", "")
    if not PS.is_enrolled(slug):
        flash("Pick an enrolled profile.")
        return redirect(url_for("main.experiments"))
    y, fs, _ = D.load_mono(PS.profile_dir(slug) / "reference_clean.wav")
    y = y[: 20 * fs]
    out = config.DATA_DIR / "experiments"
    rows = D.compression_sweep(y, fs, out_dir=out)
    _save_exp("compression", dict(profile=slug, seconds=round(len(y) / fs, 1), fs=fs, rows=D.public_rows(rows),
                                  note="First 20 s of the profile's cleaned reference speech, measured by encode -> decode."))
    return redirect(url_for("main.experiments"))


# ------------------------------------------------------------------ files
@bp.get("/files/<slug>/<path:name>")
def files(slug, name):
    _profile(slug)
    p = Path(name)
    if p.suffix.lower() not in SERVE_EXT or "xtts_latents" in name:
        abort(404)
    return send_from_directory(PS.profile_dir(slug), name, as_attachment=request.args.get("download") == "1")


@bp.get("/exp-files/<name>")
def exp_files(name):
    if Path(name).suffix.lower() not in {".png"}:
        abort(404)
    return send_from_directory(config.DATA_DIR / "experiments", name)


@bp.get("/api/status")
def api_status():
    return jsonify(_status())


@bp.get("/profiles/<slug>/export/<run_id>/<fmt>")
def export_audio(slug, run_id, fmt):
    _profile(slug)
    try:
        SY.load_run(slug, run_id)
    except SY.SynthesisError:
        abort(404)
    directory = PS.profile_dir(slug) / "outputs" / run_id
    if fmt == "zip":
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix in SERVE_EXT:
                    archive.write(path, path.name)
        buf.seek(0)
        return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"{slug}-{run_id}.zip")
    if fmt not in A.EXPORTS:
        abort(400, "Unsupported export format.")
    codec, mime = A.EXPORTS[fmt]
    if fmt == "wav":
        return send_from_directory(directory, "final.wav", as_attachment=True, download_name=f"{slug}-{run_id}.wav")
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / f"speech.{fmt}"
        try:
            A.transcode(directory / "final.wav", dest, codec)
            buf = io.BytesIO(dest.read_bytes())
        except A.AudioError as exc:
            abort(422, str(exc))
    return send_file(buf, mimetype=mime, as_attachment=True, download_name=f"{slug}-{run_id}.{fmt}")
