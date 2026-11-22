"""Speaker-adaptive speech synthesis demo (Flask). Run:  python app.py"""
import logging

from flask import Flask, g, request, jsonify

import config
from services import profile_service as PS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def create_app():
    config.initialize_runtime()
    app = Flask(__name__)
    app.secret_key = config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = 120 * 1024 * 1024
    config.PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (config.DATA_DIR / "experiments").mkdir(parents=True, exist_ok=True)
    PS.ensure_default_profiles()
    from routes.main import bp
    app.register_blueprint(bp)

    @app.before_request
    def serialize_mutations():
        # A single 4 GB GPU and reference files are shared. Protect long-running
        # inference from uploads, deletion or enrollment in another browser tab.
        if request.method == "POST":
            from services.synthesis_service import GPU_JOB_LOCK
            GPU_JOB_LOCK.acquire()
            g.gpu_job_locked = True

    @app.teardown_request
    def release_mutations(error):
        if getattr(g, "gpu_job_locked", False):
            from services.synthesis_service import GPU_JOB_LOCK
            g.gpu_job_locked = False
            GPU_JOB_LOCK.release()

    @app.errorhandler(413)
    def too_large(error):
        message = "File exceeds the 120 MB upload limit. Use a shorter or compressed recording."
        if request.headers.get("X-Requested-With") == "fetch":
            return jsonify(ok=False, error=message), 413
        return message, 413
    return app


if __name__ == "__main__":
    import sys
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Run with .venv\\Scripts\\python.exe app.py (project virtual environment required).")
    create_app().run(debug=False, host="127.0.0.1", port=5000, threaded=True)
