"""Speaker-adaptive speech synthesis demo (Flask). Run:  python app.py"""
import logging

from flask import Flask

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
    return app


if __name__ == "__main__":
    create_app().run(debug=False, host="127.0.0.1", port=5000, threaded=True)
