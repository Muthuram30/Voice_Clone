"""Profile storage. One folder per profile; nothing is shared between profiles.

profiles/<slug>/
    metadata.json            profile + enrollment state
    reference.wav            the uploaded enrollment recording (as PCM16 WAV)
    reference_clean.wav      preprocessed, speech-only audio (16 kHz)
    xtts_reference.wav       ~30 s of evenly spaced speech chunks used to condition XTTS
    speaker_embedding.npy    centroid embedding of this profile
    chunk_embeddings.npy     per-chunk embeddings (used by the PCA/SVM experiment)
    xtts_latents.pt          cached XTTS conditioning latents (created on first synthesis)
    analysis/{python,matlab}/  DSP analysis output
    outputs/<run_id>/        generated speech + metrics

"user1"/"user2" are just two default profile names; any number of profiles can exist.
"""
import json, os, re, shutil, tempfile, threading
from datetime import datetime, timezone
from pathlib import Path

import config

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_lock = threading.RLock()

# Files/folders created by enrollment; wiped whenever a new reference is uploaded.
DERIVED = ["processed.wav", "reference_clean.wav", "xtts_reference.wav", "speaker_embedding.npy",
           "chunk_embeddings.npy", "xtts_latents.pt", "analysis", "preview.png"]


class ProfileError(Exception):
    pass


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")[:32].strip("-")
    return s


def valid_slug(slug) -> bool:
    return isinstance(slug, str) and bool(SLUG_RE.match(slug))


def profile_dir(slug: str) -> Path:
    """Path of a profile folder. Raises on anything that is not a clean slug (path-traversal safe)."""
    if not valid_slug(slug):
        raise ProfileError("Invalid profile id.")
    return config.PROFILES_DIR / slug


def exists(slug) -> bool:
    return valid_slug(slug) and (profile_dir(slug) / "metadata.json").exists()


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def get(slug: str) -> dict:
    if not exists(slug):
        raise ProfileError(f"Profile '{slug}' does not exist.")
    return json.loads((profile_dir(slug) / "metadata.json").read_text(encoding="utf-8"))


def save(slug: str, meta: dict):
    with _lock:
        _write_json(profile_dir(slug) / "metadata.json", meta)


def update(slug: str, **changes) -> dict:
    """Shallow-merge top-level keys (dict values are merged one level deep)."""
    with _lock:
        meta = get(slug)
        for k, v in changes.items():
            if isinstance(v, dict) and isinstance(meta.get(k), dict):
                meta[k].update(v)
            else:
                meta[k] = v
        save(slug, meta)
        return meta


def create(name: str) -> dict:
    name = (name or "").strip()
    if not name or len(name) > 40:
        raise ProfileError("Profile name must be 1-40 characters.")
    slug = slugify(name)
    if not valid_slug(slug):
        raise ProfileError("Profile name must contain letters or digits.")
    with _lock:
        if exists(slug):
            raise ProfileError(f"A profile called '{slug}' already exists.")
        meta = {
            "slug": slug, "name": name, "created": now_iso(), "reference": None,
            "enrollment": {"status": "not_started"},
            "consent": "Voice used only with the speaker's permission.",
        }
        save(slug, meta)
    return meta


def delete(slug: str):
    with _lock:
        d = profile_dir(slug)
        if not d.exists():
            raise ProfileError(f"Profile '{slug}' does not exist.")
        shutil.rmtree(d)


def list_profiles() -> list[dict]:
    out = []
    if config.PROFILES_DIR.exists():
        for d in sorted(config.PROFILES_DIR.iterdir()):
            if d.is_dir() and (d / "metadata.json").exists() and valid_slug(d.name):
                try:
                    out.append(json.loads((d / "metadata.json").read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    continue
    return out


def ensure_default_profiles():
    """First run only: create two example profiles so the UI shows [User 1] [User 2]."""
    if config.SEED_DEFAULT_PROFILES and not list_profiles():
        for n in ("User 1", "User 2"):
            try:
                create(n)
            except ProfileError:
                pass


def clear_derived(slug: str):
    """Remove everything derived from the old reference (embeddings, latents, analysis).
    Prevents a new recording from ever being combined with a stale voice representation."""
    d = profile_dir(slug)
    for name in DERIVED:
        p = d / name
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()


def path(slug: str, *parts) -> Path:
    p = profile_dir(slug).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def is_enrolled(slug: str) -> bool:
    try:
        m = get(slug)
    except ProfileError:
        return False
    d = profile_dir(slug)
    return m.get("enrollment", {}).get("status") == "enrolled" and (d / "speaker_embedding.npy").exists() \
        and (d / "reference_clean.wav").exists()


def list_runs(slug: str) -> list[str]:
    out = profile_dir(slug) / "outputs"
    if not out.exists():
        return []
    return sorted([p.name for p in out.iterdir() if p.is_dir() and (p / "metrics.json").exists()], reverse=True)
