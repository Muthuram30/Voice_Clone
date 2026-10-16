"""All MATLAB access lives here.  MATLAB Engine API first, ``matlab -batch`` CLI fallback.

Interface
---------
Python writes a JSON **request** file to the output directory, invokes
``matlab_runner(requestJsonPath)`` (a single fixed MATLAB entry-point), waits
for the file to be overwritten with the **response** JSON, then returns the
parsed response to the caller.

Request / response schema
~~~~~~~~~~~~~~~~~~~~~~~~~
Request  (written by Python)::

    {
      "task"    : "analysis" | "postprocess",
      "wavPath" : "<absolute path to WAV>",
      "outDir"  : "<absolute output directory>",
      "params"  : {
          "targetFs"      : 16000,   // analysis only  (int)
          "hpHz"          : 70,      // postprocess     (float)
          "lpHz"          : 0,       // postprocess     (float, 0 = off)
          "targetRmsDbfs" : -20      // postprocess     (float)
      }
    }

Response (MATLAB overwrites the request file)::

    { "status": "ok",    "task": ..., "outDir": ..., ... <results.json fields> }
    { "status": "error", "task": ..., "outDir": ..., "message": "<reason>" }

MATLAB is OPTIONAL: nothing in the TTS demo depends on it.
``dsp_service.py`` implements the same DSP stages in pure Python.
"""
import json, os, shutil, subprocess, threading, time
from pathlib import Path

import config


class MatlabError(Exception):
    pass


_engine = None
_lock = threading.RLock()   # MATLAB engine calls are serialised


def _exe():
    return config.MATLAB_EXE or shutil.which("matlab") or ""


def available():
    """Cheap availability check (does NOT start MATLAB).

    Returns a dict with keys: ``engine_api``, ``cli``, ``any``, ``mode``.
    """
    try:
        import importlib.util
        eng = (importlib.util.find_spec("matlab") is not None and
               importlib.util.find_spec("matlab.engine") is not None)
    except Exception:          # noqa: BLE001
        eng = False
    exe = bool(_exe()) and Path(_exe()).exists() if config.MATLAB_EXE else bool(shutil.which("matlab"))
    return dict(engine_api=eng, cli=exe, any=eng or exe, mode=config.MATLAB_MODE)


def _get_engine():
    global _engine
    if _engine is None:
        try:
            import matlab.engine
        except ImportError:
            raise MatlabError("MATLAB Engine API for Python is not installed.")
        try:
            _engine = matlab.engine.start_matlab(
                "-desktop" if config.MATLAB_DESKTOP == "1" else "")
            _engine.addpath(str(config.MATLAB_DIR), nargout=0)
        except Exception as e:     # noqa: BLE001
            raise MatlabError(f"Could not start MATLAB engine: {e}")
    return _engine


class _EngineRunError(MatlabError):
    """MATLAB started but the .m code failed (do not mask by falling back to CLI)."""


def _run_engine(request_path: str):
    with _lock:
        eng = _get_engine()
        try:
            eng.matlab_runner(request_path, nargout=0)
        except Exception as e:     # noqa: BLE001
            raise _EngineRunError(f"MATLAB engine error in matlab_runner: {e}")


def _lit(a):
    """Format a Python value as a MATLAB literal for -batch command strings."""
    return "'" + str(a).replace("'", "''") + "'" if isinstance(a, (str, Path)) else repr(float(a))


def _run_cli(request_path: str):
    exe = _exe()
    if not exe:
        raise MatlabError(
            "matlab executable not found. Add it to PATH or set MATLAB_EXE in .env.")
    cmd = (f"addpath({_lit(config.MATLAB_DIR)});"
           f"matlab_runner({_lit(request_path)});")
    with _lock:
        try:
            r = subprocess.run(
                [exe, "-batch", cmd],
                capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            raise MatlabError("MATLAB CLI timed out after 15 minutes.")
    if r.returncode != 0:
        raise MatlabError(f"MATLAB CLI failed: {(r.stderr or r.stdout)[-600:]}")


def _invoke_runner(request: dict, out: Path) -> dict:
    """Write the JSON request file, call matlab_runner, read back the response.

    Uses Engine API when available; falls back to ``-batch`` CLI otherwise.
    Raises :class:`MatlabError` on any failure.
    """
    out.mkdir(parents=True, exist_ok=True)

    request_path = out / "_matlab_request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    mode = config.MATLAB_MODE
    if mode == "cli":
        _run_cli(str(request_path))
    elif mode == "engine":
        _run_engine(str(request_path))
    else:                          # "auto" – Engine first, CLI fallback
        try:
            _run_engine(str(request_path))
        except _EngineRunError:
            raise                  # real .m failure – surface immediately
        except MatlabError:
            _run_cli(str(request_path))   # engine not installed / failed to start

    # Read response (matlab_runner overwrites the request file in-place)
    if not request_path.exists():
        raise MatlabError(
            "MATLAB finished but the request/response file is missing.")

    try:
        res = json.loads(request_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise MatlabError(f"Could not parse MATLAB response JSON: {e}")

    if res.get("status") != "ok":
        raise MatlabError(res.get("message", "Unknown MATLAB error"))

    return res


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_analysis(wav, out, fs: int = 16000) -> dict:
    """Run the full DSP analysis pipeline via ``matlab/matlab_runner.m``.

    Calls ``main_analysis(wavPath, outDir, targetFs)`` inside MATLAB.
    Writes PNGs, ``processed.wav``, ``features.json``, ``results.json``
    into *out*, and returns the parsed ``results.json`` dict.

    Parameters
    ----------
    wav : str | Path
        Absolute path to the input WAV file.
    out : str | Path
        Directory that will receive all output artefacts.
    fs : int
        Analysis sample rate (default 16 000 Hz).
    """
    request = {
        "task":    "analysis",
        "wavPath": str(Path(wav).resolve()),
        "outDir":  str(Path(out).resolve()),
        "params":  {"targetFs": int(fs)},
    }
    return _invoke_runner(request, Path(out))


def run_postprocess(wav, out, hp_hz: float = 70, lp_hz: float = 0,
                    target_rms_dbfs: float = -20) -> dict:
    """Run post-processing + compression experiment via ``matlab/matlab_runner.m``.

    Calls ``main_postprocess(wavPath, outDir, hpHz, lpHz, targetRmsDb)``
    inside MATLAB.  Writes ``final_matlab.wav``, PNGs, and ``results.json``
    into *out*, returns the parsed result dict.

    Parameters
    ----------
    wav : str | Path
        Absolute path to the generated speech WAV file.
    out : str | Path
        Directory for output artefacts.
    hp_hz : float
        High-pass cut-off frequency in Hz (default 70 Hz, 0 = off).
    lp_hz : float
        Low-pass cut-off frequency in Hz (default 0 = off).
    target_rms_dbfs : float
        Target RMS level in dBFS (default −20 dBFS).
    """
    request = {
        "task":    "postprocess",
        "wavPath": str(Path(wav).resolve()),
        "outDir":  str(Path(out).resolve()),
        "params":  {
            "hpHz":         float(hp_hz),
            "lpHz":         float(lp_hz),
            "targetRmsDbfs": float(target_rms_dbfs),
        },
    }
    return _invoke_runner(request, Path(out))
