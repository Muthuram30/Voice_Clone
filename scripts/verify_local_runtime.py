"""Manual check with real XTTS; creates output runs in the named profiles.

Run: .venv\\Scripts\\python.exe scripts\\verify_local_runtime.py --profile <slug>
The named profiles must already be enrolled. Gemini is never used.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from app import create_app
from services import profile_service as PS, synthesis_service as SY

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--profile', action='append', required=True, help='Enrolled slug; repeat to check multiple voices')
parser.add_argument('--text', default='Hello. This is a local voice studio check.')
parser.add_argument('--language', default='en')
args = parser.parse_args()

app = create_app()
app.config['TESTING'] = True
client = app.test_client()
for slug in args.profile:
    if not PS.is_enrolled(slug):
        raise RuntimeError(f'Profile {slug!r} must be enrolled before this check.')
    before = set(PS.list_runs(slug))
    response = client.post(f'/profiles/{slug}/synthesize', data={
        'text': args.text, 'language': args.language,
        'backend': 'xtts', 'hp_hz': '70', 'target_rms': '-20',
    }, follow_redirects=True)
    new = set(PS.list_runs(slug)) - before
    complete = [SY.load_run(slug, rid) for rid in new if (PS.profile_dir(slug) / 'outputs' / rid / 'metrics.json').is_file()]
    if response.status_code != 200 or not complete:
        from html.parser import HTMLParser
        class Text(HTMLParser):
            def handle_data(self, value):
                if value.strip():
                    print(value.strip())
        Text().feed(response.get_data(as_text=True))
        raise RuntimeError(f'Real synthesis failed for {slug}; HTTP {response.status_code}')
    for metrics in complete:
        assert metrics['tts']['device'].startswith('cuda')
        assert metrics['tts']['model'] != 'TEST-DOUBLE'
        report = {key: metrics[key] for key in ('profile', 'run_id', 'tts', 'audio_quality', 'verification', 'total_time_s')}
        assert metrics['audio_quality']['clipped_fraction'] == 0, 'Generated output contains clipped samples'
        for fmt in ('wav', 'flac', 'mp3', 'ogg', 'opus', 'm4a', 'aiff', 'zip'):
            exported = client.get(f"/profiles/{slug}/export/{metrics['run_id']}/{fmt}")
            assert exported.status_code == 200, (slug, fmt, exported.status_code)
        print('REAL_SYNTHESIS_OK', json.dumps(report), flush=True)
