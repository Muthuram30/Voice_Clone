"""Regression checks for phrasing, assembly, continuous references and signal quality."""
from types import SimpleNamespace

import numpy as np
import pytest

from services import gemini_service as G, speech_quality as SQ, speaker_service as S
from services import dsp_service as D, tts_service as T


def test_artificial_text_cuts_do_not_add_full_stops_or_lose_text():
    text = ' '.join(['Natural phrasing should continue across model length limits'] * 8) + '.'
    fragments = G.split_sentences(text, 95)
    assert len(fragments) > 1
    assert ' '.join(fragments) == text
    assert sum(part.count('.') for part in fragments) == 1
    assert not any(',. ' in part for part in fragments)


def test_nearby_sentences_keep_context_and_long_passages_keep_punctuation():
    assert SQ.pack_passages(['Hello.', 'How are you?'], 200) == ['Hello. How are you?']
    text = ' '.join(['Speak clearly, keep the rhythm and preserve the words.'] * 9)
    packed = SQ.pack_passages([text], 120)
    assert all(len(part) <= 120 for part in packed)
    assert ' '.join(packed) == text
    unicode_text = 'あいうえお' * 30
    assert ''.join(SQ.pack_passages([unicode_text], 71)) == unicode_text


def test_token_budget_is_respected_without_hiding_words():
    tokenizer = SimpleNamespace(char_limits={'en': 200}, encode=lambda text, lang: list(text))
    model = SimpleNamespace(tokenizer=tokenizer, args=SimpleNamespace(gpt_max_text_tokens=40))
    text = 'A long passage with every original word retained for the listener.'
    parts = T._passages(model, [text], 'en')
    assert ' '.join(parts) == text
    assert all(len(part) < 40 for part in parts)


def test_join_preserves_model_pauses_without_adding_another_gap():
    fs = 1000
    wave = .1 * np.sin(2 * np.pi * 100 * np.arange(500) / fs)
    a = np.concatenate((wave, np.zeros(250)))
    b = np.concatenate((np.zeros(250), wave))
    joined, gaps = SQ.join_passages([a, b], ['First.', 'Second.'], fs)
    assert gaps == [0.]
    assert len(joined) == len(a) + len(b)
    assert joined[len(a)-250:len(a)+250].max() == 0


def test_join_smooths_abrupt_edges_and_rejects_one_bad_passage():
    fs = 24000
    wave = np.full(fs // 10, .25, dtype=np.float32)
    joined, gaps = SQ.join_passages([wave, -wave], ['A phrase,', 'continued.'], fs)
    assert gaps == [.08]
    assert joined[0] == joined[-1] == 0
    assert np.max(np.abs(np.diff(joined))) < .01
    with pytest.raises(ValueError, match='invalid passage'):
        SQ.join_passages([wave, np.zeros(100)], ['A.', 'B.'], fs)


def test_quality_reports_defects_without_certifying_naturalness():
    fs = 1000
    audio = np.concatenate((np.full(500, .1), np.zeros(1800), np.full(500, .1)))
    report = SQ.inspect_audio(audio, fs)
    assert report['longest_internal_pause_s'] == 1.8
    assert report['warnings'] and not report['artifact_checks_passed']
    assert 'Requires listening' in report['naturalness']
    assert SQ.inspect_audio(np.full(1000, 1.), fs)['clipped_fraction'] == 1.
    with pytest.raises(ValueError):
        SQ.inspect_audio(np.array([float('nan')]), fs)


def test_reference_windows_are_contiguous_nonoverlapping_and_avoid_silence():
    fs = 1000
    y = np.zeros(45 * fs, dtype=np.float32)
    y[5 * fs:20 * fs] = .05
    y[25 * fs:40 * fs] = .1
    windows = S.select_reference_windows(y, fs, [(5, 20), (25, 40)])
    assert len(windows) == 2
    assert all(b-a == 12 * fs for a, b in windows)
    assert windows[0][1] <= windows[1][0] or windows[1][1] <= windows[0][0]
    assert all(np.mean(np.abs(y[a:b]) > 0) > .9 for a, b in windows)
    with pytest.raises(S.EnrollmentError, match='No clean continuous'):
        S.select_reference_windows(np.zeros(30 * fs), fs, [(0, 30)])


def test_cuda_filters_remain_linear_above_full_scale():
    fs = 16000
    t = np.arange(fs) / fs
    small = (.3 * np.sin(2 * np.pi * 120 * t) + .1).astype(np.float32)
    a = D.GPU.host(D.GPU.filter_audio(small, fs, hp=70))
    b = D.GPU.host(D.GPU.filter_audio(small * 8, fs, hp=70))
    assert np.allclose(b, a * 8, atol=2e-4)
    assert np.max(np.abs(b)) > 1.


def test_synthesis_uses_model_prosody_settings_and_one_global_headroom_gain(monkeypatch):
    import config
    monkeypatch.setattr(T.PS, 'is_enrolled', lambda slug: True)
    monkeypatch.setattr(config, 'SENTENCE_MAX_CHARS', 30)
    model = SimpleNamespace(
        config=SimpleNamespace(temperature=.75, top_k=50, top_p=.85, repetition_penalty=5.),
        tokenizer=SimpleNamespace(char_limits={'en': 200}, encode=lambda text, lang: list(text)),
        args=SimpleNamespace(gpt_max_text_tokens=400))
    monkeypatch.setattr(T, '_load', lambda: SimpleNamespace(synthesizer=SimpleNamespace(tts_model=model, output_sample_rate=1000)))
    monkeypatch.setattr(T, '_latents', lambda *args: (None, None, 'test'))
    monkeypatch.setattr(T, '_device', 'cuda:0')
    calls = []

    def inference(model, text, language, gpt, speaker, settings):
        calls.append(dict(settings))
        return len(calls) * .8 * np.sin(2 * np.pi * 100 * np.arange(1000) / 1000)

    monkeypatch.setattr(T, '_run_inference', inference)
    wave, fs, report = T.synthesize('fixture', ['First passage carries words.', 'Second passage has words.'])
    assert len(calls) == report['passages'] == 2
    assert calls[0] == calls[1] == dict(temperature=.75, top_k=50, top_p=.85, repetition_penalty=5., do_sample=True, speed=1.)
    assert fs == 1000 and np.max(np.abs(wave)) <= .98001
    assert report['headroom_gain_db'] < 0
    assert np.isclose(np.max(np.abs(wave[:1000])) / np.max(np.abs(wave[-1000:])), .5, atol=.01)
