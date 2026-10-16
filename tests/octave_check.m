% Usage (from the project root):  OCT_WAV=processed.wav OCT_OUT=/tmp octave-cli tests/octave_check.m
% Run MATLAB DSP modules under Octave on an already-processed WAV and dump features for comparison with Python.
pkg load signal;
addpath(fullfile(pwd, 'matlab')); addpath(fullfile(pwd, 'tests', 'octave_shims'));
wav = getenv('OCT_WAV'); if isempty(wav), error('set OCT_WAV to a (preferably already processed) mono WAV'); end
[x, fs] = audioread(wav); y = x(:, 1); out = tempname(); if ~exist(out, 'dir'), mkdir(out); end
V = vad_analysis(y, fs); H = V.hop;
F = pitch_analysis(y, fs, out); U = vuv_analysis(y, fs, V, F, out);
emph = filter([1 -0.97], 1, y);
R = lpc_formants(emph, fs, U, H, out);
S = stft_analysis(y, fs, H, out); M = mfcc_analysis(y, fs, H, out);
[vec, names] = extract_features(M, F, S, R, V, U);
cmp = struct('speech_ratio', V.speechRatio, 'regions', size(V.regions, 1), 'voiced_ratio', U.voicedRatio, ...
    'f0_mean', mean(F.f0(~isnan(F.f0))), 'f0_std', std(F.f0(~isnan(F.f0)), 1), 'formants', R.median, 'formant_frames', R.frames, ...
    'n_features', numel(vec), 'nan_features', sum(isnan(vec)));
cmp.vec = vec(:)'; cmp.names = {names};
fid = fopen(fullfile(getenv('OCT_OUT'), 'oct_features.json'), 'w'); fwrite(fid, jsonencode(cmp), 'char'); fclose(fid); disp('OCTAVE_DONE');
