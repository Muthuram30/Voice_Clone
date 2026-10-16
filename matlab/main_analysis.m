function main_analysis(wavPath, outDir, targetFs, showFigures)
% DSP analysis pipeline - dispatched by matlab_runner (the single Python-facing entry-point).
%   main_analysis('voice.wav', 'out', 16000)
% Writes PNGs, processed.wav, features.json and results.json (same contract as dsp_service.analyze).
% Called via: Python -> matlab_service.run_analysis -> matlab_runner -> main_analysis
% Toolboxes: Signal Processing Toolbox only. MATLAB R2020a+ recommended (R2019a minimum: stft/istft).
if nargin < 3, targetFs = 16000; end
if nargin < 4, showFigures = false; end
if ~exist(outDir, 'dir'), mkdir(outDir); end
setappdata(0, 'DSP_SHOW_FIGURES', logical(showFigures));
try
    tic;
    need = {'resample', 'butter', 'filtfilt', 'stft', 'istft', 'spectrogram', 'lpc', 'dct', 'pwelch', 'medfilt1'};
    miss = need(cellfun(@(f) isempty(which(f)), need));
    if ~isempty(miss), error('Signal Processing Toolbox functions missing: %s', strjoin(miss, ', ')); end
    P = preprocess_audio(wavPath, targetFs); fs = P.fs; y = P.processed;
    V = vad_analysis(y, fs); H = V.hop;
    F = pitch_analysis(y, fs, outDir);
    U = vuv_analysis(y, fs, V, F, outDir);
    A = fft_analysis(P.raw, y, fs, U, H, outDir);
    S = stft_analysis(y, fs, H, outDir);
    M = mfcc_analysis(y, fs, H, outDir);
    R = lpc_formants(P.emph, fs, U, H, outDir);
    [vec, names] = extract_features(M, F, S, R, V, U);
    plot_waveform_local(P.raw, y, fs, V, fullfile(outDir, 'waveform.png'));
    audiowrite(fullfile(outDir, 'processed.wav'), max(min(y, 1), -1), fs);
    n = min(numel(V.mask), numel(F.f0)); sp = V.mask(1:n); f0 = F.f0(~isnan(F.f0));
    st = struct('original_fs_hz', P.origFs, 'channels', P.channels, 'analysis_fs_hz', fs, ...
        'duration_s', round(P.duration, 2), 'speech_ratio', round(V.speechRatio, 3), ...
        'speech_seconds', round(sum(sp) * H / fs, 2), 'voiced_ratio_of_speech', round(U.voicedRatio, 3), ...
        'unvoiced_ratio_of_speech', round(U.unvoicedRatio, 3), 'f0_mean_hz', nz(mean(f0)), 'f0_std_hz', nz(std(f0, 1)), ...
        'f0_median_hz', nz(median(f0)), 'spectral_centroid_hz', nz(mean(S.centroid(sp(1:min(n, numel(S.centroid)))))), ...
        'formants_F1_F2_F3_hz', round(R.median, 2), 'formant_frames', R.frames, ...
        'rms_raw_dbfs', round(20*log10(rms_(P.raw) + 1e-12), 2), 'rms_processed_dbfs', round(20*log10(rms_(y) + 1e-12), 2), ...
        'snr_estimate_db', snr_est(V), 'fft_frame_peak_hz', round(A.framePeakHz, 2), ...
        'feature_vector_length', numel(vec), 'runtime_s', round(toc, 2));
    keys = {'waveform', 'spectrogram', 'fft', 'vad', 'mfcc', 'pitch', 'formants'};
    titles = {'Original vs processed waveform', 'Spectrogram (STFT)', 'FFT / long-term spectrum', 'Voiced / unvoiced / silence', 'MFCC', 'Pitch (F0)', 'Formants (LPC)'};
    plots = cellfun(@(k, t) struct('key', k, 'title', t, 'file', [k '.png']), keys, titles, 'UniformOutput', false);
    plots = [plots{:}];
    ft = struct(); ft.names = names; ft.values = vec(:)';     % (struct('names',{cell}) would create a struct array)
    write_json(fullfile(outDir, 'features.json'), ft);
    write_json(fullfile(outDir, 'results.json'), struct('status', 'ok', 'backend', 'matlab', 'stats', st, 'plots', plots));
catch err
    setappdata(0, 'DSP_SHOW_FIGURES', false);
    write_json(fullfile(outDir, 'results.json'), struct('status', 'error', 'message', err.message));
    rethrow(err);
end
setappdata(0, 'DSP_SHOW_FIGURES', false);
end
function r = rms_(x), r = sqrt(mean(x.^2)); end
function v = nz(x), if isempty(x) || ~isfinite(x), v = NaN; else, v = round(x, 2); end, end
function v = snr_est(V)
p = 10 .^ (V.eDb / 10); m = V.mask;
if sum(m) == 0 || sum(~m) < 5, v = NaN; else, v = round(10 * log10(max(mean(p(m)) - mean(p(~m)), 1e-12) / (mean(p(~m)) + 1e-12)), 2); end
end
function write_json(path, s)
fid = fopen(path, 'w'); fwrite(fid, jsonencode(s), 'char'); fclose(fid);
end
function plot_waveform_local(raw, y, fs, V, path)
f = figure('Visible', 'off', 'Position', [100 100 1000 480]);
for k = 1:2
    subplot(2, 1, k); z = {raw, y}; s = z{k}; b = max(floor(numel(s) / 3000), 1); m = floor(numel(s) / b);
    q = reshape(s(1:m*b), b, m); x = ((0:m-1) * b) / fs; fill([x fliplr(x)], [max(q, [], 1) fliplr(min(q, [], 1))], [0.17 0.42 0.69], 'EdgeColor', 'none');
    xlim([0 numel(s)/fs]); ylabel('Amplitude'); if k == 1, title('Original (mono, resampled)'); else, title('Processed (filtered + denoised + normalised)'); xlabel('Time (s)'); end
end
subplot(2, 1, 2); hold on; t = (0:numel(V.mask)-1)' * V.hop / fs; stairs(t, V.mask * 0.9, 'Color', [0.22 0.63 0.35]);
save_fig(f, path);
end
