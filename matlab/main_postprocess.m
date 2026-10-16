function main_postprocess(wavPath, outDir, hpHz, lpHz, targetRmsDb)
% Post-process generated speech - dispatched by matlab_runner (the single Python-facing entry-point).
% Called via: Python -> matlab_service.run_postprocess -> matlab_runner -> main_postprocess
% Writes final_matlab.wav, spectrum/spectrogram plots, compression experiment PNGs, and results.json.
if nargin < 3, hpHz = 70; end, if nargin < 4, lpHz = 0; end, if nargin < 5, targetRmsDb = -20; end
if ~exist(outDir, 'dir'), mkdir(outDir); end
try
    tic; [x, fs] = audioread(wavPath); y = mean(x, 2);
    [out, m] = postprocess_audio(y, fs, hpHz, lpHz, targetRmsDb);
    audiowrite(fullfile(outDir, 'final_matlab.wav'), max(min(out, 1), -1), fs);
    [p1, f1] = pwelch(y, hann(2048), 1024, 2048, fs); [p2, f2] = pwelch(out, hann(2048), 1024, 2048, fs);
    f = figure('Visible', 'off'); plot(f1, 10*log10(p1 + 1e-14)); hold on; plot(f2, 10*log10(p2 + 1e-14));
    xlabel('Frequency (Hz)'); ylabel('dB'); title('Spectrum before / after post-processing'); legend('generated', 'final');
    save_fig(f, fullfile(outDir, 'spectrum_compare.png'));
    f = figure('Visible', 'off'); spectrogram(out, hann(512), 384, 1024, fs, 'yaxis'); title('Spectrogram of final speech');
    save_fig(f, fullfile(outDir, 'spectrogram.png'));
    C = compression_analysis(out, fs, outDir);
    r = C.rows(strcmp({C.rows.codec}, 'STFT-threshold') & abs([C.rows.keep_fraction] - 0.3) < 1e-9);
    m.compression_stft_keep30_ratio_entropy_est = r.ratio_entropy_est; m.compression_stft_keep30_snr_db = r.snr_db;
    m.compression_stft_keep30_lsd_db = r.lsd_db; m.sample_rate_hz = fs; m.duration_s = numel(y) / fs; m.runtime_s = toc;
    keys = {'spectrum_compare', 'spectrogram', 'compression_rd'}; titles = {'Spectrum before/after', 'Spectrogram', 'Rate-distortion'};
    plots = cellfun(@(k, t) struct('key', k, 'title', t, 'file', [k '.png']), keys, titles, 'UniformOutput', false); plots = [plots{:}];
    res = struct('status', 'ok', 'backend', 'matlab', 'stats', m, 'plots', plots, 'compression', C.rows);
    fid = fopen(fullfile(outDir, 'results.json'), 'w'); fwrite(fid, jsonencode(res), 'char'); fclose(fid);
catch err
    fid = fopen(fullfile(outDir, 'results.json'), 'w'); fwrite(fid, jsonencode(struct('status', 'error', 'message', err.message)), 'char'); fclose(fid);
    rethrow(err);
end
end
