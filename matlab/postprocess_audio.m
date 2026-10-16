function [out, m] = postprocess_audio(y, fs, hpHz, lpHz, targetRmsDb)
% DC removal -> high-pass -> optional low-pass -> 10 ms fades -> RMS normalisation + peak limit (0.97).
% Toolbox: Signal Processing (butter, filtfilt). Metrics mirror dsp_service.postprocess.
y = y(:); out = y - mean(y);
if hpHz > 0, [b, a] = butter(2, hpHz / (fs/2), 'high'); out = filtfilt(b, a, out); end
if lpHz > 0 && lpHz < fs/2, [b, a] = butter(4, lpHz / (fs/2), 'low'); out = filtfilt(b, a, out); end
f = min(round(0.01 * fs), floor(numel(out) / 4));
if f > 0, out(1:f) = out(1:f) .* linspace(0, 1, f)'; out(end-f+1:end) = out(end-f+1:end) .* linspace(1, 0, f)'; end
gain = 10^(targetRmsDb / 20) / (sqrt(mean(out.^2)) + 1e-12); out = out * gain;
pk = max(abs(out)) + 1e-12; if pk > 0.97, out = out * 0.97 / pk; end
g = (y' * out) / (y' * y + 1e-20); ref = g * y;
m.rms_before_dbfs = 20 * log10(sqrt(mean(y.^2)) + 1e-12); m.rms_after_dbfs = 20 * log10(sqrt(mean(out.^2)) + 1e-12);
m.peak_before = max(abs(y)); m.peak_after = max(abs(out)); m.dc_offset_before = mean(y);
m.applied_gain_db = 20 * log10(gain);
m.snr_vs_input_gain_matched_db = 10 * log10((sum(ref.^2) + 1e-20) / (sum((ref - out).^2) + 1e-20));
A = abs(spectrogram(ref, hann(512), 384, 512, fs)).^2; B = abs(spectrogram(out, hann(512), 384, 512, fs)).^2;
fl = 1e-8 * max(max(A(:)), max(B(:))); d = 10 * log10(A + fl) - 10 * log10(B + fl);
m.lsd_vs_input_gain_matched_db = mean(sqrt(mean(d.^2, 1)));
end
