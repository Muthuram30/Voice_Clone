function C = compression_analysis(y, fs, outDir, keeps)
% Rate-distortion experiment: STFT-threshold codec and block-DCT-threshold codec.
% Keep the largest `keep` fraction of coefficients, quantise log-magnitude (+ phase for STFT, + sign for DCT),
% reconstruct from the quantised values, measure SNR / log-spectral distance. Size estimates:
%   fixed_bits   = bitmap + n_kept * bits per coefficient
%   entropy_bits = Shannon-entropy estimate of the bitmap and symbol streams (ideal entropy coder)
% (The Python version additionally measures real zlib bytes.)   Toolbox: Signal Processing (stft, istft, dct)
if nargin < 4, keeps = [0.05 0.1 0.2 0.3 0.5]; end
y = y(:); N = numel(y); origBits = N * 16; rows = struct([]);
for codec = 1:2
    for keep = keeps
        if codec == 1, [rec, fixed, ent, nk] = stft_codec(y, fs, keep, 6, 5); name = 'STFT-threshold';
        else,          [rec, fixed, ent, nk] = dct_codec(y, keep, 7);        name = 'DCT-threshold'; end
        rec = rec(:); rec(end+1:N) = 0; rec = rec(1:N);
        r = struct('codec', name, 'keep_fraction', keep, 'kept_coefficients', nk, 'ratio_fixed_length', origBits / fixed, ...
            'ratio_entropy_est', origBits / max(ent, 1), 'snr_db', snr_db(y, rec), 'lsd_db', lsd_db(y, rec, fs));
        if isempty(rows), rows = r; else, rows(end+1) = r; end %#ok<AGROW>
    end
end
C.rows = rows;
f = figure('Visible', 'off', 'Position', [100 100 1000 380]);
for k = 1:2
    subplot(1, 2, k); hold on;
    for nm = {'STFT-threshold', 'DCT-threshold'}
        s = rows(strcmp({rows.codec}, nm{1}));
        if k == 1, plot([s.ratio_entropy_est], [s.snr_db], 'o-'); else, plot([s.ratio_entropy_est], [s.lsd_db], 'o-'); end
    end
    xlabel('Compression ratio (entropy estimate)'); if k == 1, ylabel('SNR (dB)'); else, ylabel('Log-spectral distance (dB)'); end
    legend('STFT-threshold', 'DCT-threshold');
end
save_fig(f, fullfile(outDir, 'compression_rd.png'));
end

function [rec, fixed, ent, nk] = stft_codec(y, fs, keep, mb, pb)
w = hann(512, 'periodic');
S = stft(y, fs, 'Window', w, 'OverlapLength', 384, 'FFTLength', 512, 'FrequencyRange', 'onesided');
mag = abs(S); ph = angle(S); k = max(1, round(keep * numel(mag)));
s = sort(mag(:), 'descend'); m = mag >= s(k); nk = nnz(m); mv = mag(m);
lo = 20 * log10(max(min(mv), 1e-9)); hi = 20 * log10(max(max(mv), 1e-9));
qm = qlog(mv, mb, lo, hi); lv = 2^pb; qp = min(max(floor((ph(m) + pi) / (2*pi) * lv), 0), lv - 1);
Sh = zeros(size(S)); Sh(m) = dqlog(qm, mb, lo, hi) .* exp(1j * (-pi + (qp + 0.5) * 2*pi / lv));
rec = real(istft(Sh, fs, 'Window', w, 'OverlapLength', 384, 'FFTLength', 512, 'FrequencyRange', 'onesided'));
fixed = numel(m) + nk * (mb + pb); ent = ent_bits(double(m(:))) + ent_bits(qm) + ent_bits(qp);
end

function [rec, fixed, ent, nk] = dct_codec(y, keep, mb)
B = 512; n = numel(y); nb = ceil(n / B); X = dct(reshape([y; zeros(nb * B - n, 1)], B, nb));
mag = abs(X); k = max(1, round(keep * numel(mag))); s = sort(mag(:), 'descend'); m = mag >= s(k); nk = nnz(m); mv = mag(m);
lo = 20 * log10(max(min(mv), 1e-9)); hi = 20 * log10(max(max(mv), 1e-9));
qm = qlog(mv, mb, lo, hi); Xh = zeros(size(X)); v = dqlog(qm, mb, lo, hi); sg = X(m) < 0; v(sg) = -v(sg); Xh(m) = v;
rec = reshape(idct(Xh), [], 1); rec = rec(1:n);
fixed = numel(m) + nk * (mb + 1); ent = ent_bits(double(m(:))) + ent_bits(qm) + nk;
end

function q = qlog(mag, bits, lo, hi)
db = 20 * log10(max(mag, 1e-9)); q = round((min(max(db, lo), hi) - lo) / max(hi - lo, 1e-9) * (2^bits - 1));
end
function mag = dqlog(q, bits, lo, hi), mag = 10 .^ ((lo + q / (2^bits - 1) * (hi - lo)) / 20); end
function b = ent_bits(sym)
if isempty(sym), b = 0; return; end
[~, ~, ic] = unique(sym); c = accumarray(ic, 1); p = c / sum(c); b = -sum(c .* log2(p));
end
function s = snr_db(a, b), s = 10 * log10((sum(a.^2) + 1e-20) / (sum((a - b).^2) + 1e-20)); end
function d = lsd_db(a, b, fs)
A = abs(spectrogram(a, hann(512), 384, 512, fs)).^2; B = abs(spectrogram(b, hann(512), 384, 512, fs)).^2;
fl = 1e-8 * max(max(A(:)), max(B(:))); dd = 10 * log10(A + fl) - 10 * log10(B + fl); d = mean(sqrt(mean(dd.^2, 1)));
end
