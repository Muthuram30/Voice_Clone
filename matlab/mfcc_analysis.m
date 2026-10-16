function M = mfcc_analysis(y, fs, H, outDir)
% 13 MFCC (+ deltas) on the 10 ms grid. Own HTK-mel filterbank (40 bands, 512-pt FFT), power -> dB, orthonormal DCT.
% Toolbox: Signal Processing (spectrogram, dct). No Audio Toolbox needed.
nfft = 512; nmel = 40;
yp = [zeros(nfft/2, 1); y(:); zeros(nfft/2, 1)];
P = abs(spectrogram(yp, hann(nfft, 'periodic'), nfft - H, nfft, fs)).^2;      % 257 x n
fb = mel_filterbank(fs, nfft, nmel);
lm = 10 * log10(max(fb * P, 1e-10)); lm = max(lm, max(lm(:)) - 80);
c = dct(lm); c = c(1:13, :);                                                   % 13 x n
M.coeffs = c; M.delta = delta_feat(c, 4);
M.hop = H;
f = figure('Visible', 'off'); imagesc((0:size(c, 2)-1) * H / fs, 0:12, c); axis xy; colorbar;
xlabel('Time (s)'); ylabel('MFCC index'); title('MFCC (13 coefficients)');
save_fig(f, fullfile(outDir, 'mfcc.png'));
end
function fb = mel_filterbank(fs, nfft, nmel)
mel = @(f) 2595 * log10(1 + f / 700); imel = @(m) 700 * (10.^(m / 2595) - 1);
edges = imel(linspace(mel(0), mel(fs/2), nmel + 2)); fq = (0:nfft/2) * fs / nfft;
fb = zeros(nmel, nfft/2 + 1);
for i = 1:nmel
    l = edges(i); c = edges(i+1); r = edges(i+2);
    fb(i, :) = max(0, min((fq - l) / (c - l), (r - fq) / (r - c)));
end
end
function d = delta_feat(c, W)
n = size(c, 2); cp = [repmat(c(:, 1), 1, W), c, repmat(c(:, end), 1, W)]; d = zeros(size(c));
for k = 1:W, d = d + k * (cp(:, W+1+k:W+n+k) - cp(:, W+1-k:W+n-k)); end
d = d / (2 * sum((1:W).^2));
end
