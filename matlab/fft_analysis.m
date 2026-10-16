function A = fft_analysis(raw, y, fs, U, H, outDir)
% FFT of the strongest voiced frame (2048 samples) + Welch long-term spectrum (raw vs processed).
N = 2048; lab = U.labels; e = movmean(y.^2, N);
c = min((0:numel(lab)-1)' * H + 1, numel(y)); sc = e(c); sc(lab ~= 2) = -1;
[mx, i0] = max(sc); if mx <= 0, i0 = max(1, round(numel(lab) / 2)); end
s = max(1, min(numel(y) - N + 1, c(i0) - N/2)); seg = y(s:s+N-1) .* hann(N);
Fm = 20 * log10(abs(fft(seg)) + 1e-9); Fm = Fm(1:N/2+1); fr = (0:N/2)' * fs / N;
[p1, f1] = pwelch(raw, hann(2048), 1024, 2048, fs); [p2, f2] = pwelch(y, hann(2048), 1024, 2048, fs);
f = figure('Visible', 'off', 'Position', [100 100 1000 380]);
subplot(1, 2, 1); plot(fr, Fm); xlabel('Frequency (Hz)'); ylabel('dB'); title('FFT of strongest voiced frame');
subplot(1, 2, 2); plot(f1, 10*log10(p1 + 1e-14)); hold on; plot(f2, 10*log10(p2 + 1e-14));
xlabel('Frequency (Hz)'); title('Long-term average spectrum (Welch)'); legend('original', 'processed');
save_fig(f, fullfile(outDir, 'fft.png'));
[~, k] = max(Fm(6:end)); A.framePeakHz = fr(k + 5);
end
