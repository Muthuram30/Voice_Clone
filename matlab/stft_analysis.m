function S = stft_analysis(y, fs, H, outDir)
% Spectrogram plot (512-pt Hann, hop 128) and 10 ms-grid spectral features (centroid, 85 % roll-off, ZCR).
[Sp, Fq, T] = spectrogram(y, hann(512), 384, 512, fs);
D = 20 * log10(abs(Sp) + 1e-9); D = max(D, max(D(:)) - 80);
f = figure('Visible', 'off'); imagesc(T, Fq, D); axis xy; colormap(magma_like()); colorbar;
xlabel('Time (s)'); ylabel('Frequency (Hz)'); title('Spectrogram (STFT, 512-pt Hann)');
save_fig(f, fullfile(outDir, 'spectrogram.png'));
yp = [zeros(256, 1); y(:); zeros(256, 1)];
G = abs(spectrogram(yp, hann(512, 'periodic'), 512 - H, 512, fs)); Fq = (0:256)' * fs / 512;
S.centroid = (sum(G .* Fq, 1) ./ (sum(G, 1) + eps))';
cs = cumsum(G, 1); [~, idx] = max(cs >= 0.85 * cs(end, :), [], 1); S.rolloff = Fq(idx);
fr = frame_signal(y, 2 * round(fs * 0.025 / 2), H);
S.zcr = (sum(abs(diff(sign(fr), 1, 1)) > 0, 1) / size(fr, 1))';
end
function c = magma_like()
c = [linspace(0, 1, 64)', linspace(0, 0.9, 64)'.^1.5, linspace(0.2, 0.6, 64)'];
end
