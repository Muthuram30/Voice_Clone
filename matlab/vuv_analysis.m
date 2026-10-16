function U = vuv_analysis(y, fs, V, F, outDir)
% Voiced / unvoiced / silence: 0 silence, 1 unvoiced speech, 2 voiced speech.
n = min(numel(V.mask), numel(F.f0));
lab = zeros(n, 1); lab(V.mask(1:n)) = 1; lab(V.mask(1:n) & F.voiced(1:n)) = 2;
U.labels = lab; sp = sum(lab >= 1);
U.voicedRatio = sum(lab == 2) / max(sp, 1); U.unvoicedRatio = sum(lab == 1) / max(sp, 1);
t = (0:n-1)' * V.hop / fs; b = max(floor(numel(y) / 3000), 1); k = floor(numel(y) / b);
z = reshape(y(1:k*b), b, k); x = ((0:k-1) * b) / fs;
f = figure('Visible', 'off'); hold on;
fill([x fliplr(x)], [max(z, [], 1) fliplr(min(z, [], 1))], [0.3 0.3 0.35], 'EdgeColor', 'none');
stairs(t, (lab == 2) * 0.9, 'Color', [0.22 0.63 0.35], 'LineWidth', 1);
stairs(t, -(lab == 1) * 0.9, 'Color', [0.87 0.42 0.13], 'LineWidth', 1);
xlabel('Time (s)'); title('Voiced (green, +) / unvoiced (orange, -) / silence'); ylim([-1 1]);
save_fig(f, fullfile(outDir, 'vad.png'));
end
