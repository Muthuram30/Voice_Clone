function R = lpc_formants(emph, fs, U, H, outDir)
% LPC-root formant estimation (autocorrelation method) on voiced 30 ms frames. Toolbox: Signal Processing (lpc)
L = round(0.03 * fs); step = round(0.02 * fs); order = 2 + round(fs / 1000);
win = hamming(L); lab = U.labels; FM = zeros(0, 3);
for s = 1:step:numel(emph) - L
    k = min(round((s - 1 + L/2) / H) + 1, numel(lab));
    if lab(k) ~= 2, continue; end
    seg = emph(s:s+L-1) .* win;
    if sqrt(mean(seg.^2)) < 1e-3, continue; end
    a = lpc(seg, order); if any(~isfinite(a)), continue; end
    r = roots(a); r = r(imag(r) > 0.01);
    fr = atan2(imag(r), real(r)) * fs / (2*pi); bw = -log(abs(r)) * fs / pi;
    [fr, o] = sort(fr); bw = bw(o); fr = fr(fr > 90 & bw < 400);
    if numel(fr) >= 3, FM(end+1, :) = fr(1:3)'; end %#ok<AGROW>
end
f = figure('Visible', 'off');
if ~isempty(FM)
    scatter(FM(:, 2), FM(:, 1), 8, 'filled', 'MarkerFaceAlpha', 0.3); set(gca, 'XDir', 'reverse', 'YDir', 'reverse');
end
xlabel('F2 (Hz)'); ylabel('F1 (Hz)'); title('Formants F1-F2 (LPC)');
save_fig(f, fullfile(outDir, 'formants.png'));
if isempty(FM), R.median = [NaN NaN NaN]; else, R.median = median(FM, 1); end
R.frames = size(FM, 1); R.F = FM;
end
