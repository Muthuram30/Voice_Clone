function [v, names] = extract_features(M, F, S, R, V, U)
% Fixed-length speaker feature vector; same 53 features, same order as dsp_service.FEATURE_NAMES (Python).
n = min([size(M.coeffs, 2), numel(V.mask), numel(S.centroid), numel(S.rolloff), numel(S.zcr), numel(F.f0)]);
sp = V.mask(1:n); if sum(sp) < 3, sp = true(n, 1); end
mf = M.coeffs(:, 1:n); mf = mf(:, sp); dm = M.delta(:, 1:n); dm = dm(:, sp);
f0 = F.f0(1:n); f0 = f0(~isnan(f0)); lab = U.labels; spn = max(sum(V.mask(1:n)), 1);
if isempty(f0), fs5 = [NaN NaN NaN NaN NaN]; else
    fs5 = [mean(f0), std(f0, 1), median(f0), pct(f0, 5), pct(f0, 95)]; end
cen = S.centroid(1:n); rol = S.rolloff(1:n); zc = S.zcr(1:n);
v = [mean(mf, 2); std(mf, 1, 2); mean(dm, 2); fs5(:); R.median(:); mean(cen(sp)); std(cen(sp), 1); mean(rol(sp)); mean(zc(sp)); ...
     mean(V.mask(1:n)); sum(lab(1:min(n, numel(lab))) == 2) / spn];
names = [arrayfun(@(i) sprintf('mfcc%d_mean', i), 0:12, 'UniformOutput', false), arrayfun(@(i) sprintf('mfcc%d_std', i), 0:12, 'UniformOutput', false), ...
    arrayfun(@(i) sprintf('dmfcc%d_mean', i), 0:12, 'UniformOutput', false), {'f0_mean_hz', 'f0_std_hz', 'f0_median_hz', 'f0_p5_hz', 'f0_p95_hz', ...
    'F1_hz', 'F2_hz', 'F3_hz', 'centroid_mean_hz', 'centroid_std_hz', 'rolloff85_mean_hz', 'zcr_mean', 'speech_ratio', 'voiced_ratio_of_speech'}];
end
function p = pct(x, q)
x = sort(x(:)); p = x(max(1, min(numel(x), round(q / 100 * numel(x)))));
end
