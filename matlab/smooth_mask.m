function m = smooth_mask(m, minGap, minRun, hang)
% Close gaps shorter than minGap frames, drop runs shorter than minRun, extend by `hang` frames.
m = m(:);
[s, e] = runs(~m);
for k = 1:numel(s)
    if s(k) > 1 && e(k) < numel(m) && (e(k) - s(k) + 1) < minGap, m(s(k):e(k)) = true; end
end
[s, e] = runs(m);
for k = 1:numel(s)
    if (e(k) - s(k) + 1) < minRun, m(s(k):e(k)) = false; end
end
idx = find(m); md = m;
for k = -hang:hang, md(min(max(idx + k, 1), numel(m))) = true; end
m = md;
end
function [s, e] = runs(b)
d = diff([0; b(:); 0]); s = find(d == 1); e = find(d == -1) - 1;
end
