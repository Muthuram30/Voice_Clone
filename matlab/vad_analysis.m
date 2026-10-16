function V = vad_analysis(y, fs)
% Energy VAD: 25 ms frames / 10 ms hop, adaptive threshold, gap closing, burst removal, hangover.
H = round(fs * 0.010); L = 2 * round(fs * 0.025 / 2);
fr = frame_signal(y, L, H);
e = 10 * log10(mean(fr.^2, 1)' + 1e-12);
es = sort(e); n = numel(e);
fl = es(max(1, round(0.10 * n))); tp = es(max(1, round(0.95 * n)));
thr = max(fl + max(0.3 * (tp - fl), 6), -55);
m = smooth_mask(e > thr, 15, 10, 2);
d = diff([0; m; 0]); s = find(d == 1); en = find(d == -1) - 1;
V.mask = m; V.eDb = e; V.thrDb = thr; V.hop = H; V.frameLen = L;
V.regions = [(s - 1) * H / fs, en * H / fs];       % seconds
V.speechRatio = mean(m);
end
