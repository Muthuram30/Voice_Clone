function F = pitch_analysis(y, fs, outDir)
% Autocorrelation pitch tracker (60-400 Hz). First strong peak (>= 0.9 of max) avoids octave errors;
% voicing needs normalised autocorrelation > 0.5.   Toolbox: Signal Processing (medfilt1)
%
% BUG FIX (2026-10-02): The original code crashed with
%   "Unable to perform assignment because the left and right sides have
%    a different number of elements."
% ROOT CAUSE: When the normalised autocorrelation array (rn) contains NaN
%   (e.g. from near-zero frames or numerical edge cases in the FFT round-
%   trip), max(s(lm)) returns NaN, so the compound condition
%     lm & s >= 0.9 * max(s(lm))
%   evaluates to all-false, and find(..., 1, 'first') returns [] (empty).
%   k = [] then propagates through y1/y2/y3/den/d, causing
%     f0(i) = fs / (lo + ([] - 1) + 0)  ->  f0(i) = []
%   which fails the scalar assignment.
%
% FIX STRATEGY:
%   1. Sanitise rn to replace NaN/Inf with 0 before the per-frame loop.
%   2. After find(), guard against empty k: treat the frame as unvoiced
%      (f0 stays NaN, strength stays 0) and continue.
%   3. Clamp k to [2, numel(s)-1] so parabolic interpolation never
%      accesses index 0 or numel(s)+1 (belt-and-suspenders; lm already
%      ensures this, but an explicit check makes the invariant visible).
%   4. Everything else (algorithm, outputs, figure) is unchanged.

fmin = 60; fmax = 400; thr = 0.50;
L = 2 * round(fs * 0.040 / 2); H = round(fs * 0.010);
w = hann(L);
fr = frame_signal(y, L, H) .* w;          % L x n windowed frames
nfft = 2^nextpow2(2 * L);

% --- Autocorrelation via power spectrum (Wiener-Khinchin) ----------------
r  = real(ifft(abs(fft(fr, nfft, 1)).^2, nfft, 1)); r = r(1:L, :);   % L x n
rw = real(ifft(abs(fft(w,  nfft)   ).^2, nfft));    rw = rw(1:L);    % L x 1

% Unbiased normalisation: divide by window autocorrelation, then normalise
% so that lag-0 == 1 for each frame.
rn = r ./ (rw + 1e-12);            % L x n  (broadcast: L×1 denominator)
rn = rn ./ (rn(1, :) + 1e-12);     % L x n  (broadcast: 1×n denominator)

% FIX 1: Sanitise NaN/Inf in rn before the search loop.
% Source of NaN: extremely quiet frames make rn(1,:) near zero, then
% dividing by (near-zero + 1e-12) is fine, but if rw has exact-zero
% entries or fft numerical noise produces tiny negatives in rn(1,:) that
% are negative and cancel 1e-12, we can get divide-by-~0 -> Inf/NaN.
rn(~isfinite(rn)) = 0;

lo = floor(fs / fmax);              % minimum lag (samples) for fmax
hi = ceil(fs / fmin);               % maximum lag (samples) for fmin
seg = rn(lo+1 : hi+1, :);          % (hi-lo+1) x n  --  row k <-> lag lo+k-1

n = size(seg, 2);
f0       = nan(n, 1);               % NaN = unvoiced / no valid peak
strength = zeros(n, 1);
rmsf     = sqrt(mean(fr.^2, 1))';   % n x 1  frame RMS

for i = 1:n
    s = seg(:, i);                  % column vector for this frame

    % --- Local-maximum mask (interior points only: indices 2..end-1) -----
    lm = false(size(s));
    lm(2:end-1) = s(2:end-1) > s(1:end-2) & s(2:end-1) >= s(3:end);

    if ~any(lm), continue; end      % no local max -> unvoiced, skip

    % --- FIX 2: Guard against NaN in s -----------------------------------
    % If s contains NaN, max(s(lm)) == NaN, the compound condition becomes
    % all-false, and find() returns [].  Replace NaN in s with 0 so the
    % search degrades gracefully to the real-valued peaks only.
    if any(~isfinite(s))
        s(~isfinite(s)) = 0;
        % Recompute lm with cleaned s
        lm = false(size(s));
        lm(2:end-1) = s(2:end-1) > s(1:end-2) & s(2:end-1) >= s(3:end);
        if ~any(lm), continue; end
    end

    % --- Pick the first peak >= 90 % of the strongest local max ----------
    % "first" selects the lowest-lag (highest-F0) candidate among the
    % strong peaks, which avoids octave doubling errors.
    k = find(lm & s >= 0.9 * max(s(lm)), 1, 'first');

    % FIX 3: Guard against empty k (no qualifying peak found).
    % This should not happen after the NaN sanitisation above, but we
    % keep this as a belt-and-suspenders safety net.
    if isempty(k), continue; end

    % FIX 4: Clamp k so parabolic interpolation never reads out of bounds.
    % (lm construction already guarantees k in [2, numel(s)-1], but an
    % explicit clamp makes the invariant explicit and prevents any future
    % regression.)
    k = max(2, min(k, numel(s) - 1));

    % --- Parabolic interpolation for sub-sample lag refinement -----------
    y1 = s(k-1);                    % all three are guaranteed scalars
    y2 = s(k);
    y3 = s(k+1);
    den = y1 - 2*y2 + y3;
    if abs(den) > 1e-12
        d = 0.5 * (y1 - y3) / den;
    else
        d = 0;
    end
    d = max(min(d, 1), -1);         % clamp interpolation offset to [-1, 1]

    % --- Store results for this frame ------------------------------------
    % k corresponds to lag = lo + (k-1) samples; d is the fractional correction.
    f0(i)       = fs / (lo + (k - 1) + d);   % F0 in Hz  (scalar / scalar)
    strength(i) = y2;                          % normalised autocorrelation peak
end

% --- Post-processing: voicing decision + median smoothing ----------------
voiced = strength > thr & isfinite(f0) & rmsf > 1e-4 & f0 >= fmin & f0 <= fmax;
f0v = f0; f0v(~voiced) = 0; f0v = medfilt1(f0v, 5);
voiced = voiced & f0v > 0; f0v(~voiced) = NaN;

% --- Output struct -------------------------------------------------------
F.f0       = f0v;
F.voiced   = voiced;
F.strength = strength;
F.t        = (0:n-1)' * H / fs;

% --- Figure --------------------------------------------------------------
f = figure('Visible', 'off');
plot(F.t, f0v, '.', 'MarkerSize', 4); ylim([50 420]);
xlabel('Time (s)'); ylabel('F0 (Hz)'); title('Pitch (autocorrelation), voiced frames only');
save_fig(f, fullfile(outDir, 'pitch.png'));
end
