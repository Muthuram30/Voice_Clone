function [S, F, T] = spectrogram(x, win, noverlap, nfft, fs)
% TEST-ONLY shim so Octave (which lacks spectrogram) can run the MATLAB DSP modules. One-sided complex STFT,
% no padding, no scaling: same layout as MATLAB's spectrogram(x, window, noverlap, nfft, fs) with a numeric window.
x = x(:); L = numel(win); H = L - noverlap; n = floor((numel(x) - L) / H) + 1;
idx = (1:L)' + (0:n-1) * H; X = x(idx) .* win(:); Y = fft(X, nfft, 1);
S = Y(1:floor(nfft/2)+1, :); F = (0:floor(nfft/2))' * fs / nfft; T = ((0:n-1) * H + L/2) / fs;
end
