function P = preprocess_audio(wavPath, targetFs)
% Load, mono, resample, DC removal, high-pass, spectral-subtraction noise reduction,
% peak normalisation, pre-emphasis.   Toolbox: Signal Processing (resample, butter, filtfilt, stft, istft)
[x, fs0] = audioread(wavPath);
P.origFs = fs0; P.channels = size(x, 2); P.duration = size(x, 1) / fs0;
x = mean(x, 2);                                  % mono
x = x - mean(x);                                 % DC removal
if fs0 ~= targetFs, x = resample(x, targetFs, fs0); end
fs = targetFs; P.fs = fs; P.raw = x;
[b, a] = butter(4, 80 / (fs/2), 'high');         % rumble filter
hp = filtfilt(b, a, x); P.hp = hp;
% spectral subtraction (noise = mean magnitude of the quietest 10 % of frames)
w = hann(512, 'periodic');
S = stft(hp, fs, 'Window', w, 'OverlapLength', 384, 'FFTLength', 512);
mag = abs(S); [~, i] = sort(sum(mag.^2, 1));
noise = mean(mag(:, i(1:max(1, round(0.1 * numel(i))))), 2);
mag2 = max(mag - 1.5 * noise, 0.05 * mag);
yd = real(istft(mag2 .* exp(1j * angle(S)), fs, 'Window', w, 'OverlapLength', 384, 'FFTLength', 512));
n = numel(x); yd(end+1:n) = 0; yd = yd(1:n);
yd = yd / (max(abs(yd)) + eps) * 0.95;           % peak normalisation
P.processed = yd;
P.emph = filter([1 -0.97], 1, yd);               % pre-emphasis (formant LPC)
end
