function F = frame_signal(y, L, H)
% Centre-padded framing -> L x n matrix, n = 1 + floor(N/H) (same grid as the Python code).
y = y(:); n = 1 + floor(numel(y) / H);
yp = [zeros(L/2, 1); y; zeros(L/2, 1)];
need = (n - 1) * H + L;
if numel(yp) < need, yp(end+1:need) = 0; end
F = yp((1:L)' + (0:n-1) * H);
end
