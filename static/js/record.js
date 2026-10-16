// Browser recording -> mono 16-bit WAV -> POST to the upload route (X-Requested-With: fetch).
function initRecorder(uploadUrl) {
  const bStart = document.getElementById('recStart'), bStop = document.getElementById('recStop'), info = document.getElementById('recInfo');
  if (!bStart) return;
  let stream, ctx, proc, src, chunks = [], t0, timer;
  function encodeWav(buffers, rate) {
    let n = 0; buffers.forEach(b => n += b.length);
    const pcm = new Int16Array(n); let o = 0;
    buffers.forEach(b => { for (let i = 0; i < b.length; i++) { const s = Math.max(-1, Math.min(1, b[i])); pcm[o++] = s < 0 ? s * 0x8000 : s * 0x7FFF; } });
    const buf = new ArrayBuffer(44 + pcm.length * 2), v = new DataView(buf);
    const w = (p, s) => { for (let i = 0; i < s.length; i++) v.setUint8(p + i, s.charCodeAt(i)); };
    w(0, 'RIFF'); v.setUint32(4, 36 + pcm.length * 2, true); w(8, 'WAVEfmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
    v.setUint16(22, 1, true); v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    w(36, 'data'); v.setUint32(40, pcm.length * 2, true); new Int16Array(buf, 44).set(pcm);
    return new Blob([buf], { type: 'audio/wav' });
  }
  bStart.onclick = async () => {
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, channelCount: 1 } }); }
    catch (e) { info.textContent = 'Microphone not available: ' + e.message; return; }
    ctx = new (window.AudioContext || window.webkitAudioContext)(); src = ctx.createMediaStreamSource(stream);
    proc = ctx.createScriptProcessor(4096, 1, 1); chunks = [];
    proc.onaudioprocess = ev => chunks.push(new Float32Array(ev.inputBuffer.getChannelData(0)));
    src.connect(proc); proc.connect(ctx.destination);
    t0 = Date.now(); timer = setInterval(() => info.textContent = 'Recording... ' + ((Date.now() - t0) / 1000).toFixed(0) + ' s', 500);
    bStart.disabled = true; bStop.disabled = false;
  };
  bStop.onclick = async () => {
    clearInterval(timer); bStop.disabled = true; proc.disconnect(); src.disconnect(); stream.getTracks().forEach(t => t.stop());
    const blob = encodeWav(chunks, ctx.sampleRate); await ctx.close(); info.textContent = 'Uploading...';
    const fd = new FormData(); fd.append('audio', blob, 'recording.wav');
    const r = await fetch(uploadUrl, { method: 'POST', body: fd, headers: { 'X-Requested-With': 'fetch' } });
    const j = await r.json().catch(() => ({}));
    if (r.ok && j.ok) location.href = j.redirect; else { info.textContent = 'Rejected: ' + (j.error || r.status); bStart.disabled = false; }
  };
}
