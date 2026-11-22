// Lossless PCM capture with explicit permission, preview and retry.
function initRecorder(uploadUrl) {
  const $ = id => document.getElementById(id);
  const start = $('recStart'), stop = $('recStop'), info = $('recInfo');
  if (!start) return;
  let stream, ctx, source, worklet, mute, timer, animation, analyser, chunks = [], blob, previewUrl;
  let recording = false, total = 0, rate = 48000;
  const limit = Number(start.dataset.maxSeconds || 300);
  const preview = $('recPreview'), upload = $('recUpload'), discard = $('recDiscard');
  const canvas = $('recWave'), meter = $('recLevel');
  const say = message => { info.textContent = message; };
  function encode() {
    const data = new ArrayBuffer(44 + total * 2), v = new DataView(data);
    const str = (p, s) => [...s].forEach((c, i) => v.setUint8(p + i, c.charCodeAt(0)));
    str(0, 'RIFF'); v.setUint32(4, 36 + total * 2, true); str(8, 'WAVEfmt ');
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    str(36, 'data'); v.setUint32(40, total * 2, true);
    let offset = 44;
    for (const chunk of chunks) for (const sample of chunk) {
      const x = Math.max(-1, Math.min(1, sample));
      v.setInt16(offset, Math.round(x * (x < 0 ? 32768 : 32767)), true); offset += 2;
    }
    return new Blob([data], {type: 'audio/wav'});
  }
  async function cleanup() {
    recording = false; clearInterval(timer); cancelAnimationFrame(animation);
    if (worklet) worklet.port.onmessage = null;
    for (const node of [source, worklet, mute, analyser]) { try { node?.disconnect(); } catch (_) {} }
    stream?.getTracks().forEach(track => track.stop());
    if (ctx && ctx.state !== 'closed') await ctx.close();
    if (meter) meter.value = 0;
  }
  function draw() {
    if (!recording) return;
    const data = new Float32Array(analyser.fftSize); analyser.getFloatTimeDomainData(data);
    let sum = 0, peak = 0;
    data.forEach(x => { sum += x * x; peak = Math.max(peak, Math.abs(x)); });
    if (meter) meter.value = Math.sqrt(sum / data.length);
    if (canvas) {
      const g = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
      g.clearRect(0, 0, w, h); g.strokeStyle = peak > .98 ? '#ff637d' : '#80e4cb'; g.lineWidth = 2; g.beginPath();
      data.forEach((x, i) => { const px = i * w / data.length, py = (1 - x) * h / 2; i ? g.lineTo(px, py) : g.moveTo(px, py); }); g.stroke();
    }
    animation = requestAnimationFrame(draw);
  }
  start.onclick = async () => {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      say('Open this page using localhost or HTTPS to allow microphone access.'); return;
    }
    if (!window.AudioWorkletNode) { say('Use a browser with AudioWorklet support, or upload audio.'); return; }
    start.disabled = true; say('Waiting for microphone permission…');
    try {
      stream = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1}});
      ctx = new AudioContext(); await ctx.resume(); rate = ctx.sampleRate;
      await ctx.audioWorklet.addModule('/static/js/pcm-worklet.js');
      source = ctx.createMediaStreamSource(stream); worklet = new AudioWorkletNode(ctx, 'pcm-recorder');
      mute = ctx.createGain(); mute.gain.value = 0; analyser = ctx.createAnalyser(); analyser.fftSize = 1024;
      chunks = []; total = 0; blob = null; recording = true;
      preview.hidden = true; upload.hidden = true; discard.hidden = true;
      worklet.port.onmessage = event => {
        if (!recording) return;
        const remaining = Math.max(0, Math.floor(limit * rate) - total);
        const chunk = event.data.subarray(0, remaining);
        if (chunk.length) { chunks.push(chunk); total += chunk.length; }
        if (total >= limit * rate) stop.click();
      };
      source.connect(analyser); source.connect(worklet); worklet.connect(mute); mute.connect(ctx.destination);
      stop.disabled = false; say('Recording… speak naturally.'); draw();
      timer = setInterval(() => say(`Recording ${(total / rate).toFixed(0)} / ${limit} seconds · watch the input level`), 500);
    } catch (e) {
      await cleanup(); start.disabled = false;
      const errors = {NotAllowedError: 'Microphone permission denied. Allow it in browser site settings, then retry.', NotFoundError: 'No microphone found. Connect one or upload audio.', NotReadableError: 'Microphone is busy. Close other recording apps and retry.'};
      say(errors[e.name] || `Could not start recording: ${e.message}`);
    }
  };
  stop.onclick = async () => {
    if (!recording) return;
    stop.disabled = true; await cleanup(); blob = encode(); chunks = [];
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(blob); preview.src = previewUrl; preview.hidden = false;
    upload.hidden = false; discard.hidden = false; start.disabled = false;
    say(`${(total / rate).toFixed(1)} seconds captured. Listen before saving to this profile.`);
  };
  upload.onclick = async () => {
    if (!blob) return;
    upload.disabled = true; start.disabled = true; discard.disabled = true; say('Uploading and checking recording…');
    try {
      const body = new FormData(); body.append('audio', blob, 'microphone.wav');
      const r = await fetch(uploadUrl, {method: 'POST', body, headers: {'X-Requested-With': 'fetch'}});
      const data = await r.json().catch(() => ({}));
      if (!r.ok || !data.ok) throw new Error(data.error || `Upload failed (${r.status}).`);
      location.assign(data.redirect);
    } catch (e) { say(`${e.message} Your recording is kept here so you can retry.`); }
    finally { upload.disabled = false; start.disabled = false; discard.disabled = false; }
  };
  discard.onclick = () => { blob = null; preview.pause(); preview.removeAttribute('src'); preview.hidden = true; upload.hidden = true; discard.hidden = true; URL.revokeObjectURL(previewUrl); say('Recording discarded.'); };
  window.addEventListener('pagehide', () => { cleanup(); if (previewUrl) URL.revokeObjectURL(previewUrl); });
}
