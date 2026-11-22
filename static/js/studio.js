(() => {
  const data = JSON.parse(document.getElementById('language-data').textContent);
  const text = document.getElementById('synth-text'), lang = document.getElementById('lang-select');
  const engine = document.getElementById('engine'), status = document.getElementById('engine-status');
  const generate = document.getElementById('generate');
  function updateEngine() {
    const current = lang.value || data.selected, indic = engine.value === 'indicf5';
    lang.replaceChildren(...Object.entries(data[engine.value]).map(([code, name]) => new Option(`${name} (${code})`, code)));
    lang.value = current in data[engine.value] ? current : indic ? 'ta' : 'en';
    document.getElementById('indic-prompt').hidden = !indic;
    for (const id of ['prompt-audio', 'prompt-text']) document.getElementById(id).required = indic;
    status.textContent = indic ? data.indic_ready ? 'Local model files found. CUDA readiness is checked on generation.' : 'Setup required. See README for .venv-indic and local model files. Tamil cannot run through XTTS.' : '17 supported languages. Accent and pronunciation quality depend on the reference and model.';
    generate.disabled = indic && !data.indic_ready;
  }
  engine.addEventListener('change', updateEngine); updateEngine();
  const updateCount = () => { document.getElementById('text-count').textContent = `${text.value.length} / ${text.maxLength}`; };
  text.addEventListener('input', updateCount); updateCount();
  document.getElementById('text-import').addEventListener('change', async event => {
    const file = event.target.files[0], message = document.getElementById('import-message');
    if (!file) return;
    try {
      if (file.size > 2*1024*1024) throw new Error('Text import is limited to 2 MB.');
      const ext = file.name.split('.').pop().toLowerCase();
      if (!['txt','srt','vtt'].includes(ext)) throw new Error('Choose TXT, SRT or VTT.');
      const raw = await file.text();
      const parsed = ext === 'txt' ? raw : raw.replace(/^\uFEFF/, '').split(/\r?\n/)
        .filter(line => !/^\s*(WEBVTT.*|\d+|.*-->.*)\s*$/.test(line)).join('\n').replace(/<[^>]*>/g, '').trim();
      if (parsed.length > text.maxLength) throw new Error(`Text has ${parsed.length} characters; split it into passages of at most ${text.maxLength}. Nothing was truncated.`);
      text.value = parsed; updateCount(); message.textContent = `Imported ${file.name}. Subtitle timing is removed; speech is generated as a continuous passage.`;
    } catch (error) { message.textContent = error.message; }
    event.target.value = '';
  });
  document.getElementById('synth-form').addEventListener('submit', () => {
    generate.disabled = true; generate.textContent = 'Generating…';
    const output = document.getElementById('generation-status'), start = Date.now();
    output.textContent = 'Loading the voice model and generating on GPU. Keep this page open.';
    setInterval(() => { output.textContent = `Generating and measuring audio · ${Math.floor((Date.now()-start)/1000)} s elapsed. Keep this page open.`; }, 1000);
  });
})();
