enum PlaygroundPage {
    static let html = #"""
    <!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>ANERun Playground</title>
    <style>
    :root {
      color-scheme: light; --paper: #f3f2ec; --surface: #fffefa; --ink: #222520;
      --muted: #555c51; --line: #d5d9cf; --green: #245746; --wash: #e7efe8; --error: #992e22;
      font: 15px/1.5 system-ui, -apple-system, sans-serif;
    }
    *, *::before, *::after { box-sizing: border-box }
    [hidden] { display: none !important }
    body { margin: 0; color: var(--ink); background: var(--paper) }
    main { width: min(1160px, 100%); margin: auto; padding: clamp(16px, 4vw, 48px) }
    header, .row, .panel-head, .question-head, .metrics { display: flex; align-items: center; gap: 10px; flex-wrap: wrap }
    header, .panel-head { justify-content: space-between }
    header { padding-bottom: 18px; border-bottom: 1px solid var(--line) }
    .brand { font-weight: 780; letter-spacing: -.035em; font-size: 1.25rem }
    .mark { display: inline-grid; place-items: center; width: 29px; height: 29px; border-radius: 8px; background: var(--green); color: white; font-size: 1rem }
    .model { background: var(--wash); color: var(--green); padding: 4px 9px; border-radius: 6px; font: .79rem ui-monospace, monospace; overflow-wrap: anywhere }
    .intro { padding: clamp(23px, 5vw, 48px) 0 24px }
    h1, h2, h3, p { margin: 0 }
    h1 { font: 500 clamp(2.3rem, 6vw, 4rem)/1.05 ui-serif, Georgia, serif; letter-spacing: -.06em }
    .intro p { margin-top: 8px; color: var(--muted) }
    .workbench { display: grid; grid-template-columns: minmax(0, 1.08fr) minmax(0, .92fr); align-items: start; gap: 14px }
    .panel { min-width: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: clamp(14px, 2.4vw, 24px); box-shadow: 0 2px 5px #28372e09 }
    .panel-head { margin-bottom: 16px }
    h2 { font-size: 1.18rem; letter-spacing: -.02em }
    h3 { font-size: .97rem; overflow-wrap: anywhere }
    label, legend { display: block; font-weight: 650; min-width: 0 }
    label > input:not([type=checkbox]), label > textarea, label > select { margin-top: 4px }
    .stack, #questions, .candidates, #result { display: grid; gap: 12px }
    .grow { flex: 1 1 140px }
    .hint, .empty, .subtle, small { color: var(--muted); font-size: .86rem }
    .empty { padding: 24px 12px; text-align: center; border: 1px dashed var(--line); border-radius: 8px }
    button, input, select, textarea { font: inherit; color: inherit }
    textarea, input:not([type=checkbox]), select { width: 100%; min-width: 0; padding: 7px 9px; border: 1px solid #b9c1b4; border-radius: 7px; background: white }
    textarea { resize: vertical; min-height: 60px }
    input[type=file] { font-size: .86rem }
    button { cursor: pointer; background: var(--surface); border: 1px solid #b9c1b4; border-radius: 7px; padding: 6px 10px; font-weight: 650 }
    button:hover:not(:disabled) { background: var(--wash) }
    button:disabled { cursor: default; opacity: .52 }
    .primary { background: var(--green); border-color: var(--green); color: white; padding: 8px 14px }
    .primary:hover:not(:disabled) { background: #174134 }
    .remove, #stop { color: var(--error) }
    :focus-visible { outline: 3px solid #447d68; outline-offset: 2px }
    fieldset { min-width: 0; margin: 0; padding: 10px; border: 1px solid var(--line); border-radius: 9px }
    legend { padding: 0 5px }
    .question-head { justify-content: space-between }
    .candidates { margin: 9px 0 }
    .candidate { display: grid; grid-template-columns: minmax(62px, 1fr) minmax(100px, 2fr) auto; align-items: end; gap: 6px }
    .candidate label { font-size: .8rem }
    .score-key { align-self: center; font: .8rem ui-monospace, monospace; color: var(--muted) }
    .candidate button { min-height: 36px }
    .actions { border-top: 1px solid var(--line); padding-top: 12px }
    .metrics { gap: 5px 12px; font: .8rem ui-monospace, monospace; color: var(--green) }
    .answer { border: 1px solid var(--line); border-radius: 9px; padding: 10px 12px }
    .value { font-weight: 750; color: var(--green); overflow-wrap: anywhere }
    .prob { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 2px 8px; margin-top: 7px; font-size: .85rem; overflow-wrap: anywhere }
    meter { grid-column: 1 / -1; width: 100%; height: 8px; accent-color: var(--green) }
    meter::-webkit-meter-optimum-value { background: var(--green) }
    .ranks { margin: 0; padding-left: 2rem }
    .ranks li { padding: 7px 0; overflow-wrap: anywhere; border-bottom: 1px solid var(--line) }
    .ranks li:last-child { border: 0 }
    .rank { display: flex; gap: 8px; justify-content: space-between }
    .rank code { color: var(--green); flex-shrink: 0 }
    #recent-runs .panel-head { margin-bottom: 6px }
    #recent-runs-list { list-style: none; margin: 0; padding: 0 }
    #recent-runs-list li { padding: 7px 0; border-bottom: 1px solid var(--line); overflow-wrap: anywhere }
    #recent-runs-list li:last-child { border-bottom: 0; padding-bottom: 0 }
    .run-summary { font-weight: 650 }
    .run-model { font: .78rem ui-monospace, monospace; color: var(--muted) }
    #status { min-height: 1.5em; margin: 10px 0 0; overflow-wrap: anywhere }
    #status.error { color: var(--error); font-weight: 600 }
    pre { max-height: 260px; overflow: auto; background: var(--paper); padding: 10px; font: .78rem ui-monospace, monospace }
    summary { cursor: pointer; font-weight: 650 }
    @media (max-width: 760px) { .workbench { grid-template-columns: minmax(0, 1fr) } }
    @media (max-width: 370px) { .candidate { grid-template-columns: minmax(0, 1fr) minmax(0, 1.45fr); } .candidate button { grid-column: 2; justify-self: end } }
    </style>
    </head>
    <body>
    <main>
      <header>
        <div class="row"><span class="mark" aria-hidden="true">∿</span><span class="brand">ANERun</span><span id="model" class="model">Connecting…</span></div>
        <button id="stop" type="button" disabled>Stop server</button>
      </header>
      <div class="intro"><h1 id="title">Playground</h1><p id="subtitle">Run the model served by this local instance.</p></div>
      <div id="workbench" class="workbench" hidden>
        <section class="panel" aria-labelledby="input-heading">
          <div class="panel-head">
            <h2 id="input-heading">Request</h2>
            <div class="row">
              <button id="example" type="button">Load example</button>
              <button class="primary" id="run" type="submit" form="form">Run</button>
            </div>
          </div>
          <form id="form" class="stack" autocomplete="off">
            <div class="stack" id="decision" hidden>
              <label>State<textarea id="state" rows="3" autocomplete="off" spellcheck="false" placeholder="Enter text or paste serialized JSON"></textarea></label>
              <label id="decision-image-field" hidden>State image <span class="hint">(optional)</span><input id="decision-image" type="file" accept="image/*"></label>
              <div id="questions"></div>
              <div><button type="button" id="add-question">Add question</button></div>
            </div>
            <div class="stack" id="embedding" hidden>
              <label>Documents<textarea id="documents" rows="5" autocomplete="off" spellcheck="false" placeholder="One document per line"></textarea></label>
              <label id="document-image-field" hidden>Image documents <span class="hint">(optional)</span><input id="document-images" type="file" accept="image/*" multiple></label>
              <p class="hint" id="cache">Document vectors are kept in this page.</p>
              <label>Search query<textarea id="query" rows="2" autocomplete="off" spellcheck="false" placeholder="Enter a search query"></textarea></label>
              <label id="query-image-field" hidden>Image query <span class="hint">(takes precedence over text)</span><input id="query-image" type="file" accept="image/*"></label>
            </div>
            <label><input id="truncate" type="checkbox" checked> Allow truncation</label>
            <div class="row actions"><button id="copy" type="button">Copy curl</button></div>
            <details id="curl-preview" hidden><summary>Request commands</summary><pre id="curl-text" tabindex="0"></pre></details>
          </form>
        </section>
        <div class="stack">
          <section class="panel" aria-labelledby="result-heading">
            <div class="panel-head"><h2 id="result-heading">Results</h2><div id="metrics" class="metrics"></div></div>
            <div id="result" aria-live="polite" aria-busy="false"><div class="empty">Run a request to see results here.</div></div>
          </section>
          <section id="recent-runs" class="panel" aria-labelledby="recent-runs-heading">
            <div class="panel-head"><h2 id="recent-runs-heading">Recent runs</h2><button id="clear-history" type="button" disabled>Clear</button></div>
            <p id="recent-runs-empty" class="hint">No runs in this page.</p>
            <ol id="recent-runs-list" aria-label="Recent runs, newest first" aria-live="polite"></ol>
          </section>
        </div>
      </div>
      <p id="status" class="subtle" role="status"></p>
    </main>
    <script>
    'use strict';
    const byId = id => document.getElementById(id);
    const node = (tag, text, className) => {
      const element = document.createElement(tag);
      if (text !== undefined) element.textContent = text;
      if (className) element.className = className;
      return element;
    };
    const form = byId('form'), questions = byId('questions'), result = byId('result');
    const run = byId('run'), copy = byId('copy'), stop = byId('stop');
    let config, revision = 0, pending, stopped = false, nextQuestion = 0;
    let vectors = new Map();
    const recentRuns = [];
    let historyVersion = 0;
    const status = (message, error = false) => {
      const element = byId('status');
      element.textContent = message;
      element.classList.toggle('error', error);
    };
    const empty = message => {
      byId('metrics').replaceChildren();
      result.replaceChildren(node('p', message, 'empty'));
    };
    function renderHistory() {
      byId('recent-runs-list').replaceChildren(...recentRuns.map(({model, summary, latency}) => {
        const item = node('li'), row = node('div', undefined, 'rank');
        row.append(node('span', summary, 'run-summary'), node('code', latency));
        item.append(row, node('div', model, 'run-model'));
        return item;
      }));
      const hasRuns = recentRuns.length > 0;
      byId('recent-runs-empty').hidden = hasRuns;
      byId('clear-history').disabled = !hasRuns;
    }
    function clearHistory() {
      historyVersion++;
      recentRuns.length = 0;
      renderHistory();
    }
    function recordRun(version, summary, latency) {
      if (version !== historyVersion) return;
      recentRuns.unshift({model: config.model, summary, latency});
      if (recentRuns.length > 8) recentRuns.pop();
      renderHistory();
    }
    byId('clear-history').onclick = () => { clearHistory(); status('Recent runs cleared.'); };
    function invalidate() {
      revision++;
      pending?.abort();
      pending = undefined;
      run.disabled = copy.disabled = stopped;
      result.setAttribute('aria-busy', 'false');
      byId('curl-preview').hidden = true;
      byId('curl-preview').open = false;
      byId('curl-text').textContent = '';
      empty('Run a request to see results here.');
      cacheStatus();
      status('');
    }
    function field(label, value, className, multiline = false) {
      const control = node(multiline ? 'textarea' : 'input');
      control.autocomplete = 'off';
      control.spellcheck = false;
      if (multiline) control.rows = 2;
      control.className = className;
      control.value = value;
      const wrap = node('label', label);
      wrap.className = 'grow';
      wrap.append(control);
      return wrap;
    }
    function button(label, action, className) {
      const element = node('button', label, className);
      element.type = 'button';
      element.dataset.action = action;
      return element;
    }
    function addCandidate(container, key = '', description = '') {
      const row = node('div', undefined, 'candidate');
      const remove = button('Remove', 'remove-candidate', 'remove');
      remove.setAttribute('aria-label', 'Remove candidate');
      const level = node('span', undefined, 'score-key');
      level.hidden = true;
      row.append(field('Key', key, 'candidate-key'), level,
        field('Description', description, 'candidate-description'), remove);
      container.append(row);
    }
    function showCandidateKeys(card) {
      const score = card.querySelector('.kind').value === 'score';
      [...card.querySelectorAll('.candidate')].forEach((row, index) => {
        row.querySelector('.candidate-key').parentElement.hidden = score;
        const level = row.querySelector('.score-key');
        level.hidden = !score;
        level.textContent = 'Level ' + index;
      });
    }
    function addQuestion(question = {}) {
      const card = node('fieldset', undefined, 'stack question');
      const number = ++nextQuestion;
      const head = node('div', undefined, 'question-head');
      const name = field('Question name', question.name ?? ('question_' + number), 'question-name');
      const kind = node('select', undefined, 'kind');
      kind.autocomplete = 'off';
      const kinds = [['choice', 'Choice'], ['noul', 'Yes / no']];
      if (config.id !== 'k-decision-9b') kinds.push(['score', 'Score']);
      for (const [value, label] of kinds) {
        const option = node('option', label);
        option.value = value;
        kind.append(option);
      }
      kind.value = question.kind ?? 'choice';
      const kindLabel = node('label', 'Answer type', 'grow');
      kindLabel.append(kind);
      head.append(name, kindLabel);
      const options = node('div', undefined, 'option-group');
      const rows = node('div', undefined, 'candidates');
      for (const candidate of question.candidates ?? [{key: '', description: ''}, {key: '', description: ''}]) {
        addCandidate(rows, candidate.key, candidate.description);
      }
      options.append(rows, button('Add candidate', 'add-candidate'));
      options.hidden = kind.value === 'noul';
      card.append(node('legend', 'Question ' + number), head,
        field('Instructions', question.instructions ?? '', 'instructions', true), options,
        button('Remove question', 'remove-question', 'remove'));
      questions.append(card);
      showCandidateKeys(card);
      return name.querySelector('input');
    }
    function loadExample() {
      const billing = {name: 'team', kind: 'choice', instructions: 'Which team should handle this request?',
        candidates: [{key: 'billing', description: 'Charges, invoices, and refunds'},
          {key: 'technical', description: 'Product errors and outages'}]};
      byId('truncate').checked = true;
      if (config.kind === 'decision') {
        questions.replaceChildren();
        nextQuestion = 0;
        byId('decision-image').value = '';
        byId('state').value = 'I was charged twice for the same order. Please refund the extra charge.';
        addQuestion(billing);
      } else {
        byId('document-images').value = byId('query-image').value = '';
        byId('documents').value = 'Refunds are issued to the original payment method within five business days.\nReset your password from the sign-in page.\nTrack shipments using the link in your dispatch email.';
        byId('query').value = 'How long will my refund take?';
      }
      invalidate();
    }
    function resetPage(showExample = false) {
      clearHistory();
      vectors.clear();
      form.reset();
      questions.replaceChildren();
      if (showExample && config && !stopped) loadExample();
      else invalidate();
    }
    window.addEventListener('pagehide', () => resetPage());
    window.addEventListener('pageshow', event => {
      if (event.persisted) resetPage(true);
    });
    questions.addEventListener('change', event => {
      if (event.target.matches('.kind')) {
        const options = event.target.closest('.question').querySelector('.option-group');
        options.hidden = event.target.value === 'noul';
        if (!options.hidden && !options.querySelector('.candidate')) {
          addCandidate(options.querySelector('.candidates'));
          addCandidate(options.querySelector('.candidates'));
        }
        showCandidateKeys(event.target.closest('.question'));
      }
    });
    questions.addEventListener('click', event => {
      const action = event.target.dataset.action;
      if (!action) return;
      const card = event.target.closest('.question');
      if (action === 'remove-question') {
        card.remove();
        (questions.querySelector('.question-name') ?? byId('add-question')).focus();
      } else if (action === 'remove-candidate') {
        event.target.closest('.candidate').remove();
        card.querySelector('[data-action=add-candidate]').focus();
      } else {
        const rows = card.querySelector('.candidates');
        addCandidate(rows);
        showCandidateKeys(card);
        rows.lastElementChild.querySelector(card.querySelector('.kind').value === 'score'
          ? '.candidate-description' : '.candidate-key').focus();
      }
      if (action === 'remove-candidate') showCandidateKeys(card);
      invalidate();
    });
    byId('add-question').onclick = () => { addQuestion().focus(); invalidate(); };
    byId('example').onclick = loadExample;
    form.addEventListener('input', invalidate);
    form.addEventListener('change', event => { if (event.target.type === 'file') invalidate(); });
    const readImage = file => new Promise((resolve, reject) => {
      if (!file.type.startsWith('image/')) { reject(new Error('Select a supported image file.')); return; }
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error('Could not read ' + file.name + '.'));
      reader.readAsDataURL(file);
    });
    const endpoints = {decision: '/v1/decisions', embedding: '/v1/embeddings'};
    async function api(path, body, signal) {
      const options = {signal, redirect: 'error', cache: 'no-store', mode: 'same-origin'};
      if (body !== undefined) {
        options.method = 'POST';
        options.headers = {'Content-Type': 'application/json'};
        options.body = JSON.stringify(body);
      }
      const response = await fetch(path, options);
      const text = await response.text();
      let data;
      try { data = text ? JSON.parse(text) : null; } catch { /* status below handles non-JSON bodies */ }
      if (!response.ok) throw new Error(data?.error?.message ?? (response.status + ' ' + response.statusText));
      if (!data && path !== '/playground/stop') throw new Error('The server did not return JSON.');
      return data;
    }
    function decisionSnapshot() {
      const items = [...questions.children].map(card => {
        const get = selector => card.querySelector(selector).value;
        const kind = get('.kind');
        const candidates = kind === 'noul' ? [] : [...card.querySelectorAll('.candidate')].map((row, i) => ({
          key: kind === 'score' ? String(i) : row.querySelector('.candidate-key').value.trim(),
          description: row.querySelector('.candidate-description').value.trim()
        }));
        return {name: get('.question-name').trim(), kind, instructions: get('.instructions'), candidates};
      });
      if (!items.length) throw new Error('Add at least one question.');
      if (items.some(item => !item.name || !item.instructions.trim()) ||
          new Set(items.map(item => item.name)).size !== items.length) {
        throw new Error('Give each question a unique name and instructions.');
      }
      for (const item of items) {
        if (item.kind === 'noul') continue;
        if (item.candidates.length < 2 || item.candidates.some(c => !c.key || !c.description) ||
            new Set(item.candidates.map(c => c.key)).size !== item.candidates.length) {
          throw new Error('Add at least two candidates with unique keys and descriptions for ' + item.name + '.');
        }
      }
      return {body: {model: config.model, state: byId('state').value, questions: items, truncate: byId('truncate').checked},
        file: config.images ? byId('decision-image').files[0] : undefined};
    }
    async function decisionBody(snapshot) {
      if (snapshot.file) snapshot.body.image = (await readImage(snapshot.file)).split(',', 2)[1];
      return snapshot.body;
    }
    function documents() {
      const texts = byId('documents').value.split(/\r?\n/).filter(text => text.trim()).map(text => ({text, label: text}));
      const images = config?.images ? [...byId('document-images').files].map(file => ({file, label: file.name})) : [];
      return [...texts, ...images];
    }
    function cacheKey(item, truncate) {
      return item.file ? item.file : JSON.stringify([truncate, item.text]);
    }
    function cacheStatus() {
      if (config?.kind !== 'embedding') return;
      const truncate = byId('truncate').checked;
      const docs = documents(), cached = docs.filter(doc => {
        const stored = vectors.get(cacheKey(doc, truncate));
        return stored?.truncate === truncate;
      }).length;
      byId('cache').textContent = docs.length
        ? cached + ' of ' + docs.length + ' documents cached in this page.'
        : 'Add text documents or supported images to search.';
    }
    function embeddingSnapshot() {
      const docs = documents();
      const query = {text: byId('query').value, file: config.images ? byId('query-image').files[0] : undefined};
      if (!docs.length) throw new Error('Add at least one document.');
      if (!query.file && !query.text.trim()) throw new Error('Enter a search query or select an image.');
      return {docs, query, truncate: byId('truncate').checked};
    }
    async function embeddingBody(item, type, truncate) {
      const body = {model: config.model, encoding_format: 'float', truncate};
      if (item.file) {
        body.input = [{content: [{type: 'image_url', image_url: {url: await readImage(item.file)}}]}];
      } else {
        body.input = item.text;
        // PPLX accepts input_type but applies neither prefix.
        if (config.id !== 'pplx-embed-v1-0.6b') body.input_type = type;
      }
      return body;
    }
    function normalized(values) {
      if (!Array.isArray(values) || !values.length || !values.every(Number.isFinite)) return null;
      let scale = 0;
      for (const v of values) scale = Math.max(scale, Math.abs(v));
      if (!scale) return null;
      const scaled = values.map(v => v / scale);
      const norm = Math.sqrt(scaled.reduce((sum, v) => sum + v * v, 0));
      return scaled.map(v => v / norm);
    }
    function cosine(a, b) {
      if (!a || !b || a.length !== b.length) return null;
      const dot = a.reduce((sum, v, i) => sum + v * b[i], 0);
      return Math.max(-1, Math.min(1, dot));
    }
    async function fetchVector(item, type, truncate, signal) {
      const body = await embeddingBody(item, type, truncate);
      const response = await api(endpoints.embedding, body, signal);
      const row = response?.data?.find?.(entry => entry.index === 0);
      const unit = normalized(row?.embedding);
      if (!unit) throw new Error('The server returned an invalid or zero-norm embedding.');
      return {unit, clipped: row.truncated === true, truncate};
    }
    async function search(snapshot, signal) {
      const nextCache = new Map();
      const rows = [];
      for (const doc of snapshot.docs) {
        signal.throwIfAborted();
        const key = cacheKey(doc, snapshot.truncate);
        let stored = nextCache.get(key) ?? vectors.get(key);
        if (!stored || stored.truncate !== snapshot.truncate) {
          stored = await fetchVector(doc, 'document', snapshot.truncate, signal);
        }
        nextCache.set(key, stored);
        rows.push(stored);
      }
      const query = await fetchVector(snapshot.query, 'query', snapshot.truncate, signal);
      const matches = rows.map((row, index) => ({label: snapshot.docs[index].label, clipped: row.clipped,
        score: cosine(query.unit, row.unit), index}));
      if (matches.some(match => match.score === null)) throw new Error('Embedding dimensions differ. Reload the page and try again.');
      matches.sort((a, b) => b.score - a.score || a.index - b.index);
      return {matches, nextCache, dimensions: query.unit.length, clipped: query.clipped};
    }
    const percent = value => (value * 100).toFixed(1) + '%';
    function probability(container, label, value) {
      if (!Number.isFinite(value)) return;
      const row = node('div', undefined, 'prob');
      const meter = node('meter');
      meter.min = 0; meter.max = 1; meter.value = value;
      meter.setAttribute('aria-label', label + ' probability');
      row.append(node('span', label), node('span', percent(value)), meter);
      container.append(row);
    }
    function decisionResult(data, items) {
      if (!data?.answers || typeof data.answers !== 'object') throw new Error('The server returned no answers.');
      let summary;
      const answers = items.map((item, index) => {
        const answer = Object.hasOwn(data.answers, item.name) ? data.answers[item.name] : null;
        if (!answer) throw new Error('The server returned no answer for ' + item.name + '.');
        const card = node('article', undefined, 'answer');
        card.append(node('h3', item.name));
        if (answer.type === 'noul' && Number.isFinite(answer.noul)) {
          const value = percent(answer.noul) + ' yes';
          card.append(node('p', value, 'value'));
          if (!index) summary = value;
          probability(card, 'Yes', answer.noul);
          probability(card, 'No', 1 - answer.noul);
        } else {
          const value = answer.choice ?? (Number.isFinite(answer.score) ? answer.score.toFixed(2) : '—');
          card.append(node('p', String(value), 'value'));
          if (!index) {
            summary = typeof answer.choice === 'string'
              ? (answer.choice.length <= 48 && !answer.choice.startsWith('data:') ? answer.choice : 'Choice selected')
              : 'Score ' + value;
          }
          if (Number.isFinite(answer.confidence)) card.append(node('small', 'Confidence ' + percent(answer.confidence)));
          for (const {key} of item.candidates) {
            const legend = answer.legend && Object.hasOwn(answer.legend, key) ? answer.legend[key] : '';
            const probabilities = answer.probabilities ?? {};
            if (Object.hasOwn(probabilities, key)) probability(card, legend ? key + ' · ' + legend : key, probabilities[key]);
          }
        }
        return card;
      });
      result.replaceChildren(...answers);
      return items.length > 1 ? summary + ' · ' + items.length + ' answers' : summary;
    }
    function ranking(data) {
      const list = node('ol', undefined, 'ranks');
      for (const match of data.matches) {
        const item = node('li'), row = node('div', undefined, 'rank');
        row.append(node('span', match.label), node('code', match.score.toFixed(4)));
        item.append(row);
        if (match.clipped) item.append(node('small', 'Document truncated'));
        list.append(item);
      }
      result.replaceChildren(list);
    }
    const elapsed = start => (performance.now() - start).toFixed(0) + ' ms';
    function metrics(latency, ...values) {
      byId('metrics').replaceChildren(...[latency, ...values]
        .filter(Boolean).map(value => node('span', value)));
    }
    async function execute() {
      if (!config || stopped || pending) return;
      const ticket = revision, historyTicket = historyVersion, controller = new AbortController(), start = performance.now();
      pending = controller;
      run.disabled = copy.disabled = true;
      result.setAttribute('aria-busy', 'true');
      status(config.kind === 'decision' ? 'Running decision…' : 'Embedding and ranking…');
      const current = () => revision === ticket && !controller.signal.aborted;
      let submitted = false;
      try {
        if (config.kind === 'decision') {
          const snapshot = decisionSnapshot(), body = await decisionBody(snapshot);
          if (!current()) return;
          submitted = true;
          const data = await api(endpoints.decision, body, controller.signal);
          if (!current()) return;
          const summary = decisionResult(data, body.questions), latency = elapsed(start);
          const usage = data.usage ?? {};
          metrics(latency, Number.isFinite(usage.input_tokens) ? usage.input_tokens + ' input tokens' : '',
            usage.truncated ? 'State truncated' : '');
          recordRun(historyTicket, summary, latency);
        } else {
          const snapshot = embeddingSnapshot();
          submitted = true;
          const data = await search(snapshot, controller.signal);
          if (!current()) return;
          vectors = data.nextCache;
          cacheStatus();
          ranking(data);
          const best = data.matches[0], latency = elapsed(start);
          metrics(latency, data.dimensions + ' dimensions', 'Cosine similarity', data.clipped ? 'Query truncated' : '');
          recordRun(historyTicket, 'Top match: document ' + (best.index + 1) + ' · ' + best.score.toFixed(4), latency);
        }
        status('Complete.');
      } catch (error) {
        if (current()) {
          empty('The request did not complete.');
          if (submitted) recordRun(historyTicket, 'Request failed', elapsed(start));
          status(error.message, true);
        }
      } finally {
        if (pending === controller) {
          pending = undefined;
          run.disabled = copy.disabled = stopped;
          result.setAttribute('aria-busy', 'false');
        }
      }
    }
    form.addEventListener('submit', event => { event.preventDefault(); void execute(); });
    const shellQuote = value => "'" + value.replaceAll("'", "'\\''") + "'";
    const curl = (path, body) => 'curl -sS ' + shellQuote(location.origin + path) +
      " -H 'Content-Type: application/json' --data-binary " + shellQuote(JSON.stringify(body));
    copy.onclick = async () => {
      const ticket = revision;
      try {
        let commands;
        if (config.kind === 'decision') {
          commands = [curl(endpoints.decision, await decisionBody(decisionSnapshot()))];
        } else {
          const {docs, query, truncate} = embeddingSnapshot();
          commands = [];
          for (const doc of docs) {
            if (ticket !== revision) return;
            commands.push(curl(endpoints.embedding, await embeddingBody(doc, 'document', truncate)));
          }
          commands.push(curl(endpoints.embedding, await embeddingBody(query, 'query', truncate)));
        }
        if (ticket !== revision) return;
        const command = commands.join('\n');
        byId('curl-text').textContent = command;
        byId('curl-preview').hidden = false;
        try {
          await navigator.clipboard.writeText(command);
          if (ticket === revision) status('Copied curl command' + (commands.length > 1 ? 's.' : '.'));
        } catch {
          if (ticket !== revision) return;
          byId('curl-preview').open = true;
          const selection = window.getSelection(), range = document.createRange();
          range.selectNodeContents(byId('curl-text'));
          selection.removeAllRanges(); selection.addRange(range);
          byId('curl-text').focus();
          status('Curl commands selected. Copy them with your keyboard.');
        }
      } catch (error) {
        if (ticket === revision) status(error.message, true);
      }
    };
    stop.onclick = async () => {
      if (stopped || !window.confirm('Stop this local ANERun server?')) return;
      stopped = true;
      stop.disabled = true;
      resetPage();
      status('Stopping server…');
      try {
        await api('/playground/stop', {});
        status('Shutdown requested.');
      } catch (error) {
        stopped = false;
        stop.disabled = run.disabled = copy.disabled = false;
        status('Could not confirm shutdown: ' + error.message, true);
      }
    };
    async function initialize() {
      try {
        const value = await api('/playground/config');
        if (typeof value.model !== 'string' || typeof value.id !== 'string' ||
            !Object.hasOwn(endpoints, value.kind) || typeof value.images !== 'boolean') {
          throw new Error('Invalid playground configuration.');
        }
        config = value;
        const deciding = config.kind === 'decision';
        byId('model').textContent = config.model;
        byId('title').textContent = deciding ? 'Decision playground' : 'Embedding playground';
        byId('subtitle').textContent = deciding
          ? 'Give the model a state and compare its answers.'
          : 'Search your documents by meaning and inspect their cosine rankings.';
        byId('decision').hidden = !deciding;
        byId('embedding').hidden = deciding;
        byId('decision-image-field').hidden = !deciding || !config.images;
        byId('document-image-field').hidden = byId('query-image-field').hidden = deciding || !config.images;
        run.textContent = deciding ? 'Run decision' : 'Rank documents';
        loadExample();
        byId('workbench').hidden = false;
        stop.disabled = false;
      } catch (error) {
        byId('model').textContent = 'Disconnected';
        status('Could not load local configuration: ' + error.message, true);
      }
    }
    void initialize();
    </script>
    </body>
    </html>
    """#
}
