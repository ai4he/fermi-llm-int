// Fermi LLM - Client Application v3.0
// Features: SSE streaming, demo samples, agent-aware progress tracking,
//           granular Master Agent progress, ngrok compatibility

let sessionId = null;
let ws = null;
let pipelinePolling = null;
let pipelineSSE = null;
let chatSSE = null;
let availableModels = [];
let demoSamples = [];
// Regenerated code awaiting user approval (feedback #7): lang -> proposed text
const pendingProposals = { yaml: null, python: null };
// Id of a Validation Agent script proposal shown in the Python diff view;
// Apply/Discard report the decision to /script_review.
let validatorProposalId = null;
// YAML as submitted to the last pipeline run, to diff against the executed
// version the Validation Agent may have repaired (feedback #8)
let lastRunYaml = null;
let lastRunPython = null;
// Token binding the next click to the exact preflight-reviewed YAML/Python.
let pendingRunApproval = null;
// Deterministic Revert (reviewer: "revert last changes fails" via LLM-only undo):
// per-editor stacks of previous content, capped at 20 entries each.
const editorHistory = { yaml: [], python: [] };
// Last "committed" content per editor, used to detect manual edits in saveCodes
// so no-op saves don't push a redundant history entry.
const editorLastKnown = { yaml: '', python: '' };

// Ngrok-compatible fetch wrapper: adds header to skip interstitial page
const NGROK_HEADERS = { 'ngrok-skip-browser-warning': 'true' };

// --- Auth state (Google Sign-In; the app is fully usable as a guest) ---
const AUTH_TOKEN_KEY = 'fermi_auth_token';
const LAST_SESSION_KEY = 'fermi_session_id';
let authToken = localStorage.getItem(AUTH_TOKEN_KEY) || null;
let currentUser = null;               // {sub,email,name,picture} when signed in
let googleClientId = null;            // from /api/auth/config

function apiFetch(url, options = {}) {
    const headers = { ...NGROK_HEADERS, ...(options.headers || {}) };
    // Attach our session token so the server can associate tasks with the user.
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
    options.headers = headers;
    return fetch(url, options);
}

function apiEventSource(url) {
    // EventSource doesn't support custom headers, so use fetch-based SSE
    return new FetchEventSource(url);
}

// Fetch-based EventSource that supports custom headers (for ngrok)
class FetchEventSource {
    constructor(url) {
        this.url = url;
        this.onmessage = null;
        this.onerror = null;
        this._abortController = new AbortController();
        this._start();
    }

    async _start() {
        try {
            const resp = await fetch(this.url, {
                headers: NGROK_HEADERS,
                signal: this._abortController.signal,
            });
            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                const lines = buffer.split('\n');
                buffer = lines.pop(); // keep incomplete line

                for (const line of lines) {
                    if (line.startsWith('data: ') && this.onmessage) {
                        this.onmessage({ data: line.slice(6) });
                    }
                    // Ignore comment lines (heartbeats)
                }
            }
        } catch (err) {
            if (err.name !== 'AbortError' && this.onerror) {
                this.onerror(err);
            }
        }
    }

    close() {
        this._abortController.abort();
    }
}

// ============================================================
// Model management
// ============================================================

async function loadModels() {
    try {
        const resp = await apiFetch('/api/models');
        const data = await resp.json();
        availableModels = data.models || [];
        populateModelSelector(
            availableModels, data.gpu_available, data.gpu_info,
            data.default_model || 'gpt-5.6-luna');
    } catch (err) {
        console.error('Failed to load models:', err);
    }
}

function populateModelSelector(models, gpuAvailable, gpuInfo, defaultModelId) {
    const select = document.getElementById('model-select');
    select.innerHTML = '';

    const groups = {};
    models.forEach(m => {
        if (!groups[m.group]) groups[m.group] = [];
        groups[m.group].push(m);
    });

    for (const [groupName, groupModels] of Object.entries(groups)) {
        const optgroup = document.createElement('optgroup');
        optgroup.label = groupName;

        groupModels.forEach(m => {
            const option = document.createElement('option');
            option.value = m.id;
            let label = m.name;
            if (!m.available) {
                label += ` (${m.status_detail || 'unavailable'})`;
                option.disabled = true;
            }
            if (m.recommended) label += ' [RECOMMENDED]';
            option.textContent = label;
            optgroup.appendChild(option);
        });

        select.appendChild(optgroup);
    }

    // Prefer the server-configured default (Luna API by default). If it is not
    // available in this deployment, fall back to another usable real model,
    // with the no-LLM template as the last resort.
    const avail = models.filter(m => m.available && m.id !== 'template');
    const best =
        avail.find(m => m.id === defaultModelId) ||
        avail.find(m => m.recommended) ||
        avail.find(m => m.backend === 'gemini') ||
        avail[0] ||
        models.find(m => m.id === 'template');
    if (best) select.value = best.id;

    const gpuEl = document.getElementById('gpu-info');
    gpuEl.textContent = gpuAvailable ? gpuInfo : 'CPU only';
    gpuEl.title = gpuAvailable ? gpuInfo : 'No GPU detected - local models unavailable';

    select.addEventListener('change', updateModelStatus);
    updateModelStatus();
}

function updateModelStatus() {
    const modelId = document.getElementById('model-select').value;
    const dot = document.getElementById('model-status');
    const model = availableModels.find(m => m.id === modelId);

    if (!model) {
        dot.className = 'model-status-dot';
        dot.title = '';
        return;
    }

    dot.className = `model-status-dot ${model.status}`;
    if (model.loaded) {
        dot.className = 'model-status-dot available';
        dot.title = 'Model loaded';
    } else if (model.available) {
        dot.className = 'model-status-dot available';
        dot.title = 'Model available';
    } else {
        dot.title = model.status_detail || 'Unavailable';
    }
}

function getSelectedModel() {
    return document.getElementById('model-select').value;
}

// ============================================================
// Demo samples
// ============================================================

async function loadDemoSamples() {
    try {
        const resp = await apiFetch('/api/demo_samples');
        const data = await resp.json();
        demoSamples = data.samples || [];
        renderDemoSamples();
    } catch (err) {
        console.error('Failed to load demo samples:', err);
    }
}

function renderDemoSamples() {
    const localContainer = document.getElementById('demo-local');
    const geminiContainer = document.getElementById('demo-gemini');
    localContainer.innerHTML = '';
    geminiContainer.innerHTML = '';

    demoSamples.forEach(sample => {
        const card = document.createElement('div');
        card.className = `demo-card ${sample.speed || ''}`;
        card.onclick = () => selectDemoSample(sample);

        const speedBadge = sample.speed === 'fast'
            ? '<span class="speed-badge fast">FAST</span>'
            : '<span class="speed-badge slow">LONGER</span>';

        const iconSvg = sample.category === 'local'
            ? '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M20 18c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2H4c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2H0v2h24v-2h-4zM4 6h16v10H4V6z"/></svg>'
            : '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96z"/></svg>';

        card.innerHTML = `
            <div class="demo-card-header">
                <span class="demo-icon ${sample.category}">${iconSvg}</span>
                <div class="demo-card-title">${escapeHtml(sample.title)}</div>
                ${speedBadge}
            </div>
            <div class="demo-card-subtitle">${escapeHtml(sample.subtitle)}</div>
            <div class="demo-card-model">▶ Runs with the model selected above</div>
        `;

        if (sample.category === 'local') {
            localContainer.appendChild(card);
        } else {
            geminiContainer.appendChild(card);
        }
    });
}

function selectDemoSample(sample) {
    // Demo samples are just example prompts: run them with whatever model is
    // currently selected in the top bar (do NOT override the selection).
    const selectedId = getSelectedModel();
    const selected = availableModels.find(m => m.id === selectedId);
    if (selected && !selected.available) {
        setStatus('The model selected in the top bar is not available — pick another.');
        return;
    }

    // Load the prompt and send with the selected model.
    document.getElementById('chat-input').value = sample.prompt;
    document.getElementById('demo-samples').classList.add('hidden');
    sendMessage();
}

// ============================================================
// Session management
// ============================================================

async function initSession() {
    const resp = await apiFetch('/api/session/create', { method: 'POST' });
    const data = await resp.json();
    setActiveSession(data.session_id);
    upsertGuestTask(data.session_id, 'New task');
    if (typeof renderTaskList === 'function') renderTaskList();
    connectWebSocket();
    await loadModels();
    await loadDemoSamples();
    loadVersions();
}

// Single place that records the active session id — also persisted to
// localStorage so a guest (or a returning user) resumes it on next load.
function setActiveSession(id) {
    sessionId = id;
    document.getElementById('session-id').textContent = `Session: ${sessionId}`;
    try { localStorage.setItem(LAST_SESSION_KEY, id); } catch (e) {}
}

async function loadVersions() {
    try {
        const resp = await apiFetch('/api/versions');
        const v = await resp.json();
        const el = document.getElementById('version-info');
        if (!el) return;
        const bits = [];
        if (v.fermipy) bits.push(`FermiPy ${v.fermipy}`);
        if (v.fermitools) bits.push(`ScienceTools ${v.fermitools}`);
        el.textContent = bits.length ? '· ' + bits.join(' · ') : '';
    } catch (err) {
        /* non-fatal */
    }
}

async function newSession() {
    if (ws) ws.close();
    if (pipelineSSE) { pipelineSSE.close(); pipelineSSE = null; }
    if (pipelinePolling) { clearInterval(pipelinePolling); pipelinePolling = null; }

    await initSession();
    document.getElementById('chat-messages').innerHTML = `
        <div class="message assistant">
            <div class="message-header">
                <span class="agent-tag master">Master Agent</span>
            </div>
            <div class="message-body">New session started. Select a model above and describe your Fermi-LAT analysis task, or pick a demo sample below.</div>
        </div>`;
    // Re-render demo samples
    const demoEl = document.getElementById('demo-samples');
    if (demoEl) {
        demoEl.classList.remove('hidden');
        renderDemoSamples();
    } else {
        // Re-create demo section
        const chatMessages = document.getElementById('chat-messages');
        const demoDiv = document.createElement('div');
        demoDiv.id = 'demo-samples';
        demoDiv.className = 'demo-samples';
        demoDiv.innerHTML = `
            <div class="demo-section-title">Demo Samples</div>
            <div class="demo-category">
                <div class="demo-category-label">Local GPU Model</div>
                <div id="demo-local" class="demo-cards"></div>
            </div>
            <div class="demo-category">
                <div class="demo-category-label">Gemini Cloud API</div>
                <div id="demo-gemini" class="demo-cards"></div>
            </div>`;
        chatMessages.appendChild(demoDiv);
        renderDemoSamples();
    }

    resetPanels();
}

// Shared reset for every stateful panel: editors, results, outputs,
// transparency log, notebook and pipeline progress. Used by both
// newSession() and loadSession() so no panel carries content over from
// the previously displayed session.
function resetPanels() {
    closeProposal('yaml');
    closeProposal('python');
    editorHistory.yaml = [];
    editorHistory.python = [];
    editorLastKnown.yaml = '';
    editorLastKnown.python = '';
    updateRevertButton('yaml');
    updateRevertButton('python');
    lastRunYaml = null;
    lastRunPython = null;
    pendingRunApproval = null;
    setEditorValue('yaml', '');
    setEditorValue('python', '');
    document.getElementById('results-content').innerHTML = '<div class="results-placeholder">Run the pipeline to see results.</div>';
    const outContent = document.getElementById('outputs-content');
    if (outContent) outContent.innerHTML = '<div class="results-placeholder">Run the pipeline to visualize the produced outputs.</div>';
    const outBadge = document.getElementById('outputs-badge');
    if (outBadge) { outBadge.className = 'pipeline-badge idle'; outBadge.textContent = 'No results yet'; }
    document.getElementById('explain-log').innerHTML = '';
    const nbContent = document.getElementById('notebook-content');
    if (nbContent) nbContent.innerHTML = '<div class="results-placeholder">Generate an analysis to export it as a Jupyter notebook.</div>';
    const nbBadge = document.getElementById('notebook-badge');
    if (nbBadge) { nbBadge.className = 'pipeline-badge idle'; nbBadge.textContent = 'Not generated'; }
    const nbDl = document.getElementById('notebook-download');
    if (nbDl) nbDl.style.display = 'none';
    updatePipelineStatus('idle');
    resetProgressSteps();
    document.getElementById('pipeline-progress').classList.add('hidden');
}

// ============================================================
// WebSocket (fallback)
// ============================================================

function connectWebSocket() {
    const connectedSessionId = sessionId;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${connectedSessionId}`;
    const socket = new WebSocket(wsUrl);
    ws = socket;

    socket.onopen = () => {
        if (socket !== ws || connectedSessionId !== sessionId) return;
        setStatus('Connected');
    };

    socket.onmessage = (event) => {
        // A run may continue after the user switches tasks. Ignore late
        // events from that task so they cannot change the current UI.
        if (socket !== ws || connectedSessionId !== sessionId) return;
        const msg = JSON.parse(event.data);
        if (msg.type === 'pipeline_result') {
            handlePipelineResult(msg.result, msg.message);
        }
    };

    socket.onclose = () => {
        if (socket !== ws || connectedSessionId !== sessionId) return;
        setStatus('Disconnected');
        setTimeout(() => {
            if (socket === ws && connectedSessionId === sessionId) connectWebSocket();
        }, 5000);
    };

    socket.onerror = () => {
        if (socket !== ws || connectedSessionId !== sessionId) return;
        setStatus('Connection error');
    };

    // Heartbeat every 30s
    setInterval(() => {
        if (socket === ws && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: 'ping' }));
        }
    }, 30000);
}

// ============================================================
// Chat
// ============================================================

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (!message) return;

    input.value = '';
    document.getElementById('send-btn').disabled = true;

    // Hide demo samples
    const demoEl = document.getElementById('demo-samples');
    if (demoEl) demoEl.classList.add('hidden');

    const selectedModel = getSelectedModel();

    addChatMessage('user', message);

    // Title the task from its first message (shown in the sidebar list).
    const _t = getGuestTasks().find(x => x.id === sessionId);
    if (!_t || !_t.title || _t.title === 'New task') {
        upsertGuestTask(sessionId, message.slice(0, 100));
    }
    renderTaskList();

    // Show Master Agent work in the global status and transparency panel.
    // The multi-step progress bar is reserved for an actual pipeline run.
    setStatus('Master Agent: Analyzing prompt...');
    setProgressStep('master', 'active', 'Analyzing prompt and consulting documentation...');
    addExplainEntry('master', `Analyzing prompt with model: ${selectedModel}...`);

    document.getElementById('model-status').className = 'model-status-dot loading';

    // Start chat SSE BEFORE sending the POST so we catch all progress events
    startChatSSE();

    try {
        const resp = await apiFetch(`/api/session/${sessionId}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, model: selectedModel }),
        });
        const data = await resp.json();

        // Stop chat SSE now that we have the final response
        stopChatSSE();

        addChatMessage('assistant', data.response, 'master');
        renderTaskList();   // signed-in: pick up the server-assigned title/status

        // If the editors already hold a config, don't overwrite it silently:
        // show a diff and ask for confirmation (Apply changes / Discard).
        if (data.yaml) proposeCode('yaml', data.yaml);
        if (data.python) proposeCode('python', data.python);

        if (data.duration_estimate && data.duration_estimate.text) {
            addExplainEntry('master', `Estimated run time: ${data.duration_estimate.text}`);
        }

        if (data.analysis) {
            const a = data.analysis;
            if (a.rationale) {
                a.rationale.forEach(r => addExplainEntry('master', r));
            }
            if (a.decisions) {
                a.decisions.forEach(d => addExplainEntry('master', `Decision: ${d}`));
            }
            if (a.rag_context) {
                addExplainEntry('master', `RAG: Consulted ${a.rag_context.join(', ')}`);
            }
        }

        if (data.gen_meta) {
            const gm = data.gen_meta;
            addExplainEntry('master', `Model: ${gm.model_name} (${gm.backend}), ${gm.gen_time || '?'}s`);
        }

        setProgressStep('master', 'pass', 'Configuration generated successfully');
        addExplainEntry('master', 'Configuration generated. Ready to run pipeline.');
        setStatus('Master Agent: Configuration ready. Click Run Pipeline.');
        hideProgressBar();

    } catch (err) {
        stopChatSSE();
        addChatMessage('assistant', `Error: ${err.message}. Please try again.`, 'system');
        setStatus('Error');
        setProgressStep('master', 'fail', `Error: ${err.message}`);
        hideProgressBar();
    }

    document.getElementById('send-btn').disabled = false;
    updateModelStatus();
}

// ============================================================
// Chat SSE: real-time Master Agent progress during generation
// ============================================================

function startChatSSE() {
    stopChatSSE();
    const url = `/api/session/${sessionId}/chat_stream`;
    chatSSE = apiEventSource(url);

    chatSSE.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleChatProgress(data);
        } catch (e) {
            // ignore
        }
    };

    chatSSE.onerror = () => {
        // Don't close — the POST response will close it
    };
}

function stopChatSSE() {
    if (chatSSE) {
        chatSSE.close();
        chatSSE = null;
    }
}

function handleChatProgress(data) {
    const stage = data.stage || '';
    const detail = data.detail || '';

    // Map stages to user-friendly Master Agent card text
    const stageMessages = {
        'analyzing':     'Analyzing prompt...',
        'analyzed':      detail,
        'rag':           'Searching documentation (RAG)...',
        'rag_done':      detail,
        'model_start':   detail,
        'prompt':        detail,
        'prompt_ready':  detail,
        'model_loading': detail,
        'model_ready':   detail,
        'inference':     detail,
        'api_call':      detail,
        'response':      detail,
        'parsing':       'Extracting YAML and Python from response...',
        'done':          detail,
        'template':      'Generating via template (no LLM)...',
        'template_done': 'Template generation complete',
        'model_error':   detail,
        'complete':      'Configuration generated successfully',
    };

    const displayText = stageMessages[stage] || detail || stage;

    // Keep Master Agent card in "active" state with updated detail
    if (stage !== 'complete') {
        setProgressStep('master', 'active', displayText);
    }
    setStatus(`Master Agent: ${displayText}`);
    addExplainEntry('master', displayText);
}

function addChatMessage(role, content, agent) {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = `message ${role}`;

    if (role === 'assistant') {
        const agentClass = agent || 'master';
        const agentLabel = agentClass === 'master' ? 'Master Agent'
            : agentClass === 'validation' ? 'Validation Agent'
            : 'System';
        div.innerHTML = `
            <div class="message-header">
                <span class="agent-tag ${agentClass}">${agentLabel}</span>
            </div>
            <div class="message-body">${formatMarkdown(content)}</div>`;
    } else {
        div.innerHTML = `<div class="message-body">${escapeHtml(content)}</div>`;
    }

    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

// ============================================================
// Pipeline execution with SSE streaming
// ============================================================

async function runPipeline(confirmed = false) {
    if (!sessionId) return;

    if (pendingProposals.yaml != null || pendingProposals.python != null) {
        setStatus('Review the proposed changes first — Apply or Discard them before running.');
        return;
    }

    await saveCodes();
    lastRunYaml = document.getElementById('yaml-editor').value;
    lastRunPython = document.getElementById('python-editor').value;

    setRunControls(true);
    updatePipelineStatus('running');
    showProgressBar();
    resetProgressSteps();

    // Set Master Agent step as done (config was already generated)
    setProgressStep('master', 'pass', 'Configuration generated');

    addExplainEntry('master', 'Handing off to Validation Agent for pipeline execution...');
    addExplainEntry('validation', 'Starting pipeline execution...');
    setStatus('Validation Agent: Starting pipeline...');

    showPanel('results');

    document.getElementById('results-content').innerHTML = `
        <div class="result-card running">
            <h4>Pipeline Running</h4>
            <p>The <strong>Validation Agent</strong> is checking the configuration and reviewing analysis.py.</p>
            <div class="pipeline-live-log" id="pipeline-live-log"></div>
        </div>`;

    try {
        const resp = await apiFetch(`/api/session/${sessionId}/run_pipeline`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                confirm: confirmed,
                approval_token: pendingRunApproval,
            }),
        });
        const data = await resp.json();

        if (!resp.ok) {
            throw new Error(data.detail || data.message || `Pipeline request failed (HTTP ${resp.status})`);
        }

        if (data.status === 'confirmation_required') {
            // Long run: back out of the "running" UI state and ask first.
            updatePipelineStatus('idle');
            hideProgressBar();
            resetProgressSteps();
            document.getElementById('results-content').innerHTML = '';
            setStatus('Awaiting confirmation — this run has a long estimated duration.');
            showRunConfirmDialog(data);
            return;
        }

        if (data.status === 'script_changes_proposed' || data.status === 'script_blocked') {
            showScriptReview(data);
            return;
        }

        if (data.status === 'review_required') {
            // Preflight may repair the YAML; the script is the model's own
            // analysis.py, reviewed by the Validation Agent, run as shown.
            pendingRunApproval = data.approval_token;
            updatePipelineStatus('review_required');
            hideProgressBar();
            resetProgressSteps();
            setEditorValue('yaml', data.yaml || '');
            setEditorValue('python', data.python || '');
            editorLastKnown.yaml = data.yaml || '';
            editorLastKnown.python = data.python || '';
            setCodeStatus('yaml', 'active');
            setCodeStatus('python', 'active');
            const repairs = (data.repairs || [])
                .map(item => `<li>${escapeHtml(item)}</li>`).join('');
            const bins = data.expected_sed_bins == null ? ''
                : `<p><strong>Expected SED bins:</strong> ${data.expected_sed_bins}</p>`;
            document.getElementById('results-content').innerHTML = `
                <div class="result-card warn">
                    <h4>Review the exact files before execution</h4>
                    <p>${escapeHtml(data.message || '')}</p>
                    ${bins}
                    ${repairs ? `<details open><summary>Validator changes to the YAML</summary><ul>${repairs}</ul></details>` :
                        '<p>No deterministic YAML repairs were needed.</p>'}
                    ${renderReviewFindings(data)}
                    <p><strong>Nothing has run yet.</strong> Review both editor
                    panels, then click <strong>Run Pipeline</strong> again.</p>
                </div>`;
            addExplainEntry(
                'validation',
                'Preflight complete — waiting for approval of the exact final YAML and analysis.py.');
            setStatus('Review required — click Run Pipeline again to execute these exact files.');
            return;
        }

        pendingRunApproval = null;
        addExplainEntry('validation', data.message || 'Pipeline started');

        if (data.estimate) startEtaTimer(data.estimate);

        // Start SSE streaming for real-time progress
        startPipelineSSE();

        // Also start polling as fallback (less frequent)
        startPipelinePolling();

    } catch (err) {
        updatePipelineStatus('error');
        hideProgressBar();
        resetProgressSteps();
        addExplainEntry('validation', `Error: ${err.message}`);
        setProgressStep('repair', 'fail', `Error: ${err.message}`);
        setStatus('Pipeline error');
    }
}

function abortPipeline() {
    if (!sessionId) return;

    const existing = document.getElementById('abort-confirm-overlay');
    if (existing) return;

    const abortSessionId = sessionId;
    addExplainEntry('system', 'Abort Run selected — waiting for confirmation.');
    setStatus('Confirm whether to stop the running pipeline.');

    const overlay = document.createElement('div');
    overlay.id = 'abort-confirm-overlay';
    overlay.innerHTML = `
        <div class="run-confirm-box abort-confirm-box" role="dialog"
             aria-modal="true" aria-labelledby="abort-confirm-title">
            <h3 id="abort-confirm-title">Abort this run?</h3>
            <p>The active pipeline and its child processes will be stopped.
               Partial output files may remain in this task.</p>
            <div class="run-confirm-actions">
                <button id="abort-confirm-cancel" class="btn-secondary">Keep Running</button>
                <button id="abort-confirm-stop" class="btn-danger">Yes, Abort Run</button>
            </div>
        </div>`;
    document.body.appendChild(overlay);

    const cancel = () => {
        overlay.remove();
        addExplainEntry('system', 'Abort cancelled — pipeline will keep running.');
        setStatus('Pipeline continues to run.');
    };
    document.getElementById('abort-confirm-cancel').onclick = cancel;
    overlay.onclick = (event) => {
        if (event.target === overlay) cancel();
    };
    document.getElementById('abort-confirm-stop').onclick = () => {
        overlay.remove();
        executeAbortPipeline(abortSessionId);
    };
    document.getElementById('abort-confirm-stop').focus();
}

async function executeAbortPipeline(abortSessionId) {
    if (!abortSessionId) return;

    setRunControls(true, true);
    setStatus('Stopping the running pipeline...');
    addExplainEntry('system', 'Abort confirmed — sending the stop request...');
    addExplainEntry('validation', 'Stopping this run and its child processes...');

    try {
        const resp = await apiFetch(`/api/session/${abortSessionId}/abort_pipeline`, {
            method: 'POST',
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(data.detail || 'Unable to abort the run');

        // The user may switch tasks while the server is stopping the worker.
        // The request still targets the task they confirmed, but its result
        // must not overwrite whichever task is now on screen.
        if (sessionId !== abortSessionId) return;
        if (pipelineSSE) { pipelineSSE.close(); pipelineSSE = null; }
        if (pipelinePolling) { clearInterval(pipelinePolling); pipelinePolling = null; }
        stopEtaTimer(false);
        addExplainEntry('system', 'Abort completed — the pipeline process was stopped.');
        handlePipelineResult(data.result, null);
    } catch (err) {
        if (sessionId !== abortSessionId) return;
        // The process may still be running, so keep Abort available for a
        // retry instead of pretending that the run stopped.
        setRunControls(true, false);
        setStatus(`Could not abort pipeline: ${err.message}`);
        addExplainEntry('system', `Abort failed — ${err.message}`);
    }
}

// ============================================================
// Time-based ETA bar for a running pipeline
// ============================================================
// The wall-clock estimate is heuristic (roughly 0.6-1.8x), so the bar
// tracks elapsed time against the estimate midpoint and never claims
// 100% until the run actually finishes.

let etaTimer = null;
let etaStartMs = null;
let etaMidMin = null;
let etaLoMin = null;
let etaHiMin = null;

function fmtMinutes(m) {
    if (m >= 90) return `${(m / 60).toFixed(1)} h`;
    return `${Math.max(1, Math.round(m))} min`;
}

function startEtaTimer(estimate, startedAtSec) {
    if (!estimate || !estimate.total_hi_min) return;
    stopEtaTimer();
    etaStartMs = startedAtSec ? startedAtSec * 1000 : Date.now();
    etaLoMin = estimate.total_lo_min;
    etaHiMin = estimate.total_hi_min;
    etaMidMin = (estimate.total_lo_min + estimate.total_hi_min) / 2;
    ensureEtaBar();
    updateEtaBar();
    etaTimer = setInterval(updateEtaBar, 1000);
}

function ensureEtaBar() {
    if (document.getElementById('eta-container')) return;
    const log = document.getElementById('pipeline-live-log');
    const host = log ? log.parentElement
        : document.getElementById('results-content');
    if (!host) return;
    const div = document.createElement('div');
    div.id = 'eta-container';
    div.innerHTML = `
        <div class="eta-track"><div class="eta-fill" id="eta-fill"></div></div>
        <div class="eta-label" id="eta-label"></div>`;
    if (log) host.insertBefore(div, log); else host.prepend(div);
}

function updateEtaBar() {
    const fill = document.getElementById('eta-fill');
    const label = document.getElementById('eta-label');
    if (!fill || !label) { ensureEtaBar(); return; }
    const elapsedMin = (Date.now() - etaStartMs) / 60000;
    const pct = Math.min(97, (elapsedMin / etaMidMin) * 100);
    fill.style.width = `${pct}%`;
    if (elapsedMin < etaHiMin) {
        const remaining = Math.max(etaMidMin - elapsedMin, 0.5);
        label.textContent =
            `elapsed ${fmtMinutes(elapsedMin)} · estimated total ` +
            `${fmtMinutes(etaLoMin)}–${fmtMinutes(etaHiMin)} · ` +
            `~${fmtMinutes(remaining)} remaining (estimate)`;
    } else {
        label.textContent =
            `elapsed ${fmtMinutes(elapsedMin)} · exceeded the ` +
            `${fmtMinutes(etaHiMin)} upper estimate — still running ` +
            `(complex fits can legitimately take longer)`;
    }
}

function stopEtaTimer(finished) {
    if (etaTimer) { clearInterval(etaTimer); etaTimer = null; }
    if (finished && etaStartMs) {
        const fill = document.getElementById('eta-fill');
        const label = document.getElementById('eta-label');
        const elapsedMin = (Date.now() - etaStartMs) / 60000;
        if (fill) fill.style.width = '100%';
        if (label) label.textContent =
            `finished in ${fmtMinutes(elapsedMin)}`;
    }
    if (!finished) etaStartMs = null;
}

function showRunConfirmDialog(data) {
    // In-app modal (not window.confirm) so the estimate breakdown is
    // readable and the page stays scriptable.
    const existing = document.getElementById('run-confirm-overlay');
    if (existing) existing.remove();

    const est = data.estimate || {};
    const rows = (est.breakdown || [])
        .map(b => `<tr><td>${b.stage}</td><td class="rc-min">~${b.minutes} min</td></tr>`)
        .join('');

    const overlay = document.createElement('div');
    overlay.id = 'run-confirm-overlay';
    overlay.innerHTML = `
        <div class="run-confirm-box">
            <h3>Long analysis — confirm before starting</h3>
            <p>${data.message || 'This run is estimated to take a long time.'}</p>
            ${rows ? `<table class="run-confirm-table">${rows}</table>` : ''}
            <p class="rc-hint">Tip: shrink <code>selection.tmin</code>/<code>tmax</code>
            in the YAML panel to analyze a shorter time range.</p>
            <div class="run-confirm-actions">
                <button id="rc-cancel" class="btn-secondary">Cancel</button>
                <button id="rc-start" class="btn-primary">Start anyway</button>
            </div>
        </div>`;
    document.body.appendChild(overlay);

    document.getElementById('rc-cancel').onclick = () => {
        overlay.remove();
        setStatus('Run cancelled — adjust the time range or start again when ready.');
    };
    document.getElementById('rc-start').onclick = () => {
        overlay.remove();
        runPipeline(true);
    };
}

function startPipelineSSE() {
    if (pipelineSSE) { pipelineSSE.close(); pipelineSSE = null; }

    const url = `/api/session/${sessionId}/pipeline_stream`;
    pipelineSSE = apiEventSource(url);

    pipelineSSE.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleProgressUpdate(data);
        } catch (e) {
            // Ignore parse errors (heartbeats, etc.)
        }
    };

    pipelineSSE.onerror = () => {
        // SSE reconnects automatically, but if pipeline is done, close it
        if (pipelineSSE) {
            pipelineSSE.close();
            pipelineSSE = null;
        }
    };
}

function handleProgressUpdate(data) {
    const step = data.step;
    const status = data.status;
    const detail = data.detail || '';
    const agent = data.agent || 'system';

    // Estimate line replayed via SSE: (re)start the ETA bar, e.g. after a
    // page reload mid-run. The event's own payload carries the start time.
    if (step === 'run_started' && data.data && data.data.estimate && !etaTimer) {
        startEtaTimer(data.data.estimate, data.data.started_at);
    }

    // Update progress stepper
    if (step && step !== 'done' && step !== 'init' && step !== 'result'
        && step !== 'error' && step !== 'level4_optimize') {
        setProgressStep(step, status, detail);
    }

    // Update live log in results panel
    appendLiveLog(detail, agent, status);

    // Update explain panel
    if (detail) {
        addExplainEntry(agent === 'master' ? 'master' : 'validation', detail);
    }

    // Update status bar
    if (detail) {
        const agentLabel = agent === 'master' ? 'Master Agent'
            : agent === 'validation' ? 'Validation Agent' : '';
        setStatus(`${agentLabel}: ${detail.substring(0, 100)}`);
    }

    // Handle final result
    if (step === 'result' && data.result) {
        stopEtaTimer(data.result.status !== 'aborted');
        // Close SSE
        if (pipelineSSE) { pipelineSSE.close(); pipelineSSE = null; }
        if (pipelinePolling) { clearInterval(pipelinePolling); pipelinePolling = null; }
        handlePipelineResult(data.result, null);
    }

    if (step === 'done') {
        stopEtaTimer(status !== 'aborted');
        if (pipelineSSE) { pipelineSSE.close(); pipelineSSE = null; }
        // Polling will pick up the final result
    }
}

function appendLiveLog(text, agent, status) {
    const log = document.getElementById('pipeline-live-log');
    if (!log) return;

    const entry = document.createElement('div');
    entry.className = `live-log-entry ${status || ''} ${agent || ''}`;

    const statusIcon = status === 'pass' ? '&#10003;'
        : status === 'fail' ? '&#10007;'
        : status === 'active' ? '&#9679;'
        : '&#8226;';

    const agentLabel = agent === 'master' ? '<span class="log-agent master">Master</span>'
        : agent === 'validation' ? '<span class="log-agent validation">Validator</span>'
        : '<span class="log-agent system">System</span>';

    entry.innerHTML = `<span class="log-status-icon ${status}">${statusIcon}</span> ${agentLabel} ${escapeHtml(text)}`;
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
}

function startPipelinePolling() {
    if (pipelinePolling) clearInterval(pipelinePolling);
    // Poll every 5 seconds as fallback (reduced from 3s for ngrok friendliness)
    pipelinePolling = setInterval(async () => {
        try {
            const resp = await apiFetch(`/api/session/${sessionId}/pipeline_status`);
            const data = await resp.json();

            if (data.status && data.status !== 'running') {
                stopEtaTimer(data.status !== 'aborted');
                clearInterval(pipelinePolling);
                pipelinePolling = null;
                if (pipelineSSE) { pipelineSSE.close(); pipelineSSE = null; }
                handlePipelineResult(data.result, null);
            }
        } catch (err) {
            // Ignore polling errors
        }
    }, 5000);
}

function handlePipelineResult(result, message) {
    // Announce the finished run so UI plugins can react (core/plugins.js).
    if (window.FermiLLM && window.FermiLLM.emit) {
        window.FermiLLM.emit('run:finished', result);
    }
    pendingRunApproval = null;
    setRunControls(false);

    if (!result) return;

    hideProgressBar();

    const status = result.status === 'complete_l4' ? 'complete' : (result.success ? 'complete' : (result.status || 'error'));
    updatePipelineStatus(status);

    // Update final progress step states
    if (result.success || result.status === 'complete_l4') {
        // Finalize the repair step (it is only ever set to "active" while the
        // validate-repair loop runs and otherwise blinks indefinitely).
        setProgressStep('repair', 'pass', 'Validation/repair complete');
        setProgressStep('level3', 'pass', 'gta.setup() completed');
        if (result.status === 'complete_l4') {
            setProgressStep('level4', 'pass', 'Science-quality fit completed');
        } else if (result.level4 && !result.level4.level4_pass) {
            setProgressStep('level4', 'partial', 'Fit attempted but did not fully pass');
        }
    }

    // Update YAML if repaired (and refresh its syntax highlight). When the
    // executed YAML differs from what was submitted, build a line diff so the
    // user can see exactly what the Validation Agent changed (feedback #8).
    let valDiffCard = '';
    if (result.final_yaml) {
        if (lastRunYaml != null && result.final_yaml !== lastRunYaml) {
            const ops = diffLines(lastRunYaml, result.final_yaml);
            if (ops) {
                const changed = ops.filter(o => o[0] !== '=').length;
                valDiffCard = `
                    <div class="result-card warn">
                        <h4>Validation Agent modified the YAML before execution (${changed} line${changed === 1 ? '' : 's'})</h4>
                        <p>The YAML panel now shows the executed version.</p>
                        <details>
                            <summary>Show diff (submitted &rarr; executed)</summary>
                            <pre class="diff-view inline">${renderDiffHtml(ops)}</pre>
                        </details>
                    </div>`;
                addExplainEntry('validation', `Executed YAML differs from the submitted one (${changed} changed lines) — see the diff in the Results panel.`);
            }
        }
        setEditorValue('yaml', result.final_yaml);
        // Keep the manual-edit tracker in sync so a subsequent Save Edits
        // doesn't mistake the Validation Agent's own rewrite for a user edit.
        editorLastKnown.yaml = result.final_yaml;
    }

    // Replace the draft Python with the runnable script generated from the
    // final validated YAML, so the panel, notebook and downloads stay aligned.
    let pythonDiffCard = '';
    if (result.final_python) {
        if (lastRunPython != null && result.final_python !== lastRunPython) {
            const ops = diffLines(lastRunPython, result.final_python);
            if (ops) {
                const changed = ops.filter(o => o[0] !== '=').length;
                pythonDiffCard = `
                    <div class="result-card info">
                        <h4>Python updated to match the final validated YAML (${changed} changed line${changed === 1 ? '' : 's'})</h4>
                        <p>The Python panel and downloads now show the runnable final script.</p>
                        <details>
                            <summary>Show diff (draft &rarr; final runnable script)</summary>
                            <pre class="diff-view inline">${renderDiffHtml(ops)}</pre>
                        </details>
                    </div>`;
            }
        }
        setEditorValue('python', result.final_python);
        editorLastKnown.python = result.final_python;
    }

    // Build results display
    let html = '';

    // Incremental-run badge: the pipeline reused a previous session's setup
    // and fit instead of redoing them from scratch.
    if (result.run_mode === 'incremental') {
        const info = result.incremental_info || {};
        const skipped = (info.skipped_stages || []).join(', ');
        html += `
            <div class="result-card info">
                <h4>Incremental run</h4>
                <p>Reused the previous setup and fit${skipped ? ` (skipped: ${escapeHtml(skipped)})` : ''}.</p>
                ${info.note ? `<p>${escapeHtml(info.note)}</p>` : ''}
            </div>`;
    }

    // Download links: the run logs are always available once a run
    // finishes. Successful validated runs also expose their exact final YAML
    // and matching standalone Python, individually and as a small bundle.
    const downloads = result.downloads || {};
    const hasAnalysisDownloads = Boolean(
        downloads.yaml || downloads.python || downloads.bundle);
    html += `
        <div class="result-card download-card">
            <div class="download-card-header">
                <span class="download-card-icon" aria-hidden="true">&#8595;</span>
                <div>
                    <h4>Downloads</h4>
                    <p>${hasAnalysisDownloads ? 'Final validated files for this run' : 'Run logs and available outputs'}</p>
                </div>
            </div>

            ${downloads.bundle ? `
                <a class="download-bundle-btn" href="${downloads.bundle}" download>
                    <span class="download-bundle-badge">ZIP</span>
                    <span class="download-btn-copy">
                        <strong>YAML + Python</strong>
                        <small>Complete reproducibility bundle</small>
                    </span>
                    <span class="download-btn-arrow" aria-hidden="true">&#8595;</span>
                </a>` : ''}

            ${(downloads.yaml || downloads.python) ? `
                <div class="download-file-grid">
                    ${downloads.yaml ? `
                        <a class="download-file-btn" href="${downloads.yaml}" download>
                            <span class="download-file-badge yaml">YML</span>
                            <span class="download-btn-copy">
                                <strong>Final config</strong>
                                <small>Validated YAML</small>
                            </span>
                            <span class="download-btn-arrow" aria-hidden="true">&#8595;</span>
                        </a>` : ''}
                    ${downloads.python ? `
                        <a class="download-file-btn" href="${downloads.python}" download>
                            <span class="download-file-badge python">PY</span>
                            <span class="download-btn-copy">
                                <strong>Analysis script</strong>
                                <small>Runnable Python</small>
                            </span>
                            <span class="download-btn-arrow" aria-hidden="true">&#8595;</span>
                        </a>` : ''}
                </div>` : ''}

            <div class="download-utilities ${hasAnalysisDownloads ? '' : 'standalone'}">
                <span class="download-section-label">Logs &amp; outputs</span>
                <div class="download-utility-links">
                    <a href="/api/session/${sessionId}/logs/llm" download>LLM log <span aria-hidden="true">&#8595;</span></a>
                    <a href="/api/session/${sessionId}/logs/fermipy" download>FermiPy log <span aria-hidden="true">&#8595;</span></a>
                    ${result.artifacts_zip ? `<a class="outputs" href="${result.artifacts_zip}" download>All outputs <span aria-hidden="true">&#8595;</span></a>` : ''}
                </div>
            </div>
        </div>`;

    if (result.status === 'aborted') {
        html += `
            <div class="result-card aborted">
                <h4>Pipeline Aborted</h4>
                <p>The analysis was stopped at your request. Your YAML and Python edits are still available, and you can start a new run whenever you are ready.</p>
            </div>`;
        const etaLabel = document.getElementById('eta-label');
        if (etaLabel) etaLabel.textContent = 'Run aborted';
        addExplainEntry('validation', 'ABORTED: Pipeline stopped by user.');
        setStatus('Pipeline aborted');
    } else if (result.status === 'complete_l4' && result.science_results) {
        const sci = result.science_results;
        const sources = result.level3?.sources || [];
        const tsStr = sci.target_ts != null ? sci.target_ts.toFixed(1) : 'N/A';
        const fluxStr = sci.target_flux != null ? sci.target_flux.toExponential(2) : 'N/A';
        const fluxErrStr = sci.target_flux_error != null ? ' +/- ' + sci.target_flux_error.toExponential(2) : '';
        const idxStr = sci.spectral_index != null ? sci.spectral_index.toFixed(2) : 'N/A';
        const idxErrStr = sci.spectral_index_error != null ? ' +/- ' + sci.spectral_index_error.toFixed(2) : '';
        const convStr = sci.convergence_ok ? 'Converged' : 'Did not converge';

        html += `
            <div class="result-card success">
                <h4>Level 4 Passed -- Science-Quality Results</h4>
                <table style="width:100%; border-collapse:collapse; margin:8px 0;">
                    <tr><td style="padding:4px 8px;"><strong>Fit Status</strong></td><td style="padding:4px 8px;">${convStr}</td></tr>
                    <tr><td style="padding:4px 8px;"><strong>Target TS</strong></td><td style="padding:4px 8px;">${tsStr}</td></tr>
                    <tr><td style="padding:4px 8px;"><strong>Flux (ph/cm2/s)</strong></td><td style="padding:4px 8px;">${fluxStr}${fluxErrStr}</td></tr>
                    <tr><td style="padding:4px 8px;"><strong>Spectral Index</strong></td><td style="padding:4px 8px;">${idxStr}${idxErrStr}</td></tr>
                </table>
            </div>`;

        const topSrc = sci.sources_summary || [];
        if (topSrc.length > 0) {
            html += `<div class="result-card"><h4>Top Sources by TS</h4><table style="width:100%; border-collapse:collapse;">
                <tr><th style="text-align:left; padding:4px 8px;">Source</th><th style="text-align:right; padding:4px 8px;">TS</th><th style="text-align:right; padding:4px 8px;">Flux</th></tr>`;
            topSrc.slice(0, 8).forEach(s => {
                const tsCell = (s.ts != null) ? s.ts.toFixed(1) : 'N/A';
                const fluxCell = (s.flux != null) ? s.flux.toExponential(2) : 'N/A';
                html += `<tr><td style="padding:4px 8px;">${escapeHtml(s.name)}</td><td style="text-align:right; padding:4px 8px;">${tsCell}</td><td style="text-align:right; padding:4px 8px;">${fluxCell}</td></tr>`;
            });
            html += `</table></div>`;
        }

        html += `<div class="result-card success"><h4>Level 3 Passed</h4>
                <p><strong>Sources in ROI:</strong> ${sources.length > 0 ? sources.join(', ') : 'N/A'}</p></div>`;

        addExplainEntry('validation', `SUCCESS: Level 4 passed. TS=${tsStr}, Flux=${fluxStr}`);
        setStatus('Pipeline complete -- science results ready');

    } else if (result.success) {
        const sources = result.level3?.sources || [];
        html += `
            <div class="result-card success">
                <h4>Level 3 Passed</h4>
                <p><strong>gta.setup()</strong> completed successfully.</p>
                <p><strong>Sources in ROI:</strong> ${sources.length > 0 ? sources.join(', ') : 'N/A'}</p>
            </div>`;

        if (result.level4) {
            const l4 = result.level4;
            if (l4.level4_pass) {
                html += `<div class="result-card success"><h4>Level 4 Passed</h4></div>`;
            } else {
                const l4err = l4.error || 'Fit did not converge';
                html += `<div class="result-card"><h4>Level 4 (Fit)</h4><p>Did not pass: ${escapeHtml(l4err.substring(0, 200))}</p></div>`;
            }
        }

        addExplainEntry('validation', `SUCCESS: Pipeline completed. ${sources.length} sources found.`);
        setStatus('Pipeline complete');
    } else {
        const error = result.level3?.error || result.error || 'Unknown error';
        html += `
            <div class="result-card fail">
                <h4>Pipeline Failed</h4>
                <pre>${escapeHtml(error)}</pre>
            </div>`;
        addExplainEntry('validation', `FAILED: ${error.substring(0, 200)}`);
        setStatus('Pipeline failed');
    }

    // What the Validation Agent changed, as a line diff
    html += valDiffCard + pythonDiffCard;

    // Repair history
    if (result.repair_history && result.repair_history.length > 0) {
        html += `<div class="result-card"><h4>Repair History (Validation Agent)</h4>`;
        result.repair_history.forEach(rh => {
            html += `<p><strong>Iteration ${rh.iteration + 1}:</strong></p><ul>`;
            (rh.repairs || []).forEach(r => {
                html += `<li>${escapeHtml(r)}</li>`;
            });
            html += `</ul>`;
        });
        html += `</div>`;
    }

    // Validation levels
    if (result.validation_results) {
        result.validation_results.forEach((vr, i) => {
            const l1 = vr.level1 || {};
            const l2 = vr.level2 || {};
            const l3 = vr.level3 || {};
            html += `
                <div class="result-card ${l3.setup_ok ? 'success' : 'fail'}">
                    <h4>Validation Iteration ${i + 1}</h4>
                    <p>Level 1 (YAML): ${l1.fermipy_load ? 'PASS' : 'FAIL'} ${l1.error ? `- ${l1.error.substring(0, 100)}` : ''}</p>
                    <p>Level 2 (Init): ${l2.gta_init ? 'PASS' : 'FAIL'} ${l2.error ? `- ${l2.error.substring(0, 100)}` : ''}</p>
                    <p>Level 3 (Setup): ${l3.setup_ok ? 'PASS' : 'FAIL'} ${l3.error ? `- ${l3.error.substring(0, 100)}` : ''}</p>
                </div>`;
        });
    }

    html += renderScriptExecution(result.level4 || {});

    // ROI sources table (top N by TS), when the backend supplied it.
    const roiSources = (result.level4 && result.level4.roi_sources)
        || (result.science_results && result.science_results.roi_sources)
        || [];
    if (Array.isArray(roiSources) && roiSources.length > 0) {
        html += renderRoiSourcesTable(roiSources);
    }

    document.getElementById('results-content').innerHTML = html;

    // Populate the dedicated Outputs panel (plots + charts + science cards).
    // Guarded so a rendering error here can never blank the panel or block
    // the rest of the completion handler.
    try {
        renderOutputs(result);
    } catch (e) {
        console.error('renderOutputs failed:', e);
        const oc = document.getElementById('outputs-content');
        if (oc) oc.innerHTML = `<div class="results-placeholder">Outputs could not be rendered (${escapeHtml(e.message)}), but the run completed. See the Results panel.</div>`;
    }

    if (message) {
        addChatMessage('assistant', message, 'validation');
    }

    // Keep the notebook in sync with the new results/plots if the tab is open.
    refreshNotebookIfVisible();
}

// Collapsible ROI sources table (same <details> pattern as the validation
// diff card). Values are formatted defensively since fields may be null.
function renderRoiSourcesTable(sources) {
    const rows = sources.map(s => {
        const ts = (s.ts != null) ? s.ts.toFixed(1) : '&mdash;';
        const npred = (s.npred != null) ? s.npred.toExponential(2) : '&mdash;';
        const flux = (s.flux != null)
            ? `${s.flux.toExponential(2)}${s.flux_unit ? ' ' + escapeHtml(s.flux_unit) : ''}`
            : '&mdash;';
        const spectrum = s.spectrum_type ? escapeHtml(s.spectrum_type) : '&mdash;';
        const offset = (s.offset_deg != null) ? s.offset_deg.toFixed(2) : '&mdash;';
        return `<tr>
            <td>${escapeHtml(s.name || '?')}</td>
            <td class="num">${ts}</td>
            <td class="num">${npred}</td>
            <td class="num">${flux}</td>
            <td>${spectrum}</td>
            <td class="num">${offset}</td>
        </tr>`;
    }).join('');

    return `
        <div class="result-card">
            <details>
                <summary>ROI sources (top ${sources.length} by TS)</summary>
                <div class="table-scroll">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Name</th><th class="num">TS</th><th class="num">Npred</th>
                                <th class="num">Flux</th><th>Spectrum</th><th class="num">Offset (deg)</th>
                            </tr>
                        </thead>
                        <tbody>${rows}</tbody>
                    </table>
                </div>
            </details>
        </div>`;
}

// ============================================================
// Outputs panel: visualize produced analysis products
// ============================================================

function renderOutputs(result) {
    const content = document.getElementById('outputs-content');
    const badge = document.getElementById('outputs-badge');
    if (!content) return;

    const sci = result.science_results || {};
    const l4 = result.level4 || {};
    const artifacts = (sci.artifacts && sci.artifacts.length ? sci.artifacts
                       : (l4.artifacts || []));
    const sedData = sci.sed_data || l4.sed_data || null;
    const sources = sci.sources_summary || l4.sources_summary || [];

    const hasFit = result.status === 'complete_l4'
        || (l4 && (l4.fit_ok || l4.target_ts != null));

    if (!hasFit) {
        // No science fit produced (pipeline failed earlier or only reached L3).
        const reason = result.status === 'complete'
            ? 'Level 3 (gta.setup) passed, but the Level 4 science fit did not run to completion, so no science outputs were produced.'
            : 'No analysis outputs were produced &mdash; the pipeline did not reach the science-fit stage. See the Results panel for details.';
        content.innerHTML = `<div class="results-placeholder">${reason}</div>`;
        if (badge) { badge.className = 'pipeline-badge idle'; badge.textContent = 'No outputs'; }
        return;
    }

    let html = '';

    // ---- Warnings: target mismatch / validation substitution ----
    if (sci.target_unmatched || result.target_unmatched) {
        html += `<div class="output-warning">⚠ The requested target is not one of the
            three bundled sources (Mrk 421, Vela, Crab), which are the only ones with
            local data. The analysis ran against the default dataset, so these results
            may not correspond to the requested source.</div>`;
    }
    if (sci.target_warning || l4.target_warning) {
        html += `<div class="output-warning">⚠ ${escapeHtml(sci.target_warning || l4.target_warning)}</div>`;
    }

    // ---- What was requested vs. what was produced ----
    const requested = sci.requested_products || l4.requested_products || [];
    if (requested.length) {
        const label = l4.script_calls ? 'Products computed by analysis.py'
            : 'Requested analysis products';
        html += `<div class="output-note">${label}:
            <strong>${requested.map(escapeHtml).join(', ')}</strong></div>`;
    }

    // ---- Transparency: did validation modify the submitted config? ----
    const repairs = (result.repair_history || [])
        .flatMap(r => r.repairs || []);
    if (repairs.length) {
        html += `<div class="output-note">The Validation Agent modified the submitted
            configuration before running it (${repairs.length} change${repairs.length === 1 ? '' : 's'}).
            The YAML panel now shows the exact version that was executed; see the Results
            panel for the full list of changes.</div>`;
    }

    // ---- Headline science metrics as cards ----
    html += renderScienceCards(sci, l4);

    // ---- SED plot: native FermiPy PNG if present, else inline SVG ----
    const sedArtifact = artifacts.find(a => a.kind === 'sed');
    if (sedArtifact) {
        html += outputCard('Spectral Energy Distribution (SED)',
            artifactImg(sedArtifact));
    } else if (sedData && sedData.e_ctr && sedData.e_ctr.length) {
        html += outputCard('Spectral Energy Distribution (SED)',
            buildSedSvg(sedData));
    }

    // ---- Other plot artifacts (TS map, residual map, light curve, etc.) ----
    artifacts.filter(a => a.kind !== 'sed').forEach(a => {
        html += outputCard(a.caption || a.filename, artifactImg(a));
    });

    // ---- Top sources by TS as a bar chart ----
    if (sources && sources.length) {
        html += outputCard('Top sources by TS', buildSourcesBar(sources));
    }

    if (artifacts.length === 0 && !(sedData && sedData.e_ctr && sedData.e_ctr.length)) {
        html += `<div class="output-note">Plot images were not generated for this run,
            but the science-quality fit results are shown above.</div>`;
    }

    content.innerHTML = html;
    if (badge) {
        const n = artifacts.length;
        badge.className = 'pipeline-badge complete';
        badge.textContent = n > 0 ? `${n} plot${n === 1 ? '' : 's'}` : 'Results ready';
    }
    showPanel('outputs');
}

function outputCard(title, innerHtml) {
    return `<div class="output-card">
        <div class="output-card-title">${escapeHtml(title)}</div>
        <div class="output-card-body">${innerHtml}</div>
    </div>`;
}

function artifactImg(a) {
    const url = `/api/session/${sessionId}/artifact/${encodeURIComponent(a.filename)}`;
    return `<a href="${url}" target="_blank" rel="noopener" class="output-img-link" title="Open full size">
        <img class="output-img" src="${url}" alt="${escapeHtml(a.caption || a.filename)}" loading="lazy">
    </a>
    <div class="output-img-caption"><a href="${url}" target="_blank" rel="noopener" download>${escapeHtml(a.filename)}</a></div>`;
}

function renderScienceCards(sci, l4) {
    const ts = sci.target_ts != null ? sci.target_ts : l4.target_ts;
    const flux = sci.target_flux != null ? sci.target_flux : l4.target_flux;
    const fluxErr = sci.target_flux_error != null ? sci.target_flux_error : l4.target_flux_error;
    const fluxUl95 = sci.target_flux_ul95 != null ?
        sci.target_flux_ul95 : l4.target_flux_ul95;
    const detectionStatus = sci.detection_status || l4.detection_status || '';
    const idx = sci.spectral_index != null ? sci.spectral_index : l4.spectral_index;
    const idxErr = sci.spectral_index_error != null ? sci.spectral_index_error : l4.spectral_index_error;
    const conv = (sci.convergence_ok != null ? sci.convergence_ok : l4.convergence_ok);
    const loglike = sci.loglike != null ? sci.loglike : l4.loglike;
    const catFlux = sci.catalog_flux != null ? sci.catalog_flux : l4.catalog_flux;
    const ratio = sci.flux_ratio != null ? sci.flux_ratio : l4.flux_ratio;

    const sigma = (ts != null && ts >= 0) ? Math.sqrt(ts).toFixed(1) : null;

    // Adaptive summary: only show cards that are meaningful for this fit.
    // - Spectral index is omitted for curved spectra (e.g. LogParabola,
    //   PLSuperExpCutoff) where a single index is not the headline quantity.
    // - Log-likelihood is only shown when there is no catalog comparison
    //   available, since the reviewer found it redundant otherwise.
    const specModel = (sci.target_spectral_model || l4.target_spectral_model || '');
    const indexIsMeaningful = idx != null &&
        !/logparabola|superexp|plsuperexp|dmfit|broken/i.test(specModel);

    const cards = [];
    if (ts != null) {
        let tsDetail = '';
        if (detectionStatus === 'detected') {
            tsDetail = sigma != null ? '~' + sigma + ' sigma detection' : 'Detected';
        } else if (detectionStatus === 'subthreshold') {
            tsDetail = 'Below TS detection threshold (~' + sigma + ' sigma)';
        } else if (detectionStatus === 'not_detected') {
            tsDetail = 'Non-detection';
        }
        cards.push(metricCard('Target TS', ts.toFixed(1),
            tsDetail, detectionStatus === 'detected' ? 'good' : 'warn'));
    }
    if (flux != null) {
        cards.push(metricCard('Flux (ph cm⁻² s⁻¹)', flux.toExponential(2),
            fluxErr != null ? `± ${fluxErr.toExponential(1)}` : ''));
    }
    if (fluxUl95 != null && detectionStatus !== 'detected') {
        cards.push(metricCard('Flux upper limit (95%)',
            fluxUl95.toExponential(2), 'ph cm⁻² s⁻¹'));
    }
    if (indexIsMeaningful) {
        cards.push(metricCard('Spectral index', idx.toFixed(2),
            idxErr != null ? `± ${idxErr.toFixed(2)}` : ''));
    }
    if (specModel) {
        cards.push(metricCard('Spectral model', specModel, ''));
    }
    cards.push(metricCard('Fit convergence', conv ? 'Converged' : 'Check',
        conv ? 'fit_quality OK' : 'did not fully converge', conv ? 'good' : 'warn'));
    if (catFlux != null && ratio != null) {
        cards.push(metricCard('Flux vs 4FGL', `${ratio.toFixed(2)}×`,
            `catalog ${catFlux.toExponential(1)}`));
    } else if (loglike != null) {
        cards.push(metricCard('Log-likelihood', loglike.toFixed(1), ''));
    }

    return `<div class="metric-grid">${cards.join('')}</div>`;
}

function metricCard(label, value, sub, tone) {
    return `<div class="metric-card ${tone || ''}">
        <div class="metric-value">${escapeHtml(String(value))}</div>
        <div class="metric-label">${label}</div>
        ${sub ? `<div class="metric-sub">${sub}</div>` : ''}
    </div>`;
}

// Inline log-log SED plot (no external chart library; offline-safe).
function buildSedSvg(sed) {
    const W = 560, H = 360, ml = 70, mr = 20, mt = 20, mb = 50;
    const pw = W - ml - mr, ph = H - mt - mb;

    const pts = [];
    for (let i = 0; i < sed.e_ctr.length; i++) {
        const e = sed.e_ctr[i];
        const isUl = sed.is_ul && sed.is_ul[i];
        let y = sed.e2dnde ? sed.e2dnde[i] : null;
        let yerr = sed.e2dnde_err ? sed.e2dnde_err[i] : null;
        if (isUl && sed.e2dnde_ul95 && sed.e2dnde_ul95[i] != null) y = sed.e2dnde_ul95[i];
        if (e != null && e > 0 && y != null && y > 0) {
            pts.push({ e, y, yerr, isUl });
        }
    }
    if (pts.length === 0) {
        return '<div class="output-note">SED data available but not plottable.</div>';
    }

    const xs = pts.map(p => Math.log10(p.e));
    const yvals = [];
    pts.forEach(p => {
        yvals.push(p.y);
        if (p.yerr && p.y - p.yerr > 0) yvals.push(p.y - p.yerr);
        if (p.yerr) yvals.push(p.y + p.yerr);
    });
    const ys = yvals.map(v => Math.log10(v));

    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    let ymin = Math.min(...ys), ymax = Math.max(...ys);
    const xpad = (xmax - xmin) * 0.05 || 0.5;
    const ypad = (ymax - ymin) * 0.1 || 0.5;
    const x0 = xmin - xpad, x1 = xmax + xpad;
    const y0 = ymin - ypad, y1 = ymax + ypad;

    const sx = v => ml + ((Math.log10(v) - x0) / (x1 - x0)) * pw;
    const sy = v => mt + ph - ((Math.log10(v) - y0) / (y1 - y0)) * ph;

    let svg = `<svg viewBox="0 0 ${W} ${H}" class="sed-svg" preserveAspectRatio="xMidYMid meet">`;
    svg += `<rect x="${ml}" y="${mt}" width="${pw}" height="${ph}" fill="#ffffff" stroke="#d1d5db"/>`;

    // X gridlines/ticks at each decade
    for (let d = Math.ceil(x0); d <= Math.floor(x1); d++) {
        const x = ml + ((d - x0) / (x1 - x0)) * pw;
        svg += `<line x1="${x}" y1="${mt}" x2="${x}" y2="${mt + ph}" stroke="#eceff3"/>`;
        svg += `<text x="${x}" y="${mt + ph + 16}" fill="#6b7280" font-size="10" text-anchor="middle">10^${d}</text>`;
    }
    // Y gridlines/ticks at each decade
    for (let d = Math.ceil(y0); d <= Math.floor(y1); d++) {
        const y = mt + ph - ((d - y0) / (y1 - y0)) * ph;
        svg += `<line x1="${ml}" y1="${y}" x2="${ml + pw}" y2="${y}" stroke="#eceff3"/>`;
        svg += `<text x="${ml - 6}" y="${y + 3}" fill="#6b7280" font-size="10" text-anchor="end">10^${d}</text>`;
    }

    // Axis labels
    svg += `<text x="${ml + pw / 2}" y="${H - 6}" fill="#374151" font-size="12" text-anchor="middle">Energy [MeV]</text>`;
    svg += `<text x="14" y="${mt + ph / 2}" fill="#374151" font-size="12" text-anchor="middle" transform="rotate(-90 14 ${mt + ph / 2})">E² dN/dE [MeV cm⁻² s⁻¹]</text>`;

    // Data points
    pts.forEach(p => {
        const px = sx(p.e);
        const py = sy(p.y);
        if (p.isUl) {
            // upper-limit: downward arrow
            svg += `<line x1="${px}" y1="${py}" x2="${px}" y2="${py + 18}" stroke="#d97706" stroke-width="1.5"/>`;
            svg += `<path d="M${px - 4},${py + 12} L${px + 4},${py + 12} L${px},${py + 18} Z" fill="#d97706"/>`;
        } else {
            if (p.yerr && p.y - p.yerr > 0) {
                svg += `<line x1="${px}" y1="${sy(p.y - p.yerr)}" x2="${px}" y2="${sy(p.y + p.yerr)}" stroke="#4f46e5" stroke-width="1.4"/>`;
            }
            svg += `<circle cx="${px}" cy="${py}" r="3.5" fill="#4f46e5" stroke="#ffffff" stroke-width="0.8"/>`;
        }
    });

    svg += `</svg>`;
    svg += `<div class="output-note">Filled circles: detected bins (TS &ge; 4); arrows: 95% upper limits.</div>`;
    return svg;
}

// Horizontal bar chart of top sources by TS.
function buildSourcesBar(sources) {
    const top = sources.slice(0, 8);
    const maxTs = Math.max(...top.map(s => s.ts || 0), 1);
    const rowH = 26, W = 560, labelW = 170, barMax = W - labelW - 70;
    const H = top.length * rowH + 10;
    let svg = `<svg viewBox="0 0 ${W} ${H}" class="bar-svg" preserveAspectRatio="xMidYMid meet">`;
    top.forEach((s, i) => {
        const y = i * rowH + 6;
        const w = Math.max(2, (Math.max(s.ts, 0) / maxTs) * barMax);
        const name = (s.name || '').length > 24 ? s.name.slice(0, 23) + '…' : s.name;
        svg += `<text x="0" y="${y + rowH / 2 + 1}" fill="#374151" font-size="11">${escapeHtml(name)}</text>`;
        svg += `<rect x="${labelW}" y="${y + 3}" width="${w}" height="${rowH - 10}" rx="2" fill="#4f46e5"/>`;
        svg += `<text x="${labelW + w + 6}" y="${y + rowH / 2 + 1}" fill="#6b7280" font-size="10">TS ${(s.ts || 0).toFixed(1)}</text>`;
    });
    svg += `</svg>`;
    return svg;
}

// ============================================================
// Progress stepper
// ============================================================

function showProgressBar() {
    document.getElementById('pipeline-progress').classList.remove('hidden');
}

function hideProgressBar() {
    document.getElementById('pipeline-progress').classList.add('hidden');
}

function resetProgressSteps() {
    const steps = ['master', 'repair', 'level1', 'level2', 'level3', 'level4'];
    steps.forEach(s => {
        const el = document.getElementById(`step-${s}`);
        if (el) {
            el.className = 'progress-step';
            el.dataset.status = '';
        }
        const detail = document.getElementById(`step-${s}-detail`);
        if (detail) {
            const defaults = {
                master: 'Prompt analysis & code generation',
                repair: 'Iterative config repair',
                level1: 'ConfigManager validation',
                level2: 'Target / ROI model check',
                level3: 'Real Fermi-LAT data processing',
                level4: 'gta.optimize() + gta.fit()',
            };
            detail.textContent = defaults[s] || '';
        }
    });
}

function setProgressStep(step, status, detail) {
    const el = document.getElementById(`step-${step}`);
    if (!el) return;

    // Map statuses
    const cssStatus = status === 'pass' ? 'pass'
        : status === 'fail' ? 'fail'
        : status === 'active' ? 'active'
        : status === 'partial' ? 'partial'
        : '';

    el.className = `progress-step ${cssStatus}`;
    el.dataset.status = cssStatus;

    if (detail) {
        const detailEl = document.getElementById(`step-${step}-detail`);
        if (detailEl) {
            // Strip agent prefix for display in stepper
            const cleanDetail = detail.replace(/^(Master Agent|Validation Agent|System):\s*/i, '');
            detailEl.textContent = cleanDetail.substring(0, 120);
        }
    }
}

// ============================================================
// Code management
// ============================================================

async function saveCodes() {
    if (!sessionId) return;
    // The action-bar button doubles as "Apply changes" while a regeneration
    // proposal is pending: accept all pending proposals, then persist.
    let applied = false;
    for (const lang of ['yaml', 'python']) {
        if (pendingProposals[lang] != null) {
            const prevValue = document.getElementById(`${lang}-editor`).value;
            pushHistory(lang, prevValue);
            setEditorValue(lang, pendingProposals[lang]);
            editorLastKnown[lang] = pendingProposals[lang];
            setCodeStatus(lang, 'active');
            closeProposal(lang);
            applied = true;
        } else {
            // Manual edit detection: only push when the editor actually
            // changed since the last known/committed value, so plain
            // re-saves don't grow the history stack with no-ops.
            const current = document.getElementById(`${lang}-editor`).value;
            if (current !== editorLastKnown[lang]) {
                pushHistory(lang, editorLastKnown[lang]);
                editorLastKnown[lang] = current;
            }
        }
    }
    const yamlCode = document.getElementById('yaml-editor').value;
    const pythonCode = document.getElementById('python-editor').value;

    await apiFetch(`/api/session/${sessionId}/update_code`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ yaml: yamlCode, python: pythonCode }),
    });
    setStatus(applied ? 'Proposed changes applied' : 'Edits saved');
    if (applied && validatorProposalId) await resolveValidatorProposal('accept');
}

// ============================================================
// Regeneration proposals (feedback #7): when the model produces a new
// YAML/Python while the editor already holds one, show a line diff and
// require explicit Apply / Discard instead of silently overwriting.
// ============================================================

// Line-based LCS diff. Returns a list of ['=', line] / ['-', line] /
// ['+', line] ops, or null when the inputs are too large to diff cheaply.
function diffLines(oldText, newText) {
    const a = (oldText || '').split('\n');
    const b = (newText || '').split('\n');
    const n = a.length, m = b.length;
    if (n * m > 4000000) return null;

    const dp = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
    for (let i = n - 1; i >= 0; i--) {
        for (let j = m - 1; j >= 0; j--) {
            dp[i][j] = a[i] === b[j]
                ? dp[i + 1][j + 1] + 1
                : Math.max(dp[i + 1][j], dp[i][j + 1]);
        }
    }
    const ops = [];
    let i = 0, j = 0;
    while (i < n && j < m) {
        if (a[i] === b[j]) { ops.push(['=', a[i]]); i++; j++; }
        else if (dp[i + 1][j] >= dp[i][j + 1]) { ops.push(['-', a[i]]); i++; }
        else { ops.push(['+', b[j]]); j++; }
    }
    while (i < n) ops.push(['-', a[i++]]);
    while (j < m) ops.push(['+', b[j++]]);
    return ops;
}

function renderDiffHtml(ops) {
    return ops.map(([t, line]) => {
        const cls = t === '+' ? 'diff-add' : t === '-' ? 'diff-del' : 'diff-ctx';
        const prefix = t === '=' ? '  ' : t + ' ';
        return `<span class="diff-line ${cls}">${escHtml(prefix + line)}</span>`;
    }).join('');
}

function proposeCode(lang, newText) {
    const ed = document.getElementById(`${lang}-editor`);
    // First generation (or identical content): apply directly, nothing to confirm.
    if (!ed.value.trim() || ed.value === newText) {
        if (pendingProposals[lang] != null) closeProposal(lang);
        // Only the empty->content transition is a meaningful revert point;
        // ed.value === newText is a true no-op, so skip it there.
        if (!ed.value.trim()) pushHistory(lang, ed.value);
        setEditorValue(lang, newText);
        editorLastKnown[lang] = newText;
        setCodeStatus(lang, 'active');
        return;
    }
    const ops = diffLines(ed.value, newText);
    pendingProposals[lang] = newText;
    const diffEl = document.getElementById(`${lang}-diff`);
    diffEl.innerHTML = ops ? renderDiffHtml(ops) : escHtml(newText);
    diffEl.classList.remove('hidden');
    ed.classList.add('hidden');
    document.getElementById(`${lang}-highlight`).classList.add('hidden');
    document.getElementById(`${lang}-proposal`).classList.remove('hidden');
    setCodeStatus(lang, 'running');
    updateSaveButton();
    const changed = ops ? ops.filter(o => o[0] !== '=').length : '?';
    setStatus(`Proposed ${lang.toUpperCase()} changes (${changed} lines) — review, then Apply or Discard`);
}

async function applyProposal(lang) {
    if (pendingProposals[lang] == null) return;
    const prevValue = document.getElementById(`${lang}-editor`).value;
    pushHistory(lang, prevValue);
    setEditorValue(lang, pendingProposals[lang]);
    editorLastKnown[lang] = pendingProposals[lang];
    setCodeStatus(lang, 'active');
    closeProposal(lang);
    await saveCodes();
    setStatus(`${lang.toUpperCase()} changes applied`);
    if (lang === 'python') await resolveValidatorProposal('accept');
}

function discardProposal(lang) {
    if (pendingProposals[lang] == null) return;
    setCodeStatus(lang, 'active');
    closeProposal(lang);
    setStatus(`${lang.toUpperCase()} changes discarded — keeping the current version`);
    if (lang === 'python') resolveValidatorProposal('decline');
}

function closeProposal(lang) {
    pendingProposals[lang] = null;
    document.getElementById(`${lang}-diff`).classList.add('hidden');
    document.getElementById(`${lang}-proposal`).classList.add('hidden');
    document.getElementById(`${lang}-editor`).classList.remove('hidden');
    document.getElementById(`${lang}-highlight`).classList.remove('hidden');
    updateSaveButton();
}

// ============================================================
// Validation Agent script review: the model's analysis.py is executed as
// written, so proposed fixes are shown as a diff and need Apply/Discard.
// ============================================================

function renderReviewFindings(data) {
    const findings = data.findings || [];
    const note = data.llm_note
        ? `<p class="review-note">${escapeHtml(data.llm_note)}</p>` : '';
    const summary = data.summary
        ? `<p><strong>Reviewer:</strong> ${escapeHtml(data.summary)}</p>` : '';
    if (!findings.length) {
        return `${summary}<p>The Validation Agent found no issues in analysis.py.</p>${note}`;
    }
    const items = findings.map(f => `
        <li class="finding finding-${escapeHtml(f.severity)}">
            <span class="finding-tag">${escapeHtml(f.severity)} · ${escapeHtml(f.category)}</span>
            ${f.line ? `<span class="finding-line">line ${f.line}</span>` : ''}
            ${escapeHtml(f.message)}
        </li>`).join('');
    const open = findings.some(f => f.severity !== 'info') ? ' open' : '';
    return `${summary}<details class="review-findings"${open}>
        <summary>Script review findings (${findings.length})</summary>
        <ul>${items}</ul></details>${note}`;
}

function showScriptReview(data) {
    pendingRunApproval = null;
    hideProgressBar();
    resetProgressSteps();
    if (data.yaml != null) {
        setEditorValue('yaml', data.yaml);
        editorLastKnown.yaml = data.yaml;
    }
    const proposed = data.status === 'script_changes_proposed';
    updatePipelineStatus(proposed ? 'review_required' : 'error');
    let body;
    if (proposed) {
        validatorProposalId = data.proposal_id;
        const editorWasEmpty = !document.getElementById('python-editor').value.trim();
        proposeCode('python', data.proposed_python || '');
        if (editorWasEmpty) {
            // Nothing to overwrite: like a first generation, apply directly.
            saveCodes().then(() => resolveValidatorProposal('accept'));
        }
        const kind = data.proposal_kind === 'incremental'
            ? 'Suggested: reuse the previous fit'
            : 'Proposed fix for analysis.py';
        body = `
            <h4>${escapeHtml(kind)}</h4>
            <p>${escapeHtml(data.reason || '')}</p>
            <p>${escapeHtml(data.message || '')} The Python panel shows the diff.</p>
            ${data.blocking ? '<p><strong>The current script cannot run as is</strong> — Discard keeps it blocked until you fix the errors.</p>' : ''}`;
        setStatus('Validation Agent proposes analysis.py changes — Apply or Discard them.');
        addExplainEntry('validation', 'Script review: proposed changes to analysis.py are waiting for your decision.');
    } else {
        validatorProposalId = null;
        body = `
            <h4>analysis.py cannot run yet</h4>
            <p>${escapeHtml(data.message || '')}</p>`;
        setStatus('Script review found errors — nothing was run.');
        addExplainEntry('validation', 'Script review blocked execution: fix the listed errors.');
    }
    document.getElementById('results-content').innerHTML = `
        <div class="result-card ${proposed ? 'warn' : 'fail'}">
            ${body}
            ${renderReviewFindings(data)}
        </div>`;
    showPanel('results');
}

async function resolveValidatorProposal(decision) {
    const proposalId = validatorProposalId;
    if (!proposalId || !sessionId) return;
    validatorProposalId = null;
    try {
        const resp = await apiFetch(`/api/session/${sessionId}/script_review`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ proposal_id: proposalId, decision }),
        });
        if (!resp.ok) return;
        addExplainEntry('validation', decision === 'accept'
            ? 'Validator changes to analysis.py applied.'
            : 'Validator changes to analysis.py declined — the script stays as written.');
        setStatus(decision === 'accept'
            ? 'Changes applied — click Run Pipeline to review and run.'
            : 'Changes discarded — click Run Pipeline to continue with your script.');
    } catch (err) {
        setStatus(`Could not record the decision: ${err.message}`);
    }
}

function renderScriptExecution(l4) {
    let html = '';
    if (l4.script_error) {
        html += `<div class="result-card fail"><h4>analysis.py raised an error</h4>
            <pre class="script-output">${escapeHtml(l4.script_traceback || l4.script_error)}</pre></div>`;
    }
    const calls = l4.script_calls || [];
    if (calls.length) {
        const rows = calls.map(c => {
            const args = (c.args || []).concat(
                Object.entries(c.kwargs || {}).map(([k, v]) => `${k}=${v}`));
            const state = c.ok ? '✓' : (c.error ? '✗' : '…');
            return `<tr class="${c.ok ? '' : 'call-failed'}">
                <td>${state}</td><td><code>gta.${escapeHtml(c.method)}(${escapeHtml(args.join(', '))})</code></td>
                <td>${c.seconds == null ? '' : `${c.seconds}s`}</td>
                <td>${escapeHtml(c.error || '')}</td></tr>`;
        }).join('');
        html += `<details class="result-card"><summary>FermiPy calls made by analysis.py (${calls.length})</summary>
            <table class="call-trace"><tbody>${rows}</tbody></table></details>`;
    }
    if ((l4.script_stdout || '').trim()) {
        html += `<details class="result-card" open><summary>Output printed by analysis.py</summary>
            <pre class="script-output">${escapeHtml(l4.script_stdout)}</pre></details>`;
    }
    return html;
}

// ============================================================
// Deterministic Revert: per-editor undo stacks (independent of the LLM).
// ============================================================

function pushHistory(lang, snapshot) {
    const stack = editorHistory[lang];
    stack.push(snapshot);
    if (stack.length > 20) stack.shift();
    updateRevertButton(lang);
}

function updateRevertButton(lang) {
    const btn = document.getElementById(`${lang}-revert-btn`);
    if (!btn) return;
    btn.disabled = editorHistory[lang].length === 0;
}

function revertEditor(lang) {
    // A pending proposal takes precedence: discard it rather than mixing it
    // with a stack-restore in the same click (avoids a dangling proposal UI).
    if (pendingProposals[lang] != null) {
        discardProposal(lang);
        return;
    }
    const stack = editorHistory[lang];
    if (stack.length === 0) return;
    const prev = stack.pop();
    setEditorValue(lang, prev);
    editorLastKnown[lang] = prev;
    setCodeStatus(lang, 'active');
    updateRevertButton(lang);
    setStatus(`${lang.toUpperCase()} reverted to previous version`);
}

function updateSaveButton() {
    const btn = document.getElementById('save-btn');
    if (!btn) return;
    const pending = pendingProposals.yaml != null || pendingProposals.python != null;
    btn.textContent = pending ? 'Apply changes' : 'Save Edits';
    btn.classList.toggle('btn-attention', pending);
}

function copyCode(type) {
    const editor = document.getElementById(`${type}-editor`);
    navigator.clipboard.writeText(editor.value).then(() => {
        setStatus(`${type.toUpperCase()} copied to clipboard`);
    });
}

// ============================================================
// Panel management
// ============================================================

function togglePanel(name) {
    const panel = document.getElementById(`panel-${name}`);
    const btn = document.querySelector(`[data-panel="${name}"]`);
    if (panel.classList.contains('hidden')) {
        panel.classList.remove('hidden');
        btn.classList.add('active');
        // Lazily build the notebook the first time its tab is enabled.
        if (name === 'notebook') loadNotebook();
    } else {
        panel.classList.add('hidden');
        btn.classList.remove('active');
    }
}

// ============================================================
// Notebook panel (hidden by default; enabled via the Notebook tab)
// ============================================================

async function loadNotebook() {
    const content = document.getElementById('notebook-content');
    const badge = document.getElementById('notebook-badge');
    const dl = document.getElementById('notebook-download');
    if (!content) return;
    if (!sessionId) {
        content.innerHTML = '<div class="results-placeholder">Start a session and generate an analysis first.</div>';
        return;
    }
    badge.className = 'pipeline-badge running';
    badge.textContent = 'Building...';
    if (dl) dl.style.display = 'none';
    try {
        const resp = await fetch(`/api/session/${sessionId}/notebook`);
        if (resp.status === 400) {
            content.innerHTML = '<div class="results-placeholder">No analysis to export yet &mdash; generate a configuration first.</div>';
            badge.className = 'pipeline-badge idle';
            badge.textContent = 'Not generated';
            return;
        }
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const nb = await resp.json();
        const fname = resp.headers.get('X-Notebook-Filename') || `notebook_${sessionId}.ipynb`;
        if (dl) {
            dl.href = `/api/session/${sessionId}/notebook`;
            dl.setAttribute('download', fname);
            dl.style.display = '';
        }
        content.innerHTML = renderNotebookPreview(nb);
        badge.className = 'pipeline-badge complete';
        badge.textContent = `${(nb.cells || []).length} cells`;
    } catch (e) {
        content.innerHTML = `<div class="results-placeholder">Could not build the notebook (${escapeHtml(e.message)}).</div>`;
        badge.className = 'pipeline-badge error';
        badge.textContent = 'Error';
    }
}

function refreshNotebookIfVisible() {
    const panel = document.getElementById('panel-notebook');
    if (panel && !panel.classList.contains('hidden')) loadNotebook();
}

function renderNotebookPreview(nb) {
    let html = '<div class="nb-preview">';
    (nb.cells || []).forEach(cell => {
        const src = Array.isArray(cell.source) ? cell.source.join('') : (cell.source || '');
        if (cell.cell_type === 'code') {
            html += `<div class="nb-cell nb-code"><span class="nb-gutter">In [ ]:</span><pre>${escapeHtml(src)}</pre></div>`;
        } else {
            html += `<div class="nb-cell nb-md">${renderNotebookMarkdown(src)}</div>`;
        }
    });
    html += '</div>';
    return html;
}

// Minimal markdown renderer for the notebook preview. The markdown is authored
// server-side (controlled), so a small subset is enough: fenced code, images
// (data URIs), headings, bold, inline code, and blockquotes.
function renderNotebookMarkdown(md) {
    const blocks = [];
    md = md.replace(/```[a-z]*\n([\s\S]*?)```/g, (m, code) => {
        blocks.push(`<pre class="nb-fence">${escapeHtml(code.replace(/\n$/, ''))}</pre>`);
        return `@@BLOCK${blocks.length - 1}@@`;
    });
    const imgs = [];
    md = md.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (m, alt, src) => {
        // Only allow inline data: images (what the server embeds).
        if (!/^data:image\//.test(src)) return '';
        imgs.push(`<img class="nb-img" alt="${escapeHtml(alt)}" src="${src}">`);
        return `@@IMG${imgs.length - 1}@@`;
    });
    let h = escapeHtml(md);
    h = h.replace(/^### (.*)$/gm, '<h4>$1</h4>')
         .replace(/^## (.*)$/gm, '<h3>$1</h3>')
         .replace(/^# (.*)$/gm, '<h3>$1</h3>')
         .replace(/^&gt; ?(.*)$/gm, '<blockquote>$1</blockquote>')
         .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
         .replace(/`([^`]+)`/g, '<code>$1</code>');
    h = h.split(/\n{2,}/).map(p => {
        const t = p.trim();
        if (!t) return '';
        if (/^@@(BLOCK|IMG)\d+@@$/.test(t)) return p;
        if (/^<(h3|h4|blockquote)/.test(t)) return p;
        return '<p>' + p.replace(/\n/g, '<br>') + '</p>';
    }).join('');
    h = h.replace(/@@BLOCK(\d+)@@/g, (m, i) => blocks[i]);
    h = h.replace(/@@IMG(\d+)@@/g, (m, i) => imgs[i]);
    return h;
}

function showPanel(name) {
    const panel = document.getElementById(`panel-${name}`);
    const btn = document.querySelector(`[data-panel="${name}"]`);
    panel.classList.remove('hidden');
    if (btn) btn.classList.add('active');
}

// ============================================================
// UI helpers
// ============================================================

function setStatus(text) {
    document.getElementById('global-status').textContent = text;
}

function setCodeStatus(type, status) {
    const dot = document.getElementById(`${type}-status`);
    dot.className = `status-dot ${status}`;
}

function updatePipelineStatus(status) {
    const badge = document.getElementById('pipeline-status-badge');
    badge.className = `pipeline-badge ${status}`;
    const labels = {
        idle: 'Idle', running: 'Running...', complete: 'Complete',
        error: 'Error', validation_failed: 'Failed', aborted: 'Aborted',
        review_required: 'Review required',
    };
    badge.textContent = labels[status] || status;
    setRunControls(status === 'running');
}

function setRunControls(running, aborting = false) {
    const runBtn = document.getElementById('run-btn');
    const abortBtn = document.getElementById('abort-btn');
    if (runBtn) runBtn.disabled = running;
    if (!abortBtn) return;
    const abortDisabled = !running || aborting;
    abortBtn.disabled = abortDisabled;
    abortBtn.setAttribute('aria-disabled', String(abortDisabled));
    abortBtn.textContent = aborting ? 'Aborting...' : 'Abort Run';
    abortBtn.title = aborting ? 'The pipeline is being stopped'
        : running ? 'Stop the running pipeline' : 'No pipeline is currently running';
}

function addExplainEntry(agent, text) {
    const log = document.getElementById('explain-log');
    const div = document.createElement('div');
    div.className = `explain-entry ${agent}`;
    const now = new Date().toLocaleTimeString();
    const agentLabel = agent === 'master' ? 'Master'
        : agent === 'validation' ? 'Validator'
        : 'System';
    div.innerHTML = `
        <span class="entry-time">${now}</span>
        <span class="entry-agent ${agent}">${agentLabel}</span>
        <div>${escapeHtml(text)}</div>`;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
}

function formatMarkdown(text) {
    return escapeHtml(text)
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>')
        .replace(/- (.+?)(?=<br>|$)/g, '&bull; $1');
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && document.activeElement.id === 'chat-input') {
        e.preventDefault();
        sendMessage();
    }
});

// Allow tab in code editors
document.querySelectorAll('.code-editor').forEach(editor => {
    editor.addEventListener('keydown', (e) => {
        if (e.key === 'Tab') {
            e.preventDefault();
            const start = editor.selectionStart;
            const end = editor.selectionEnd;
            editor.value = editor.value.substring(0, start) + '  ' + editor.value.substring(end);
            editor.selectionStart = editor.selectionEnd = start + 2;
            syncHighlight(editor);
        }
    });
});

// ============================================================
// Lightweight syntax highlighting for the YAML / Python editors.
// The textarea remains the editable source of truth (transparent text +
// caret); a <pre> backdrop shows the colorized copy. No external libraries.
// ============================================================

function escHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

const HIGHLIGHT_SPECS = {
    python: [
        ['comment', /#[^\n]*/],
        ['string', /"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/],
        ['deco', /@[A-Za-z_][\w.]*/],
        ['kw', /\b(?:import|from|as|def|return|if|elif|else|for|while|in|is|not|and|or|with|try|except|finally|raise|class|lambda|pass|break|continue|global|nonlocal|yield|assert|del|True|False|None)\b/],
        ['fn', /\b[A-Za-z_]\w*(?=\s*\()/],
        ['number', /\b\d+\.?\d*(?:[eE][+-]?\d+)?\b/],
    ],
    yaml: [
        ['comment', /#[^\n]*/],
        ['key', /^[ \t]*-?[ \t]*[\w.\-]+(?=\s*:)/m],
        ['string', /"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/],
        ['bool', /\b(?:true|false|null|True|False|None|yes|no)\b/],
        ['number', /-?\b\d+\.?\d*(?:[eE][+-]?\d+)?\b/],
    ],
};

function highlightCode(text, lang) {
    const specs = HIGHLIGHT_SPECS[lang];
    if (!specs) return escHtml(text);
    // Each spec source uses only non-capturing groups, so the combined regex
    // has exactly one capturing group per spec, in order.
    const combined = new RegExp(specs.map(s => '(' + s[1].source + ')').join('|'), 'gm');
    let out = '', last = 0, m;
    while ((m = combined.exec(text)) !== null) {
        out += escHtml(text.slice(last, m.index));
        let cls = 'plain';
        for (let i = 1; i < m.length; i++) {
            if (m[i] !== undefined) { cls = specs[i - 1][0]; break; }
        }
        out += `<span class="tok-${cls}">${escHtml(m[0])}</span>`;
        last = m.index + m[0].length;
        if (m[0].length === 0) combined.lastIndex++;  // guard against zero-width
    }
    out += escHtml(text.slice(last));
    return out;
}

function syncHighlight(editor) {
    try {
        if (!editor || !editor.dataset) return;
        const lang = editor.dataset.lang;
        const hl = document.getElementById(`${lang}-highlight`);
        if (!hl) return;
        // Trailing newline keeps the last line aligned during scroll.
        hl.innerHTML = highlightCode(editor.value || '', lang) + '\n';
        hl.scrollTop = editor.scrollTop;
        hl.scrollLeft = editor.scrollLeft;
    } catch (e) {
        /* highlighting is purely cosmetic; never block editing */
    }
}

// Set an editor's value programmatically AND refresh its highlight backdrop.
function setEditorValue(lang, val) {
    const ed = document.getElementById(`${lang}-editor`);
    if (!ed) return;
    ed.value = val || '';
    syncHighlight(ed);
}

function initEditorHighlighting() {
    document.querySelectorAll('.code-editor').forEach(ed => {
        ed.addEventListener('input', () => syncHighlight(ed));
        ed.addEventListener('scroll', () => {
            const hl = document.getElementById(`${ed.dataset.lang}-highlight`);
            if (hl) { hl.scrollTop = ed.scrollTop; hl.scrollLeft = ed.scrollLeft; }
        });
        syncHighlight(ed);
    });
}
initEditorHighlighting();

// ============================================================
// Resizable panels (feedback #2): draggable splitters between panels.
// Replaces the old CSS `resize: horizontal` corner handle.
// ============================================================

function initSplitters() {
    const addSplittersIn = (container) => {
        if (!container) return;
        const kids = Array.from(container.children).filter(el =>
            el.classList.contains('panel') || el.classList.contains('code-panels'));
        for (let k = 0; k < kids.length - 1; k++) {
            const splitter = document.createElement('div');
            splitter.className = 'splitter';
            splitter.title = 'Drag to resize';
            container.insertBefore(splitter, kids[k].nextSibling);
            attachSplitterDrag(splitter);
        }
    };
    addSplittersIn(document.querySelector('.main-layout'));
    addSplittersIn(document.querySelector('.code-panels'));
    updateSplitterVisibility();
    // Panels are shown/hidden by toggling the `hidden` class; keep splitters in sync.
    const layout = document.querySelector('.main-layout');
    if (layout) {
        new MutationObserver(updateSplitterVisibility).observe(layout, {
            subtree: true, attributes: true, attributeFilter: ['class'],
        });
    }
}

const isPanelShown = (el) => !!el && !el.classList.contains('hidden') && el.getClientRects().length > 0;

// Nearest shown panel (or .code-panels) on one side of a splitter, skipping
// hidden panels and other splitters. dir = -1 (left) or +1 (right).
function splitterNeighbor(splitter, dir) {
    let el = dir < 0 ? splitter.previousElementSibling : splitter.nextElementSibling;
    while (el && (el.classList.contains('splitter') || !isPanelShown(el))) {
        el = dir < 0 ? el.previousElementSibling : el.nextElementSibling;
    }
    return el;
}

// Show a splitter only if it directly follows a shown panel and has a shown
// panel somewhere to its right, so exactly one splitter sits between any two
// visible neighbors and none dangles at an edge.
function updateSplitterVisibility() {
    document.querySelectorAll('.splitter').forEach(sp => {
        const prev = sp.previousElementSibling;
        const show = isPanelShown(prev) && !!splitterNeighbor(sp, +1);
        sp.classList.toggle('splitter-hidden', !show);
    });
}

// The panel whose edge touches the splitter. For the .code-panels container
// that is its last (left side) or first (right side) visible code panel, so
// dragging the YAML|Python block's outer edge resizes only the adjacent editor.
function splitterEdgePanel(el, dir) {
    if (!el || !el.classList.contains('code-panels')) return el;
    const shown = Array.from(el.children).filter(c => c.classList.contains('panel') && isPanelShown(c));
    if (!shown.length) return el;
    return dir < 0 ? shown[shown.length - 1] : shown[0];
}

function attachSplitterDrag(splitter) {
    splitter.addEventListener('pointerdown', (e) => {
        const leftEl = splitterEdgePanel(splitterNeighbor(splitter, -1), -1);
        const rightEl = splitterEdgePanel(splitterNeighbor(splitter, +1), +1);
        if (!leftEl || !rightEl) return;
        e.preventDefault();
        // Disable the panels' flex transition first: the measurements below
        // (and minOf's collapse-and-measure) must see the new sizes immediately.
        document.body.classList.add('col-resizing');

        // Main-layout panels get a fixed px basis. Code panels (YAML / Python)
        // get a flex-grow proportional to their width with a 0 basis, so they
        // always fill .code-panels. .code-panels itself is pinned to an explicit
        // basis that follows the drag: when the layout is wider than the window
        // (horizontal scroll) it would otherwise sit at its min-content width and
        // not absorb the change, so the drag would move the *outer* edges of the
        // two neighbors instead of the boundary between them. flex-grow stays 1
        // so it still takes up extra room when the window is widened.
        const isCode = (el) => el.classList.contains('panel-code');
        const isContainer = (el) => el.classList.contains('code-panels');
        const codePanels = document.querySelector('.code-panels');
        const inCode = (el) => !!codePanels && codePanels.contains(el);
        // +1: the container is on the left of the boundary (grows with dx),
        // -1: on the right (shrinks with dx), 0: not involved or on both sides.
        const codeSign = (inCode(leftEl) ? 1 : 0) - (inCode(rightEl) ? 1 : 0);
        let startC = 0;
        if (inCode(leftEl) || inCode(rightEl)) {
            // Read every width before writing any: each write relayouts the others.
            const codes = Array.from(codePanels.querySelectorAll('.panel-code')).filter(isPanelShown)
                .map(c => [c, c.getBoundingClientRect().width]);
            codes.forEach(([c, w]) => { c.style.flex = `${w} 1 0px`; });
            startC = codePanels.getBoundingClientRect().width;
            codePanels.style.flex = `1 0 ${startC}px`;
            // Its automatic min-width (min-content) exceeds the sum of the code
            // panels' own minimums and would stop it shrinking with the drag; the
            // code panels' min-widths (clamped below) are the real limit.
            codePanels.style.minWidth = '0';
        }
        const widthOf = (el) => el.getBoundingClientRect().width;
        // Real minimum width: the larger of the CSS min-width and the content's
        // min-content width (e.g. a header's buttons), measured by collapsing the
        // basis to 0 for a moment. Clamping to the CSS value alone would let the
        // other neighbor keep moving after this one has stopped shrinking.
        const minOf = (el) => {
            if (isContainer(el)) return 0;
            const saved = el.style.flex;
            el.style.flex = '0 0 0px';
            const w = el.getBoundingClientRect().width;
            el.style.flex = saved;
            return w;
        };
        const startL = widthOf(leftEl), startR = widthOf(rightEl);
        // Never snap a panel that is already below its min (e.g. squeezed by a narrow window).
        const minL = Math.min(minOf(leftEl), startL), minR = Math.min(minOf(rightEl), startR);
        const setWidth = (el, w) => {
            if (isContainer(el)) return; // no visible code panel: sized via its basis below
            el.style.flex = isCode(el) ? `${w} 1 0px` : `0 0 ${w}px`;
        };
        const startX = e.clientX;

        splitter.classList.add('active');
        splitter.setPointerCapture(e.pointerId);

        const onMove = (ev) => {
            let dx = ev.clientX - startX;
            dx = Math.max(dx, minL - startL);   // left panel can't go below its min
            dx = Math.min(dx, startR - minR);   // nor can the right one
            setWidth(leftEl, startL + dx);
            setWidth(rightEl, startR - dx);
            if (codeSign) codePanels.style.flex = `1 0 ${startC + codeSign * dx}px`;
        };
        const onUp = () => {
            splitter.classList.remove('active');
            document.body.classList.remove('col-resizing');
            splitter.removeEventListener('pointermove', onMove);
            splitter.removeEventListener('pointerup', onUp);
            splitter.removeEventListener('pointercancel', onUp);
        };
        splitter.addEventListener('pointermove', onMove);
        splitter.addEventListener('pointerup', onUp);
        splitter.addEventListener('pointercancel', onUp);
    });
}
initSplitters();

// ============================================================
// Authentication (Google Sign-In) + task ownership
// ============================================================

async function loadAuthConfig() {
    try {
        const resp = await apiFetch('/api/auth/config');
        const cfg = await resp.json();
        googleClientId = cfg.enabled ? cfg.client_id : null;
    } catch (e) { googleClientId = null; }
}

function initGoogleSignIn() {
    const wrap = document.getElementById('google-signin-btn');
    if (!googleClientId) {
        // Sign-in not configured on the server → guest-only; hide the button.
        if (wrap) wrap.style.display = 'none';
        return;
    }
    if (!(window.google && google.accounts && google.accounts.id)) {
        setTimeout(initGoogleSignIn, 300);  // GIS script still loading
        return;
    }
    google.accounts.id.initialize({
        client_id: googleClientId,
        callback: handleGoogleCredential,
    });
    if (wrap && !currentUser) {
        wrap.style.display = '';
        wrap.innerHTML = '';
        google.accounts.id.renderButton(wrap, {
            type: 'standard', theme: 'outline', size: 'medium',
            text: 'signin_with', shape: 'pill',
        });
    }
}

async function handleGoogleCredential(response) {
    try {
        const resp = await apiFetch('/api/auth/google', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ credential: response.credential }),
        });
        if (!resp.ok) {
            const e = await resp.json().catch(() => ({}));
            throw new Error(e.detail || `HTTP ${resp.status}`);
        }
        const data = await resp.json();
        authToken = data.token;
        currentUser = data.user;
        try { localStorage.setItem(AUTH_TOKEN_KEY, authToken); } catch (e) {}
        renderAuthUI();
        // Attach the task the user was working on (as a guest) to their account.
        if (sessionId) {
            try { await apiFetch(`/api/session/${sessionId}/claim`, { method: 'POST' }); } catch (e) {}
        }
        renderTaskList();       // switch to the user's server-side task list
        setStatus(`Signed in as ${currentUser.name || currentUser.email}`);
    } catch (e) {
        setStatus(`Sign-in failed: ${e.message}`);
    }
}

async function checkExistingAuth() {
    if (!authToken) { currentUser = null; return; }
    try {
        const resp = await apiFetch('/api/auth/me');
        const data = await resp.json();
        currentUser = data.authenticated ? data.user : null;
        if (!currentUser) { authToken = null; localStorage.removeItem(AUTH_TOKEN_KEY); }
    } catch (e) { currentUser = null; }
}

function signOut() {
    authToken = null;
    currentUser = null;
    try { localStorage.removeItem(AUTH_TOKEN_KEY); } catch (e) {}
    if (window.google && google.accounts && google.accounts.id) {
        try { google.accounts.id.disableAutoSelect(); } catch (e) {}
    }
    renderAuthUI();
    initGoogleSignIn();
    renderTaskList();       // back to the local guest task list
    setStatus('Signed out — continuing as guest.');
}

function renderAuthUI() {
    const menu = document.getElementById('user-menu');
    const btn = document.getElementById('google-signin-btn');
    if (currentUser) {
        if (menu) menu.classList.remove('hidden');
        if (btn) btn.style.display = 'none';
        const nameEl = document.getElementById('user-name');
        const avEl = document.getElementById('user-avatar');
        if (nameEl) nameEl.textContent = currentUser.name || currentUser.email || 'Signed in';
        if (avEl) {
            if (currentUser.picture) { avEl.src = currentUser.picture; avEl.style.display = ''; }
            else { avEl.style.display = 'none'; }
        }
    } else {
        if (menu) menu.classList.add('hidden');
        if (btn && googleClientId) btn.style.display = '';
    }
}

// ============================================================
// Task sidebar (ChatGPT-style conversation list)
// ============================================================

const SIDEBAR_KEY = 'fermi_sidebar_open';
const GUEST_TASKS_KEY = 'fermi_tasks';

function toggleSidebar() {
    const sb = document.getElementById('task-sidebar');
    if (!sb) return;
    const open = sb.classList.toggle('collapsed') === false;
    try { localStorage.setItem(SIDEBAR_KEY, open ? '1' : '0'); } catch (e) {}
    updateSidebarToggle(open);
}

function updateSidebarToggle(open) {
    const btn = document.getElementById('sidebar-toggle');
    if (!btn) return;
    const label = open ? 'Close sidebar' : 'Open sidebar';
    btn.title = label;
    btn.setAttribute('aria-label', label);
}

function applySidebarState() {
    const sb = document.getElementById('task-sidebar');
    if (!sb) return;
    // Default: open on wide screens, collapsed on narrow ones.
    let open = localStorage.getItem(SIDEBAR_KEY);
    if (open === null) open = window.innerWidth > 900 ? '1' : '0';
    sb.classList.toggle('collapsed', open !== '1');
    updateSidebarToggle(open === '1');
}

// Guests have no server-side task list, so we remember their tasks locally.
function getGuestTasks() {
    try { return JSON.parse(localStorage.getItem(GUEST_TASKS_KEY) || '[]'); }
    catch (e) { return []; }
}
function saveGuestTasks(arr) {
    try { localStorage.setItem(GUEST_TASKS_KEY, JSON.stringify(arr.slice(0, 100))); }
    catch (e) {}
}
function upsertGuestTask(id, title, linked = false) {
    if (!id) return;
    const arr = getGuestTasks();
    const i = arr.findIndex(t => t.id === id);
    if (i >= 0) {
        if (title) arr[i].title = title;
        if (linked) arr[i].linked = true;
    } else {
        arr.unshift({ id, title: title || 'New task', created_at: nowIso(), linked });
    }
    saveGuestTasks(arr);
}
function nowIso() { try { return new Date().toISOString(); } catch (e) { return ''; } }

async function getTaskList() {
    // Explicitly opened links remain in this browser, without changing ownership.
    const localTasks = getGuestTasks();
    const asSession = t => ({
        session_id: t.id, title: t.title, created_at: t.created_at,
        status: 'idle', has_config: false,
    });
    if (currentUser) {
        let owned = [];
        try {
            const resp = await apiFetch('/api/my/sessions');
            if (resp.ok) owned = (await resp.json()).sessions || [];
        } catch (e) {}
        const ownedIds = new Set(owned.map(t => t.session_id));
        return [...owned, ...localTasks
            .filter(t => t.linked && !ownedIds.has(t.id)).map(asSession)];
    }
    return localTasks.map(asSession);
}

async function renderTaskList() {
    const list = document.getElementById('task-list');
    const hint = document.getElementById('sidebar-hint');
    if (!list) return;
    if (hint) {
        hint.textContent = currentUser
            ? `Signed in as ${currentUser.name || currentUser.email}`
            : 'Guest — sign in to keep tasks across devices';
    }
    const items = await getTaskList();
    if (!items.length) {
        list.innerHTML = '<div class="sidebar-empty">No tasks yet. Click “New task” to begin.</div>';
        return;
    }
    list.innerHTML = items.map(s => `
        <div class="task-row ${s.session_id === sessionId ? 'active' : ''}"
             onclick="loadSession('${s.session_id}')" title="${escapeHtml(s.title || 'Untitled task')}">
            <span class="task-row-title">${escapeHtml(s.title || 'Untitled task')}</span>
            <span class="task-row-date">${formatDate(s.created_at)}</span>
        </div>`).join('');
}

async function startNewTask() {
    await newSession();          // creates a fresh session + resets the UI
    upsertGuestTask(sessionId, 'New task');
    await renderTaskList();
    if (window.innerWidth <= 900) {  // auto-collapse on mobile after choosing
        const sb = document.getElementById('task-sidebar');
        if (sb) sb.classList.add('collapsed');
    }
}

function formatDate(iso) {
    if (!iso) return '';
    try {
        const d = new Date(iso);
        const today = new Date();
        const sameDay = d.toDateString() === today.toDateString();
        return sameDay ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                       : d.toLocaleDateString();
    } catch (e) { return iso; }
}

// ---- Load an existing task into the UI ----
async function loadSession(id, rememberLink = false) {
    try {
        const resp = await apiFetch(`/api/session/${id}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const s = await resp.json();
        if (pipelineSSE) { pipelineSSE.close(); pipelineSSE = null; }
        if (pipelinePolling) { clearInterval(pipelinePolling); pipelinePolling = null; }
        stopEtaTimer(false);
        setActiveSession(id);
        connectWebSocket();

        // Clear every panel before restoring: outputs, transparency log and
        // notebook otherwise keep showing the previous session (they are only
        // overwritten below when the loaded session has a pipeline_result).
        resetPanels();

        // Restore chat history
        const chat = document.getElementById('chat-messages');
        chat.innerHTML = '';
        (s.chat_history || []).forEach(m => {
            if (m.role === 'user') addChatMessage('user', m.content);
            else addChatMessage('assistant', m.content, (m.analysis ? 'validation' : 'master'));
        });
        const demoEl = document.getElementById('demo-samples');
        if (demoEl && (s.chat_history || []).length) demoEl.classList.add('hidden');

        // Restore editors
        setEditorValue('yaml', s.current_yaml || '');
        setEditorValue('python', s.current_python || '');
        editorHistory.yaml = []; editorHistory.python = [];
        editorLastKnown.yaml = s.current_yaml || '';
        editorLastKnown.python = s.current_python || '';

        // Restore only this task's pipeline UI. A saved result from an older
        // run must not override a currently-running status, and an idle task
        // must never inherit the progress bar or disabled Run button from the
        // task that was open before it.
        const pipelineStatus = s.pipeline_status || 'idle';
        updatePipelineStatus(pipelineStatus);
        if (pipelineStatus === 'running') {
            showProgressBar();
            resetProgressSteps();
            setProgressStep('master', 'pass', 'Configuration generated');
            document.getElementById('results-content').innerHTML = `
                <div class="result-card running">
                    <h4>Pipeline Running</h4>
                    <p>This task has an active pipeline run.</p>
                    <div class="pipeline-live-log" id="pipeline-live-log"></div>
                </div>`;
            startPipelineSSE();
            startPipelinePolling();
        } else if (s.pipeline_result) {
            hideProgressBar();
            resetProgressSteps();
            try { handlePipelineResult(s.pipeline_result, null); } catch (e) {}
        } else {
            hideProgressBar();
            resetProgressSteps();
            setRunControls(false);
            document.getElementById('results-content').innerHTML =
                '<div class="results-placeholder">Run the pipeline to see results.</div>';
        }
        upsertGuestTask(id, s.title || (s.current_prompt || '').slice(0, 100), rememberLink);
        renderTaskList();       // update active highlight
        setStatus(`Loaded task ${id}`);
        refreshNotebookIfVisible();
        if (window.innerWidth <= 900) {
            const sb = document.getElementById('task-sidebar');
            if (sb) sb.classList.add('collapsed');
        }
        return true;
    } catch (e) {
        setStatus(`Could not load task: ${e.message}`);
        return false;
    }
}

// ============================================================
// Initialize — resume a previous session (guest or signed-in) or start fresh
// ============================================================

async function bootstrap() {
    applySidebarState();
    await loadAuthConfig();
    await checkExistingAuth();
    renderAuthUI();
    initGoogleSignIn();
    renderTaskList();

    const taskUrl = new URL(window.location.href);
    const requestedId = taskUrl.searchParams.get('session');
    if (requestedId !== null) {
        // Consume the link once: later task switches should survive a reload.
        taskUrl.searchParams.delete('session');
        window.history.replaceState(null, '', taskUrl);
        if (/^[0-9a-f]{8}$/i.test(requestedId)) {
            await loadModels();
            await loadDemoSamples();
            loadVersions();
            if (await loadSession(requestedId, true)) return;
        } else {
            setStatus('Could not load task: invalid session link.');
        }
        // Keep an unsuccessful link visible as an error; do not replace the
        // user's last task or silently create a new one.
        return;
    }

    const lastId = localStorage.getItem(LAST_SESSION_KEY);
    if (lastId) {
        try {
            const resp = await apiFetch(`/api/session/${lastId}`);
            if (resp.ok) {
                await loadModels();
                await loadDemoSamples();
                loadVersions();
                await loadSession(lastId);
                return;
            }
        } catch (e) { /* fall through to a fresh session */ }
    }
    await initSession();
}

bootstrap();
