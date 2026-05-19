// Mic capture → WAV encode → upload.
// We capture at whatever the AudioContext gives us (typically 48 kHz). The
// pipeline calls librosa.load(sr=16000), which resamples on load, so we don't
// resample in the browser.

class Recorder {
    constructor() {
        this.audioContext = null;
        this.mediaStream = null;
        this.workletNode = null;
        this.chunks = [];
        this.sampleRate = 0;
        this.recording = false;
    }

    async start() {
        if (this.recording) return;
        this.chunks = [];

        this.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: false,
            },
        });

        this.audioContext = new AudioContext();
        this.sampleRate = this.audioContext.sampleRate;
        await this.audioContext.audioWorklet.addModule("/static/worklet.js");

        const source = this.audioContext.createMediaStreamSource(this.mediaStream);
        this.workletNode = new AudioWorkletNode(this.audioContext, "capture");
        this.workletNode.port.onmessage = (e) => this.chunks.push(e.data);

        source.connect(this.workletNode);
        // Worklet must connect somewhere for `process` to be called. A muted
        // GainNode keeps it alive without echoing the mic to the speakers.
        const sink = this.audioContext.createGain();
        sink.gain.value = 0;
        this.workletNode.connect(sink);
        sink.connect(this.audioContext.destination);

        this.recording = true;
    }

    async stop() {
        if (!this.recording) return null;
        this.recording = false;

        this.workletNode.disconnect();
        this.mediaStream.getTracks().forEach((t) => t.stop());
        await this.audioContext.close();

        const merged = mergeChunks(this.chunks);
        return encodeWAV(merged, this.sampleRate);
    }
}

function mergeChunks(chunks) {
    let total = 0;
    for (const c of chunks) total += c.length;
    const out = new Float32Array(total);
    let offset = 0;
    for (const c of chunks) {
        out.set(c, offset);
        offset += c.length;
    }
    return out;
}

// 16-bit PCM mono WAV.
function encodeWAV(float32, sampleRate) {
    const numChannels = 1;
    const bitsPerSample = 16;
    const bytesPerSample = bitsPerSample / 8;
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = float32.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + dataSize, true);
    writeString(view, 8, "WAVE");
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);             // PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);
    writeString(view, 36, "data");
    view.setUint32(40, dataSize, true);

    let offset = 44;
    for (let i = 0; i < float32.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, float32[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([buffer], { type: "audio/wav" });
}

function writeString(view, offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

// ---------------- App glue ----------------

const state = {
    questions: [],
    qIndex: 0,
    caseId: null,
    ageMonths: 66,
    recorder: new Recorder(),
    lastBlob: null,
    lastUrl: null,
};

const $ = (id) => document.getElementById(id);

async function init() {
    const r = await fetch("/api/questions");
    const data = await r.json();
    state.questions = data.questions;
    $("ageYears").value = Math.round(state.ageMonths / 12);
    $("startBtn").onclick = startSession;
    $("recBtn").onclick = toggleRecord;
    $("acceptBtn").onclick = acceptAndUpload;
    $("retryBtn").onclick = resetTake;
    $("assessBtn").onclick = runAssessment;

    $("tabSummary").onclick = () => switchTab("summary");
    $("tabRaw").onclick = () => switchTab("raw");
    $("reassessBtn").onclick = reassess;
    $("downloadBtn").onclick = () => {
        if (state.caseId) window.location.href = `/api/cases/${state.caseId}/raw`;
    };
    $("newSessionBtn").onclick = () => window.location.reload();

    loadCases();
}

async function startSession() {
    const years = parseInt($("ageYears").value, 10) || 5;
    state.ageMonths = years * 12;
    const r = await fetch("/api/cases", { method: "POST" });
    const { case_id } = await r.json();
    state.caseId = case_id;
    $("caseId").textContent = case_id;
    $("caseId2").textContent = case_id;
    show("setup", false);
    show("cases", false);
    show("recording", true);
    renderQuestion();
}

function renderQuestion() {
    const q = state.questions[state.qIndex];
    $("qNum").textContent = `${state.qIndex + 1} / ${state.questions.length}`;
    $("qType").textContent = q.type;
    $("qText").textContent = q.text;
    resetTake();
}

function resetTake() {
    state.lastBlob = null;
    if (state.lastUrl) URL.revokeObjectURL(state.lastUrl);
    state.lastUrl = null;
    $("playback").src = "";
    $("playback").hidden = true;
    $("acceptBtn").disabled = true;
    $("retryBtn").disabled = true;
    $("recBtn").disabled = false;
    $("recBtn").textContent = "● Record";
    $("status").textContent = "Ready.";
}

async function toggleRecord() {
    if (!state.recorder.recording) {
        $("playback").hidden = true;
        $("acceptBtn").disabled = true;
        $("retryBtn").disabled = true;
        try {
            await state.recorder.start();
            $("recBtn").textContent = "■ Stop";
            $("status").textContent = "Recording…";
        } catch (e) {
            $("status").textContent = `Mic error: ${e.message}`;
        }
        return;
    }
    const blob = await state.recorder.stop();
    state.lastBlob = blob;
    state.lastUrl = URL.createObjectURL(blob);
    $("playback").src = state.lastUrl;
    $("playback").hidden = false;
    $("recBtn").textContent = "● Re-record";
    $("acceptBtn").disabled = false;
    $("retryBtn").disabled = false;
    $("status").textContent = `Captured ${(blob.size / 1024).toFixed(0)} KB. Listen, then accept or re-record.`;
}

async function acceptAndUpload() {
    if (!state.lastBlob) return;
    $("acceptBtn").disabled = true;
    $("status").textContent = "Uploading…";

    const fd = new FormData();
    fd.append("q", String(state.qIndex + 1));
    fd.append("file", state.lastBlob, `q${String(state.qIndex + 1).padStart(2, "0")}.wav`);
    const r = await fetch(`/api/cases/${state.caseId}/upload`, { method: "POST", body: fd });
    if (!r.ok) {
        $("status").textContent = `Upload failed: ${r.status}`;
        $("acceptBtn").disabled = false;
        return;
    }

    state.qIndex += 1;
    if (state.qIndex >= state.questions.length) {
        show("recording", false);
        show("postRecordingPrompt", true);
    } else {
        renderQuestion();
    }
}

async function runAssessment() {
    $("assessBtn").disabled = true;
    $("assessStatus").textContent = "Analysing… this can take 10–30 seconds.";

    const fd = new FormData();
    fd.append("child_age_months", String(state.ageMonths));
    const r = await fetch(`/api/cases/${state.caseId}/assess`, { method: "POST", body: fd });
    const data = await r.json();

    if (!r.ok) {
        $("assessStatus").textContent = `Error: ${data.detail || r.statusText}`;
        $("assessBtn").disabled = false;
        return;
    }
    $("viewerCaseId").textContent = state.caseId;
    renderSummary(data.summary);
    renderRaw(data.raw);
    switchTab("summary");
    show("postRecordingPrompt", false);
    show("resultPane", true);
}

// ---------------- Case list + viewer ----------------

async function loadCases() {
    const r = await fetch("/api/cases");
    const cases = await r.json();
    const list = $("caseList");
    list.innerHTML = "";

    if (cases.length === 0) {
        const li = document.createElement("li");
        li.className = "case-empty";
        li.textContent = "No previous cases yet.";
        list.appendChild(li);
        return;
    }

    for (const c of cases) {
        const li = document.createElement("li");
        li.className = "case-row";
        const color = c.color || "incomplete";
        const verdict = c.has_result ? c.verdict : "Incomplete";
        const ageYears = c.child_age_months ? Math.round(c.child_age_months / 12) : null;
        const ageLabel = ageYears ? `age ${ageYears}` : `${c.num_recordings}/12 recordings`;
        const when = c.created_at ? formatDate(c.created_at) : c.case_id;

        li.innerHTML = `
            <button class="case-link" data-case="${c.case_id}" data-age="${c.child_age_months || ''}">
                <span class="case-when">${escapeHtml(when)}</span>
                <span class="case-meta">${escapeHtml(ageLabel)}</span>
                <span class="case-pill" data-color="${color}">${escapeHtml(verdict)}</span>
                <span class="case-chev">›</span>
            </button>`;
        const btn = li.querySelector("button");
        btn.onclick = () => openCase(c);
        list.appendChild(li);
    }
}

function formatDate(iso) {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    return d.toLocaleString(undefined, {
        year: "numeric", month: "short", day: "numeric",
        hour: "2-digit", minute: "2-digit",
    });
}

async function openCase(c) {
    state.caseId = c.case_id;
    state.ageMonths = c.child_age_months || state.ageMonths;
    $("viewerCaseId").textContent = c.case_id;

    if (!c.has_result) {
        // No assessment yet — show a stub asking user to assess.
        renderSummary({
            headline: { tier: "insufficient_data", color: "gray", verdict: "Not assessed yet",
                        subtext: `${c.num_recordings} recordings on disk. Click "Re-run pipeline" to analyse.` },
            alerts: [],
            groups: [],
            confidence: { level: "—", score: 0, note: "Pipeline has not been run on this case." },
            session: { usable: c.num_recordings, total: c.num_recordings, rejection_notes: [] },
            next_steps: [],
            disclaimer: "This is a screening tool, not a diagnosis. Results should always be reviewed by a qualified specialist.",
        });
        renderRaw(null);
    } else {
        const [sumR, rawR] = await Promise.all([
            fetch(`/api/cases/${c.case_id}/summary`),
            fetch(`/api/cases/${c.case_id}/raw`),
        ]);
        const summary = await sumR.json();
        const raw = await rawR.json();
        renderSummary(summary);
        renderRaw(raw);
    }

    switchTab("summary");
    show("setup", false);
    show("cases", false);
    show("resultPane", true);
}

function switchTab(name) {
    $("tabSummary").classList.toggle("active", name === "summary");
    $("tabRaw").classList.toggle("active", name === "raw");
    show("summaryView", name === "summary");
    show("rawView", name === "raw");
}

function renderRaw(raw) {
    if (raw === null) {
        $("rawJson").textContent = "(no asd_result.json saved yet — re-run the pipeline to generate one)";
        return;
    }
    $("rawJson").textContent = JSON.stringify(raw, null, 2);
}

async function reassess() {
    if (!state.caseId) return;
    const ageMonths = state.ageMonths || 60;
    $("reassessBtn").disabled = true;
    const originalText = $("reassessBtn").textContent;
    $("reassessBtn").textContent = "Re-running…";

    try {
        const fd = new FormData();
        fd.append("child_age_months", String(ageMonths));
        const r = await fetch(`/api/cases/${state.caseId}/assess`, { method: "POST", body: fd });
        const data = await r.json();
        if (!r.ok) {
            alert(`Re-assess failed: ${data.detail || r.statusText}`);
            return;
        }
        renderSummary(data.summary);
        renderRaw(data.raw);
    } finally {
        $("reassessBtn").disabled = false;
        $("reassessBtn").textContent = originalText;
    }
}

// ---------------- Result rendering ----------------

const ICONS = {
    check:    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    warn:     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><circle cx="12" cy="17" r="0.5" fill="currentColor"/></svg>',
    info:     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="11"/><circle cx="12" cy="7.5" r="0.5" fill="currentColor"/></svg>',
    download: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
};

const HEADLINE_ICON_BY_COLOR = { green: "check", yellow: "warn", red: "warn", gray: "info" };
const GROUP_ICON_BY_STATUS   = { typical: "check", atypical: "warn", not_computed: "info" };

function renderSummary(s) {
    const headline = $("headline");
    headline.dataset.color = s.headline.color;
    $("headlineIcon").innerHTML = ICONS[HEADLINE_ICON_BY_COLOR[s.headline.color] || "info"];
    $("verdict").textContent = s.headline.verdict;
    $("subtext").textContent = s.headline.subtext;

    const stamp = s.headline.speech_status || { label: "—", color: "gray" };
    $("speechStamp").textContent = stamp.label;
    $("speechStamp").dataset.color = stamp.color;

    const alertsEl = $("alerts");
    alertsEl.innerHTML = "";
    for (const a of s.alerts) {
        const div = document.createElement("div");
        div.className = "alert";
        div.dataset.level = a.level;
        div.innerHTML = `${ICONS.warn}<div><strong>${escapeHtml(a.title)}</strong><p>${escapeHtml(a.body)}</p></div>`;
        alertsEl.appendChild(div);
    }

    const groupsEl = $("groupCards");
    groupsEl.innerHTML = "";
    for (const g of s.groups) {
        const card = document.createElement("div");
        card.className = "group-card";
        card.dataset.status = g.status;
        const iconKey = GROUP_ICON_BY_STATUS[g.status] || "info";
        card.innerHTML = `
            <div class="group-icon">${ICONS[iconKey]}</div>
            <div class="group-body">
                <div class="group-name">${escapeHtml(g.name)}</div>
                <div class="group-plain">${escapeHtml(g.plain)}</div>
            </div>`;
        groupsEl.appendChild(card);
    }

    $("confLevel").textContent = s.confidence.level;
    $("confLevel").dataset.level = s.confidence.level;
    $("confNote").textContent = s.confidence.note;
    $("sessionRecs").textContent = `${s.session.usable} of ${s.session.total}`;
    $("sessionNote").textContent = s.session.rejection_notes.length
        ? s.session.rejection_notes.join("; ")
        : "All recordings usable.";

    const stepsEl = $("nextSteps");
    stepsEl.innerHTML = "";
    for (const step of s.next_steps) {
        const li = document.createElement("li");
        li.textContent = step;
        stepsEl.appendChild(li);
    }

    $("disclaimer").textContent = s.disclaimer;
    $("downloadBtn").innerHTML = `${ICONS.download}<span>Download raw results</span>`;

    renderMarkerDetails(s.groups);
}

// ---------------- Marker details + SVG range charts ----------------

function renderMarkerDetails(groups) {
    const root = $("biomarkerDetails");
    root.innerHTML = "";

    let any = false;
    for (const g of groups) {
        const markers = g.markers || [];
        if (markers.length === 0) continue;
        any = true;

        const wrap = document.createElement("div");
        wrap.className = "marker-group";
        wrap.dataset.group = g.key;
        wrap.dataset.status = g.status;

        const header = document.createElement("div");
        header.className = "marker-group-header";
        header.innerHTML = `<h4>${escapeHtml(g.name)}</h4>
            <span class="marker-pill" data-status="${g.status}">${statusPillText(g.status)}</span>`;
        wrap.appendChild(header);

        for (const m of markers) {
            wrap.appendChild(renderMarkerRow(m));
        }
        root.appendChild(wrap);
    }

    if (!any) {
        root.innerHTML = `<p class="meta">No biomarkers computed for this case.</p>`;
    }
}

function statusPillText(status) {
    return ({ typical: "Typical", atypical: "Atypical", not_computed: "Not computed" })[status] || status;
}

function renderMarkerRow(m) {
    const div = document.createElement("div");
    div.className = "marker-row";
    div.dataset.atypical = String(m.atypical === true);
    div.dataset.computed = String(m.computed === true);

    const status = !m.computed
        ? { label: "Not computed", cls: "not_computed" }
        : m.atypical
            ? { label: "Atypical", cls: "atypical" }
            : { label: "Typical", cls: "typical" };

    const valueText = m.computed
        ? `${formatVal(m.value)}${m.unit ? " " + m.unit : ""}`
        : "—";

    let rangeText = "";
    if (m.norm_low !== undefined && m.norm_low !== null) {
        rangeText = `Typical: ${formatVal(m.norm_low)} – ${formatVal(m.norm_high)}${m.unit ? " " + m.unit : ""}`;
    } else if (m.threshold !== undefined && m.threshold !== null) {
        rangeText = `Threshold: ${formatVal(m.threshold)}${m.unit ? " " + m.unit : ""}`;
    }

    div.innerHTML = `
        <div class="marker-head">
            <div class="marker-name">
                <div class="marker-label">${escapeHtml(m.label)}</div>
                <div class="marker-tech">${escapeHtml(m.tech_label)}</div>
            </div>
            <span class="marker-pill" data-status="${status.cls}">${status.label}</span>
        </div>
        <p class="marker-desc">${escapeHtml(m.description)}</p>
        <div class="marker-chart-row">
            ${renderMarkerChart(m)}
            <div class="marker-numbers">
                <div class="marker-value">${valueText}</div>
                <div class="marker-range">${rangeText}</div>
            </div>
        </div>
    `;
    return div;
}

function formatVal(v) {
    if (v === null || v === undefined) return "—";
    const a = Math.abs(v);
    if (a === 0) return "0";
    if (a < 0.01) return v.toExponential(1);
    if (a < 1) return v.toFixed(3);
    if (a < 100) return v.toFixed(1);
    return Math.round(v).toLocaleString();
}

function renderMarkerChart(m) {
    const W = 260, H = 40;
    const PAD = 10;
    const trackY = 20;
    const trackH = 8;

    if (!m.computed) {
        return `<svg viewBox="0 0 ${W} ${H}" class="marker-chart" preserveAspectRatio="none">
            <rect x="${PAD}" y="${trackY - trackH / 2}" width="${W - 2 * PAD}" height="${trackH}" fill="var(--gray-bg)" stroke="var(--border)"/>
            <text x="${W / 2}" y="${trackY + 4}" text-anchor="middle" class="chart-na">not computed</text>
        </svg>`;
    }

    // Threshold-based charts
    if (m.direction === "threshold") {
        const tVal = m.threshold;
        const axisMax = Math.max(tVal * 2, m.value * 1.1);
        const tX = PAD + (tVal / axisMax) * (W - 2 * PAD);
        const vX = PAD + Math.min(1, m.value / axisMax) * (W - 2 * PAD);
        const valColor = m.atypical ? "var(--red-fg)" : "var(--green-fg)";
        return `<svg viewBox="0 0 ${W} ${H}" class="marker-chart" preserveAspectRatio="none">
            <rect x="${PAD}" y="${trackY - trackH / 2}" width="${tX - PAD}" height="${trackH}" fill="var(--green-bg)" stroke="var(--green-border)"/>
            <rect x="${tX}" y="${trackY - trackH / 2}" width="${W - PAD - tX}" height="${trackH}" fill="var(--red-bg)" stroke="var(--red-border)"/>
            <line x1="${tX}" y1="${trackY - trackH / 2 - 3}" x2="${tX}" y2="${trackY + trackH / 2 + 3}" stroke="var(--gray-fg)" stroke-dasharray="2 2"/>
            <text x="${tX}" y="${trackY + trackH / 2 + 13}" text-anchor="middle" class="chart-tick">threshold</text>
            <circle cx="${vX}" cy="${trackY}" r="6" fill="${valColor}" stroke="white" stroke-width="2.5"/>
        </svg>`;
    }

    // Norm-based charts. Axis covers mean ± 4 SD so even outliers are visible.
    const axisLow = m.norm_mean - 4 * m.norm_std;
    const axisHigh = m.norm_mean + 4 * m.norm_std;
    const range = axisHigh - axisLow || 1;
    const xFor = (v) => PAD + Math.max(0, Math.min(1, (v - axisLow) / range)) * (W - 2 * PAD);
    const lowX = xFor(m.norm_low);
    const highX = xFor(m.norm_high);
    const valX = xFor(m.value);
    const valColor = m.atypical ? "var(--red-fg)" : "var(--green-fg)";

    let bands = "";
    if (m.direction === "high") {
        bands = `
            <rect x="${PAD}" y="${trackY - trackH / 2}" width="${highX - PAD}" height="${trackH}" fill="var(--green-bg)" stroke="var(--green-border)"/>
            <rect x="${highX}" y="${trackY - trackH / 2}" width="${W - PAD - highX}" height="${trackH}" fill="var(--red-bg)" stroke="var(--red-border)"/>`;
    } else if (m.direction === "low") {
        bands = `
            <rect x="${PAD}" y="${trackY - trackH / 2}" width="${lowX - PAD}" height="${trackH}" fill="var(--red-bg)" stroke="var(--red-border)"/>
            <rect x="${lowX}" y="${trackY - trackH / 2}" width="${W - PAD - lowX}" height="${trackH}" fill="var(--green-bg)" stroke="var(--green-border)"/>`;
    } else {
        bands = `
            <rect x="${PAD}" y="${trackY - trackH / 2}" width="${lowX - PAD}" height="${trackH}" fill="var(--red-bg)" stroke="var(--red-border)"/>
            <rect x="${lowX}" y="${trackY - trackH / 2}" width="${highX - lowX}" height="${trackH}" fill="var(--green-bg)" stroke="var(--green-border)"/>
            <rect x="${highX}" y="${trackY - trackH / 2}" width="${W - PAD - highX}" height="${trackH}" fill="var(--red-bg)" stroke="var(--red-border)"/>`;
    }

    // Tick labels under the typical-zone edges.
    const ticks = `
        <text x="${lowX}" y="${trackY + trackH / 2 + 13}" text-anchor="middle" class="chart-tick">${formatVal(m.norm_low)}</text>
        <text x="${highX}" y="${trackY + trackH / 2 + 13}" text-anchor="middle" class="chart-tick">${formatVal(m.norm_high)}</text>`;

    return `<svg viewBox="0 0 ${W} ${H}" class="marker-chart" preserveAspectRatio="none">
        ${bands}
        ${ticks}
        <circle cx="${valX}" cy="${trackY}" r="6" fill="${valColor}" stroke="white" stroke-width="2.5"/>
    </svg>`;
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
}

function show(id, visible) {
    $(id).hidden = !visible;
}

document.addEventListener("DOMContentLoaded", init);
