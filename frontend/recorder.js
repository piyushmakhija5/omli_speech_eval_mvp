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

// ============================================================================
// App glue
// ============================================================================

const state = {
    questions: [],
    qIndex: 0,
    caseId: null,
    ageMonths: 66,
    recorder: new Recorder(),
    lastBlob: null,
    lastUrl: null,
    // Cached raw results so the Raw JSON tab can switch instantly without
    // re-fetching. Populated by openCase / runAssessment / reassess.
    rawByPipeline: { asd: null, speech_delay: null },
    rawTabWhich: "asd",
};

const $ = (id) => document.getElementById(id);
const show = (id, visible) => { $(id).hidden = !visible; };

const ICONS = {
    check:    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    warn:     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><circle cx="12" cy="17" r="0.5" fill="currentColor"/></svg>',
    info:     '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="11"/><circle cx="12" cy="7.5" r="0.5" fill="currentColor"/></svg>',
    download: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
};

const HEADLINE_ICON_BY_COLOR = { green: "check", yellow: "warn", red: "warn", gray: "info" };
const ASD_GROUP_ICON_BY_STATUS = { typical: "check", atypical: "warn", not_computed: "info" };

// Speech-delay status → CSS data-status (reuses ASD's pill colour classes):
// on_track stays typical-green, behind goes to "atypical" yellow/red, etc.
const SD_STATUS_TO_DATA = {
    on_track: "typical",
    behind: "atypical",
    significantly_behind: "atypical",
    not_computed: "not_computed",
};
const SD_STATUS_LABEL = {
    on_track: "On track",
    behind: "Behind",
    significantly_behind: "Significantly behind",
    not_computed: "Not computed",
};
const SD_STATUS_ICON = {
    on_track: "check",
    behind: "warn",
    significantly_behind: "warn",
    not_computed: "info",
};

async function init() {
    const r = await fetch("/api/questions");
    state.questions = (await r.json()).questions;
    $("ageYears").value = Math.round(state.ageMonths / 12);

    $("startBtn").onclick = startSession;
    $("recBtn").onclick = toggleRecord;
    $("acceptBtn").onclick = acceptAndUpload;
    $("retryBtn").onclick = resetTake;
    $("assessBtn").onclick = runAssessment;

    $("tabSummary").onclick = () => switchTab("summary");
    $("tabRaw").onclick = () => switchTab("raw");

    $("rawTabAsd").onclick = () => switchRawTab("asd");
    $("rawTabSd").onclick = () => switchRawTab("speech_delay");

    $("reassessBtn").onclick = reassess;
    $("downloadAsdBtn").onclick = () => {
        if (state.caseId) window.location.href = `/api/cases/${state.caseId}/raw?which=asd`;
    };
    $("downloadSdBtn").onclick = () => {
        if (state.caseId) window.location.href = `/api/cases/${state.caseId}/raw?which=speech_delay`;
    };
    $("newSessionBtn").onclick = () => window.location.reload();

    loadCases();
}

// ----------------------- Recording flow ------------------------------------

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
    $("assessStatus").textContent = "Analysing… this can take 30–60 seconds (both pipelines).";

    const fd = new FormData();
    fd.append("child_age_months", String(state.ageMonths));
    const r = await fetch(`/api/cases/${state.caseId}/assess`, { method: "POST", body: fd });
    const data = await r.json();

    if (!r.ok) {
        $("assessStatus").textContent = `Error: ${data.detail || r.statusText}`;
        $("assessBtn").disabled = false;
        return;
    }
    presentResults(state.caseId, data.asd, data.speech_delay);
    show("postRecordingPrompt", false);
    show("resultPane", true);
}

// ----------------------- Case list + viewer --------------------------------

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

        const ageYears = c.child_age_months ? Math.round(c.child_age_months / 12) : null;
        const meta = ageYears ? `age ${ageYears}` : `${c.num_recordings}/12 recordings`;
        const when = c.created_at ? formatDate(c.created_at) : c.case_id;

        const asdLabel = c.has_asd_result ? (c.asd_verdict || "—") : "Not assessed";
        const asdColor = c.has_asd_result ? (c.asd_color || "gray") : "incomplete";
        const sdLabel = c.has_speech_delay_result
            ? (SD_STATUS_LABEL[c.speech_delay_status] || c.speech_delay_status || "—")
            : "Not assessed";
        const sdColor = c.has_speech_delay_result ? (c.speech_delay_color || "gray") : "incomplete";

        li.innerHTML = `
            <button class="case-link" data-case="${c.case_id}">
                <div class="case-when-meta">
                    <span class="case-when">${escapeHtml(when)}</span>
                    <span class="case-meta">${escapeHtml(meta)}</span>
                </div>
                <div class="case-pills">
                    <span class="case-pill" data-color="${asdColor}">ASD: ${escapeHtml(asdLabel)}</span>
                    <span class="case-pill" data-color="${sdColor}">Speech delay: ${escapeHtml(sdLabel)}</span>
                </div>
                <span class="case-chev">›</span>
            </button>`;
        li.querySelector("button").onclick = () => openCase(c);
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

    // Pull both summaries from /summary (joined response).
    let asdSummary = null;
    let sdSummary = null;
    const sumR = await fetch(`/api/cases/${c.case_id}/summary`);
    if (sumR.ok) {
        const data = await sumR.json();
        asdSummary = data.asd;
        sdSummary = data.speech_delay;
    }

    // Pull raw payloads for both pipelines, but only if their result file exists.
    const rawByPipeline = { asd: null, speech_delay: null };
    if (c.has_asd_result) {
        const r = await fetch(`/api/cases/${c.case_id}/raw?which=asd`);
        if (r.ok) rawByPipeline.asd = await r.json();
    }
    if (c.has_speech_delay_result) {
        const r = await fetch(`/api/cases/${c.case_id}/raw?which=speech_delay`);
        if (r.ok) rawByPipeline.speech_delay = await r.json();
    }

    presentResults(c.case_id, { raw: rawByPipeline.asd, summary: asdSummary },
                              { raw: rawByPipeline.speech_delay, summary: sdSummary });

    show("setup", false);
    show("cases", false);
    show("resultPane", true);
}

function presentResults(caseId, asdResult, sdResult) {
    state.caseId = caseId;
    $("viewerCaseId").textContent = caseId;
    state.rawByPipeline.asd = asdResult ? asdResult.raw : null;
    state.rawByPipeline.speech_delay = sdResult ? sdResult.raw : null;

    renderASDSummary(asdResult ? asdResult.summary : null);
    renderSpeechDelaySummary(sdResult ? sdResult.summary : null);

    // Disclaimer is shared between both columns — pull from whichever summary is present.
    const disclaimerText = (sdResult && sdResult.summary && sdResult.summary.disclaimer)
        || (asdResult && asdResult.summary && asdResult.summary.disclaimer)
        || "";
    $("disclaimer").textContent = disclaimerText;

    switchTab("summary");
    switchRawTab(state.rawTabWhich);
}

function switchTab(name) {
    $("tabSummary").classList.toggle("active", name === "summary");
    $("tabRaw").classList.toggle("active", name === "raw");
    show("summaryView", name === "summary");
    show("rawView", name === "raw");
}

function switchRawTab(which) {
    state.rawTabWhich = which;
    $("rawTabAsd").classList.toggle("active", which === "asd");
    $("rawTabSd").classList.toggle("active", which === "speech_delay");
    const raw = state.rawByPipeline[which];
    $("rawJson").textContent = raw
        ? JSON.stringify(raw, null, 2)
        : `(no ${which} result saved — run the pipeline first)`;
}

// ----------------------- ASD summary renderer ------------------------------

function renderASDSummary(s) {
    const colHeadline = $("asdHeadline");
    if (!s) {
        colHeadline.dataset.color = "gray";
        $("asdHeadlineIcon").innerHTML = ICONS.info;
        $("asdVerdict").textContent = "Not assessed";
        $("asdSubtext").textContent = "ASD pipeline has not been run for this case.";
        $("asdSpeechStamp").textContent = "—";
        $("asdSpeechStamp").dataset.color = "gray";
        $("asdAlerts").innerHTML = "";
        $("asdGroupCards").innerHTML = "";
        $("asdConfLevel").textContent = "—";
        $("asdConfNote").textContent = "—";
        $("asdSessionRecs").textContent = "—";
        $("asdSessionNote").textContent = "";
        $("asdBiomarkerDetails").innerHTML = `<p class="meta">No ASD result.</p>`;
        $("asdNextSteps").innerHTML = "";
        return;
    }

    colHeadline.dataset.color = s.headline.color;
    $("asdHeadlineIcon").innerHTML = ICONS[HEADLINE_ICON_BY_COLOR[s.headline.color] || "info"];
    $("asdVerdict").textContent = s.headline.verdict;
    $("asdSubtext").textContent = s.headline.subtext;

    const stamp = s.headline.speech_status || { label: "—", color: "gray" };
    $("asdSpeechStamp").textContent = stamp.label;
    $("asdSpeechStamp").dataset.color = stamp.color;

    renderAlerts($("asdAlerts"), s.alerts);

    const groupsEl = $("asdGroupCards");
    groupsEl.innerHTML = "";
    for (const g of s.groups) {
        const card = document.createElement("div");
        card.className = "group-card";
        card.dataset.status = g.status;
        const iconKey = ASD_GROUP_ICON_BY_STATUS[g.status] || "info";
        card.innerHTML = `
            <div class="group-icon">${ICONS[iconKey]}</div>
            <div class="group-body">
                <div class="group-name">${escapeHtml(g.name)}</div>
                <div class="group-plain">${escapeHtml(g.plain)}</div>
            </div>`;
        groupsEl.appendChild(card);
    }

    $("asdConfLevel").textContent = s.confidence.level;
    $("asdConfLevel").dataset.level = s.confidence.level;
    $("asdConfNote").textContent = s.confidence.note;
    $("asdSessionRecs").textContent = `${s.session.usable} of ${s.session.total}`;
    $("asdSessionNote").textContent = s.session.rejection_notes.length
        ? s.session.rejection_notes.join("; ")
        : "All recordings usable.";

    renderASDMarkerDetails(s.groups);
    renderStepsList($("asdNextSteps"), s.next_steps);
}

function renderASDMarkerDetails(groups) {
    const root = $("asdBiomarkerDetails");
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
        const statusLabel = ({ typical: "Typical", atypical: "Atypical", not_computed: "Not computed" })[g.status] || g.status;
        wrap.innerHTML = `<div class="marker-group-header">
            <h5>${escapeHtml(g.name)}</h5>
            <span class="marker-pill" data-status="${g.status}">${statusLabel}</span>
        </div>`;
        for (const m of markers) {
            wrap.appendChild(renderASDMarkerRow(m));
        }
        root.appendChild(wrap);
    }
    if (!any) root.innerHTML = `<p class="meta">No biomarkers computed for this case.</p>`;
}

function renderASDMarkerRow(m) {
    const div = document.createElement("div");
    div.className = "marker-row";
    div.dataset.atypical = String(m.atypical === true);
    div.dataset.computed = String(m.computed === true);

    const status = !m.computed
        ? { label: "Not computed", cls: "not_computed" }
        : m.atypical
            ? { label: "Atypical", cls: "atypical" }
            : { label: "Typical", cls: "typical" };

    const valueText = m.computed ? `${formatVal(m.value)}${m.unit ? " " + m.unit : ""}` : "—";
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
            ${renderASDMarkerChart(m)}
            <div class="marker-numbers">
                <div class="marker-value">${valueText}</div>
                <div class="marker-range">${rangeText}</div>
            </div>
        </div>`;
    return div;
}

function renderASDMarkerChart(m) {
    const W = 260, H = 40, PAD = 10, trackY = 20, trackH = 8;

    if (!m.computed) {
        return `<svg viewBox="0 0 ${W} ${H}" class="marker-chart" preserveAspectRatio="none">
            <rect x="${PAD}" y="${trackY - trackH / 2}" width="${W - 2 * PAD}" height="${trackH}" fill="var(--gray-bg)" stroke="var(--border)"/>
            <text x="${W / 2}" y="${trackY + 4}" text-anchor="middle" class="chart-na">not computed</text>
        </svg>`;
    }

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
    const ticks = `
        <text x="${lowX}" y="${trackY + trackH / 2 + 13}" text-anchor="middle" class="chart-tick">${formatVal(m.norm_low)}</text>
        <text x="${highX}" y="${trackY + trackH / 2 + 13}" text-anchor="middle" class="chart-tick">${formatVal(m.norm_high)}</text>`;

    return `<svg viewBox="0 0 ${W} ${H}" class="marker-chart" preserveAspectRatio="none">
        ${bands}
        ${ticks}
        <circle cx="${valX}" cy="${trackY}" r="6" fill="${valColor}" stroke="white" stroke-width="2.5"/>
    </svg>`;
}

// ----------------------- Speech-delay summary renderer --------------------

function renderSpeechDelaySummary(s) {
    if (!s) {
        $("sdHeadline").dataset.color = "gray";
        $("sdHeadlineIcon").innerHTML = ICONS.info;
        $("sdVerdict").textContent = "Not assessed";
        $("sdSubtext").textContent = "Speech-delay pipeline has not been run for this case.";
        $("sdDelayLabel").textContent = "";
        $("sdAlerts").innerHTML = "";
        $("sdDomainCards").innerHTML = "";
        $("sdConfLevel").textContent = "—";
        $("sdConfNote").textContent = "—";
        $("sdSessionRecs").textContent = "—";
        $("sdSessionNote").textContent = "";
        $("sdMetricDetails").innerHTML = `<p class="meta">No speech-delay result.</p>`;
        $("sdNextSteps").innerHTML = "";
        return;
    }

    $("sdHeadline").dataset.color = s.headline.color;
    $("sdHeadlineIcon").innerHTML = ICONS[HEADLINE_ICON_BY_COLOR[s.headline.color] || "info"];
    $("sdVerdict").textContent = s.headline.verdict;
    $("sdSubtext").textContent = s.headline.subtext;
    $("sdDelayLabel").textContent = s.headline.delay_label || "";

    renderAlerts($("sdAlerts"), s.alerts);

    const domainsEl = $("sdDomainCards");
    domainsEl.innerHTML = "";
    for (const d of s.domains) {
        const card = document.createElement("div");
        card.className = "group-card";
        card.dataset.status = SD_STATUS_TO_DATA[d.status] || "not_computed";
        const iconKey = SD_STATUS_ICON[d.status] || "info";
        const percText = d.percentile !== null && d.percentile !== undefined ? `<strong>p${d.percentile}</strong>` : "";
        card.innerHTML = `
            <div class="group-icon">${ICONS[iconKey]}</div>
            <div class="group-body">
                <div class="group-name">${escapeHtml(d.name)} ${percText}</div>
                <div class="group-plain">${escapeHtml(d.description)} <span class="domain-counts">(${d.computed_count}/${d.total_count} metrics)</span></div>
            </div>`;
        domainsEl.appendChild(card);
    }

    $("sdConfLevel").textContent = s.confidence.level;
    $("sdConfLevel").dataset.level = s.confidence.level;
    $("sdConfNote").textContent = s.confidence.note;
    $("sdSessionRecs").textContent = `${s.session.usable} of ${s.session.total}`;
    $("sdSessionNote").textContent = s.session.rejection_notes.length
        ? s.session.rejection_notes.join("; ")
        : "All recordings usable.";

    renderSpeechDelayMetricDetails(s.metrics);
    renderStepsList($("sdNextSteps"), s.next_steps);
}

function renderSpeechDelayMetricDetails(metrics) {
    const root = $("sdMetricDetails");
    root.innerHTML = "";
    if (!metrics || metrics.length === 0) {
        root.innerHTML = `<p class="meta">No metrics computed for this case.</p>`;
        return;
    }

    // Group metrics by domain so the details list reads "Articulation: X, Y; Language: …"
    const byDomain = { articulation: [], language: [], fluency: [] };
    for (const m of metrics) {
        (byDomain[m.domain] || (byDomain[m.domain] = [])).push(m);
    }
    const DOMAIN_NAMES = { articulation: "Articulation", language: "Language", fluency: "Fluency & voice" };

    for (const domainKey of ["articulation", "language", "fluency"]) {
        const ms = byDomain[domainKey] || [];
        if (ms.length === 0) continue;
        const wrap = document.createElement("div");
        wrap.className = "marker-group";
        wrap.dataset.group = domainKey;
        const computedCount = ms.filter((m) => m.computed).length;
        wrap.innerHTML = `<div class="marker-group-header">
            <h5>${escapeHtml(DOMAIN_NAMES[domainKey] || domainKey)}</h5>
            <span class="marker-pill" data-status="${computedCount === 0 ? 'not_computed' : 'typical'}">${computedCount}/${ms.length} computed</span>
        </div>`;
        for (const m of ms) {
            wrap.appendChild(renderSpeechDelayMetricRow(m));
        }
        root.appendChild(wrap);
    }
}

function renderSpeechDelayMetricRow(m) {
    const div = document.createElement("div");
    div.className = "marker-row";
    div.dataset.computed = String(m.computed === true);
    const dataStatus = SD_STATUS_TO_DATA[m.status] || "not_computed";
    const statusLabel = SD_STATUS_LABEL[m.status] || m.status;

    const valueText = m.computed
        ? `${formatVal(m.value)}${m.unit ? " " + m.unit : ""}`
        : "—";
    const percText = (m.percentile !== null && m.percentile !== undefined) ? `p${m.percentile}` : "";

    const modeNote = m.mode_note ? `<br><em class="mode-note">${escapeHtml(m.mode_note)}</em>` : "";
    const reasonNote = !m.computed && m.reason ? `<br><em class="mode-note">${escapeHtml(m.reason)}</em>` : "";

    div.innerHTML = `
        <div class="marker-head">
            <div class="marker-name">
                <div class="marker-label">${escapeHtml(m.label)}</div>
                <div class="marker-tech">${escapeHtml(m.tech_label)}</div>
            </div>
            <span class="marker-pill" data-status="${dataStatus}">${statusLabel}</span>
        </div>
        <p class="marker-desc">${escapeHtml(m.description)}${modeNote}${reasonNote}</p>
        <div class="marker-chart-row">
            ${renderPercentileChart(m)}
            <div class="marker-numbers">
                <div class="marker-value">${valueText}</div>
                <div class="marker-range">${percText}</div>
            </div>
        </div>`;
    return div;
}

function renderPercentileChart(m) {
    const W = 260, H = 40, PAD = 10, trackY = 20, trackH = 8;

    if (!m.computed || m.percentile === null || m.percentile === undefined) {
        return `<svg viewBox="0 0 ${W} ${H}" class="marker-chart" preserveAspectRatio="none">
            <rect x="${PAD}" y="${trackY - trackH / 2}" width="${W - 2 * PAD}" height="${trackH}" fill="var(--gray-bg)" stroke="var(--border)"/>
            <text x="${W / 2}" y="${trackY + 4}" text-anchor="middle" class="chart-na">not computed</text>
        </svg>`;
    }

    // Axis 0-100. Red < p10, yellow p10-p25, green ≥ p25.
    const xFor = (p) => PAD + (Math.max(0, Math.min(100, p)) / 100) * (W - 2 * PAD);
    const x0 = xFor(0), x10 = xFor(10), x25 = xFor(25), x100 = xFor(100);
    const vX = xFor(m.percentile);

    const valColor = m.status === "on_track"
        ? "var(--green-fg)"
        : m.status === "behind"
            ? "var(--yellow-fg)"
            : "var(--red-fg)";

    return `<svg viewBox="0 0 ${W} ${H}" class="marker-chart" preserveAspectRatio="none">
        <rect x="${x0}"  y="${trackY - trackH / 2}" width="${x10 - x0}"  height="${trackH}" fill="var(--red-bg)"    stroke="var(--red-border)"/>
        <rect x="${x10}" y="${trackY - trackH / 2}" width="${x25 - x10}" height="${trackH}" fill="var(--yellow-bg)" stroke="var(--yellow-border)"/>
        <rect x="${x25}" y="${trackY - trackH / 2}" width="${x100 - x25}" height="${trackH}" fill="var(--green-bg)"  stroke="var(--green-border)"/>
        <text x="${x10}" y="${trackY + trackH / 2 + 13}" text-anchor="middle" class="chart-tick">p10</text>
        <text x="${x25}" y="${trackY + trackH / 2 + 13}" text-anchor="middle" class="chart-tick">p25</text>
        <circle cx="${vX}" cy="${trackY}" r="6" fill="${valColor}" stroke="white" stroke-width="2.5"/>
    </svg>`;
}

// ----------------------- Shared helpers ------------------------------------

function renderAlerts(rootEl, alerts) {
    rootEl.innerHTML = "";
    for (const a of (alerts || [])) {
        const div = document.createElement("div");
        div.className = "alert";
        div.dataset.level = a.level || "warning";
        div.innerHTML = `${ICONS.warn}<div><strong>${escapeHtml(a.title)}</strong><p>${escapeHtml(a.body)}</p></div>`;
        rootEl.appendChild(div);
    }
}

function renderStepsList(rootEl, steps) {
    rootEl.innerHTML = "";
    for (const s of (steps || [])) {
        const li = document.createElement("li");
        li.textContent = s;
        rootEl.appendChild(li);
    }
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

function escapeHtml(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
}

// ----------------------- Re-assess + flash ---------------------------------

let _lastRerunAt = null;
let _rerunTickInterval = null;

function setReassessStatus(text, state) {
    const el = $("reassessStatus");
    el.textContent = text;
    el.dataset.state = state;
    el.hidden = false;
}

function clearRerunTicker() {
    if (_rerunTickInterval) { clearInterval(_rerunTickInterval); _rerunTickInterval = null; }
}

function tickRerunStatus() {
    if (!_lastRerunAt) return;
    const ageS = Math.round((Date.now() - _lastRerunAt) / 1000);
    let label;
    if (ageS < 5) label = "just now";
    else if (ageS < 60) label = `${ageS} seconds ago`;
    else if (ageS < 3600) label = `${Math.round(ageS / 60)} min ago`;
    else { clearRerunTicker(); label = `at ${new Date(_lastRerunAt).toLocaleTimeString()}`; }
    setReassessStatus(`✓ Re-ran ${label} — both screenings refreshed.`, "success");
}

function flashResults() {
    const target = $("summaryView");
    target.classList.remove("flash");
    void target.offsetWidth;
    target.classList.add("flash");
}

async function reassess() {
    if (!state.caseId) return;
    const ageMonths = state.ageMonths || 60;
    $("reassessBtn").disabled = true;
    const originalText = $("reassessBtn").textContent;
    $("reassessBtn").textContent = "Re-running…";
    clearRerunTicker();
    setReassessStatus("Re-running both pipelines… (30–60 seconds)", "running");

    try {
        const fd = new FormData();
        fd.append("child_age_months", String(ageMonths));
        const r = await fetch(`/api/cases/${state.caseId}/assess`, { method: "POST", body: fd });
        const data = await r.json();
        if (!r.ok) {
            setReassessStatus(`✗ Re-run failed: ${data.detail || r.statusText}`, "error");
            return;
        }
        presentResults(state.caseId, data.asd, data.speech_delay);
        flashResults();
        _lastRerunAt = Date.now();
        tickRerunStatus();
        _rerunTickInterval = setInterval(tickRerunStatus, 5000);
    } catch (e) {
        setReassessStatus(`✗ Re-run failed: ${e.message}`, "error");
    } finally {
        $("reassessBtn").disabled = false;
        $("reassessBtn").textContent = originalText;
    }
}

document.addEventListener("DOMContentLoaded", init);
