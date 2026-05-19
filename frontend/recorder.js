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
        show("done", true);
    } else {
        renderQuestion();
    }
}

async function runAssessment() {
    $("assessBtn").disabled = true;
    $("result").textContent = "Running pipeline… (this can take 10–30s)";

    const fd = new FormData();
    fd.append("child_age_months", String(state.ageMonths));
    const r = await fetch(`/api/cases/${state.caseId}/assess`, { method: "POST", body: fd });
    const data = await r.json();

    if (!r.ok) {
        $("result").textContent = `Error: ${data.detail || r.statusText}`;
        $("assessBtn").disabled = false;
        return;
    }
    $("result").textContent = JSON.stringify(data, null, 2);
}

function show(id, visible) {
    $(id).hidden = !visible;
}

document.addEventListener("DOMContentLoaded", init);
