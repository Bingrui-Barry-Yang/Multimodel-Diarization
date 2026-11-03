"use strict";

const $ = (id) => document.getElementById(id);
const STAGE_BG = "#121418"; // must match render.BACKGROUND_BGR
const STAGES = ["prepare", "track", "pose", "render", "voice", "mux"];
// which rendered video the processed layer shows in each mode
const PROC_FILE = { pose: "pose.mp4", blobpose: "blobpose.mp4" };
const NEEDS_ORIGINAL = ["original", "overlay", "split"];
const NEEDS_POSE = ["pose", "blobpose"];

const state = {
  config: null,
  file: null,
  classes: ["person"],
  jobs: [],
  currentJobId: null,
  // viewer
  job: null,
  tracks: null,
  poses: null,
  mode: "blobs",
  audio: "altered",
  selected: null,
  hidden: new Set(),
  split: 50,
  frame: -1,
  hasOriginal: true,
};

/* ---------------------------------------------------------------- helpers */

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return res.json();
}

function fmtTime(t) {
  if (!isFinite(t)) t = 0;
  const m = Math.floor(t / 60);
  const s = (t - m * 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

function fileUrl(job, name) {
  return `/api/jobs/${job.id}/files/${name}`;
}

/* --------------------------------------------------------------- settings */

async function loadConfig() {
  state.config = await api("/api/config");
  const eng = $("engine");
  eng.innerHTML = "";
  for (const e of state.config.engines) {
    const o = new Option(e.label + (e.available ? "" : " — not installed"), e.id);
    o.disabled = !e.available;
    if (e.hint && !e.available) o.title = e.hint;
    eng.add(o);
  }
  const style = $("style");
  for (const s of state.config.styles) style.add(new Option(s, s));
  const add = $("class-add");
  for (const c of state.config.classes) add.add(new Option(c, c));
  renderClassChips();

  for (const [id, fmt] of [["smooth", (v) => v], ["alpha", (v) => v], ["pitch", (v) => `${v > 0 ? "+" : ""}${v} st`]]) {
    const input = $(id), out = $(`${id}-out`);
    const upd = () => (out.textContent = fmt(input.value));
    input.addEventListener("input", upd);
    upd();
  }
}

function renderClassChips() {
  const box = $("classes");
  box.innerHTML = "";
  for (const c of state.classes) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = c;
    const x = document.createElement("button");
    x.textContent = "×";
    x.title = "remove";
    x.onclick = () => {
      if (state.classes.length > 1) { state.classes = state.classes.filter((k) => k !== c); renderClassChips(); }
    };
    chip.append(x);
    box.append(chip);
  }
}

$("class-add").addEventListener("change", (e) => {
  const v = e.target.value;
  if (v && !state.classes.includes(v)) { state.classes.push(v); renderClassChips(); }
  e.target.value = "";
});

function collectParams() {
  const engine = $("engine").value;
  return {
    engine,
    discard_original: $("discard").checked,
    track: {
      classes: state.classes,
      model_size: $("model-size").value,
      text_prompt: state.classes.join(", "),
    },
    blob: { style: $("style").value, smooth: parseFloat($("smooth").value) },
    pose: { enabled: $("pose").checked },
    voice: { mcadams_alpha: parseFloat($("alpha").value), pitch_semitones: parseFloat($("pitch").value) },
  };
}

/* ----------------------------------------------------------------- upload */

function setFile(f) {
  if (!f) return;
  state.file = f;
  $("file-name").textContent = `${f.name} · ${(f.size / 1e6).toFixed(1)} MB`;
  $("process-btn").disabled = false;
}

const dz = $("dropzone");
$("file-input").addEventListener("change", (e) => setFile(e.target.files[0]));
for (const ev of ["dragenter", "dragover"]) dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); });
for (const ev of ["dragleave", "drop"]) dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); });
dz.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
// allow dropping anywhere on the page
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => { e.preventDefault(); if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });

$("process-btn").addEventListener("click", async () => {
  if (!state.file) return;
  const btn = $("process-btn");
  btn.disabled = true;
  btn.textContent = "Uploading…";
  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("params", JSON.stringify(collectParams()));
  try {
    const job = await api("/api/jobs", { method: "POST", body: fd });
    state.file = null;
    $("file-input").value = "";
    $("file-name").textContent = "mp4, mov, webm, avi…";
    await refreshJobs();
    openJob(job.id);
  } catch (err) {
    alert(`Upload failed: ${err.message}`);
    btn.disabled = false;
  } finally {
    btn.textContent = "Process video";
  }
});

/* ------------------------------------------------------------------- jobs */

async function refreshJobs() {
  state.jobs = await api("/api/jobs");
  renderJobList();
  const cur = state.jobs.find((j) => j.id === state.currentJobId);
  if (cur) updateJobView(cur);
}

function renderJobList() {
  const ul = $("job-list");
  ul.innerHTML = "";
  if (!state.jobs.length) { ul.innerHTML = '<li class="muted small">No jobs yet</li>'; return; }
  for (const j of state.jobs) {
    const li = document.createElement("li");
    li.className = "job" + (j.id === state.currentJobId ? " active" : "");
    const pct = j.status === "running" ? ` ${Math.round(j.progress * 100)}%` : "";
    const sub = j.result ? `${j.result.objects} objects · ${fmtTime(j.result.duration)}` : (j.error || j.message || "");
    li.innerHTML = `
      <span class="job-name"></span>
      <span class="job-status ${j.status}">${j.status}${pct}</span>
      <span class="job-sub"><span class="job-sub-text"></span><button class="job-del" title="Delete job and all its files">delete</button></span>`;
    li.querySelector(".job-name").textContent = j.filename;
    li.querySelector(".job-sub-text").textContent = sub;
    li.onclick = () => openJob(j.id);
    li.querySelector(".job-del").onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete "${j.filename}" and all its outputs?`)) return;
      try { await api(`/api/jobs/${j.id}`, { method: "DELETE" }); } catch (err) { alert(err.message); }
      if (state.currentJobId === j.id) showView("empty");
      refreshJobs();
    };
    ul.append(li);
  }
}

function openJob(id) {
  state.currentJobId = id;
  if (new URLSearchParams(location.hash.slice(1)).get("job") !== id) history.replaceState(null, "", `#job=${id}`);
  state.job = null;
  renderJobList();
  const j = state.jobs.find((x) => x.id === id);
  if (j) updateJobView(j);
}

function updateJobView(j) {
  if (j.status === "done") {
    if (!state.job || state.job.id !== j.id || state.job.original_discarded !== j.original_discarded) loadViewer(j);
    return;
  }
  showView("progress");
  $("progress-title").textContent = j.status === "error" ? `Failed: ${j.filename}` : `Processing ${j.filename}`;
  $("bar-fill").style.width = `${Math.round(j.progress * 100)}%`;
  $("progress-msg").textContent = j.status === "error" ? j.error : (j.status === "queued" ? "waiting in queue…" : j.message);
  const ol = $("stages");
  ol.innerHTML = "";
  const cur = STAGES.indexOf(j.stage);
  STAGES.forEach((s, i) => {
    const li = document.createElement("li");
    li.textContent = s;
    if (i < cur || (i === cur && j.stage_progress >= 1)) li.className = "done";
    else if (i === cur) li.className = "active";
    ol.append(li);
  });
}

function showView(which) {
  $("empty").hidden = which !== "empty";
  $("progress-view").hidden = which !== "progress";
  $("viewer").hidden = which !== "viewer";
  if (which !== "viewer") { vOrig.pause(); vProc.pause(); }
}

async function poll() {
  try {
    const before = JSON.stringify(state.jobs.map((j) => [j.id, j.status, j.progress]));
    await refreshJobs();
    const busy = state.jobs.some((j) => j.status === "queued" || j.status === "running");
    const after = JSON.stringify(state.jobs.map((j) => [j.id, j.status, j.progress]));
    setTimeout(poll, busy || before !== after ? 800 : 4000);
  } catch {
    setTimeout(poll, 4000);
  }
}

/* ----------------------------------------------------------------- viewer */

const vOrig = $("v-orig");
const vProc = $("v-proc");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const stage = $("stage");

async function loadViewer(job) {
  state.job = job;
  state.selected = null;
  state.hidden = new Set();
  state.frame = -1;
  state.hasOriginal = !job.original_discarded && job.files.includes("original.mp4");
  showView("viewer");
  $("viewer-title").textContent = job.filename;
  const r = job.result;
  $("viewer-meta").textContent =
    `${r.width}×${r.height} · ${r.fps.toFixed(2)} fps · ${fmtTime(r.duration)} · engine ${job.params.engine || "rfdetr"}` +
    (state.hasOriginal ? "" : " · original deleted");

  state.tracks = await api(fileUrl(job, "tracks.json"));
  state.hasPose = job.files.includes("poses.json");
  state.poses = state.hasPose ? await api(fileUrl(job, "poses.json")) : null;
  stage.style.aspectRatio = `${r.width} / ${r.height}`;
  // keep tall (phone) videos from overflowing: width follows the max height
  stage.style.maxWidth = `calc((100vh - 230px) * ${r.width / r.height})`;

  vProc.src = fileUrl(job, "processed.mp4");
  vProc.dataset.file = "processed.mp4";
  if (state.hasOriginal) vOrig.src = fileUrl(job, "original.mp4");
  else vOrig.removeAttribute("src");
  vOrig.load(); vProc.load();

  for (const b of document.querySelectorAll("#mode-seg button")) b.disabled = !modeAvailable(b.dataset.mode);
  for (const b of document.querySelectorAll("#audio-seg button")) {
    b.disabled = (b.dataset.audio === "original" && !(state.hasOriginal && r.has_audio)) ||
                 (b.dataset.audio === "altered" && !r.has_audio);
  }
  if (!modeAvailable(state.mode)) state.mode = "blobs";
  if (!r.has_audio) state.audio = "mute";
  else if (state.audio === "original" && !state.hasOriginal) state.audio = "altered";

  $("dl-video").href = fileUrl(job, "processed.mp4");
  $("dl-tracks").href = fileUrl(job, "tracks.json");
  $("dl-audio").href = fileUrl(job, "altered.wav");
  $("dl-audio").setAttribute("aria-disabled", String(!job.files.includes("altered.wav")));
  for (const [id, name] of [["dl-pose", "pose.mp4"], ["dl-blobpose", "blobpose.mp4"], ["dl-poses", "poses.json"]]) {
    $(id).href = fileUrl(job, name);
    $(id).hidden = !job.files.includes(name);
  }
  $("discard-btn").hidden = !state.hasOriginal;

  renderObjectList();
  applyDeepLink();
  setMode(state.mode);
  setAudio(state.audio);
  resizeCanvas();
  drawPresence();
}

// #job=<id>&mode=overlay&t=4.2&sel=3 — shareable view state
function applyDeepLink() {
  const p = new URLSearchParams(location.hash.slice(1));
  const mode = p.get("mode");
  if (["original", "blobs", "pose", "blobpose", "overlay", "split"].includes(mode) && modeAvailable(mode)) state.mode = mode;
  const sel = parseInt(p.get("sel"), 10);
  if (state.tracks.objects.some((o) => o.id === sel)) selectObject(sel);
  const t = parseFloat(p.get("t"));
  if (isFinite(t)) {
    const go = () => seekTo(t);
    vProc.readyState >= 1 ? go() : vProc.addEventListener("loadedmetadata", go, { once: true });
  }
}

function master() { return state.audio === "original" && state.hasOriginal ? vOrig : vProc; }
function follower() { return master() === vOrig ? vProc : (state.hasOriginal ? vOrig : null); }
function duration() { return vProc.duration || (state.job ? state.job.result.duration : 0); }
function frameAt(t) {
  const n = state.tracks.frame_count;
  return Math.max(0, Math.min(n - 1, Math.floor(t * state.tracks.fps + 1e-3)));
}

function modeAvailable(mode) {
  return !(NEEDS_ORIGINAL.includes(mode) && !state.hasOriginal) && !(NEEDS_POSE.includes(mode) && !state.hasPose);
}

// swap the processed layer between blob / pose / blob+pose renders, keeping time and play state
function setProcFile(name) {
  if (vProc.dataset.file === name) return;
  const t = master().currentTime, playing = !master().paused;
  vProc.dataset.file = name;
  vProc.src = fileUrl(state.job, name);
  vProc.addEventListener("loadedmetadata", () => {
    vProc.currentTime = t;
    if (playing) playBoth();
  }, { once: true });
}

function setMode(mode) {
  state.mode = mode;
  for (const b of document.querySelectorAll("#mode-seg button")) b.classList.toggle("active", b.dataset.mode === mode);
  setProcFile(PROC_FILE[mode] || "processed.mp4");
  vOrig.style.visibility = (NEEDS_ORIGINAL.includes(mode) && state.hasOriginal) ? "visible" : "hidden";
  vProc.style.visibility = (mode === "original" || mode === "overlay") ? "hidden" : "visible";
  $("split-handle").hidden = mode !== "split";
  applySplit();
  $("stage-badge").textContent = { original: "ORIGINAL", blobs: "PROCESSED", pose: "POSE", blobpose: "BLOBS + POSE",
                                   overlay: "OVERLAY", split: "ORIGINAL ◀ ▶ PROCESSED" }[mode];
  draw(true);
}

function applySplit() {
  vProc.style.clipPath = state.mode === "split" ? `inset(0 0 0 ${state.split}%)` : "none";
  $("split-handle").style.left = `${state.split}%`;
}

function setAudio(which) {
  state.audio = which;
  for (const b of document.querySelectorAll("#audio-seg button")) b.classList.toggle("active", b.dataset.audio === which);
  vProc.muted = which !== "altered";
  vOrig.muted = which !== "original";
}

document.querySelectorAll("#mode-seg button").forEach((b) => (b.onclick = () => !b.disabled && setMode(b.dataset.mode)));
document.querySelectorAll("#audio-seg button").forEach((b) => (b.onclick = () => {
  if (b.disabled) return;
  const playing = !master().paused;
  const t = master().currentTime;
  setAudio(b.dataset.audio);
  // the audible video becomes the clock; bring it to the shared time
  master().currentTime = t;
  if (playing) playBoth();
}));

/* playback */

function playBoth() {
  const f = follower();
  if (f) f.currentTime = master().currentTime;
  master().playbackRate = f ? (f.playbackRate = parseFloat($("rate").value)) : parseFloat($("rate").value);
  master().play().catch(() => {});
  if (f) f.play().catch(() => {});
  $("play").textContent = "❚❚";
}
function pauseBoth() {
  vOrig.pause(); vProc.pause();
  const f = follower();
  if (f) f.currentTime = master().currentTime;
  $("play").textContent = "▶";
}
function togglePlay() { master().paused ? playBoth() : pauseBoth(); }
function seekTo(t) {
  t = Math.max(0, Math.min(duration(), t));
  master().currentTime = t;
  const f = follower();
  if (f) f.currentTime = t;
  draw(true);
}
function stepFrames(n) {
  pauseBoth();
  const fps = state.tracks.fps;
  const f = frameAt(master().currentTime) + n;
  seekTo((f + 0.5) / fps);
}

$("play").onclick = togglePlay;
$("step-back").onclick = () => stepFrames(-1);
$("step-fwd").onclick = () => stepFrames(1);
$("rate").onchange = () => { vOrig.playbackRate = vProc.playbackRate = parseFloat($("rate").value); };
$("seek").addEventListener("input", (e) => seekTo((e.target.value / 1000) * duration()));
vProc.addEventListener("ended", pauseBoth);
vOrig.addEventListener("ended", pauseBoth);
vProc.addEventListener("loadedmetadata", () => { resizeCanvas(); draw(true); });

document.addEventListener("keydown", (e) => {
  if ($("viewer").hidden || e.target.matches("input[type=text], select")) return;
  if (e.code === "Space") { e.preventDefault(); togglePlay(); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); stepFrames(e.shiftKey ? -10 : -1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); stepFrames(e.shiftKey ? 10 : 1); }
  else if (e.key === "Escape") selectObject(null);
  else if ("123456".includes(e.key)) {
    const b = document.querySelectorAll("#mode-seg button")[+e.key - 1];
    if (b && !b.disabled) setMode(b.dataset.mode);
  }
});

/* drawing */

function resizeCanvas() {
  const r = stage.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(r.width * dpr));
  canvas.height = Math.max(1, Math.round(r.height * dpr));
  draw(true);
}
new ResizeObserver(resizeCanvas).observe(stage);

function objectsAt(frame) {
  return (state.tracks && state.tracks.frames[frame]) || [];
}
function objMeta(id) {
  return state.tracks.objects.find((o) => o.id === id);
}

function tracePolys(polys) {
  const W = canvas.width, H = canvas.height;
  ctx.beginPath();
  for (const p of polys) {
    ctx.moveTo(p[0] * W, p[1] * H);
    for (let i = 2; i < p.length; i += 2) ctx.lineTo(p[i] * W, p[i + 1] * H);
    ctx.closePath();
  }
}

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

function draw(force = false) {
  if (!state.tracks) return;
  const t = master().currentTime;
  const frame = frameAt(t);
  if (!force && frame === state.frame) return;
  state.frame = frame;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const dpr = window.devicePixelRatio || 1;
  const rows = objectsAt(frame);
  const sel = state.selected;
  const mode = state.mode;
  const splitX = (state.split / 100) * canvas.width;

  for (const [id, ...polys] of rows) {
    const meta = objMeta(id);
    const hidden = state.hidden.has(id);
    tracePolys(polys);
    if (mode === "blobs" || mode === "split" || NEEDS_POSE.includes(mode)) {
      // the rendered video already shows blobs; paint over hidden / de-emphasised ones
      if (mode === "split") { ctx.save(); ctx.beginPath(); ctx.rect(splitX, 0, canvas.width, canvas.height); ctx.clip(); tracePolys(polys); }
      if (hidden) {
        ctx.fillStyle = STAGE_BG; ctx.strokeStyle = STAGE_BG; ctx.lineWidth = 3 * dpr; ctx.fill(); ctx.stroke();
      } else if (sel !== null && sel !== id) {
        ctx.fillStyle = "rgba(18, 20, 24, 0.72)"; ctx.fill();
      }
      if (mode === "split") ctx.restore();
    } else if (mode === "overlay" && !hidden) {
      const a = sel === null ? 0.55 : (sel === id ? 0.7 : 0.15);
      ctx.fillStyle = hexA(meta.color, a); ctx.fill();
      ctx.strokeStyle = hexA(meta.color, Math.min(1, a + 0.3)); ctx.lineWidth = 1.5 * dpr; ctx.stroke();
    }
  }

  if (mode === "overlay" && state.poses) drawSkeletons(frame);

  // selection outline + label, in every mode
  const selRow = rows.find((r) => r[0] === sel);
  if (selRow && !state.hidden.has(sel)) {
    const meta = objMeta(sel);
    tracePolys(selRow.slice(1));
    ctx.lineJoin = "round";
    ctx.strokeStyle = "rgba(0,0,0,.6)"; ctx.lineWidth = 5 * dpr; ctx.stroke();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 2.5 * dpr; ctx.stroke();
    let minX = 1, minY = 1;
    for (const p of selRow.slice(1)) for (let i = 0; i < p.length; i += 2) { minX = Math.min(minX, p[i]); minY = Math.min(minY, p[i + 1]); }
    const label = `${meta.label} ${meta.id}`;
    ctx.font = `${12 * dpr}px system-ui, sans-serif`;
    const tw = ctx.measureText(label).width + 12 * dpr;
    const x = minX * canvas.width, y = Math.max(0, minY * canvas.height - 22 * dpr);
    ctx.fillStyle = meta.color; ctx.fillRect(x, y, tw, 18 * dpr);
    ctx.fillStyle = "#111"; ctx.fillText(label, x + 6 * dpr, y + 13 * dpr);
  }

  updateLiveDots(rows);
}

// overlay mode: stick figures from poses.json over the original, in each blob's colour
function drawSkeletons(frame) {
  const W = canvas.width, H = canvas.height, dpr = window.devicePixelRatio || 1;
  const sk = state.poses.skeleton;
  for (const [id, ...k] of state.poses.frames[frame] || []) {
    if (state.hidden.has(id)) continue;
    const meta = objMeta(id);
    const a = state.selected === null || state.selected === id ? 1 : 0.25;
    const ok = (i) => k[i * 3 + 2] >= 0.3;
    ctx.strokeStyle = ctx.fillStyle = meta ? hexA(meta.color, a) : `rgba(255,255,255,${a})`;
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    for (const [p, q] of sk) {
      if (!ok(p) || !ok(q)) continue;
      ctx.moveTo(k[p * 3] * W, k[p * 3 + 1] * H);
      ctx.lineTo(k[q * 3] * W, k[q * 3 + 1] * H);
    }
    ctx.stroke();
    for (let i = 0; i < 17; i++) {
      if (!ok(i)) continue;
      ctx.beginPath(); ctx.arc(k[i * 3] * W, k[i * 3 + 1] * H, 2.5 * dpr, 0, 2 * Math.PI); ctx.fill();
    }
  }
}

function tick() {
  if (state.tracks && !$("viewer").hidden) {
    const m = master(), f = follower();
    if (f && !m.paused && Math.abs(f.currentTime - m.currentTime) > 0.08) f.currentTime = m.currentTime;
    const d = duration();
    if (d && document.activeElement !== $("seek")) $("seek").value = Math.round((m.currentTime / d) * 1000);
    $("time").textContent = `${fmtTime(m.currentTime)} / ${fmtTime(d)}`;
    draw();
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

/* split handle */
let dragging = false;
$("split-handle").addEventListener("pointerdown", (e) => { dragging = true; e.target.setPointerCapture(e.pointerId); });
$("split-handle").addEventListener("pointerup", () => (dragging = false));
$("split-handle").addEventListener("pointermove", (e) => {
  if (!dragging) return;
  const r = stage.getBoundingClientRect();
  state.split = Math.max(0, Math.min(100, ((e.clientX - r.left) / r.width) * 100));
  applySplit();
  draw(true);
});

/* click-to-select */
function pointInPoly(x, y, p) {
  let inside = false;
  for (let i = 0, j = p.length - 2; i < p.length; j = i, i += 2) {
    const xi = p[i], yi = p[i + 1], xj = p[j], yj = p[j + 1];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function polyArea(p) {
  let a = 0;
  for (let i = 0, j = p.length - 2; i < p.length; j = i, i += 2) a += (p[j] + p[i]) * (p[j + 1] - p[i + 1]);
  return Math.abs(a / 2);
}
canvas.addEventListener("click", (e) => {
  const r = canvas.getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
  let best = null, bestArea = Infinity;
  for (const [id, ...polys] of objectsAt(state.frame)) {
    if (state.hidden.has(id)) continue;
    for (const p of polys) {
      if (pointInPoly(x, y, p)) {
        const a = polyArea(p);
        if (a < bestArea) { best = id; bestArea = a; }
      }
    }
  }
  selectObject(best === state.selected ? null : best);
});

/* object list */

function renderObjectList() {
  const ul = $("obj-list");
  ul.innerHTML = "";
  const objs = state.tracks.objects;
  const n = Math.max(1, state.tracks.frame_count);
  $("obj-count").textContent = `(${objs.length})`;
  for (const o of objs) {
    const li = document.createElement("li");
    li.className = "obj";
    li.dataset.id = o.id;
    const secs = (o.frames_visible / state.tracks.fps).toFixed(1);
    li.innerHTML = `
      <span class="sw" style="background:${o.color}"></span>
      <span class="name">${o.label} ${o.id}<span class="live" title="visible in this frame"></span>
        <span class="muted small"> · ${secs}s</span></span>
      <button class="eye" title="Hide / show this blob">👁</button>
      <span class="span"><i style="left:${(o.first_frame / n) * 100}%;width:${Math.max(0.6, ((o.last_frame - o.first_frame + 1) / n) * 100)}%;background:${o.color}"></i></span>`;
    li.onclick = () => {
      const id = o.id;
      selectObject(state.selected === id ? null : id);
      if (state.selected === id && !objectsAt(state.frame).some((r) => r[0] === id)) {
        pauseBoth();
        seekTo((o.first_frame + 0.5) / state.tracks.fps);
      }
    };
    li.querySelector(".eye").onclick = (e) => {
      e.stopPropagation();
      state.hidden.has(o.id) ? state.hidden.delete(o.id) : state.hidden.add(o.id);
      li.classList.toggle("hidden-obj", state.hidden.has(o.id));
      draw(true);
    };
    ul.append(li);
  }
}

function selectObject(id) {
  state.selected = id;
  for (const li of document.querySelectorAll(".obj")) li.classList.toggle("selected", +li.dataset.id === id);
  const el = document.querySelector(`.obj[data-id="${id}"]`);
  if (el) el.scrollIntoView({ block: "nearest" });
  drawPresence();
  draw(true);
}

let liveKey = "";
function updateLiveDots(rows) {
  const key = rows.map((r) => r[0]).join(",");
  if (key === liveKey) return;
  liveKey = key;
  const ids = new Set(rows.map((r) => r[0]));
  for (const li of document.querySelectorAll(".obj")) li.classList.toggle("visible-now", ids.has(+li.dataset.id));
}

$("show-all").onclick = () => {
  state.hidden.clear();
  document.querySelectorAll(".obj").forEach((li) => li.classList.remove("hidden-obj"));
  selectObject(null);
};

/* presence strip behind the seek bar: object count per frame, selected object highlighted */
function drawPresence() {
  const c = $("presence");
  if (!state.tracks) return;
  const dpr = window.devicePixelRatio || 1;
  const r = c.getBoundingClientRect();
  c.width = Math.max(1, Math.round(r.width * dpr));
  c.height = Math.max(1, Math.round(r.height * dpr));
  const g = c.getContext("2d");
  const frames = state.tracks.frames;
  const n = frames.length;
  if (!n) return;
  const maxCount = Math.max(1, ...frames.map((f) => f.length));
  const cols = c.width;
  for (let x = 0; x < cols; x++) {
    const f = Math.min(n - 1, Math.floor((x / cols) * n));
    const row = frames[f];
    const h = (row.length / maxCount) * c.height * 0.8;
    g.fillStyle = "rgba(124,156,255,.28)";
    g.fillRect(x, c.height - h, 1, h);
    if (state.selected !== null && row.some((rr) => rr[0] === state.selected)) {
      g.fillStyle = objMeta(state.selected).color;
      g.fillRect(x, 0, 1, 4 * dpr);
    }
  }
}
new ResizeObserver(drawPresence).observe($("presence"));

/* privacy: delete originals */
$("discard-btn").onclick = async () => {
  if (!confirm("Permanently delete the original video and audio for this job? Only the blob and pose videos, altered audio and tracks remain.")) return;
  try {
    await api(`/api/jobs/${state.job.id}/discard-original`, { method: "POST" });
    await refreshJobs();
  } catch (err) { alert(err.message); }
};

/* ------------------------------------------------------------------- boot */

(async function boot() {
  try { await loadConfig(); } catch (err) { console.error(err); }
  await refreshJobs().catch(() => {});
  const m = location.hash.match(/job=([\w-]+)/);
  if (m && state.jobs.some((j) => j.id === m[1])) openJob(m[1]);
  else showView("empty");
  poll();
})();
