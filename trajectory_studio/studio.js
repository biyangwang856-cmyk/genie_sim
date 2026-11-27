"use strict";

const $ = (selector) => document.querySelector(selector);
const elements = {
  form: $("#request-form"),
  input: $("#instruction"),
  submit: $("#submit-button"),
  formError: $("#form-error"),
  connection: $("#connection"),
  connectionText: $("#connection-text"),
  badge: $("#job-badge"),
  message: $("#job-message"),
  detail: $("#job-detail"),
  episodes: $("#job-episodes"),
  history: $("#history"),
  historyList: $("#history-list"),
  player: $("#player"),
  empty: $("#video-empty"),
  videoLoading: $("#video-loading"),
  videoTitle: $("#video-title"),
  videoSubtitle: $("#video-subtitle"),
  videoError: $("#video-error"),
  viewerTag: $("#viewer-tag"),
  sampleGrid: $("#sample-grid"),
  sampleCount: $("#sample-count"),
  libraryMessage: $("#library-message"),
};

const HISTORY_KEY = "nexus-studio-jobs-v1";
let history = [];
let samples = [];
let activeEpisode = null;
let activeCamera = "head";
let activeJob = null;
let pollTimer = null;
let connected = false;
let cameraRequest = 0;

try {
  const saved = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  history = Array.isArray(saved) ? saved.filter((item) =>
    typeof item.id === "string" && /^[a-f0-9]{32}$/.test(item.id)
    && typeof item.instruction === "string").slice(0, 12) : [];
} catch {
  history = [];
}

function saveHistory() {
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(history)); } catch { /* Optional browser storage. */ }
}

function renderHistory() {
  elements.history.hidden = history.length === 0;
  elements.historyList.replaceChildren();
  for (const item of history) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "history-item";
    const title = document.createElement("span");
    title.textContent = item.instruction;
    const date = document.createElement("small");
    date.textContent = new Date(item.created).toLocaleString("zh-CN", {
      month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit",
    });
    button.append(title, date);
    button.addEventListener("click", () => showJob(item.id));
    elements.historyList.append(button);
  }
}

async function getJSON(url, options) {
  const response = await fetch(url, { cache: "no-store", ...options });
  let data;
  try { data = await response.json(); } catch { throw new Error("服务器返回了无法读取的数据"); }
  if (!response.ok) {
    const missing = ["missing_skills", "missing_assets", "missing_robot_capabilities"]
      .flatMap((key) => Array.isArray(data[key]) ? data[key] : []);
    throw new Error([data.message || data.error || data.reason || "请求未能完成", missing.join("、")]
      .filter(Boolean).join(" · "));
  }
  return data;
}

function renderConnection(isReady) {
  connected = isReady;
  elements.connection.classList.toggle("online", isReady);
  elements.connection.classList.toggle("offline", !isReady);
  elements.connectionText.textContent = isReady ? "采集服务已连接" : "采集服务未连接";
  elements.submit.disabled = !isReady;
}

async function checkConnection() {
  try {
    const health = await getJSON("/studio/health");
    renderConnection(health.worker_ready === true);
  } catch {
    renderConnection(false);
  }
}

function setStatus(label, message, tone = "neutral", details = "") {
  elements.badge.textContent = label;
  elements.badge.className = "status-pill " + tone;
  elements.message.textContent = message;
  elements.detail.hidden = !details;
  elements.detail.textContent = details;
}

function drawCameraButtons() {
  for (const button of document.querySelectorAll(".camera")) {
    const selected = button.dataset.camera === activeCamera;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
    button.disabled = !activeEpisode || !activeEpisode.videos[button.dataset.camera];
  }
}

async function setCamera(camera, keepPosition = false) {
  if (!activeEpisode || !activeEpisode.videos[camera]) return;
  const request = ++cameraRequest;
  const url = activeEpisode.videos[camera];
  const time = keepPosition && Number.isFinite(elements.player.currentTime) ? elements.player.currentTime : 0;
  const resume = keepPosition && !elements.player.paused;
  activeCamera = camera;
  elements.videoError.hidden = true;
  elements.videoLoading.hidden = false;
  elements.player.pause();
  drawCameraButtons();
  try {
    let state;
    do {
      const result = await getJSON(url.replace("/studio/media/", "/studio/prepare/"));
      state = result.state;
      if (state === "failed") throw new Error("视频转码失败");
      if (state === "preparing") await new Promise((resolve) => setTimeout(resolve, 2000));
      if (request !== cameraRequest) return;
    } while (state === "preparing");
    if (state !== "ready") throw new Error("视频暂时不可用");
  } catch {
    if (request === cameraRequest) {
      elements.videoLoading.hidden = true;
      elements.videoError.hidden = false;
    }
    return;
  }
  if (request !== cameraRequest) return;
  elements.videoLoading.hidden = true;
  elements.player.src = url;
  elements.player.poster = camera === "head" ? (activeEpisode.poster || "") : "";
  elements.player.load();
  if (time > 0 || resume) {
    elements.player.addEventListener("loadedmetadata", () => {
      if (time > 0 && time < elements.player.duration) elements.player.currentTime = time;
      if (resume) elements.player.play().catch(() => {});
    }, { once: true });
  }
  drawCameraButtons();
}

function selectEpisode(episode, type, scroll = false) {
  activeEpisode = episode;
  elements.empty.hidden = true;
  elements.viewerTag.textContent = type === "sample" ? "样例轨迹" : "本次生成";
  elements.videoTitle.textContent = episode.title;
  elements.videoSubtitle.textContent = episode.scene_label + "  ·  " + episode.frames.toLocaleString("zh-CN") + " 帧";
  for (const card of elements.sampleGrid.querySelectorAll(".sample-card")) {
    card.classList.toggle("selected", type === "sample" && card.dataset.id === episode.id);
  }
  setCamera("head");
  if (scroll) $("#viewer-heading").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderSamples() {
  elements.sampleGrid.replaceChildren();
  elements.sampleCount.textContent = samples.length + " 条样例";
  elements.libraryMessage.hidden = samples.length > 0;
  elements.libraryMessage.textContent = "暂时没有可用的轨迹样例。";
  for (const episode of samples) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "sample-card";
    card.dataset.id = episode.id;
    card.setAttribute("aria-label", "查看" + episode.title + "，" + episode.scene_label);
    const image = document.createElement("div");
    image.className = "sample-image";
    const video = document.createElement("img");
    video.alt = "";
    video.loading = "lazy";
    if (episode.poster) video.src = episode.poster;
    const play = document.createElement("span");
    play.className = "sample-play";
    play.textContent = "▶";
    image.append(video, play);
    const meta = document.createElement("div");
    meta.className = "sample-meta";
    const copy = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = episode.title;
    const scene = document.createElement("small");
    scene.textContent = episode.scene_label + "  ·  G2CRSB90D";
    copy.append(title, scene);
    const tag = document.createElement("span");
    tag.className = "sample-chip";
    tag.textContent = "已生成";
    meta.append(copy, tag);
    card.append(image, meta);
    card.addEventListener("click", () => selectEpisode(episode, "sample", true));
    elements.sampleGrid.append(card);
  }
  if (samples.length && !activeEpisode) selectEpisode(samples[0], "sample");
}

function describeFailure(result) {
  if (!result || typeof result !== "object") return "";
  const parts = [];
  for (const [key, label] of [
    ["missing_skills", "缺少技能"], ["missing_assets", "缺少资产"],
    ["missing_robot_capabilities", "缺少机器人能力"], ["missing_verifiers", "缺少验证器"],
  ]) {
    if (Array.isArray(result[key]) && result[key].length) parts.push(label + " " + result[key].join("、"));
  }
  const reason = result.message || result.reason || result.error;
  if (typeof reason === "string") parts.unshift(reason);
  return parts.join("；");
}

function renderJob(job, shouldSelect) {
  const native = job.native || job.result || {};
  const nativeStatus = native.status || "";
  const partial = nativeStatus === "partial";
  const videos = Array.isArray(job.episodes) ? job.episodes : [];
  elements.episodes.replaceChildren();
  for (const episode of videos) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "episode-button";
    button.textContent = "查看视频  ·  " + episode.title + "  ·  " + episode.scene_label;
    button.addEventListener("click", () => selectEpisode(episode, "job", true));
    elements.episodes.append(button);
  }

  if (job.state === "queued") {
    setStatus("排队中", "任务已提交，正在等待采集服务处理。", "working");
  } else if (job.state === "running") {
    const phase = native.phase || nativeStatus;
    let message = "正在解析需求、准备场景和执行仿真，请稍候。";
    if (/waiting_for/.test(phase)) message = "任务正在等待服务器资源或可用的采集槽位。";
    if (/collecting|trial/.test(phase)) message = "机器人正在仿真环境中执行任务，系统会自动检查结果。";
    if (/finalizing/.test(phase)) message = "正在整理轨迹、视频和训练数据。";
    setStatus("生成中", message, "working", "作业 " + job.id.slice(0, 12));
  } else if (job.state === "finished" && videos.length) {
    setStatus(partial ? "部分完成" : "已完成",
      partial ? "已有通过检查的轨迹，其他轨迹未完成。点击下方视频查看。"
        : "轨迹已完成并通过读取检查，点击下方视频查看。",
      "working", "共 " + videos.length + " 条可播放轨迹");
    if (shouldSelect) selectEpisode(videos[0], "job");
  } else if (job.state === "finished") {
    setStatus("未生成视频", "任务已结束，目前没有通过检查的可播放轨迹。", "failed",
      describeFailure(native) || (nativeStatus ? "任务状态 " + nativeStatus : ""));
  } else if (job.state === "failed" || job.state === "interrupted") {
    const result = job.result || native;
    const label = {
      ASSET_NOT_FOUND: "缺少匹配资产",
      ASSET_GEOMETRY_INCOMPATIBLE: "资产几何不匹配",
      SKILL_NOT_REGISTERED: "缺少可用技能",
      TASK_GOAL_UNSUPPORTED: "任务未支持",
    }[result.code];
    setStatus(job.state === "interrupted" ? "已中断" : (label || "未完成"),
      "任务没有完成，请查看返回的信息。", "failed", describeFailure(job.result) || describeFailure(native));
  } else {
    setStatus("状态未知", "正在获取任务状态。", "neutral");
  }
}

async function showJob(id, selectOnCompletion = true) {
  activeJob = id;
  clearTimeout(pollTimer);
  try {
    const job = await getJSON("/studio/jobs/" + encodeURIComponent(id));
    if (activeJob !== id) return;
    const alreadySelected = activeEpisode && activeEpisode.id.startsWith(id + "/");
    renderJob(job, selectOnCompletion && !alreadySelected);
    if (job.state === "queued" || job.state === "running") {
      pollTimer = setTimeout(() => showJob(id, true), 4000);
    }
  } catch (error) {
    if (activeJob !== id) return;
    setStatus("连接中断", "暂时无法获取任务状态，正在尝试重新连接。", "failed", error.message);
    pollTimer = setTimeout(() => showJob(id, true), 8000);
  }
}

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const instruction = elements.input.value.trim();
  if (!instruction || !connected) return;
  elements.formError.hidden = true;
  elements.submit.disabled = true;
  try {
    const job = await getJSON("/v1/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Idempotency-Key": crypto.randomUUID() },
      body: JSON.stringify({ instruction, simulator: "isaac_physx", robot_id: "g2crsb90d", execute: true }),
    });
    history = [{ id: job.id, instruction, created: Date.now() }, ...history.filter((item) => item.id !== job.id)]
      .slice(0, 12);
    saveHistory();
    renderHistory();
    await showJob(job.id);
  } catch (error) {
    elements.formError.textContent = error.message;
    elements.formError.hidden = false;
  } finally {
    elements.submit.disabled = !connected;
  }
});

for (const button of document.querySelectorAll(".prompt-chip")) {
  button.addEventListener("click", () => {
    elements.input.value = button.dataset.prompt;
    elements.input.focus();
  });
}
for (const button of document.querySelectorAll(".camera")) {
  button.addEventListener("click", () => setCamera(button.dataset.camera, true));
}
elements.player.addEventListener("error", () => {
  if (activeEpisode) elements.videoError.hidden = false;
});

drawCameraButtons();
renderHistory();
checkConnection();
setInterval(checkConnection, 15000);
getJSON("/studio/samples").then((data) => {
  samples = Array.isArray(data) ? data : [];
  renderSamples();
}).catch(() => {
  elements.sampleCount.textContent = "暂不可用";
  elements.libraryMessage.hidden = false;
  elements.libraryMessage.textContent = "样例暂时无法加载，请检查服务器连接。";
});
if (history.length) showJob(history[0].id);
