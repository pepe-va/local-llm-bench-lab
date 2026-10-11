// llb lab, shared by every tab: DOM and API helpers, status polling, machine cards and the
// running job. Text from the Hub, a model or a log is only set through textContent.
"use strict";

const HEADERS = { "Content-Type": "application/json", "X-LLB-Lab": "1" };
const $ = (id) => document.getElementById(id);

function el(tag, props, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "value") node.value = v;
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c !== null && c !== undefined && c !== false) node.append(c instanceof Node ? c : String(c));
  }
  return node;
}

class ApiError extends Error {
  constructor(message, line) {
    super(message);
    this.line = line;
  }
}

async function api(method, url, body) {
  const r = await fetch(url, { method, headers: HEADERS, body: body ? JSON.stringify(body) : undefined });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const d = data.detail;
    if (d && typeof d === "object" && !Array.isArray(d)) throw new ApiError(d.message, d.line);
    throw new ApiError(typeof d === "string" ? d : d ? d.map((e) => `${e.loc.slice(-1)}: ${e.msg}`).join("\n") : r.statusText);
  }
  return data;
}

function say(text, ok) {
  const box = el("div", { class: ok ? "ok" : "", text });
  $("msg").append(box);
  setTimeout(() => box.remove(), ok ? 4000 : 10000);
}

async function act(fn, okText) {
  try {
    const out = await fn();
    if (okText) say(okText, true);
    return out;
  } catch (e) {
    say(e.message);
    return null;
  }
}

const gb = (n) => (n == null ? "-" : n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB` : `${(n / 1e6).toFixed(0)} MB`);
const num = (v, d = 0) => (v == null ? "-" : Number(v).toFixed(d));
const intOrNull = (v) => (v === "" || v == null ? null : parseInt(v, 10));
const FIT = { ok: ["ok", "fits"], tight: ["warn", "tight"], no: ["ko", "too big"] };
const param = (name) => new URLSearchParams(location.search).get(name);

// -- status -----------------------------------------------------------------------

const Lab = {
  listeners: [],
  status: null,
  onStatus(fn) {
    this.listeners.push(fn);
    if (this.status) fn(this.status);
  },
};

function renderMachine(m) {
  const card = (value, label, cls) => el("div", { class: `card ${cls || ""}` }, el("b", { text: value }), el("span", { text: label }));
  $("machine").replaceChildren(
    card(m.temp_c != null ? `${num(m.temp_c, 1)} °C` : "-", "CPU temperature", m.temp_c >= 80 ? "hl" : ""),
    card(`${(m.ram_used_mb / 1024).toFixed(1)} / ${(m.ram_total_mb / 1024).toFixed(1)} GB`, "RAM used"),
    card(m.cpu_mhz != null ? `${num(m.cpu_mhz)} MHz` : "-", "CPU clock"),
    card(m.governor || "-", "governor"),
    card(m.throttled == null ? "-" : m.throttled ? "yes" : "no", "throttled (Pi)"),
  );
}

let jobWasRunning = false;
let logOpen = true;

function renderActivity(s) {
  const box = $("activity");
  box.replaceChildren();
  const job = s.job;
  if (job) {
    const cls = { running: "live warn", ok: "ok", failed: "ko", interrupted: "warn", finished: "dim" }[job.state];
    const log = el("details", { open: logOpen, ontoggle: (e) => (logOpen = e.target.open) }, el("summary", { text: "output" }), el("pre", { text: job.tail || "(no output yet)" }));
    box.append(
      el("div", { class: "panel" },
        el("div", { class: "row" },
          el("span", { class: cls, text: ` ${job.title}: ${job.state}${job.state === "failed" && job.returncode != null ? ` (exit ${job.returncode})` : ""}` }),
          el("span", { class: "muted", text: `started ${job.started_at.replace("T", " ").slice(0, 19)}` }),
          job.running ? el("button", { class: "btn small red", onclick: cancelJob, text: "cancel" }) : null,
          job.results ? el("a", { href: job.results, text: "→ results" }) : null,
        ),
        log,
      ),
    );
    const pre = log.querySelector("pre");
    pre.scrollTop = pre.scrollHeight;
  }
  const running = s.downloads.filter((d) => d.state === "running");
  if (running.length && !location.pathname.endsWith("/models")) {
    box.append(el("div", { class: "panel" }, el("span", { class: "live", text: ` downloading ${running.map((d) => d.file).join(", ")}` }), " ", el("a", { href: "/lab/models", text: "→ models" })));
  }
  if (s.chat && s.chat.state !== "stopped" && !location.pathname.endsWith("/chat")) {
    box.append(el("div", { class: "panel" }, el("span", { class: s.chat.state === "ready" ? "ok" : "warn", text: `chat: ${s.chat.name} · ${s.chat.quant} (${s.chat.state})` }), " ", el("a", { href: "/lab/chat", text: "→ chat" })));
  }
  if (job) jobWasRunning = job.running;
}

async function cancelJob() {
  if (confirm("Stop the job? A run keeps what it already measured (continue it later with --resume).")) {
    await act(() => api("POST", "/lab/api/job/cancel"));
  }
}

let timer = null;

async function poll() {
  clearTimeout(timer);
  try {
    const s = await api("GET", "/lab/api/status");
    const finished = jobWasRunning && s.job && !s.job.running;
    Lab.status = s;
    renderMachine(s.machine);
    renderActivity(s);
    for (const fn of Lab.listeners) fn(s, { jobFinished: finished });
    const busy = (s.chat && s.chat.state === "loading") || (s.job && s.job.running) || s.downloads.some((d) => d.state === "running");
    timer = setTimeout(poll, busy ? 1500 : 4000);
  } catch (e) {
    timer = setTimeout(poll, 5000);
  }
}

document.addEventListener("DOMContentLoaded", poll);
