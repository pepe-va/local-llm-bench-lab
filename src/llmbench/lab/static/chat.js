// llb lab · chat: load a model in llama-server and talk to it, with metrics per answer.
"use strict";

let conversation = [];
let controller = null;
let chatKey = null;

async function loadModelOptions() {
  const data = await act(() => api("GET", "/lab/api/models"));
  if (!data) return;
  const usable = data.models.filter((m) => !(m.info || {}).embedding);
  $("chat-model").replaceChildren(...usable.map((m) => el("option", { value: m.key, text: `${m.name} · ${m.quant} · ${gb(m.size)}` })));
  if (!usable.length) $("chat-model").append(el("option", { value: "", text: "no models yet: download one in the models tab" }));
  const wanted = param("model");
  if (wanted && usable.some((m) => m.key === wanted)) {
    $("chat-model").value = wanted;
    if (param("load")) {
      history.replaceState(null, "", "/lab/chat");
      loadChat();
    }
  }
}

async function loadBuilds() {
  const list = await act(() => api("GET", "/lab/api/builds"));
  if (!list) return;
  const was = $("chat-build").value;
  $("chat-build").replaceChildren(...list.map((b) => el("option", { value: b.id, text: b.label })));
  if (!list.length) $("chat-build").append(el("option", { value: "", text: "none built yet" }));
  if (list.some((b) => b.id === was)) $("chat-build").value = was;
  $("build-hint").hidden = list.length > 0;
}

async function loadChat() {
  const model = $("chat-model").value;
  if (!model) return say("Download a model first (models tab).");
  const body = {
    model,
    build: $("chat-build").value || null,
    ctx_size: intOrNull($("chat-ctx").value) || 4096,
    threads: intOrNull($("chat-threads").value),
    reasoning: $("chat-reasoning").value,
  };
  if (await act(() => api("POST", "/lab/api/chat/start", body))) poll();
}

function renderChat(chat) {
  const box = $("chat-state");
  box.replaceChildren();
  const ready = chat && chat.state === "ready";
  $("send").disabled = !ready || controller !== null;
  if (!chat || chat.state === "stopped") {
    box.append(el("span", { class: "muted", text: "no model loaded" }));
    chatKey = null;
    return;
  }
  if (ready && chat.model !== chatKey) {
    if (chatKey !== null || conversation.length) note(`now chatting with ${chat.name} · ${chat.quant}`);
    chatKey = chat.model;
  }
  const s = chat.settings;
  const line = `${chat.name} · ${chat.quant} · ctx ${s.ctx_size} · threads ${s.threads || "auto"} · reasoning ${s.reasoning}`;
  if (chat.state === "loading") box.append(el("span", { class: "live warn", text: ` loading ${line}…` }));
  if (ready) box.append(el("span", { class: "ok", text: `● ${line}` }), el("span", { class: "muted", text: ` · loaded in ${chat.load_s}s · ${num(chat.rss_mb)} MB` }));
  if (chat.state === "error") box.append(el("span", { class: "ko", text: `could not load ${line}` }), el("pre", { text: chat.error }));
}

function note(text) {
  $("messages").append(el("div", { class: "m note", text }));
}

function bubble(role, text) {
  const node = el("div", { class: `m ${role}` }, el("div", { class: "who", text: role === "user" ? "you" : "model" }), el("div", { class: "text", text: text || "" }));
  $("messages").append(node);
  $("messages").scrollTop = $("messages").scrollHeight;
  return node;
}

function stats(m) {
  return [
    m.ttft_ms != null ? `first token ${m.ttft_ms} ms` : null,
    m.gen_tps != null ? `${num(m.gen_tps, 1)} tok/s` : null,
    m.predicted_n != null ? `${m.predicted_n} tokens` : null,
    m.prompt_n != null ? `prompt ${m.prompt_n} tok @ ${num(m.prompt_tps, 0)} tok/s` : null,
    `${num(m.total_s, 1)} s`,
    m.rss_mb != null ? `${num(m.rss_mb)} MB` : null,
    m.temp_c != null ? `${num(m.temp_c, 1)} °C` : null,
  ].filter(Boolean).join(" · ");
}

async function send(ev) {
  ev.preventDefault();
  const text = $("prompt").value.trim();
  if (!text || controller) return;
  $("prompt").value = "";
  if (!conversation.length) $("messages").replaceChildren();
  conversation.push({ role: "user", content: text });
  bubble("user", text);
  const node = bubble("assistant", "");
  const out = node.querySelector(".text");
  let thinking = null;
  let answer = "";
  const system = $("system").value.trim();
  const messages = system ? [{ role: "system", content: system }, ...conversation] : conversation;
  controller = new AbortController();
  $("send").disabled = true;
  $("abort").disabled = false;
  try {
    const r = await fetch("/lab/api/chat", {
      method: "POST",
      headers: HEADERS,
      signal: controller.signal,
      body: JSON.stringify({ messages, temperature: parseFloat($("temp").value) || 0, max_tokens: intOrNull($("max-tokens").value) || 1024 }),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const line = buffer.slice(0, cut).trim();
        buffer = buffer.slice(cut + 2);
        if (!line.startsWith("data: ")) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.reasoning) {
          if (!thinking) {
            thinking = el("details", { open: true }, el("summary", { text: "thinking" }), el("span"));
            node.insertBefore(thinking, out);
          }
          thinking.lastChild.textContent += ev.reasoning;
        }
        if (ev.content) {
          if (thinking && thinking.open && !answer) thinking.open = false;
          answer += ev.content;
          out.textContent = answer;
        }
        if (ev.error) node.append(el("div", { class: "err", text: ev.error }));
        if (ev.metrics) node.append(el("div", { class: "stats", text: stats(ev.metrics) }));
        $("messages").scrollTop = $("messages").scrollHeight;
      }
    }
  } catch (e) {
    const aborted = e.name === "AbortError";
    node.append(el("div", { class: aborted ? "muted" : "err", text: aborted ? "(stopped)" : e.message }));
  } finally {
    controller = null;
    $("abort").disabled = true;
    $("send").disabled = false;
    if (answer) conversation.push({ role: "assistant", content: answer });
    else conversation.pop();
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $("chat-load").addEventListener("click", loadChat);
  $("chat-unload").addEventListener("click", async () => {
    await act(() => api("POST", "/lab/api/chat/stop"));
    poll();
  });
  $("chat-form").addEventListener("submit", send);
  $("prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("chat-form").requestSubmit();
    }
  });
  $("abort").addEventListener("click", () => controller && controller.abort());
  $("clear").addEventListener("click", () => {
    conversation = [];
    $("messages").replaceChildren();
  });
  Lab.onStatus((s, ev) => {
    renderChat(s.chat);
    if (ev && ev.jobFinished) loadBuilds();
  });
  loadModelOptions();
  loadBuilds();
});
