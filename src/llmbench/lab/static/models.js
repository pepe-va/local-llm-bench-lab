// llb lab · models: search the Hub or pull a reference, download, inspect, keep, delete, prune.
"use strict";

async function loadModels() {
  const data = await act(() => api("GET", "/lab/api/models"));
  if (!data) return;
  const repos = [...new Set(data.models.filter((m) => m.where === "cache").map((m) => m.repo))].sort();
  $("prune-repos").replaceChildren(...repos.map((r) => el("option", { value: r })));
  const body = $("t-models").tBodies[0];
  body.replaceChildren();
  if (!data.models.length) {
    body.append(el("tr", {}, el("td", { colspan: 7, class: "muted", text: "No models yet: search above and download one." })));
  }
  for (const m of data.models) {
    const info = m.info || {};
    const what = [info.architecture, info.params_b ? `${info.params_b}B` : null, info.context_length ? `ctx ${info.context_length}` : null].filter(Boolean).join(" · ");
    const tags = [];
    if (info.reasoning) tags.push(el("span", { class: "tag v", text: "reasoning" }));
    if (info.embedding) tags.push(el("span", { class: "tag", text: "embeddings" }));
    else if (info.chat === false) tags.push(el("span", { class: "tag", text: "no chat template" }));
    if (m.in_chat) tags.push(el("span", { class: "tag v", text: "in chat" }));
    const [cls, label] = FIT[m.fit];
    body.append(
      el("tr", {},
        el("td", { class: "wrapcell" }, el("b", { text: m.name }), el("div", { class: "muted", text: m.file + (m.shards > 1 ? ` (+${m.shards - 1} shards)` : "") })),
        el("td", { text: m.quant }),
        el("td", { class: "num", text: gb(m.size) }),
        el("td", { class: "wrapcell" }, el("div", { text: what || "-" }), tags),
        el("td", { class: cls, text: label }),
        el("td", { text: m.where === "favorite" ? "★ models/" : m.where === "local" ? "models/" : "cache" }),
        el("td", {},
          el("button", { class: "btn small", title: "GGUF metadata, like llb models inspect", onclick: () => inspect(m), text: "inspect" }), " ",
          info.embedding ? null : el("a", { class: "btn small", href: `/lab/chat?model=${m.key}&load=1`, text: "chat" }), " ",
          el("a", { class: "btn small", href: `/lab/test?model=${m.key}`, text: "test" }), " ",
          m.where === "cache" ? el("button", { class: "btn small", title: "move into models/: never deleted automatically", onclick: () => keep(m), text: "★ keep" }) : null, " ",
          el("button", { class: "btn small red", onclick: () => remove(m), text: "delete" }),
        ),
      ),
    );
  }
}

async function inspect(m) {
  const box = $("inspect");
  box.replaceChildren(el("p", { class: "muted", text: `reading ${m.file}…` }));
  const data = await act(() => api("GET", `/lab/api/models/${m.key}/inspect`));
  box.replaceChildren();
  if (!data) return;
  const fmt = (v) => (typeof v === "number" && Number.isInteger(v) ? v.toLocaleString("en-US") : String(v));
  const kv = (k, v) => [el("dt", { text: k }), el("dd", { text: v })];
  const summary = Object.entries(data.summary).flatMap(([k, v]) => kv(k, fmt(v)));
  const types = data.tensor_types.map((t) => `${t.type} ${t.share == null ? "-" : `${Math.round(100 * t.share)}%`}`).join(", ");
  box.append(
    el("div", { class: "panel" },
      el("div", { class: "row" },
        el("span", { class: "ok", style: "word-break:break-all", text: `✓ ${data.path}` }),
        el("span", { class: "muted", text: `(${data.where}, ${gb(data.size)}${data.shards > 1 ? `, ${data.shards} shards` : ""})` }),
        el("button", { class: "btn small", style: "margin-left:auto", onclick: () => box.replaceChildren(), text: "close" }),
      ),
      data.source !== data.path ? el("div", { class: "muted", style: "margin-top:4px", text: data.source }) : null,
      el("div", { style: "height:10px" }),
      el("dl", { class: "kv" }, summary, kv("tensors", fmt(data.tensors)), kv("tensor types", types || "-")),
      el("details", {},
        el("summary", { text: `raw metadata (${data.metadata.length} keys)` }),
        el("div", { class: "wrap" }, el("table", {},
          el("thead", {}, el("tr", {}, el("th", { text: "key" }), el("th", { text: "type" }), el("th", { text: "value" }))),
          el("tbody", {}, data.metadata.map((e) => el("tr", {},
            el("td", { text: e.key }),
            el("td", { class: "muted", text: e.type + (e.length != null && e.length > 200 ? ` · ${e.length.toLocaleString("en-US")} chars` : "") }),
            el("td", { class: "v" }, e.type === "string" && (e.value.includes("\n") || e.value.length > 200) ? el("pre", { text: e.value }) : e.value),
          ))),
        )),
      ),
    ),
  );
  box.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function keep(m) {
  if (await act(() => api("POST", `/lab/api/models/${m.key}/favorite`), `${m.name} ${m.quant} moved into models/`)) loadModels();
}

async function remove(m) {
  let text = `Delete ${m.name} ${m.quant} (${gb(m.size)})?`;
  if (m.also.length) text += `\n\nThe same Hugging Face revision also holds these files, deleted with it:\n${m.also.join("\n")}`;
  if (!confirm(text)) return;
  if (await act(() => api("DELETE", `/lab/api/models/${m.key}`), "deleted")) loadModels();
}

async function search(ev) {
  ev.preventDefault();
  const q = $("q").value.trim();
  if (q.length < 2) return;
  const out = $("search-results");
  out.replaceChildren(el("p", { class: "muted", text: "searching…" }));
  $("repo").replaceChildren();
  const hits = await act(() => api("GET", `/lab/api/search?q=${encodeURIComponent(q)}`));
  out.replaceChildren();
  if (!hits) return;
  if (hits.length === 1 && hits[0].direct) return openRepo(hits[0].repo);
  if (!hits.length) {
    out.append(el("p", { class: "muted", text: `No GGUF repos match "${q}". Try fewer words, or the model's name plus GGUF.` }));
    return;
  }
  out.append(
    el("div", { class: "wrap" }, el("table", {},
      el("thead", {}, el("tr", {}, el("th", { text: "repo" }), el("th", { text: "task" }), el("th", { class: "num", text: "downloads" }), el("th", { class: "num", text: "likes" }), el("th", {}))),
      el("tbody", {}, hits.map((h) => el("tr", {},
        el("td", {}, el("a", { href: "#", onclick: (e) => { e.preventDefault(); openRepo(h.repo); }, text: h.repo }), h.gated ? el("span", { class: "tag v", text: "gated: needs HF_TOKEN" }) : null),
        el("td", { class: "muted", text: h.pipeline || "" }),
        el("td", { class: "num", text: h.downloads.toLocaleString() }),
        el("td", { class: "num", text: h.likes.toLocaleString() }),
        el("td", {}, el("button", { class: "btn small", onclick: () => openRepo(h.repo), text: "files" })),
      ))),
    )),
  );
}

async function openRepo(repo) {
  const box = $("repo");
  box.replaceChildren(el("p", { class: "muted", text: `listing ${repo}…` }));
  const data = await act(() => api("GET", `/lab/api/repo?id=${encodeURIComponent(repo)}`));
  box.replaceChildren();
  if (!data) return;
  box.append(el("h3", {}, data.repo, " ", el("a", { href: `https://huggingface.co/${data.repo}`, target: "_blank", rel: "noopener", text: "↗" })));
  if (!data.choices.length) {
    const name = data.repo.split("/")[1];
    box.append(el("p", { class: "warn", text: `This repo has no GGUF files, so llama.cpp cannot run it as is. Most models have a GGUF version: search for "${name} GGUF".` }));
    return;
  }
  if (data.projectors.length) box.append(el("p", { class: "hint", text: "This repo also ships a vision projector (mmproj); the lab chat is text-only for now." }));
  box.append(
    el("div", { class: "wrap" }, el("table", {},
      el("thead", {}, el("tr", {}, el("th", { text: "quant" }), el("th", { text: "file" }), el("th", { class: "num", text: "size" }), el("th", { text: "fits" }), el("th", {}))),
      el("tbody", {}, data.choices.map((c) => {
        const [cls, label] = FIT[c.fit];
        return el("tr", {},
          el("td", {}, el("b", { text: c.quant })),
          el("td", { class: "muted", text: c.file + (c.shards > 1 ? ` (${c.shards} shards)` : "") }),
          el("td", { class: "num", text: gb(c.size) }),
          el("td", { class: cls, text: label }),
          el("td", {}, el("button", { class: "btn small", onclick: () => download(data.repo, c), text: "download" })),
        );
      })),
    )),
  );
}

async function pull(ev) {
  ev.preventDefault();
  const ref = $("pull-ref").value.trim();
  if (!ref) return;
  const favorite = $("pull-favorite").checked;
  const d = await act(() => api("POST", "/lab/api/models/pull", { ref, favorite }));
  if (!d) return;
  say(`downloading ${d.repo}/${d.file} (${gb(d.size)})${favorite ? " into models/" : ""}`, true);
  $("pull-ref").value = "";
  poll();
}

async function download(repo, c) {
  if (c.fit === "no" && !confirm(`${gb(c.size)} will most likely not fit in this machine's RAM. Download anyway?`)) return;
  if (await act(() => api("POST", "/lab/api/downloads", { repo, file: c.file }), `downloading ${c.file}`)) poll();
}

let downloadsRunning = new Set();

function renderDownloads(list) {
  const box = $("downloads");
  box.replaceChildren();
  const running = new Set();
  for (const d of list) {
    if (d.state === "running") running.add(d.id);
    const pct = d.size ? Math.round((100 * d.done) / d.size) : 0;
    const label = d.state === "running" ? `${gb(d.done)} / ${gb(d.size)} · ${pct}% · ${d.elapsed_s}s` : d.state === "done" ? `done · ${gb(d.size)}` : "failed";
    box.append(
      el("div", { class: "panel", style: "margin:6px 0" },
        el("span", { class: d.state === "error" ? "ko" : d.state === "done" ? "ok" : "live", text: ` ${d.repo}/${d.file}` }), " ",
        d.favorite ? el("span", { class: "tag v", text: "★ models/" }) : null,
        el("span", { class: "muted", text: label }),
        d.state === "running" ? el("div", { class: "bar" }, el("i", { style: `width:${pct}%` })) : null,
        d.error ? el("pre", { class: "ko", text: d.error }) : null,
      ),
    );
  }
  const finished = [...downloadsRunning].some((id) => !running.has(id));
  downloadsRunning = running;
  if (finished) loadModels();
}

// -- prune ------------------------------------------------------------------------------

let pruneBusy = false;

function pruneRequest(dryRun, expect) {
  const days = $("prune-days").value;
  return { older_than_days: days === "" ? null : Number(days), repo: $("prune-repo").value.trim() || null, dry_run: dryRun, expect };
}

async function prunePreview(ev) {
  if (ev) ev.preventDefault();
  const out = $("prune-out");
  const req = pruneRequest(true);
  const data = await act(() => api("POST", "/lab/api/models/prune", req));
  out.replaceChildren();
  if (!data) return;
  if (!data.files.length) {
    out.append(el("p", { class: "muted", text: "Nothing to prune with these filters." }));
    return;
  }
  const inChat = data.files.some((f) => f.in_chat);
  out.append(
    el("div", { class: "wrap", style: "margin-top:10px" }, el("table", {},
      el("thead", {}, el("tr", {}, el("th", { text: "repo" }), el("th", { text: "file" }), el("th", { text: "revision" }), el("th", { class: "num", text: "size" }), el("th", { class: "num", text: "unused" }))),
      el("tbody", {}, data.files.map((f) => el("tr", {},
        el("td", { text: f.repo }),
        el("td", {}, f.name, f.in_chat ? el("span", { class: "tag v", style: "margin-left:6px", text: "in chat" }) : null),
        el("td", { class: "muted", text: f.commit.slice(0, 12) }),
        el("td", { class: "num", text: gb(f.size) }),
        el("td", { class: "num", text: `${f.idle_days} d` }),
      ))),
    )),
    el("div", { class: "row", style: "margin-top:10px" },
      el("span", { text: `${data.files.length} file(s) in ${data.revisions.length} revision(s), ${gb(data.size)}` }),
      inChat ? el("span", { class: "warn", text: "the chat is using one of them: unload it first" }) : null,
      el("button", { class: "btn red", type: "button", id: "prune-run", "data-in-chat": inChat ? "1" : false, disabled: inChat || pruneBusy, onclick: () => pruneRun(req, data), text: `delete ${gb(data.size)}` }),
    ),
  );
}

async function pruneRun(req, preview) {
  if (!confirm(`Delete ${preview.files.length} file(s) (${gb(preview.size)}) from the Hugging Face cache?\n\n${preview.revisions.join("\n")}`)) return;
  const done = await act(() => api("POST", "/lab/api/models/prune", { ...req, dry_run: false, expect: preview.revisions }));
  if (!done) return;
  say(`freed ${gb(done.freed)}`, true);
  $("prune-out").replaceChildren();
  loadModels();
}

document.addEventListener("DOMContentLoaded", () => {
  $("search-form").addEventListener("submit", search);
  $("pull-form").addEventListener("submit", pull);
  $("prune-form").addEventListener("submit", prunePreview);
  Lab.onStatus((s, ev) => {
    pruneBusy = Boolean(s.job && s.job.running) || s.downloads.some((d) => d.state === "running");
    const pruneBtn = $("prune-run");
    if (pruneBtn) pruneBtn.disabled = pruneBusy || pruneBtn.dataset.inChat === "1";
    renderDownloads(s.downloads);
    if (ev && ev.jobFinished) loadModels();
  });
  loadModels();
});
