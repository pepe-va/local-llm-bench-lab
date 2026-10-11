// llb lab · suites: an editor for suites/*.yaml (the generic form lives in form.js). The YAML
// text is the source of truth: the form is generated from Suite's JSON schema and each
// change is applied server-side as a patch that keeps the YAML's comments. "validate &
// preview" lists the requests a run would send; "try it" sends the first ones to the chat
// model and scores the answers.
"use strict";

let meta = null;
let text = "";
let saved = null; // text as last opened/saved; null = never saved
let sourceTab = null;
let trying = null; // AbortController of a running try
const openItems = new Set();
for (const key of ["suite", "source"]) Form.openSections.add(key);

const SOURCES = { items: "items (inline)", synthetic: "synthetic", file: "file", hf_dataset: "Hugging Face dataset" };
const ITEM_FIELDS = {
  completion: ["id", "category", "prompt", "input", "reference", "max_tokens", "evaluator"],
  chat: ["id", "category", "system", "prompt", "image", "input", "reference", "max_tokens", "evaluator"],
  embeddings: ["id", "category", "a", "b", "label"],
  rerank: ["id", "category", "query", "documents", "relevant"],
};
const SUITE_FIELDS = {
  completion: ["name", "description", "endpoint", "template", "max_tokens"],
  chat: ["name", "description", "endpoint", "system", "template", "max_tokens"],
  embeddings: ["name", "description", "endpoint", "pooling"],
  rerank: ["name", "description", "endpoint", "k"],
};
const LONG = new Set(["prompt", "system", "input", "reference", "template", "rubric", "a", "b", "query", "description"]);
const LINES = new Set(["documents", "values"]);

const endpoint = () => (ITEM_FIELDS[Form.doc.endpoint] ? Form.doc.endpoint : "completion");
const generates = () => ["completion", "chat"].includes(endpoint());
const sourceKind = () => Object.keys(SOURCES).find((k) => getAt(Form.doc, ["source", k]) != null) || null;
const errorText = (e) => (e.line ? `line ${e.line}: ${e.message}` : e.message);

function multiline(name, path, kind) {
  if (kind === "string") return LONG.has(name);
  if (kind === "list") return LINES.has(name);
  return kind === "object" && name === "schema"; // json_schema evaluator
}

function evalLabel(e) {
  if (!e || typeof e !== "object") return "";
  if ((e.type === "all" || e.type === "any") && Array.isArray(e.of)) return `${e.type}(${e.of.map(evalLabel).join(", ")})`;
  if (e.type === "weighted" && Array.isArray(e.of)) return `weighted(${e.of.map((p) => evalLabel(p && p.evaluator)).join(", ")})`;
  return String(e.type || "?");
}

// -- text and patches ---------------------------------------------------------------

function setText(t, { fromTextarea = false } = {}) {
  text = t;
  if (!fromTextarea) $("yaml").value = t;
  renderDirty();
}

function renderDirty() {
  const d = $("dirty");
  d.replaceChildren();
  if (saved === null) d.append(el("span", { class: "dirty", text: text ? "● not saved" : "" }));
  else if (text !== saved) d.append(el("span", { class: "dirty", text: "● unsaved changes" }));
  else d.append(el("span", { class: "ok", text: "saved" }));
}

async function change(path, value, { rerender = false } = {}) {
  try {
    const r = await api("POST", "/lab/api/suites/patch", { text, path, value });
    Form.doc = r.doc;
    setText(r.text);
    $("yaml-state").replaceChildren();
    if (rerender || path[0] === "endpoint" || (path[0] === "source" && path.length === 1)) renderForm();
  } catch (e) {
    say(e.message);
  }
}

let parseTimer = null;

function onYamlInput() {
  setText($("yaml").value, { fromTextarea: true });
  clearTimeout(parseTimer);
  parseTimer = setTimeout(async () => {
    try {
      Form.doc = (await api("POST", "/lab/api/editor/parse", { text })).doc;
      $("yaml-state").replaceChildren(el("span", { class: "ok", text: "YAML ok" }));
      renderForm();
    } catch (e) {
      $("yaml-state").replaceChildren(el("span", { class: "ko", text: errorText(e) }));
    }
  }, 600);
}

// -- the form -------------------------------------------------------------------------

function renderForm() {
  const form = $("form");
  const scroll = window.scrollY;
  form.replaceChildren();
  const schema = resolve(meta.schema);
  const props = schema.properties;
  const ep = endpoint();

  form.append(section("suite", "suite", "Name (how results and reports call it), the endpoint the requests go to and their defaults.", (d) => {
    const shown = SUITE_FIELDS[ep];
    const extra = Object.keys(props).filter((k) => !shown.includes(k) && !["version", "source", "columns", "evaluator", "sampling"].includes(k) && Form.doc[k] !== undefined);
    const grid = el("div", { class: "fields" });
    for (const k of [...shown, ...extra]) renderField(grid, k, props[k], [k]);
    d.append(grid);
  }));

  form.append(section("source", "requests", resolve(Form.defs.SuiteSource).description + " Each tab is one kind of source.", renderSource));

  if (generates() || Form.doc.evaluator !== undefined) {
    form.append(section("evaluator", "default evaluator", "How answers are scored when an item has no evaluator of its own (and for every row of a file or dataset). Deterministic checks come first; llm_judge is scored later by `llb judge`.", (d) => {
      const grid = el("div", { class: "fields" });
      renderField(grid, "evaluator", props.evaluator, ["evaluator"]);
      d.append(grid);
    }));
  }

  if (generates() || Form.doc.sampling !== undefined) {
    form.append(section("sampling", "sampling", `${props.sampling.description || ""} Empty = the experiment's value (or llama.cpp's default). "try it" uses these too.`, (d) => renderObject(d, resolve(Form.defs.SamplingConfig), ["sampling"])));
  }
  $("fetch").hidden = sourceKind() !== "hf_dataset";
  window.scrollTo(0, scroll);
}

function starter(kind) {
  const name = Form.doc.name || "my-suite";
  if (kind === "items") return [newItem([])];
  if (kind === "synthetic") return { prompt_tokens: { dist: "fixed", value: 512 }, output_tokens: { dist: "fixed", value: 128 }, count: 20, seed: 42 };
  if (kind === "file") return { path: `data/${name}.jsonl` };
  return { repo: "org/dataset", config: "default", split: "test", limit: 20 };
}

function renderSource(d) {
  const set = sourceKind();
  const tab = sourceTab || set || "items";
  d.append(tabs(Object.keys(SOURCES), tab, (k) => (k === set ? `${SOURCES[k]} ●` : SOURCES[k]), (k) => { sourceTab = k; renderForm(); }));
  const sch = resolve(Form.defs.SuiteSource).properties[tab];
  const desc = unwrap(sch).s.description;
  if (tab !== set) {
    d.append(el("p", { class: "hint", text: desc || "" }));
    d.append(el("p", {}, set ? `This suite reads its requests from ${SOURCES[set]}. A suite has exactly one source. ` : "No source yet. ",
      el("button", { class: "btn small", type: "button", onclick: () => {
        if (set && !confirm(`Replace the ${SOURCES[set]} source with ${SOURCES[tab]}? The current requests are removed from the YAML.`)) return;
        sourceTab = tab;
        change(["source"], { [tab]: starter(tab) }, { rerender: true });
      }, text: `use ${SOURCES[tab]}` })));
    return;
  }
  if (tab === "items") return renderItems(d);
  if (desc) d.append(el("p", { class: "hint", text: desc }));
  const def = { synthetic: "Synthetic", file: "FileSource", hf_dataset: "HfDataset" }[tab];
  renderObject(d, resolve(Form.defs[def]), ["source", tab]);
  if (tab !== "synthetic") {
    const cols = el("details", { class: "sub", open: true }, el("summary", { text: "columns: which column is the prompt, the reference…" }));
    cols.append(el("p", { class: "hint", text: resolve(Form.defs.Columns).description }));
    renderObject(cols, resolve(Form.defs.Columns), ["columns"]);
    d.append(cols);
  }
}

// -- items ----------------------------------------------------------------------------

function newItem(items) {
  const ep = endpoint();
  const ids = new Set(items.map((it) => it && it.id));
  let n = items.length + 1;
  while (ids.has(`item-${n}`)) n++;
  if (ep === "embeddings") return { id: `item-${n}`, a: "", b: "", label: 1.0 };
  if (ep === "rerank") return { id: `item-${n}`, query: "", documents: [""], relevant: [0] };
  return { id: `item-${n}`, prompt: "" };
}

function itemSummary(item, ep) {
  if (!item || typeof item !== "object") return "";
  const clip = (s) => { const t = String(s || "").replace(/\s+/g, " ").trim(); return t.length > 80 ? `${t.slice(0, 80)}…` : t; };
  if (ep === "embeddings") return `${clip(item.a)} ↔ ${clip(item.b)}`;
  if (ep === "rerank") return `${clip(item.query)} · ${(item.documents || []).length} documents`;
  const ev = item.evaluator ? ` · ${evalLabel(item.evaluator)}` : "";
  return `${clip(item.prompt) || "(no prompt)"}${ev}`;
}

function renderItems(d) {
  const items = getAt(Form.doc, ["source", "items"]) || [];
  const schema = resolve(Form.defs.Item);
  const ep = endpoint();
  d.append(el("p", { class: "hint", text: `${schema.description} Fields shown for the ${ep} endpoint; the rest are under "other fields".` }));
  const swap = async (i, j) => {
    const a = items[i];
    const b = items[j];
    await change(["source", "items", i], b);
    await change(["source", "items", j], a);
    const oi = openItems.has(i);
    const oj = openItems.has(j);
    openItems.delete(i); openItems.delete(j);
    if (oi) openItems.add(j);
    if (oj) openItems.add(i);
    renderForm();
  };
  items.forEach((item, i) => {
    const path = ["source", "items", i];
    const det = el("details", { class: "req", open: openItems.has(i), ontoggle: (e) => (e.target.open ? openItems.add(i) : openItems.delete(i)) },
      el("summary", {}, el("b", { text: `#${i + 1} ${(item && item.id) || ""}` }), el("span", { class: "muted", text: itemSummary(item, ep) })));
    det.append(el("div", { class: "req-actions" },
      el("button", { class: "btn small", type: "button", disabled: i === 0, onclick: () => swap(i, i - 1), text: "↑" }),
      el("button", { class: "btn small", type: "button", disabled: i === items.length - 1, onclick: () => swap(i, i + 1), text: "↓" }),
      el("button", { class: "btn small", type: "button", title: "copy it to the end of the list", onclick: () => {
        const copy = JSON.parse(JSON.stringify(item || {}));
        if (copy.id) copy.id = `${copy.id}-copy`;
        openItems.add(items.length);
        change(["source", "items", items.length], copy, { rerender: true });
      }, text: "duplicate" }),
      el("button", { class: "btn small red", type: "button", onclick: () => {
        if (!confirm(`Remove item #${i + 1}${item && item.id ? ` (${item.id})` : ""}?`)) return;
        openItems.clear();
        change(path, null, { rerender: true });
      }, text: "remove" })));
    if (item && typeof item === "object") {
      const wanted = ITEM_FIELDS[ep];
      const shown = [...wanted, ...Object.keys(item).filter((k) => !wanted.includes(k) && schema.properties[k])];
      const grid = el("div", { class: "fields" });
      for (const k of shown) renderField(grid, k, schema.properties[k], [...path, k]);
      det.append(grid);
      const rest = Object.keys(schema.properties).filter((k) => !shown.includes(k));
      if (rest.length) {
        const more = el("details", { class: "sub" }, el("summary", { text: `other fields (${rest.join(", ")})` }));
        renderObject(more, schema, path, shown);
        det.append(more);
      }
    } else {
      yamlOnly(det, `item #${i + 1}`, item, "not a mapping");
    }
    d.append(det);
  });
  d.append(el("button", { class: "btn small", type: "button", onclick: () => {
    openItems.add(items.length);
    change(["source", "items", items.length], newItem(items), { rerender: true });
  }, text: "+ add item" }));
}

// -- files ----------------------------------------------------------------------------

function renderFiles(files, current) {
  $("files").replaceChildren(el("option", { value: "", text: files.length ? "open a suite…" : "(no suites yet)" }), ...files.map((f) => el("option", { value: f, text: f })));
  if (current && files.includes(current)) $("files").value = current;
}

function renderUsed() {
  const name = $("name").value.trim();
  const users = (meta.used_by || {})[name] || [];
  const box = $("used");
  box.replaceChildren();
  if (!name) return;
  if (!users.length) {
    box.append("not used by any experiment yet · ", el("a", { href: "/lab/test", text: "→ test" }));
    return;
  }
  box.append("used by ");
  users.forEach((f, i) => box.append(i ? ", " : "", el("a", { href: `/lab/test?file=${encodeURIComponent(f)}`, text: f })));
}

async function openFile(name) {
  if (!name) return;
  if (text !== (saved ?? "") && text && !confirm("Discard the unsaved changes?")) return;
  const r = await act(() => api("GET", `/lab/api/suites/file?name=${encodeURIComponent(name)}`));
  if (!r) return;
  $("name").value = r.name;
  saved = r.text;
  sourceTab = null;
  openItems.clear();
  setText(r.text);
  $("out").replaceChildren();
  await reparseNow();
  renderUsed();
  history.replaceState(null, "", `/lab/suites?file=${encodeURIComponent(name)}`);
}

async function reparseNow() {
  try {
    Form.doc = (await api("POST", "/lab/api/editor/parse", { text })).doc;
    $("yaml-state").replaceChildren();
  } catch (e) {
    Form.doc = {};
    $("yaml-state").replaceChildren(el("span", { class: "ko", text: errorText(e) }));
  }
  renderForm();
}

async function save() {
  let name = $("name").value.trim();
  if (!name) name = `${Form.doc.name || "suite"}.yaml`;
  if (!/\.ya?ml$/.test(name)) name += ".yaml";
  $("name").value = name;
  const exists = meta.files.includes(name);
  const ours = saved !== null && param("file") === name;
  if (exists && !ours && !confirm(`${name} already exists. Overwrite it?`)) return null;
  const r = await act(() => api("POST", "/lab/api/suites/save", { name, text, overwrite: exists }), `saved ${name}`);
  if (!r) return null;
  meta.files = r.files;
  saved = text;
  renderFiles(r.files, name);
  renderDirty();
  renderUsed();
  history.replaceState(null, "", `/lab/suites?file=${encodeURIComponent(name)}`);
  return name;
}

function renderKinds() {
  $("kinds").replaceChildren(...meta.kinds.map((k, i) => el("label", {}, el("input", { type: "radio", name: "kind", value: k.key, checked: i === 0 }), el("span", {}, el("b", { text: k.title }), el("br"), el("small", { text: k.help })))));
}

async function createNew() {
  if (text && text !== (saved ?? "") && !confirm("Discard the unsaved changes?")) return;
  const name = $("new-name").value.trim() || "my-suite";
  const kind = (document.querySelector("input[name=kind]:checked") || {}).value || "chat";
  const r = await act(() => api("POST", "/lab/api/suites/new", { kind, name }));
  if (!r) return;
  Form.doc = r.doc;
  saved = null;
  sourceTab = null;
  openItems.clear();
  openItems.add(0);
  setText(r.text);
  $("name").value = `${name}.yaml`;
  $("new-box").hidden = true;
  $("out").replaceChildren();
  history.replaceState(null, "", "/lab/suites");
  renderForm();
  renderUsed();
}

// -- validate & preview -----------------------------------------------------------------

function closeButton() {
  return el("button", { class: "btn small", type: "button", onclick: () => { if (trying) trying.abort(); $("out").replaceChildren(); }, text: "close" });
}

function head(cols) {
  return el("thead", {}, el("tr", {}, cols.map(([h, w]) => el("th", { text: h, style: `width:${w}` }))));
}

function pre(t) {
  return el("pre", { text: t });
}

function requestCell(r) {
  const cell = el("td");
  if (r.prompt_tokens != null && r.prompt == null) cell.append(el("span", { text: `${r.prompt_tokens} prompt tokens` }), el("span", { class: "muted", text: " (cut from the corpus with the model's tokenizer)" }));
  if (r.system) cell.append(el("div", { class: "muted", text: `system: ${r.system}` }));
  if (r.prompt != null) cell.append(pre(r.prompt));
  if (r.image) cell.append(el("div", { class: "muted", text: `image: ${r.image}` }));
  if (r.a != null) cell.append(pre(`a: ${r.a}\nb: ${r.b}`), el("div", { class: "muted", text: `label: ${r.label ?? "-"}` }));
  if (r.query != null) {
    const rel = new Set(r.relevant || []);
    cell.append(pre(`query: ${r.query}\n${(r.documents || []).map((doc, i) => `${rel.has(i) ? "✓" : " "} [${i}] ${doc}`).join("\n")}`));
  }
  if (r.input != null) cell.append(el("details", {}, el("summary", { class: "muted", text: "input" }), pre(r.input)));
  if (r.reference != null) cell.append(el("details", {}, el("summary", { class: "muted", text: "reference" }), pre(r.reference)));
  return cell;
}

function showPreview(p) {
  const title = `valid · ${p.name} · ${p.endpoint} · ${SOURCES[p.source] || p.source}${p.count != null ? ` · ${p.count} request${p.count === 1 ? "" : "s"}` : ""}`;
  const panel = el("div", { class: "panel", style: "margin-top:10px" }, el("div", { class: "row" }, el("b", { class: "ok", text: title }), closeButton()));
  const notes = [...p.notes];
  if (p.evaluator) notes.unshift(`default evaluator: ${p.evaluator}`);
  if (Object.keys(p.sampling || {}).length) notes.push(`sampling overrides: ${JSON.stringify(p.sampling)}`);
  if (notes.length) panel.append(el("ul", { class: "notes" }, notes.map((n) => el("li", { text: n }))));
  if (p.shapes) {
    const s = p.shapes;
    panel.append(el("div", { class: "totals" },
      el("span", {}, "prompt tokens ", el("b", { text: `${s.prompt_tokens.min} / ${s.prompt_tokens.mean} / ${s.prompt_tokens.max}` }), " (min / mean / max)"),
      el("span", {}, "output tokens ", el("b", { text: `${s.output_tokens.min} / ${s.output_tokens.mean} / ${s.output_tokens.max}` }))));
  }
  if (p.dataset) panel.append(pre(Object.entries(p.dataset).map(([k, v]) => `${k}: ${v}`).join("\n")));
  if (p.requests && p.requests.length) {
    const more = p.count > p.requests.length ? ` (first ${p.requests.length} of ${p.count})` : "";
    panel.append(el("p", { class: "hint", text: `requests a run would send${more}` }));
    panel.append(el("table", {},
      head([["#", "3%"], ["id", "17%"], ["request", "52%"], ["max tokens", "8%"], ["evaluator", "20%"]]),
      el("tbody", {}, p.requests.map((r, i) => el("tr", {},
        el("td", { text: String(i + 1) }), el("td", { text: r.id }), requestCell(r), el("td", { text: r.max_tokens != null ? String(r.max_tokens) : "" }),
        el("td", {}, r.evaluator_spec ? el("details", {}, el("summary", { text: r.evaluator }), pre(JSON.stringify(r.evaluator_spec, null, 2))) : el("span", { class: "muted", text: r.evaluator || "-" })))))));
  }
  $("out").replaceChildren(panel);
}

function showError(title, e) {
  $("out").replaceChildren(el("div", { class: "panel", style: "margin-top:10px" }, el("div", { class: "row" }, el("b", { class: "ko", text: title }), closeButton()), pre(errorText(e))));
  if (e.line) $("yaml-state").replaceChildren(el("span", { class: "ko", text: `line ${e.line}` }));
}

async function check(fetchDataset) {
  $("out").replaceChildren(el("p", { class: "muted", text: fetchDataset ? "downloading the dataset…" : "checking…" }));
  try {
    showPreview(await api("POST", "/lab/api/suites/check", { text, fetch: fetchDataset }));
  } catch (e) {
    showError("not valid", e);
  }
}

// -- try it -----------------------------------------------------------------------------

function verdict(r) {
  if (!r.ok) return el("span", { class: "ko", text: "error" });
  if (r.passed === true) return el("span", { class: "ok", text: `pass${r.score != null && r.score < 1 ? ` (${r.score})` : ""}` });
  if (r.passed === false) return el("span", { class: "ko", text: `fail (${r.score})` });
  if (r.judge) return el("span", { class: "warn", text: "judge only" });
  return el("span", { class: "muted", text: "not scored" });
}

function tryRow(r) {
  const answer = el("details", {}, el("summary", { text: r.ok ? (r.answer || "(empty answer)").replace(/\s+/g, " ").slice(0, 90) : r.error || "error" }));
  if (r.prompt != null) answer.append(el("div", { class: "muted", text: "prompt" }), pre(r.prompt));
  else if (r.prompt_tokens != null) answer.append(el("div", { class: "muted", text: `prompt: ${r.prompt_tokens} tokens` }));
  if (r.reasoning) answer.append(el("div", { class: "muted", text: "reasoning" }), pre(r.reasoning));
  if (r.ok) answer.append(el("div", { class: "muted", text: "answer" }), pre(r.answer || ""));
  if (r.reference != null) answer.append(el("div", { class: "muted", text: "reference" }), pre(r.reference));
  if (r.error) answer.append(el("div", { class: "ko", text: r.error }));
  const why = [r.evaluator ? `evaluator: ${r.evaluator}` : "", r.detail || "", r.judge ? "llm_judge parts are scored by `llb judge` in a run" : ""].filter(Boolean).join(" · ");
  return el("tr", {},
    el("td", { text: String(r.i + 1) }),
    el("td", { text: r.id }),
    el("td", {}, verdict(r), why ? el("div", { class: "muted", text: why }) : null),
    el("td", { text: r.e2e_ms != null ? `${(r.e2e_ms / 1000).toFixed(2)} s` : "-", title: r.ttft_ms != null ? `time to first token ${r.ttft_ms} ms` : "" }),
    el("td", { text: r.predicted_n != null ? `${r.predicted_n}${r.gen_tps ? ` @ ${r.gen_tps} t/s` : ""}${r.stop === "length" || r.stop === "limit" ? " (cut)" : ""}` : "-" }),
    el("td", {}, answer));
}

function totalsLine(t) {
  const part = (label, value) => el("span", {}, `${label} `, el("b", { text: String(value) }));
  return el("div", { class: "totals" },
    part("sent", t.sent), part("ok", t.ok), t.failed ? el("span", { class: "ko", text: `${t.failed} failed` }) : null,
    t.scored ? part("passed", `${t.passed} / ${t.scored}`) : el("span", { class: "muted", text: "nothing scored" }),
    t.mean_score != null ? part("mean score", t.mean_score) : null,
    t.judge_pending ? part("judge pending", t.judge_pending) : null,
    t.mean_e2e_ms != null ? part("mean latency", `${(t.mean_e2e_ms / 1000).toFixed(2)} s`) : null,
    t.mean_ttft_ms != null ? part("mean TTFT", `${t.mean_ttft_ms} ms`) : null,
    part("wall", `${t.wall_s} s`));
}

async function tryIt() {
  if (trying) return;
  const body = {
    text,
    limit: Math.max(1, Math.min(meta.try_max, intOrNull($("try-limit").value) || 5)),
    temperature: parseFloat($("try-temp").value) || 0,
    max_tokens: intOrNull($("try-max").value),
  };
  const status = el("b", { class: "live warn", text: " starting…" });
  const tbody = el("tbody");
  const foot = el("div");
  const stop = el("button", { class: "btn small red", type: "button", onclick: () => trying && trying.abort(), text: "stop" });
  const panel = el("div", { class: "panel", style: "margin-top:10px" },
    el("div", { class: "row" }, el("b", { text: "try it" }), status, stop, closeButton()),
    el("p", { class: "hint", text: "A quick check with the chat model and its settings: not a measurement, nothing is stored. Measure with an experiment in the test tab." }),
    el("table", {}, head([["#", "3%"], ["id", "17%"], ["result", "28%"], ["latency", "7%"], ["tokens", "12%"], ["answer", "33%"]]), tbody),
    foot);
  $("out").replaceChildren(panel);
  trying = new AbortController();
  $("try").disabled = true;
  let count = 0;
  let done = 0;
  const handle = (ev) => {
    if (ev.model) status.textContent = ` ${ev.model.name} · ${ev.model.quant}`;
    if (ev.start) {
      count = ev.start.count;
      status.textContent += ` · 0 / ${count}${ev.start.of > count ? ` (of ${ev.start.of})` : ""}`;
    }
    if (ev.result) {
      done++;
      tbody.append(tryRow(ev.result));
      status.textContent = status.textContent.replace(/· \d+ \/ /, `· ${done} / `);
    }
    if (ev.done) {
      status.className = "ok";
      foot.replaceChildren(totalsLine(ev.done));
    }
    if (ev.error) {
      status.className = "ko";
      foot.replaceChildren(...[el("p", { class: "ko", text: ev.error }), ev.totals ? totalsLine(ev.totals) : null].filter(Boolean));
    }
  };
  try {
    const r = await fetch("/lab/api/suites/try", { method: "POST", headers: HEADERS, body: JSON.stringify(body), signal: trying.signal });
    if (!r.ok) {
      const data = await r.json().catch(() => ({}));
      const d = data.detail;
      const e = d && typeof d === "object" && !Array.isArray(d) ? new ApiError(d.message, d.line) : new ApiError(typeof d === "string" ? d : r.statusText);
      showError("try it", e);
      return;
    }
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done: end } = await reader.read();
      if (end) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (line) handle(JSON.parse(line));
      }
    }
  } catch (e) {
    if (e.name === "AbortError") {
      status.className = "warn";
      status.textContent += " · stopped";
    } else {
      status.className = "ko";
      foot.replaceChildren(el("p", { class: "ko", text: e.message }));
    }
  } finally {
    trying = null;
    stop.remove();
    renderTryState(Lab.status);
  }
}

function renderTryState(s) {
  const chat = s && s.chat;
  const ready = chat && chat.state === "ready";
  const busy = Boolean(s && s.job && s.job.running);
  $("try").disabled = Boolean(trying) || !ready || busy;
  const box = $("try-model");
  box.replaceChildren();
  if (busy) box.append(el("span", { class: "warn", text: "a job is running" }));
  else if (ready) box.append(`with ${chat.name} · ${chat.quant}`);
  else if (chat && chat.state === "loading") box.append(el("span", { class: "warn", text: `loading ${chat.name}…` }));
  else box.append("no model loaded: ", el("a", { href: "/lab/chat", text: "→ load one in the chat" }));
}

// -- start ------------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", async () => {
  const m = await act(() => api("GET", "/lab/api/suites/meta"));
  if (!m) return;
  meta = m;
  Form.defs = meta.schema.$defs;
  Form.change = change;
  Form.multiline = multiline;
  $("folder").textContent = meta.folder;
  $("try-limit").max = String(meta.try_max);
  renderFiles(meta.files, param("file"));
  renderKinds();

  $("yaml").addEventListener("input", onYamlInput);
  $("yaml").addEventListener("keydown", (e) => {
    if (e.key === "Tab") {
      e.preventDefault();
      document.execCommand("insertText", false, "  ");
    }
  });
  $("open").addEventListener("click", () => openFile($("files").value));
  $("new-toggle").addEventListener("click", () => ($("new-box").hidden = !$("new-box").hidden));
  $("new-create").addEventListener("click", createNew);
  $("save").addEventListener("click", save);
  $("name").addEventListener("change", renderUsed);
  $("check").addEventListener("click", () => check(false));
  $("fetch").addEventListener("click", () => check(true));
  $("try").addEventListener("click", tryIt);
  window.addEventListener("beforeunload", (e) => {
    if (text && text !== saved) e.preventDefault();
  });
  Lab.onStatus(renderTryState);
  renderTryState(null);

  if (param("file")) await openFile(param("file"));
  else {
    $("new-box").hidden = false;
    $("form").replaceChildren(el("p", { class: "muted", text: "Open a suite above, or create a new one." }));
    renderDirty();
  }
});
