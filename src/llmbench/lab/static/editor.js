// llb lab · test: an experiment editor (the generic form lives in form.js). The YAML text is the source of truth; the form is
// generated from Experiment's JSON schema (so every field is reachable) and each change is
// applied server-side as a patch that keeps the YAML's comments.
"use strict";

let meta = null;
let text = "";
let saved = null; // text as last opened/saved; null = never saved
let localModels = [];
let llamaFields = []; // [{key: "load.threads", schema, help}]
for (const key of ["experiment", "models", "defaults", "phases"]) Form.openSections.add(key);
let defaultsGroup = "load";
let phaseTab = "bench";

function helpFor(path, prop) {
  if (path[0] === "defaults" && path.length === 3) return meta.help[`${path[1]}.${path[2]}`] || prop.description;
  return prop.description;
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
  if (saved === null) d.append(el("span", { class: "dirty", text: "● not saved" }));
  else if (text !== saved) d.append(el("span", { class: "dirty", text: "● unsaved changes" }));
  else d.append(el("span", { class: "ok", text: "saved" }));
}

async function change(path, value, { rerender = false } = {}) {
  try {
    const r = await api("POST", "/lab/api/editor/patch", { text, path, value });
    Form.doc = r.doc;
    setText(r.text);
    $("yaml-state").replaceChildren();
    renderRunModels();
    if (rerender) renderForm();
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
      renderRunModels();
    } catch (e) {
      $("yaml-state").replaceChildren(el("span", { class: "ko", text: e.line ? `line ${e.line}: ${e.message}` : e.message }));
    }
  }, 600);
}

// -- sections ------------------------------------------------------------------------

function renderForm() {
  const form = $("form");
  const scroll = window.scrollY;
  form.replaceChildren();
  const props = resolve(meta.schema).properties;

  form.append(section("experiment", "experiment", "Name (also the results folder), description and tags.", (d) => renderObject(d, resolve(props.experiment), ["experiment"])));

  form.append(section("machine", "machine", "Profiles bring the llama.cpp build and the hardware handling (cooldown, telemetry, governor). Later entries override earlier ones; this file overrides them all.", (d) => {
    const chosen = new Set(Form.doc.extends || []);
    const checks = el("div", { class: "checks" }, meta.profiles.map((p) => el("label", {}, el("input", { type: "checkbox", value: p, checked: chosen.has(p) }), p.replace("_profiles/", ""))));
    checks.addEventListener("change", () => {
      const picked = meta.profiles.filter((p) => checks.querySelector(`input[value="${CSS.escape(p)}"]`).checked);
      const custom = (Form.doc.extends || []).filter((p) => !meta.profiles.includes(p));
      const all = [...picked, ...custom];
      change(["extends"], all.length ? all : null);
    });
    d.append(el("label", { class: "f" }, el("span", { class: "k", text: "extends" }), checks));
    const hw = el("details", { class: "sub" }, el("summary", { text: "hardware (overrides the profile)" }));
    renderObject(hw, resolve(props.hardware), ["hardware"]);
    d.append(hw);
    const builds = el("details", { class: "sub" }, el("summary", { text: "llama.cpp builds (overrides the profile)" }));
    builds.append(el("p", { class: "hint", text: resolve(Form.defs.BuildSpec).description }));
    renderObjectArray(builds, "builds", ["llama_cpp", "builds"], resolve(Form.defs.BuildSpec), getAt(Form.doc, ["llama_cpp", "builds"]));
    d.append(builds);
  }));

  form.append(section("models", "models", "Every model is crossed with every variant. Pick downloaded ones or type any reference: hf://org/repo:QUANT, hf://org/repo/file.gguf or a path (downloaded on first use).", (d) => {
    const pick = el("select", {}, localModels.map((m) => el("option", { value: m.key, text: `${m.name} · ${m.quant}` })));
    const source = el("input", { placeholder: "hf://org/repo:Q4_K_M", size: 32 });
    const add = (src, id) => {
      const list = Form.doc.models || [];
      change(["models", list.length], { id, source: src }, { rerender: true });
    };
    const adders = el("div", { class: "row" },
      el("label", {}, "downloaded", pick),
      el("button", { class: "btn small", type: "button", onclick: () => { const m = localModels.find((x) => x.key === pick.value); if (m) add(m.source, slug(`${m.name}-${m.quant}`)); }, text: "+ add" }),
      el("label", {}, "or a reference", source),
      el("button", { class: "btn small", type: "button", onclick: () => source.value.trim() && add(source.value.trim(), slug(source.value.split(/[/:]/).slice(-2).join("-"))), text: "+ add" }),
    );
    renderObjectArray(d, "models", ["models"], resolve(Form.defs.ModelSpec), Form.doc.models, adders);
  }));

  form.append(section("defaults", "llama.cpp settings", "Shared by every run (the matrix varies some of them). Empty = llama.cpp's default. Each field shows the flag it becomes.", (d) => {
    const llama = resolve(props.defaults);
    const groups = Object.keys(llama.properties);
    d.append(tabs(groups, defaultsGroup, (g) => (getAt(Form.doc, ["defaults", g]) !== undefined ? `${g} ●` : g), (g) => { defaultsGroup = g; renderForm(); }));
    const prop = llama.properties[defaultsGroup];
    const { s } = unwrap(prop);
    if (s.description) d.append(el("p", { class: "hint", text: s.description }));
    if (s.properties) renderObject(d, s, ["defaults", defaultsGroup]);
    else {
      const grid = el("div", { class: "fields" });
      yamlOnly(grid, defaultsGroup, getAt(Form.doc, ["defaults", defaultsGroup]), "environment variables for the llama.cpp processes");
      d.append(grid);
    }
  }));

  form.append(section("matrix", "matrix", "What varies: each axis lists values of one setting, and axes are crossed (2 x 3 = 6 variants). zip, include and exclude are edited in the YAML.", renderMatrix));

  form.append(section("phases", "phases", "What to measure. Each enabled phase runs for every configuration.", (d) => {
    const phases = resolve(props.phases);
    const names = Object.keys(phases.properties);
    const enabled = (p) => {
      const v = getAt(Form.doc, ["phases", p, "enabled"]);
      return v === undefined ? resolve(phases.properties[p]).properties.enabled.default : v;
    };
    d.append(tabs(names, phaseTab, (p) => (enabled(p) ? `${p} ●` : p), (p) => { phaseTab = p; renderForm(); }));
    const ps = resolve(phases.properties[phaseTab]);
    if (ps.description) d.append(el("p", { class: "hint", text: ps.description }));
    renderObject(d, ps, ["phases", phaseTab]);
  }));

  form.append(section("gates", "gates and winner", resolve(Form.defs.Gate).description, (d) => {
    renderObjectArray(d, "gates", ["gates"], resolve(Form.defs.Gate), Form.doc.gates);
    const sel = el("details", { class: "sub", open: Form.doc.select !== undefined }, el("summary", { text: "select (how llb guide picks the winner)" }));
    renderObject(sel, resolve(Form.defs.Select), ["select"]);
    d.append(sel);
  }));

  form.append(section("output", "output", resolve(Form.defs.Output).description, (d) => renderObject(d, resolve(Form.defs.Output), ["output"])));
  window.scrollTo(0, scroll);
}

function renderMatrix(d) {
  const matrix = Form.doc.matrix || {};
  const special = ["zip", "include", "exclude"];
  const options = (current) => el("select", {}, llamaFields.map((f) => el("option", { value: f.key, text: f.key, selected: f.key === current })));
  const parse = (key, raw) => {
    const f = llamaFields.find((x) => x.key === key);
    const { s } = unwrap(f ? f.schema : {});
    const parts = raw.split(",").map((x) => x.trim()).filter((x) => x !== "");
    if (!parts.length) return null;
    return parts.map((x) => (s.type === "integer" ? parseInt(x, 10) : s.type === "number" ? parseFloat(x) : s.type === "boolean" ? x === "true" : x));
  };
  for (const [key, values] of Object.entries(matrix)) {
    if (special.includes(key)) continue;
    const sel = options(key);
    const input = el("input", { value: Array.isArray(values) ? values.join(", ") : show(values) });
    sel.addEventListener("change", async () => {
      await change(["matrix", key], null);
      await change(["matrix", sel.value], parse(sel.value, input.value), { rerender: true });
    });
    input.addEventListener("change", () => change(["matrix", key], parse(key, input.value), { rerender: true }));
    const f = llamaFields.find((x) => x.key === key);
    d.append(el("div", { class: "axis" }, el("label", { class: "f" }, el("span", { class: "k set", text: "setting" }), sel, f && f.help ? el("small", { text: f.help }) : null), el("label", { class: "f" }, el("span", { class: "k", text: "values" }), input), el("button", { class: "btn small red", type: "button", onclick: () => change(["matrix", key], null, { rerender: true }), text: "remove" })));
  }
  const sel = options(null);
  const input = el("input", { placeholder: "e.g. 2, 4, 8" });
  d.append(el("div", { class: "axis" }, el("label", { class: "f" }, el("span", { class: "k", text: "new axis" }), sel), el("label", { class: "f" }, el("span", { class: "k", text: "values" }), input), el("button", { class: "btn small", type: "button", onclick: () => { const v = parse(sel.value, input.value); if (v) change(["matrix", sel.value], v, { rerender: true }); else say("give at least one value"); }, text: "+ add axis" })));
  for (const key of special) if (matrix[key] !== undefined && (matrix[key].length || Object.keys(matrix[key]).length)) yamlOnly(d, key, matrix[key]);
}


// -- files and commands ---------------------------------------------------------------

function renderFiles(files, current) {
  $("files").replaceChildren(el("option", { value: "", text: files.length ? "open an experiment…" : "(no experiments yet)" }), files.map((f) => el("option", { value: f, text: f })));
  if (current && files.includes(current)) $("files").value = current;
}

async function openFile(name) {
  if (!name) return;
  if (text !== (saved ?? "") && text && !confirm("Discard the unsaved changes?")) return;
  const r = await act(() => api("GET", `/lab/api/editor/file?name=${encodeURIComponent(name)}`));
  if (!r) return;
  $("name").value = r.name;
  saved = r.text;
  setText(r.text);
  await reparseNow();
  history.replaceState(null, "", `/lab/test?file=${encodeURIComponent(name)}`);
}

async function reparseNow() {
  try {
    Form.doc = (await api("POST", "/lab/api/editor/parse", { text })).doc;
    $("yaml-state").replaceChildren();
  } catch (e) {
    Form.doc = {};
    $("yaml-state").replaceChildren(el("span", { class: "ko", text: e.line ? `line ${e.line}: ${e.message}` : e.message }));
  }
  renderForm();
  renderRunModels();
}

async function save() {
  let name = $("name").value.trim();
  if (!name) {
    name = `${(Form.doc.experiment && Form.doc.experiment.name) || "experiment"}.yaml`;
    $("name").value = name;
  }
  if (!/\.ya?ml$/.test(name)) {
    name += ".yaml";
    $("name").value = name;
  }
  const exists = meta.files.includes(name);
  const ours = saved !== null && new URLSearchParams(location.search).get("file") === name;
  if (exists && !ours && !confirm(`${name} already exists. Overwrite it?`)) return null;
  const r = await act(() => api("POST", "/lab/api/editor/save", { name, text, overwrite: exists }), `saved ${name}`);
  if (!r) return null;
  meta.files = r.files;
  saved = text;
  renderFiles(r.files, name);
  renderDirty();
  history.replaceState(null, "", `/lab/test?file=${encodeURIComponent(name)}`);
  return name;
}

function selectedModels() {
  return [...$("run-models").querySelectorAll("input:checked")].map((c) => c.value);
}

function renderRunModels() {
  const ids = (Form.doc.models || []).map((m) => m && m.id).filter(Boolean);
  const before = new Set(selectedModels());
  $("run-models").replaceChildren(...ids.map((id) => el("label", { title: "only these models (-m); none ticked = all" }, el("input", { type: "checkbox", value: id, checked: before.has(id) }), id)));
}

function showOutput(title, result) {
  $("output").replaceChildren(
    el("div", { class: "panel", style: "margin-top:10px" },
      el("div", { class: "row" }, el("b", { class: result.ok ? "ok" : "ko", text: `${title}: ${result.ok ? "ok" : `failed${result.code != null ? ` (exit ${result.code})` : ""}`}` }), el("button", { class: "btn small", onclick: () => $("output").replaceChildren(), text: "close" })),
      el("pre", { text: result.output }),
    ),
  );
}

async function check(command) {
  $("output").replaceChildren(el("p", { class: "muted", text: `llb ${command}…` }));
  const r = await act(() => api("POST", "/lab/api/editor/check", { text, command, models: selectedModels() }));
  if (r) showOutput(`llb ${command}`, r);
  else $("output").replaceChildren();
}

async function command(cmd, extra) {
  const name = text !== saved || !$("name").value ? await save() : $("name").value;
  if (!name) return;
  const r = await act(() => api("POST", "/lab/api/editor/command", { name, command: cmd, ...extra }));
  if (!r) return;
  if (r.sync) showOutput(`llb ${cmd}`, r.sync);
  else {
    say(r.stopped_chat ? `${cmd} started; the chat was unloaded so it does not skew the numbers` : `${cmd} started`, true);
    poll();
  }
}

// -- new experiment -------------------------------------------------------------------

function renderNewBox() {
  const usable = meta.profiles.filter((p) => !p.endsWith("/deterministic.yaml"));
  $("new-profile").replaceChildren(...usable.map((p) => el("option", { value: p, text: p.replace("_profiles/", "") })), el("option", { value: "", text: "none (write everything yourself)" }));
  if (meta.default_profile) $("new-profile").value = meta.default_profile;
  const wanted = param("model");
  $("new-models").replaceChildren(...(localModels.length ? localModels.map((m) => el("label", {}, el("input", { type: "checkbox", value: m.key, checked: m.key === wanted }), `${m.name} · ${m.quant}`)) : [el("span", { class: "muted", text: "no models downloaded yet: add references in the form afterwards" })]));
  $("new-presets").replaceChildren(...meta.presets.map((p, i) => el("label", { title: p.help }, el("input", { type: "checkbox", value: p.key, checked: i < 2 }), p.title)));
}

async function createNew() {
  if (text && text !== (saved ?? "") && !confirm("Discard the unsaved changes?")) return;
  const body = {
    models: [...$("new-models").querySelectorAll("input:checked")].map((c) => c.value),
    profile: $("new-profile").value || null,
    presets: [...$("new-presets").querySelectorAll("input:checked")].map((c) => c.value),
  };
  const r = await act(() => api("POST", "/lab/api/editor/new", body));
  if (!r) return;
  Form.doc = r.doc;
  saved = null;
  setText(r.text);
  $("name").value = `${Form.doc.experiment.name}.yaml`;
  $("new-box").hidden = true;
  history.replaceState(null, "", "/lab/test");
  renderForm();
  renderRunModels();
}

// -- start ----------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", async () => {
  const [m, models] = await Promise.all([act(() => api("GET", "/lab/api/editor/meta")), act(() => api("GET", "/lab/api/models"))]);
  if (!m) return;
  meta = m;
  Form.defs = meta.schema.$defs;
  Form.help = helpFor;
  Form.change = change;
  Form.paths = { suites: meta.suites };
  localModels = models ? models.models : [];
  const llama = resolve(resolve(meta.schema).properties.defaults);
  for (const [group, prop] of Object.entries(llama.properties)) {
    const g = unwrap(prop).s;
    for (const [field, fprop] of Object.entries(g.properties || {})) {
      const { s } = unwrap(fprop);
      if (["integer", "number", "string", "boolean"].includes(s.type) || s.enum) llamaFields.push({ key: `${group}.${field}`, schema: fprop, help: meta.help[`${group}.${field}`] });
    }
  }
  $("folder").textContent = meta.folder;
  renderFiles(meta.files, param("file"));
  renderNewBox();

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
  $("validate").addEventListener("click", () => check("validate"));
  $("plan").addEventListener("click", () => check("plan"));
  $("run").addEventListener("click", () => command("run", { models: selectedModels(), resume: $("resume").checked, no_build: $("no-build").checked }));
  $("guide").addEventListener("click", () => command("guide", {}));
  $("report").addEventListener("click", () => command("report", { formats: [...document.querySelectorAll("input[name=fmt]:checked")].map((c) => c.value) }));
  $("judge").addEventListener("click", () => command("judge", { force: $("force").checked }));
  window.addEventListener("beforeunload", (e) => {
    if (text && text !== saved) e.preventDefault();
  });
  Lab.onStatus((s) => {
    const busy = Boolean(s.job && s.job.running);
    for (const id of ["run", "judge", "report"]) $(id).disabled = busy;
  });

  if (param("file")) await openFile(param("file"));
  else if (param("model")) await createNew();
  else {
    $("new-box").hidden = false;
    $("form").replaceChildren(el("p", { class: "muted", text: "Open an experiment above, or create a new one." }));
    renderDirty();
  }
});
