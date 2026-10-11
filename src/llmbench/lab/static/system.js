// llb lab · system: environment and disk use, llama.cpp builds (list, build, flags).
"use strict";

let buildOptions = null;
let jobRunning = false;

async function loadEnv() {
  const e = await act(() => api("GET", "/lab/api/system"));
  if (!e) return;
  const kv = (k, v, cls) => [el("dt", { text: k }), el("dd", { class: cls, text: v })];
  $("env").replaceChildren(
    ...kv("llb", e.llb),
    ...kv("python", `${e.python} (${e.python_path})`),
    ...kv("machine", e.machine),
    ...kv("system", e.system),
    ...kv("cpus", e.cpus ?? "-"),
  );
  const extras = el("dd", {}, e.extras.map((x) => el("span", { class: `tag ${x.installed ? "ok" : "muted"}`, title: x.modules.join(", "), text: `${x.installed ? "✓" : "✗"} ${x.name}` })));
  const programs = el("dd", {}, Object.entries(e.programs).map(([name, path]) => el("span", { class: `tag ${path ? "ok" : "muted"}`, title: path || "not found in PATH", text: `${path ? "✓" : "✗"} ${name}` })));
  $("extras").replaceChildren(
    el("dt", { text: "extras" }), extras,
    el("dt", { text: "build tools" }), programs,
  );
  const missing = e.extras.filter((x) => !x.installed).map((x) => `--extra ${x.name}`);
  if (missing.length) $("extras").append(el("dt", { text: "install" }), el("dd", {}, el("code", { text: `uv sync ${missing.join(" ")}` })));
  $("t-storage").tBodies[0].replaceChildren(
    ...e.locations.map((l) =>
      el("tr", {},
        el("td", { text: l.label }),
        el("td", { class: "wrapcell path" }, el("span", { class: l.exists ? "" : "muted", text: l.path }), l.exists ? null : el("span", { class: "muted", text: " (not created yet)" })),
        el("td", { class: "num", text: l.used == null ? "" : gb(l.used) }),
        el("td", { class: `num ${l.free != null && l.free < 5e9 ? "warn" : ""}`, text: l.free == null ? "-" : `${gb(l.free)} of ${gb(l.disk_total)}` }),
      ),
    ),
  );
}

function extraCmake(b) {
  if (!buildOptions || !b.backend) return Object.entries(b.cmake);
  const defaults = { ...buildOptions.base_cmake, ...(buildOptions.backend_cmake[b.backend] || {}) };
  return Object.entries(b.cmake).filter(([k, v]) => defaults[k] !== v);
}

async function loadBuilds() {
  const data = await act(() => api("GET", "/lab/api/system/builds"));
  if (!data) return;
  const first = buildOptions === null;
  buildOptions = data.options;
  if (first) initForm();
  const body = $("t-builds").tBodies[0];
  body.replaceChildren();
  if (!data.builds.length) {
    body.append(el("tr", {}, el("td", { colspan: 9, class: "muted", text: "No llama.cpp build yet: build one below." })));
  }
  for (const b of data.builds) {
    const extra = extraCmake(b);
    body.append(
      el("tr", {},
        el("td", { class: "dir" }, el("b", { text: b.id }), el("div", { class: "muted", title: "bin dir", text: b.bin_dir })),
        el("td", { text: b.ref || (b.managed ? "-" : "given with --llama-server") }),
        el("td", { text: b.commit ? b.commit.slice(0, 12) : "-" }),
        el("td", { text: b.version || "-" }),
        el("td", { text: b.backend || "-" }),
        el("td", { class: "cmake" },
          extra.length ? extra.map(([k, v]) => el("span", { class: "tag v", text: `${k}=${v}` })) : el("span", { class: "muted", text: b.managed ? "defaults" : "-" }),
          Object.keys(b.cmake).length ? el("details", {}, el("summary", { text: "all" }), el("pre", { text: Object.entries(b.cmake).map(([k, v]) => `-D${k}=${v}`).join("\n") })) : null,
        ),
        el("td", { class: "tools" }, Object.entries(b.tools).map(([t, ok]) => el("span", { class: `tag ${ok ? "" : "no"}`, title: ok ? "present" : "missing", text: t.replace("llama-", "") }))),
        el("td", { class: "muted", text: b.built_at ? b.built_at.replace("T", " ").slice(0, 16) : "-" }),
        el("td", {},
          b.tools["llama-bench"] ? el("button", { class: "btn small", onclick: () => showFlags(b.id, "llama-bench"), text: "bench flags" }) : null, " ",
          b.tools["llama-server"] ? el("button", { class: "btn small", onclick: () => showFlags(b.id, "llama-server"), text: "server flags" }) : null,
        ),
      ),
    );
  }
  const sel = $("f-build");
  const was = sel.value;
  sel.replaceChildren(...data.builds.map((b) => el("option", { value: b.id, text: `${b.id}${b.version ? ` · ${b.version}` : ""}` })));
  if (!data.builds.length) sel.append(el("option", { value: "", text: "no builds yet" }));
  if (data.builds.some((b) => b.id === was)) sel.value = was;
}

// -- build form -----------------------------------------------------------------------

function initForm() {
  $("b-backend").replaceChildren(...buildOptions.backends.map((b) => el("option", { value: b, text: b })));
  $("b-ref").value = buildOptions.default_ref;
  $("b-jobs").placeholder = String(buildOptions.cpus || "auto");
  for (const id of ["b-ref", "b-backend", "b-cmake", "b-jobs", "b-force"]) {
    $(id).addEventListener("input", renderCommand);
    $(id).addEventListener("change", renderCommand);
  }
  renderCommand();
}

function buildRequest() {
  const cmake = $("b-cmake").value.split("\n").map((s) => s.trim()).filter(Boolean);
  return { ref: $("b-ref").value.trim(), backend: $("b-backend").value, cmake, jobs: intOrNull($("b-jobs").value), force: $("b-force").checked };
}

function renderCommand() {
  const r = buildRequest();
  const parts = ["llb llama build", `--ref ${r.ref || "<ref>"}`, `--backend ${r.backend}`, ...r.cmake.map((d) => `-D ${d}`)];
  if (r.jobs) parts.push(`--jobs ${r.jobs}`);
  if (r.force) parts.push("--force");
  $("b-cmd").textContent = parts.join(" ");
  const defaults = { ...buildOptions.base_cmake, ...(buildOptions.backend_cmake[r.backend] || {}) };
  $("b-defaults").textContent = `always set: ${Object.entries(defaults).map(([k, v]) => `${k}=${v}`).join(" ")} (a define above overrides them)`;
}

async function build(ev) {
  ev.preventDefault();
  const r = buildRequest();
  if (!r.ref) return say("Give a tag, branch or commit (e.g. b6500).");
  const out = await act(() => api("POST", "/lab/api/system/build", r), `building llama.cpp ${r.ref} (${r.backend})`);
  if (out) {
    if (out.stopped_chat) say("The chat was unloaded to free the machine.", true);
    poll();
  }
}

// -- flags ----------------------------------------------------------------------------

let lastFlags = null;

async function showFlags(build, tool) {
  if (build) $("f-build").value = build;
  if (tool) $("f-tool").value = tool;
  const id = $("f-build").value;
  if (!id) return say("No build to read flags from yet.");
  const box = $("flags-out");
  box.replaceChildren(el("p", { class: "muted", text: `reading ${$("f-tool").value} --help…` }));
  if (build) box.scrollIntoView({ behavior: "smooth", block: "start" });
  const q = new URLSearchParams({ build: id, tool: $("f-tool").value, raw: $("f-raw").checked ? "true" : "false" });
  const data = await act(() => api("GET", `/lab/api/system/flags?${q}`));
  box.replaceChildren();
  if (!data) return;
  lastFlags = data;
  const filter = el("input", { class: "filter", placeholder: "filter flags", "aria-label": "filter flags", autocomplete: "off", oninput: renderFlags });
  box.append(
    el("div", { class: "panel" },
      el("div", { class: "toolbar" },
        el("span", {}, el("b", { text: data.tool }), el("span", { class: "muted", text: ` ${data.version || ""} · ${data.binary}` })),
      ),
      el("div", { class: "toolbar" }, filter, el("span", { class: "count", id: "flags-count" })),
      el("div", { class: "flags", id: "flags-list" }),
      data.help != null ? el("details", {}, el("summary", { text: "--help output" }), el("pre", { class: "help-text", text: data.help })) : null,
    ),
  );
  renderFlags();
  filter.focus();
}

function renderFlags() {
  const input = document.querySelector("#flags-out input.filter");
  const q = (input ? input.value : "").trim().toLowerCase();
  const shown = lastFlags.flags.filter((f) => !q || f.toLowerCase().includes(q));
  $("flags-list").replaceChildren(...shown.map((f) => el("span", { class: "tag", text: f })));
  $("flags-count").textContent = `${shown.length} of ${lastFlags.flags.length} flags`;
}

document.addEventListener("DOMContentLoaded", () => {
  $("build-form").addEventListener("submit", build);
  $("flags-form").addEventListener("submit", (e) => {
    e.preventDefault();
    showFlags();
  });
  Lab.onStatus((s, ev) => {
    jobRunning = Boolean(s.job && s.job.running);
    $("b-run").disabled = jobRunning;
    $("b-run").title = jobRunning ? "a job is running" : "";
    if (ev && ev.jobFinished) {
      loadBuilds();
      loadEnv();
    }
  });
  loadEnv();
  loadBuilds();
});
