// llb lab: a form generated from a pydantic JSON schema, bound to a YAML document.
// Pages set `Form.defs`, `Form.doc`, `Form.change(path, value, opts)` and optionally
// `Form.help(path, prop)` and `Form.paths` (choices for path lists, e.g. suites).
// `Form.multiline(name, path, kind)` may return true to edit a string ("string"), a list of
// strings ("list", one per line) or a free-form mapping ("object", as JSON) in a textarea.
// Discriminated unions (`oneOf` + `discriminator`, e.g. evaluators) get a type selector.
// Every edit goes through `Form.change`, which patches the YAML on the server.
"use strict";

const Form = {
  defs: {},
  doc: {},
  change: async () => {},
  help: (path, prop) => prop.description,
  paths: {},
  multiline: () => false,
  openSections: new Set(),
};

// -- schema helpers -----------------------------------------------------------------

function resolve(s) {
  while (s && s.$ref) s = Form.defs[s.$ref.split("/").pop()];
  return s || {};
}

function unwrap(prop) {
  let s = resolve(prop);
  let nullable = false;
  if (s.anyOf) {
    const opts = s.anyOf.filter((o) => o.type !== "null");
    nullable = opts.length < s.anyOf.length;
    s = opts.length === 1 ? resolve(opts[0]) : { anyOf: opts };
  }
  return { s, nullable };
}

const getAt = (obj, path) => path.reduce((o, k) => (o == null ? undefined : o[k]), obj);
const show = (v) => (v === null || v === undefined ? "" : typeof v === "object" ? JSON.stringify(v) : String(v));

// -- generic fields -------------------------------------------------------------------

function fieldLabel(name, path, prop, input, wide) {
  const help = Form.help(path, prop);
  const set = getAt(Form.doc, path) !== undefined;
  return el("label", { class: `f${wide ? " wide" : ""}` }, el("span", { class: `k${set ? " set" : ""}`, text: name }), input, help ? el("small", { text: help }) : null);
}

function defaultLabel(def) {
  return def === undefined || def === null ? "(default)" : `(default: ${show(def)})`;
}

function renderField(box, name, prop, path) {
  const { s } = unwrap(prop);
  const cur = getAt(Form.doc, path);
  const def = prop.default !== undefined ? prop.default : s.default;
  if (s.oneOf && s.discriminator) {
    renderUnion(box, name, prop, s, path, unwrap(prop).nullable);
  } else if (s.anyOf && s.anyOf.every((o) => resolve(o).type === "string")) {
    stringField(box, name, prop, path, cur, def);
  } else if (s.enum) {
    const sel = el("select", {}, el("option", { value: "", text: defaultLabel(def) }), s.enum.map((v) => el("option", { value: String(v), text: String(v) })));
    sel.value = cur === undefined || cur === null ? "" : String(cur);
    sel.addEventListener("change", () => Form.change(path, sel.value === "" ? null : s.enum.find((v) => String(v) === sel.value)));
    box.append(fieldLabel(name, path, prop, sel));
  } else if (s.type === "boolean") {
    const sel = el("select", {}, el("option", { value: "", text: defaultLabel(def) }), el("option", { value: "true", text: "true" }), el("option", { value: "false", text: "false" }));
    sel.value = cur === undefined || cur === null ? "" : String(cur);
    sel.addEventListener("change", () => Form.change(path, sel.value === "" ? null : sel.value === "true"));
    box.append(fieldLabel(name, path, prop, sel));
  } else if (s.type === "integer" || s.type === "number") {
    const input = el("input", { type: "number", step: s.type === "integer" ? "1" : "any", placeholder: show(def), value: show(cur) });
    input.addEventListener("change", () => Form.change(path, input.value === "" ? null : s.type === "integer" ? parseInt(input.value, 10) : parseFloat(input.value)));
    box.append(fieldLabel(name, path, prop, input));
  } else if (s.type === "string") {
    stringField(box, name, prop, path, cur, def);
  } else if (s.type === "array") {
    renderArray(box, name, prop, s, path, cur, def);
  } else if (s.type === "object" && s.properties) {
    const sub = el("details", { class: "sub wide", open: cur !== undefined }, el("summary", { text: name }));
    if (s.description) sub.append(el("p", { class: "hint", text: s.description }));
    renderObject(sub, s, path);
    box.append(sub);
  } else if (s.type === "object" && Form.multiline(name, path, "object")) {
    const area = el("textarea", { rows: 4, placeholder: "JSON", value: cur === undefined ? "" : JSON.stringify(cur, null, 2) });
    area.addEventListener("change", () => {
      if (area.value.trim() === "") return Form.change(path, null);
      let v;
      try {
        v = JSON.parse(area.value);
      } catch (e) {
        return say(`${name}: not valid JSON (${e.message})`);
      }
      Form.change(path, v);
    });
    box.append(fieldLabel(name, path, prop, area, true));
  } else {
    yamlOnly(box, name, cur, prop.description || s.description);
  }
}

function stringField(box, name, prop, path, cur, def) {
  const multi = Form.multiline(name, path, "string");
  const input = multi ? el("textarea", { rows: 3, placeholder: show(def), value: show(cur) }) : el("input", { placeholder: show(def), value: show(cur) });
  input.addEventListener("change", () => Form.change(path, input.value === "" ? null : input.value));
  box.append(fieldLabel(name, path, prop, input, multi || name === "description" || name === "source"));
}

// -- discriminated unions (evaluators, distributions) ----------------------------------

function variants(s) {
  return Object.entries(s.discriminator.mapping).map(([tag, ref]) => ({ tag, schema: resolve({ $ref: ref }) }));
}

function variantOf(s, cur) {
  const key = s.discriminator.propertyName;
  const all = variants(s);
  if (cur && typeof cur === "object") return all.find((v) => v.tag === cur[key]) || all.find((v) => cur[key] === undefined && getAt(v.schema, ["properties", key, "default"]) === v.tag);
  // a scalar shorthand (e.g. `prompt_tokens: 512`) stands for the variant with a default tag
  if (cur !== undefined && cur !== null) return all.find((v) => getAt(v.schema, ["properties", key, "default"]) === v.tag);
  return undefined;
}

function renderUnion(box, name, prop, s, path, nullable, head) {
  const key = s.discriminator.propertyName;
  const cur = getAt(Form.doc, path);
  const current = variantOf(s, cur);
  const sel = el("select", {}, cur === undefined || cur === null || nullable ? el("option", { value: "", text: "(none)" }) : null, variants(s).map((v) => el("option", { value: v.tag, text: v.tag })));
  sel.value = current ? current.tag : "";
  sel.addEventListener("change", () => {
    const v = variants(s).find((x) => x.tag === sel.value);
    Form.change(path, v ? switchVariant(key, v, cur) : null, { rerender: true });
  });
  const wrap = el("div", { class: "union wide" }, head || null);
  const top = el("div", { class: "fields" });
  top.append(fieldLabel(name, path, prop, sel));
  wrap.append(top);
  if (current) {
    if (current.schema.description) wrap.append(el("p", { class: "hint", text: current.schema.description }));
    if (cur && typeof cur === "object") renderObject(wrap, current.schema, path, [key]);
    else {
      const input = el("input", { type: "number", value: show(cur) });
      input.addEventListener("change", () => Form.change(path, input.value === "" ? null : Number(input.value)));
      const grid = el("div", { class: "fields" });
      grid.append(fieldLabel(`${name} (${current.tag})`, path, { description: `short for { ${key}: ${current.tag}, value: ${show(cur)} }` }, input));
      wrap.append(grid);
    }
  }
  box.append(wrap);
}

// The new variant keeps the fields it shares with the old value (e.g. `values` from contains
// to not_contains); a lone number (`value: 512`, or the `512` shorthand) seeds its required
// sizes (uniform min/max, normal mean), so switching does not start from nonsense.
function switchVariant(key, v, old) {
  const out = { [key]: v.tag, ...template(v.schema) };
  const prev = old && typeof old === "object" ? old : {};
  for (const [k, val] of Object.entries(prev)) if (k !== key && v.schema.properties[k]) out[k] = val;
  const seed = typeof old === "number" ? old : typeof prev.value === "number" ? prev.value : null;
  if (seed !== null) {
    for (const k of v.schema.required || []) {
      const { s } = unwrap(v.schema.properties[k]);
      // only sizes (> 0): a spread such as normal's std (>= 0) starts at its minimum
      if ((s.type === "integer" || s.type === "number") && s.exclusiveMinimum === 0 && !(k in prev)) out[k] = seed;
    }
  }
  return out;
}

function renderUnionArray(box, name, prop, s, path, cur) {
  const wrap = el("div", { class: "wide" }, el("span", { class: "k", text: name }));
  if (prop.description) wrap.append(el("small", { class: "hint", text: ` ${prop.description}` }));
  (cur || []).forEach((_, i) => {
    const head = el("div", { class: "head" }, el("span", { text: `${name} #${i + 1}` }), el("button", { class: "btn small red", type: "button", onclick: () => Form.change([...path, i], null, { rerender: true }), text: "remove" }));
    const div = el("div", { class: "item" });
    renderUnion(div, "type", {}, s, [...path, i], false, head);
    wrap.append(div);
  });
  const first = variants(s)[0];
  wrap.append(el("button", { class: "btn small", type: "button", onclick: () => Form.change([...path, (cur || []).length], { [s.discriminator.propertyName]: first.tag, ...template(first.schema) }, { rerender: true }), text: `+ add to ${name}` }));
  box.append(wrap);
}

function yamlOnly(box, name, cur, help) {
  box.append(
    el("div", { class: "yamlonly wide" },
      el("span", { class: "k", text: `${name}: ` }), cur === undefined ? "not set" : el("code", { text: show(cur) }), " · edit it in the YAML",
      help ? el("div", { text: help }) : null,
    ),
  );
}

function scalarList(items, raw, sep) {
  const parts = raw.split(sep).map((x) => x.trim()).filter((x) => x !== "");
  if (!parts.length) return null;
  if (items.type === "integer") return parts.map((x) => parseInt(x, 10));
  if (items.type === "number") return parts.map((x) => parseFloat(x));
  return parts;
}

function renderArray(box, name, prop, s, path, cur, def) {
  const items = resolve(s.items || {});
  if (items.format === "path" && Form.paths[name]) {
    const chosen = new Set(cur || []);
    const all = [...new Set([...Form.paths[name], ...chosen])];
    const checks = el("div", { class: "checks" }, all.map((suite) => {
      const cb = el("input", { type: "checkbox", value: suite, checked: chosen.has(suite) });
      return el("label", {}, cb, suite.replace(/^(\.\.\/)+suites\//, ""));
    }));
    checks.addEventListener("change", () => {
      const picked = [...checks.querySelectorAll("input:checked")].map((c) => c.value);
      Form.change(path, picked.length ? picked : null);
    });
    box.append(fieldLabel(name, path, prop, checks, true));
  } else if (items.enum) {
    const chosen = new Set(cur || def || []);
    const checks = el("div", { class: "checks" }, items.enum.map((v) => el("label", {}, el("input", { type: "checkbox", value: v, checked: chosen.has(v) }), v)));
    checks.addEventListener("change", () => {
      const picked = [...checks.querySelectorAll("input:checked")].map((c) => c.value);
      Form.change(path, picked.length ? picked : null);
    });
    box.append(fieldLabel(name, path, prop, checks, true));
  } else if (items.oneOf && items.discriminator) {
    renderUnionArray(box, name, prop, items, path, cur);
  } else if (items.type === "string" && Form.multiline(name, path, "list")) {
    const area = el("textarea", { rows: Math.max(2, Math.min(8, (cur || []).length + 1)), placeholder: def && def.length ? def.join("\n") : "one per line", value: cur ? cur.join("\n") : "" });
    area.addEventListener("change", () => Form.change(path, scalarList(items, area.value, "\n")));
    box.append(fieldLabel(name, path, prop, area, true));
  } else if (["integer", "number", "string"].includes(items.type)) {
    const spaces = path.includes("extra_args");
    const sep = spaces ? /\s+/ : ",";
    const input = el("input", { placeholder: def && def.length ? def.join(spaces ? " " : ", ") : spaces ? "flags separated by spaces" : "comma-separated", value: cur ? cur.join(spaces ? " " : ", ") : "" });
    input.addEventListener("change", () => Form.change(path, scalarList(items, input.value, sep)));
    box.append(fieldLabel(name, path, prop, input, spaces));
  } else if (items.properties) {
    renderObjectArray(box, name, path, items, cur);
  } else {
    yamlOnly(box, name, cur, prop.description);
  }
}

function renderObject(box, schema, path, skip = []) {
  const grid = el("div", { class: "fields" });
  for (const [name, prop] of Object.entries(schema.properties || {})) {
    if (!skip.includes(name)) renderField(grid, name, prop, [...path, name]);
  }
  box.append(grid);
}

function template(schema) {
  const out = {};
  for (const name of schema.required || []) {
    const { s } = unwrap(schema.properties[name]);
    if (s.const !== undefined) out[name] = s.const;
    else if (s.enum) out[name] = s.enum[0];
    else if (s.type === "number" || s.type === "integer") out[name] = s.minimum !== undefined ? s.minimum : s.exclusiveMinimum !== undefined ? s.exclusiveMinimum + 1 : 0;
    else if (s.oneOf && s.discriminator) {
      const v = variants(s)[0];
      out[name] = { [s.discriminator.propertyName]: v.tag, ...template(v.schema) };
    } else if (s.type === "object" && s.properties) out[name] = template(s);
    else if (s.type === "object") out[name] = {};
    else if (s.type === "array") out[name] = [];
    else out[name] = "";
  }
  return out;
}

function renderObjectArray(box, name, path, items, cur, extraAdd) {
  const wrap = el("div", { class: "wide" });
  (cur || []).forEach((item, i) => {
    const head = el("div", { class: "head" }, el("span", { text: `${name} #${i + 1}` }), el("button", { class: "btn small red", onclick: () => Form.change([...path, i], null, { rerender: true }), text: "remove" }));
    const div = el("div", { class: "item" }, head);
    renderObject(div, items, [...path, i]);
    wrap.append(div);
  });
  wrap.append(extraAdd || el("button", { class: "btn small", onclick: () => Form.change([...path, (cur || []).length], template(items), { rerender: true }), text: `+ add ${name.replace(/s$/, "")}` }));
  box.append(wrap);
}

// -- layout -----------------------------------------------------------------------------

function section(key, title, help, fill) {
  const d = el("details", { class: "sec", open: Form.openSections.has(key), ontoggle: (e) => (e.target.open ? Form.openSections.add(key) : Form.openSections.delete(key)) }, el("summary", { text: title }));
  if (help) d.append(el("p", { class: "hint", text: help }));
  fill(d);
  return d;
}

function tabs(names, current, label, onPick) {
  return el("div", { class: "tabsel" }, names.map((n) => el("button", { type: "button", class: n === current ? "on" : "", onclick: () => onPick(n), text: label(n) })));
}

const slug = (s) => s.toLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/^[-.]+|[-.]+$/g, "").slice(0, 80) || "model";
