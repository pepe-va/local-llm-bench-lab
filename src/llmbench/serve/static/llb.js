// llb dashboard: sortable/filterable tables, compare baseline, experiment picker.
// Table state is kept by table id, so tables re-rendered by htmx polling keep their sort/filter.
(() => {
  "use strict";
  const state = {};

  const cellValue = (cell) => {
    if (!cell) return "";
    const raw = cell.dataset.v ?? cell.textContent.trim();
    if (raw === "" || raw === "-") return null;
    const n = Number(raw.replace(/[^0-9eE.+-]/g, ""));
    return /^[\s\[(+-]?[0-9.]/.test(raw) && !Number.isNaN(n) ? n : raw.toLowerCase();
  };

  const compare = (a, b) => {
    if (a === b) return 0;
    if (a === null) return 1; // empty cells always last
    if (b === null) return -1;
    if (typeof a === typeof b) return a < b ? -1 : 1;
    return typeof a === "number" ? -1 : 1;
  };

  function sortTable(table, col, dir) {
    const body = table.tBodies[0];
    if (!body) return;
    const rows = [...body.rows];
    rows.sort((r1, r2) => {
      const c = compare(cellValue(r1.cells[col]), cellValue(r2.cells[col]));
      return dir === "desc" && c !== 0 ? -c : c;
    });
    rows.forEach((r) => body.appendChild(r));
    table.querySelectorAll("thead th").forEach((th, i) => {
      th.dataset.sort = i === col ? dir : "";
    });
  }

  function filterTable(table, query) {
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    let shown = 0;
    const rows = table.tBodies[0] ? [...table.tBodies[0].rows] : [];
    rows.forEach((row) => {
      const text = row.textContent.toLowerCase();
      const ok = terms.every((t) => (t.startsWith("-") && t.length > 1 ? !text.includes(t.slice(1)) : text.includes(t)));
      row.hidden = !ok;
      shown += ok ? 1 : 0;
    });
    document.querySelectorAll(`[data-count="${table.id}"]`).forEach((el) => {
      el.textContent = `${shown}/${rows.length} rows`;
    });
  }

  function apply(table) {
    const s = state[table.id] || {};
    if (s.col !== undefined) sortTable(table, s.col, s.dir);
    const input = document.querySelector(`input[data-filter="${table.id}"]`);
    filterTable(table, input ? input.value : "");
  }

  function init(root) {
    root.querySelectorAll("table.sortable").forEach((table) => {
      if (!table.id) return;
      table.querySelectorAll("thead th").forEach((th, col) => {
        if (th.classList.contains("nosort")) return;
        th.tabIndex = 0;
        const go = () => {
          const s = state[table.id] || {};
          const dir = s.col === col && s.dir === "asc" ? "desc" : "asc";
          state[table.id] = { col, dir };
          sortTable(table, col, dir);
        };
        th.addEventListener("click", go);
        th.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
      });
      apply(table);
    });
  }

  document.addEventListener("input", (e) => {
    const id = e.target.dataset && e.target.dataset.filter;
    const table = id && document.getElementById(id);
    if (table) filterTable(table, e.target.value);
  });

  // --- compare: deltas against the chosen baseline row ---------------------------------
  function deltas() {
    const table = document.getElementById("t-compare");
    if (!table) return;
    const base = table.querySelector("input[name=baseline]:checked");
    const baseRow = base && base.closest("tr");
    table.querySelectorAll("td[data-metric]").forEach((td) => {
      td.querySelector(".delta")?.remove();
      td.closest("tr").classList.toggle("baseline", td.closest("tr") === baseRow);
      if (!baseRow || td.closest("tr") === baseRow) return;
      const ref = baseRow.querySelector(`td[data-metric="${td.dataset.metric}"]`);
      const a = Number(td.dataset.v), b = ref ? Number(ref.dataset.v) : NaN;
      if (!td.dataset.v || !ref || !ref.dataset.v || !b) return;
      const pct = ((a - b) / Math.abs(b)) * 100;
      const better = td.dataset.better === "lower" ? pct < 0 : pct > 0;
      const span = document.createElement("span");
      span.className = "delta " + (Math.abs(pct) < 0.5 ? "same" : better ? "up" : "down");
      span.textContent = `${pct >= 0 ? "+" : ""}${pct.toFixed(1)}%`;
      td.appendChild(span);
    });
  }
  document.addEventListener("change", (e) => {
    if (e.target.name === "baseline") deltas();
    if (e.target.name === "pick") picker();
  });

  // --- index: pick experiments to compare --------------------------------------------------
  function picker() {
    const btn = document.getElementById("compare-btn");
    if (!btn) return;
    const picked = [...document.querySelectorAll("input[name=pick]:checked")].map((i) => i.value);
    btn.disabled = picked.length < 1;
    btn.textContent = picked.length ? `compare ${picked.length} →` : "select experiments to compare";
    btn.dataset.href = "/compare?" + picked.map((v) => "e=" + encodeURIComponent(v)).join("&");
  }
  document.addEventListener("click", (e) => {
    if (e.target.id === "compare-btn" && e.target.dataset.href) location.href = e.target.dataset.href;
  });

  document.addEventListener("DOMContentLoaded", () => { init(document); deltas(); picker(); });
  document.addEventListener("htmx:afterSwap", (e) => init(e.target));
})();
