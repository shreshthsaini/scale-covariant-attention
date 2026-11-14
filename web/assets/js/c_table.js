/* SCAComp.table(el, options): the paper's main comparison (Table 1) and generality table (Table 3) as an
   interactive table: sortable columns, our rows shaded, metric groups toggled, the Holm record drawn as ten cells.
   Vanilla JS, no dependencies, works from file://.

   Data: assets/data/table/data.js (window.SCA_DATA.table), written by web/build/v5_build_table.py from the paper's
   generated table snippets (cross-checked against results/). Cells print exactly as in the paper: bold = best,
   underline = second per group, double dagger = campaign mean. No number is typed here.

   var c = SCAComp.table(el, {tab: "main" | "gen", hide: ["set"], lazy: true, base: "<data dir url>"});
   options  tab      tab shown first (default "main")
            compact  true: Table 1 only, with Method, Seconds, ImageReward, HPSv2.1, PickScore and the record; no tabs,
                     no group chips, no sorting
            hide  column groups hidden at first (main: "time", "pref", "detail", "set", "holm")
   Interaction: tabs; group chips show or hide columns; a column header sorts best first, then worst first, then
   back to the paper's order. */
(function () {
  "use strict";
  var NAME = "table", P = "c-table-";
  var SCAComp = window.SCAComp = window.SCAComp || {};
  var CUR = document.currentScript && document.currentScript.src;
  var JSDIR = CUR ? CUR.replace(/[^\/]*$/, "") : "assets/js/";
  var DEF_BASE = JSDIR.replace(/js\/$/, "") + "data/" + NAME + "/";
  var UID = 0;

  function el(tag, cls, attrs, kids) {
    var e = document.createElement(tag);
    if (cls) e.className = cls.split(" ").map(function (c) { return P + c; }).join(" ");
    if (attrs) Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (v === null || v === undefined || v === false) return;
      if (k === "text") e.textContent = v; else if (k === "html") e.innerHTML = v; else e.setAttribute(k, v === true ? "" : v);
    });
    (kids || []).forEach(function (c) { if (c) e.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return e;
  }
  function loadScript(src, ok, bad) { var s = document.createElement("script"); s.src = src; s.async = true; s.onload = ok; s.onerror = bad; document.head.appendChild(s); }
  function whenNear(node, cb) {
    if (!("IntersectionObserver" in window)) { cb(); return null; }
    var io = new IntersectionObserver(function (es) {
      for (var i = 0; i < es.length; i++) if (es[i].isIntersecting) { io.disconnect(); cb(); return; }
    }, { rootMargin: "900px 0px" });
    io.observe(node);
    return io;
  }

  function sub(node, text) {             /* "M_c" -> M<sub>c</sub>; the rest as text */
    var parts = String(text).split(/_([A-Za-z0-9]+)/);
    parts.forEach(function (q, i) { if (!q) return; node.appendChild(i % 2 ? el("sub", null, { text: q }) : document.createTextNode(q)); });
    return node;
  }

  function mount(root, opts) {
    opts = opts || {};
    var base = opts.base || DEF_BASE;
    var D = null, alive = true, io = null, ro = null, tab = opts.tab === "gen" ? "gen" : "main";
    var hidden = {}, sort = { k: null, dir: 0 };
    (opts.hide || []).forEach(function (g) { hidden[g] = true; });
    if (!opts.hide && !opts.compact && (root.clientWidth || 800) < 560) { hidden.detail = true; hidden.set = true; }   /* phones: fewer columns at first */
    var compact = !!opts.compact;
    if (compact) { tab = "main"; root.classList.add(P + "compact"); }
    root.classList.add("c-" + NAME);
    root.innerHTML = "";

    var tabs = el("div", "tabs", { role: "tablist", "aria-label": "Table" });
    var chips = el("div", "chips", { role: "group", "aria-label": "Columns shown" });
    var top = el("div", "top", null, [tabs, chips]);
    var wrap = el("div", "wrap", { tabindex: "0", role: "region", "aria-label": "Results table" });
    var tableEl = el("table", "t");
    wrap.appendChild(tableEl);
    var note = el("p", "note");
    var live = el("p", "sr", { "aria-live": "polite" });
    root.appendChild(top); root.appendChild(wrap); root.appendChild(note); root.appendChild(live);
    root.classList.add(P + "loading");          /* reserves the final heights until the data arrives (no layout shift) */
    for (var s = 0; s < 9; s++) tableEl.appendChild(el("tr", "skel-r", null, [el("td", null, { html: "&nbsp;" })]));

    function fail() {
      if (!alive) return;
      root.classList.remove(P + "loading");
      var fig = root.closest ? root.closest("figure") : null;
      if (fig && fig.querySelector(".comp-fallback")) { fig.classList.remove("is-live"); root.innerHTML = ""; return; }
      root.innerHTML = "";
      root.appendChild(el("p", "note", { text: "The table could not load; the paper's Table 1 has the same numbers." }));
    }
    function start() {
      if (window.SCA_DATA && window.SCA_DATA[NAME]) { ready(); return; }
      loadScript(base + "data.js", function () { if (window.SCA_DATA && window.SCA_DATA[NAME]) ready(); else fail(); }, fail);
    }
    if (opts.lazy === false) start(); else io = whenNear(root, start);

    function ready() {
      if (!alive) return;
      D = window.SCA_DATA[NAME];
      if (compact) { top.hidden = true; setTab("main", true); root.classList.remove(P + "loading"); return; }
      [["main", D.main.title], ["gen", D.gen.title]].forEach(function (q) {
        var b = el("button", "tab", { type: "button", role: "tab", id: P + q[0] + "-" + (++UID), "aria-selected": String(q[0] === tab), text: q[1] });
        b.addEventListener("click", function () { setTab(q[0]); });
        tabs.appendChild(b);
      });
      tabs.addEventListener("keydown", function (e) {
        if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
        e.preventDefault(); setTab(tab === "main" ? "gen" : "main"); tabs.querySelector('[aria-selected="true"]').focus();
      });
      setTab(tab, true);
      root.classList.remove(P + "loading");
    }
    function T() { return D[tab]; }
    function setTab(t, first) {
      tab = t; sort = { k: null, dir: 0 };
      Array.prototype.forEach.call(tabs.children, function (b, i) { var on = (i === 0) === (t === "main"); b.setAttribute("aria-selected", String(on)); b.tabIndex = on ? 0 : -1; });
      wrap.setAttribute("aria-label", T().title);
      if (!compact) buildChips();
      render();
      note.innerHTML = ""; sub(note, compact && T().compact ? T().compact.note : T().note);
    }
    function buildChips() {
      chips.innerHTML = "";
      T().colgroups.forEach(function (g) {
        if (!g.label) return;
        var b = el("button", "chip", { type: "button", "aria-pressed": String(!hidden[g.k]), text: g.label });
        b.addEventListener("click", function () {
          var vis = T().colgroups.filter(function (x) { return x.label && !hidden[x.k]; });
          if (!hidden[g.k] && vis.length <= 1) return;                    /* keep at least one group */
          hidden[g.k] = !hidden[g.k]; b.setAttribute("aria-pressed", String(!hidden[g.k])); render();
        });
        chips.appendChild(b);
      });
    }
    function cols() {
      if (compact && T().compact) { var keep = T().compact.cols; return T().cols.filter(function (c) { return keep.indexOf(c.k) >= 0; }); }
      return T().cols.filter(function (c) { return !hidden[c.g]; });
    }
    function keyOf(r, c) {
      var x = r.cells[c.k];
      if (!x) return null;
      if (c.k === "holm") return x.plus === undefined ? null : x.plus * 100 - x.minus;
      return x.v === undefined ? null : x.v;
    }
    function ordered() {
      var rows = T().rows.slice();
      if (!sort.k) return rows;
      var c = T().cols.filter(function (q) { return q.k === sort.k; })[0], sgn = (c.dir || 1) * sort.dir;
      return rows.map(function (r, i) { return [r, i]; }).sort(function (a, b) {
        var x = keyOf(a[0], c), y = keyOf(b[0], c);
        if (x === null && y === null) return a[1] - b[1];
        if (x === null) return 1; if (y === null) return -1;
        return (y - x) * sgn || a[1] - b[1];
      }).map(function (q) { return q[0]; });
    }
    function holmCell(x) {
      if (x.plus === undefined) return el("span", "ref", { text: x.t });
      var box = el("span", "holm", { role: "img", "aria-label": "better on " + x.plus + ", worse on " + x.minus + " of ten" });
      var bars = el("span", "holm-b", { "aria-hidden": "true" });
      for (var i = 0; i < 10; i++) bars.appendChild(el("i", i < x.plus ? "hp" : i < x.plus + x.minus ? "hm" : "h0"));
      box.appendChild(bars);
      box.appendChild(el("span", "holm-t", { "aria-hidden": "true", text: x.t }));
      return box;
    }
    function render() {
      var C = cols(), t = T();
      tableEl.innerHTML = "";
      var cg = el("colgroup"); cg.appendChild(el("col", "col-name"));
      C.forEach(function (c) { cg.appendChild(el("col", "col-" + c.g)); }); tableEl.appendChild(cg);
      var thead = el("thead"), r1 = el("tr", "gh"), r2 = el("tr", "ch");
      if (compact && t.compact) {             /* one plain header row: full names, no sorting */
        r2.appendChild(el("th", "corner corner-c", { scope: "col", text: "Method" }));
        C.forEach(function (c) { r2.appendChild(el("th", "c c-" + c.k + " cc", { scope: "col", text: t.compact.labels[c.k] || c.label })); });
        thead.appendChild(r2); tableEl.appendChild(thead);
      } else {
      r1.appendChild(el("th", "corner", { scope: "col", rowspan: "2", text: tab === "main" ? "Method" : "Setting" }));
      t.colgroups.forEach(function (g) {
        var n = C.filter(function (c) { return c.g === g.k; }).length;
        if (n) r1.appendChild(el("th", "gl g-" + g.k, { scope: "colgroup", colspan: String(n), text: g.label || "" }));
      });
      C.forEach(function (c, i) {
        var on = sort.k === c.k, prev = C[i - 1];
        var th = el("th", "c c-" + c.k + (prev && prev.g !== c.g ? " gs" : ""), { scope: "col", "aria-sort": on ? (sort.dir > 0 ? (c.dir < 0 ? "ascending" : "descending") : (c.dir < 0 ? "descending" : "ascending")) : null });
        var b = el("button", "sort" + (on ? " on" : ""), { type: "button", title: "Sort by " + c.label + (c.dir ? (c.dir > 0 ? " (higher is better)" : " (lower is better)") : "") }, [
          sub(el("span"), c.label), el("span", "arrow", { "aria-hidden": "true", text: on ? (sort.dir > 0 ? "\u25be" : "\u25b4") : c.dir ? (c.dir > 0 ? "\u2191" : "\u2193") : "" })]);
        b.addEventListener("click", function () {
          if (sort.k !== c.k) sort = { k: c.k, dir: 1 }; else if (sort.dir === 1) sort.dir = -1; else sort = { k: null, dir: 0 };
          render();
          live.textContent = sort.k ? "Sorted by " + c.label + (sort.dir > 0 ? ", best first" : ", worst first") : "Paper order";
          var again = tableEl.querySelector("." + P + "c-" + c.k + " button"); if (again) again.focus();
        });
        th.appendChild(b); r2.appendChild(th);
      });
      thead.appendChild(r1); thead.appendChild(r2); tableEl.appendChild(thead);
      }
      var tb = el("tbody"), last = null, flat = !!sort.k;
      ordered().forEach(function (r) {
        if (!flat && r.group !== last) {
          last = r.group;
          tb.appendChild(el("tr", "grp", null, [el("th", null, { scope: "colgroup", colspan: String(C.length + 1) }, [el("span", "grp-t", { text: r.group })])]));
        }
        var tr = el("tr", "r" + (r.ours ? " ours" : "") + (r.id ? " m-" + r.id : ""));
        var nm = el("th", "name", { scope: "row" }, [el("span", "nm" + (r.name_b ? " nb" : ""), { text: r.name })]);
        if (flat && r.group) nm.appendChild(el("span", "tag", { text: r.group.split(/[,(]/)[0].trim() }));
        tr.appendChild(nm);
        C.forEach(function (c, i) {
          var x = r.cells[c.k] || { t: "" }, prev = C[i - 1];
          var td = el("td", "v v-" + c.k + (x.b ? " b" : "") + (x.u ? " u" : "") + (x.sig ? " sig-" + x.sig : "") + (prev && prev.g !== c.g ? " gs" : ""));
          if (c.k === "holm") td.appendChild(holmCell(x));
          else {
            td.appendChild(el("span", "val", { text: x.t }));
            if (x.camp) td.appendChild(el("sup", "camp", { title: "campaign mean on shared nodes (no clean time)", text: "\u2021" }));
          }
          tr.appendChild(td);
        });
        tb.appendChild(tr);
      });
      tableEl.appendChild(tb);
      overflow();
    }
    function overflow() { root.classList.toggle(P + "scrolls", wrap.scrollWidth > wrap.clientWidth + 2); wrap.style.setProperty("--c-table-vw", wrap.clientWidth + "px"); }
    if ("ResizeObserver" in window) { ro = new ResizeObserver(function () { overflow(); }); ro.observe(wrap); }

    return {
      tab: function (t) { if (D) setTab(t === "gen" ? "gen" : "main"); },
      destroy: function () { alive = false; if (io) io.disconnect(); if (ro) ro.disconnect(); root.innerHTML = ""; root.classList.remove("c-" + NAME); }
    };
  }

  SCAComp[NAME] = mount;
})();
