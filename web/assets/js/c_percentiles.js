/* SCAComp.percentiles(el, options): whole images across the distribution of progressive SCA's ImageReward gain
   over AReA. Vanilla JS, no dependencies beyond assets/js/zoomgrid.js (loaded on demand), works from file://.

   Data: assets/data/percentiles/data.js (window.SCA_DATA.percentiles), written by web/build/v5_build_percentiles.py
   from the paper's per-image files with the rule of its decile figures: nearest-rank 10th, 20th, ..., 90th
   percentile of the paired difference, progressive SCA minus AReA, over the 195 evaluation prompts at seed 0.
   No number is typed here; every value shown comes from that file.

   var c = SCAComp.percentiles(el, {start: 50, lazy: true, base: "<data dir url>"});  c.set(10); c.destroy();
   options  start  percentile shown first (10..90, default 50, the median)
            lazy   load data and images only when the mount comes within 900 px of the viewport (default true)
            base   URL of the data directory (default: ../data/percentiles/ next to this script)
            compact  true: one short line under the images instead of the selection rule (the page prints the rule)
   Interaction: slider, arrow keys, Home/End, click on the strip chart, swipe on the images (whole view);
   click or tap an image to fly all four to native pixels at that point (SCAZoom.grid), drag to pan,
   double-click or Esc to return to the whole images. */
(function () {
  "use strict";
  var NAME = "percentiles", P = "c-percentiles-";
  var SCAComp = window.SCAComp = window.SCAComp || {};
  var CUR = document.currentScript && document.currentScript.src;
  var JSDIR = CUR ? CUR.replace(/[^\/]*$/, "") : "assets/js/";
  var DEF_BASE = JSDIR.replace(/js\/$/, "") + "data/" + NAME + "/";
  var REDUCED = !!(window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches);
  var NARROW = 560;                      /* below this width zoomgrid lays four panels out 2 x 2 */

  function el(tag, cls, attrs, kids) {
    var e = document.createElement(tag);
    if (cls) e.className = cls.split(" ").map(function (c) { return c.charAt(0) === "!" ? c.slice(1) : P + c; }).join(" ");
    if (attrs) Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (v === null || v === undefined || v === false) return;
      if (k === "text") e.textContent = v; else if (k === "html") e.innerHTML = v; else e.setAttribute(k, v === true ? "" : v);
    });
    (kids || []).forEach(function (c) { if (c) e.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return e;
  }
  function svg(tag, attrs) {
    var e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    return e;
  }
  function loadScript(src, ok, bad) {
    var s = document.createElement("script");
    s.src = src; s.async = true;
    s.onload = ok; s.onerror = bad;
    document.head.appendChild(s);
  }
  function loadCss(href) {
    if (document.querySelector('link[href="' + href + '"]')) return;
    var l = document.createElement("link"); l.rel = "stylesheet"; l.href = href; document.head.appendChild(l);
  }
  function whenNear(node, cb) {
    if (!("IntersectionObserver" in window)) { cb(); return null; }
    var io = new IntersectionObserver(function (es) {
      for (var i = 0; i < es.length; i++) if (es[i].isIntersecting) { io.disconnect(); cb(); return; }
    }, { rootMargin: "900px 0px" });
    io.observe(node);
    return io;
  }
  function ordinal(p) { return p + (p % 10 === 1 && p !== 11 ? "st" : p % 10 === 2 && p !== 12 ? "nd" : p % 10 === 3 && p !== 13 ? "rd" : "th"); }
  function num(x, d) { return (x < 0 ? "\u2212" : "") + Math.abs(x).toFixed(d); }
  function signed(x, d) { var s = Math.abs(x).toFixed(d); return (x < 0 ? "\u2212" : "+") + s; }
  function colsFor(w) {                  /* zoomgrid's own layout rule for four panels, maxCols 4, minW 150 */
    var gap = 6, minW = w < NARROW ? 140 : 150;
    var cols = Math.max(1, Math.min(4, Math.floor((w + gap) / (minW + gap))));
    return Math.ceil(4 / Math.ceil(4 / cols));
  }

  function mount(root, opts) {
    opts = opts || {};
    var base = opts.base || DEF_BASE;
    var D = null, cur = -1, grid = null, slot = null, caps = [], mo = null, ro = null, io = null, pending = 0, alive = true;
    var preloaded = {};
    root.classList.add("c-" + NAME);
    root.innerHTML = "";

    /* ------------------------------------------------------------ skeleton (final geometry, no data yet) */
    var head = el("div", "head");
    var stopEl = el("div", "stop", null, [el("span", "kicker", { text: "Paired ImageReward gain" }), el("span", "pct", { text: "\u00a0" })]);
    var diffV = el("span", "diff-v", { text: "\u00a0" });
    var diffL = el("span", "diff-l", { text: "" });
    var pill = el("span", "pill", { text: "" });
    var diffEl = el("div", "diff", null, [diffV, el("span", "diff-t", null, [diffL, pill])]);
    head.appendChild(stopEl); head.appendChild(diffEl);
    var promptT = el("span", "prompt-t");
    var more = el("button", "more", { type: "button", "aria-expanded": "false", text: "Full prompt" });
    var prompt = el("p", "prompt", null, [promptT, more]);
    var chartBox = el("div", "chart", { role: "img" });
    var chartSvg = svg("svg", { "class": P + "svg", viewBox: "0 0 1000 100", preserveAspectRatio: "none", "aria-hidden": "true", focusable: "false" });
    chartBox.appendChild(chartSvg);
    var legend = el("div", "legend", { "aria-hidden": "true" }, [
      el("span", "lg lg-up", { text: "progressive SCA ahead" }), el("span", "lg lg-dn", { text: "AReA ahead" }), el("span", "lg-n", { text: "" })]);
    chartBox.appendChild(legend);
    var mark = el("div", "mark", { "aria-hidden": "true" }, [el("span", "mark-v")]);
    chartBox.appendChild(mark);
    var meanLab = el("span", "mean-l", { "aria-hidden": "true" });
    chartBox.appendChild(meanLab);
    var range = el("input", "range", { type: "range", min: "0", max: "8", step: "1", value: "4", "aria-label": "Percentile of the ImageReward gain", disabled: true });
    var ticks = el("div", "ticks", { "aria-hidden": "true" });
    for (var t = 1; t <= 9; t++) ticks.appendChild(el("span", "tick", { text: ordinal(t * 10), style: "left:" + (t * 10) + "%" }));
    var prevB = el("button", "step step-p", { type: "button", "aria-label": "Previous percentile", disabled: true, html: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14.5 6l-6 6 6 6" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>' });
    var nextB = el("button", "step step-n", { type: "button", "aria-label": "Next percentile", disabled: true, html: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9.5 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>' });
    var slider = el("div", "slider", null, [prevB, range, ticks, nextB]);
    var dist = el("div", "dist", null, [chartBox, slider]);
    var stage = el("div", "stage");
    var skel = el("div", "skel", { "aria-hidden": "true" });
    for (var s = 0; s < 4; s++) skel.appendChild(el("div", "skel-p", null, [el("div", "skel-img"), el("div", "skel-cap")]));
    stage.appendChild(skel);
    stage.appendChild(el("div", "skel-bar", { "aria-hidden": "true" }));
    var rule = el("p", "rule");
    var live = el("p", "sr", { "aria-live": "polite" });
    root.appendChild(head); root.appendChild(prompt); root.appendChild(dist); root.appendChild(stage); root.appendChild(rule); root.appendChild(live);

    function fit() {
      var w = stage.clientWidth || root.clientWidth || 800, cols = colsFor(w);
      root.classList.toggle(P + "narrow", w < NARROW || cols <= 2);
      skel.style.setProperty("--c-pct-cols", cols);
    }
    fit();
    root.classList.add(P + "loading");          /* reserves the final heights until the data arrives (no layout shift) */
    if (opts.compact) root.classList.add(P + "compact");

    /* ------------------------------------------------------------ data */
    function fail() {
      if (!alive) return;
      root.classList.remove(P + "loading");
      var fig = root.closest ? root.closest("figure") : null;
      if (fig && fig.querySelector(".comp-fallback")) { fig.classList.remove("is-live"); root.innerHTML = ""; return; }
      root.innerHTML = "";
      root.appendChild(el("img", "fallback", { src: base + "fallback.webp", alt: "Four whole 4096 by 4096 images of one evaluation prompt: AReA, UltraImage, HRDiT and progressive SCA.", loading: "lazy" }));
    }
    function start() {
      var need = 2, bad = false;
      function one() { if (--need === 0 && !bad) ready(); }
      function no() { if (!bad) { bad = true; fail(); } }
      if (window.SCA_DATA && window.SCA_DATA[NAME]) one(); else loadScript(base + "data.js", function () { if (window.SCA_DATA && window.SCA_DATA[NAME]) one(); else no(); }, no);
      if (window.SCAZoom && window.SCAZoom.grid) one();
      else { loadCss(JSDIR.replace(/js\/$/, "css/") + "zoomgrid.css"); loadScript(JSDIR + "zoomgrid.js", function () { if (window.SCAZoom) one(); else no(); }, no); }
    }
    if (opts.lazy === false) start(); else io = whenNear(root, start);

    function ready() {
      if (!alive) return;
      D = window.SCA_DATA[NAME];
      var M = {}; D.methods.forEach(function (m) { M[m.id] = m; });
      D._m = M;
      drawChart();
      rule.innerHTML = "";
      var camp = D.methods.filter(function (m) { return m.secs_kind !== "clean"; }).map(function (m) { return m.label; });
      if (opts.compact) {
        root.classList.add(P + "compact");
        rule.appendChild(document.createTextNode("Seconds per 4096\u00b2 image" + (camp.length ? " (\u2021 campaign mean)" : "") + ". "));
        rule.appendChild(el("span", "hint", { html: '<span class="' + P + 'h-m">Click an image to zoom all four to native pixels.</span><span class="' + P + 'h-t">Tap an image to zoom all four to native pixels.</span>' }));
      } else {
      var full = el("span", "rule-more", { text:
        " Each stop is the prompt at that nearest-rank decile of the paired ImageReward difference, progressive SCA minus AReA, over the "
        + D.n + " evaluation prompts at seed " + D.seed + ", as in the paper's decile figures; the four images share prompt and initial noise. "
        + "Under each image: its ImageReward (best bold) and the method's seconds per 4096\u00b2 image, clean batch-1 median"
        + (camp.length ? " (\u2021 campaign mean: " + camp.join(", ") + ")" : "") + "." });
      var tog = el("button", "rule-tog", { type: "button", "aria-expanded": "false", text: "How rows are chosen" });
      tog.addEventListener("click", function () { var o = rule.classList.toggle(P + "rule-open"); tog.setAttribute("aria-expanded", String(o)); });
      rule.appendChild(el("span", "rule-short", { text: "Rows by rule: nearest-rank deciles of the gain, " + D.n + " prompts, seed " + D.seed + "." }));
      rule.appendChild(full);
      rule.appendChild(tog);
      rule.appendChild(el("span", "hint", { html: ' <span class="' + P + 'h-m">Click an image to zoom all four to native pixels; drag to pan; double-click or Esc returns.</span><span class="' + P + 'h-t">Tap an image to zoom all four to native pixels; double-tap returns; swipe for the next percentile.</span>' }));
      }
      range.disabled = false;
      var startP = +opts.start || 50, k = 4;
      D.stops.forEach(function (st, i) { if (st.p === startP) k = i; });
      range.value = String(k);
      show(k, true);
      root.classList.remove(P + "loading");
    }

    /* ------------------------------------------------------------ strip chart of all sorted differences */
    var yTop = 0, yBot = 0, hl = null;
    function Y(v) { return (yTop - v) / (yTop - yBot) * 100; }
    function drawChart() {
      var a = D.sorted, n = a.length, lo = Math.min(0, a[0]), hi = Math.max(0, a[n - 1]);
      var pad = (hi - lo) * 0.06;
      yTop = hi + pad; yBot = lo - pad;
      while (chartSvg.firstChild) chartSvg.removeChild(chartSvg.firstChild);
      var w = 1000 / n, pos = "", neg = "";
      for (var i = 0; i < n; i++) {
        var x = (i * w + w * 0.14).toFixed(2), bw = (w * 0.72).toFixed(2), y0 = Y(0), y1 = Y(a[i]);
        var top = Math.min(y0, y1).toFixed(2), h = Math.max(0.6, Math.abs(y1 - y0)).toFixed(2);
        var d = "M" + x + " " + top + "h" + bw + "v" + h + "h-" + bw + "z";
        if (a[i] >= 0) pos += d; else neg += d;
      }
      chartSvg.appendChild(svg("path", { d: neg, "class": P + "bar-neg" }));
      chartSvg.appendChild(svg("path", { d: pos, "class": P + "bar-pos" }));
      chartSvg.appendChild(svg("line", { x1: 0, x2: 1000, y1: Y(0), y2: Y(0), "class": P + "zero", "vector-effect": "non-scaling-stroke" }));
      chartSvg.appendChild(svg("line", { x1: 0, x2: 1000, y1: Y(D.mean), y2: Y(D.mean), "class": P + "meanline", "vector-effect": "non-scaling-stroke" }));
      D.stops.forEach(function (st) {
        var x = ((st.rank - 0.5) / n * 1000).toFixed(2);
        chartSvg.appendChild(svg("line", { x1: x, x2: x, y1: 100, y2: 94, "class": P + "dtick", "vector-effect": "non-scaling-stroke" }));
      });
      hl = svg("path", { d: "", "class": P + "bar-hl" });
      chartSvg.appendChild(hl);
      meanLab.textContent = "mean " + signed(D.mean, 2);
      meanLab.style.top = Y(D.mean) + "%";
      legend.lastChild.textContent = "all " + n + " prompts, sorted by the difference";
      var npos = a.filter(function (v) { return v > 0; }).length;
      chartBox.setAttribute("aria-label", "All " + n + " paired ImageReward differences, progressive SCA minus AReA, sorted from lowest to highest; " + npos + " of " + n + " are above zero; mean " + signed(D.mean, 2) + ".");
    }
    function markStop(st) {
      var n = D.sorted.length, w = 1000 / n, i = st.rank - 1;
      var x = (i * w).toFixed(2), y0 = Y(0), y1 = Y(st.diff), top = Math.min(y0, y1), h = Math.max(1.2, Math.abs(y1 - y0));
      hl.setAttribute("d", "M" + x + " " + top.toFixed(2) + "h" + w.toFixed(2) + "v" + h.toFixed(2) + "h-" + w.toFixed(2) + "z");
      hl.setAttribute("class", P + "bar-hl " + (st.diff >= 0 ? P + "up" : P + "dn"));
      var xp = (st.rank - 0.5) / n * 100;
      mark.style.left = xp + "%";
      mark.style.setProperty("--c-pct-y", Y(st.diff) + "%");
      mark.classList.toggle(P + "below", st.diff < 0);
      mark.classList.toggle(P + "flip", xp > 82);
      mark.firstChild.textContent = signed(st.diff, 2);
    }
    chartBox.addEventListener("click", function (e) {
      if (!D) return;
      var r = chartBox.getBoundingClientRect(), f = (e.clientX - r.left) / r.width;
      setIdx(Math.max(0, Math.min(8, Math.round(f * 10) - 1)));
    });

    /* ------------------------------------------------------------ one stop */
    function preload(k, cb) {
      var st = D.stops[k], left = 0, done = false;
      function fin() { if (!done) { done = true; preloaded[k] = true; if (cb) cb(); } }
      if (preloaded[k]) { fin(); return; }
      D.methods.forEach(function (m) {
        left++;
        var im = new Image();
        im.decoding = "async";
        im.onload = im.onerror = function () { if (--left === 0) fin(); };
        im.src = base + st.cells[m.id].view;
      });
      setTimeout(fin, 2500);           /* never wait forever on a slow image */
    }
    function show(k, first) {
      cur = k;
      var st = D.stops[k];
      stopEl.lastChild.textContent = st.p === 50 ? "Median" : ordinal(st.p) + " percentile";
      diffV.textContent = signed(st.diff, 2);
      diffEl.classList.toggle(P + "dn", st.diff < 0);
      diffEl.classList.toggle(P + "up", st.diff >= 0);
      diffL.innerHTML = "";
      diffL.appendChild(el("span", "diff-what", { text: "progressive SCA minus AReA \u00b7 " }));
      diffL.appendChild(document.createTextNode("rank " + st.rank + " of " + D.n));
      pill.textContent = st.diff < 0 ? "AReA ahead" : "Progressive SCA ahead";
      promptT.textContent = "\u201c" + st.prompt + "\u201d";
      prompt.classList.remove(P + "open"); more.setAttribute("aria-expanded", "false"); more.textContent = "Full prompt";
      more.hidden = false;
      requestAnimationFrame(function () { more.hidden = !(prompt.scrollHeight > prompt.clientHeight + 2); });
      markStop(st);
      for (var q = 0; q < ticks.children.length; q++) ticks.children[q].classList.toggle(P + "on", q === k);
      prevB.disabled = k <= 0; nextB.disabled = k >= D.stops.length - 1;
      range.setAttribute("aria-valuetext", (st.p === 50 ? "median" : ordinal(st.p) + " percentile") + ", difference " + signed(st.diff, 2) + (st.diff < 0 ? ", AReA ahead" : ", progressive SCA ahead"));
      if (!first) live.textContent = (st.p === 50 ? "Median" : ordinal(st.p) + " percentile") + ": " + (st.diff < 0 ? "AReA ahead by " : "progressive SCA ahead by ") + Math.abs(st.diff).toFixed(2) + " ImageReward.";
      var my = ++pending;
      preload(k, function () { if (my === pending && alive) buildGrid(st); });
      [k - 1, k + 1].forEach(function (j) { if (j >= 0 && j < D.stops.length) setTimeout(function () { if (alive) preload(j); }, 400); });
    }
    function setIdx(k) {
      k = Math.max(0, Math.min(D.stops.length - 1, k));
      if (k === cur) return;
      range.value = String(k);
      show(k);
    }
    range.addEventListener("input", function () { if (D) setIdx(+range.value); });
    prevB.addEventListener("click", function () { if (D) setIdx(cur - 1); });
    nextB.addEventListener("click", function () { if (D) setIdx(cur + 1); });
    more.addEventListener("click", function () {
      var o = prompt.classList.toggle(P + "open");
      more.setAttribute("aria-expanded", String(o)); more.textContent = o ? "Less" : "Full prompt";
    });

    function buildGrid(st) {
      if (grid) { grid.destroy(); grid = null; }
      if (mo) { mo.disconnect(); mo = null; }
      stage.innerHTML = "";
      var regs = (st.regions || []).map(function (r) { return r.slice(); });
      slot = regs.length ? regs[0].slice() : [st.size[0] / 2, st.size[1] / 2, 640];
      regs.push(slot);                          /* the last stop is where the visitor clicked (see zoomAt) */
      var panels = D.methods.map(function (m) {
        var c = st.cells[m.id];
        return { label: m.label, ours: !!m.ours, view: c.view, vs: c.vs, tiles: c.tiles };
      });
      grid = SCAZoom.grid({ size: st.size, panels: panels, regions: regs, base: base, eager: true, maxCols: 4, minW: 150,
                            barBelow: true, cls: P + "zg", alt: st.prompt,
                            label: "The four images of the " + (st.p === 50 ? "median" : ordinal(st.p) + " percentile") + " prompt; zoom and pan move all four together" });
      var g = grid.el, gridEl = g.querySelector(".zg-grid");
      var hint = g.querySelector(".zg-hint");
      if (hint) hint.parentNode.removeChild(hint);
      stage.appendChild(g);
      if (!REDUCED) { g.classList.add(P + "enter"); setTimeout(function () { g.classList.remove(P + "enter"); }, 400); }
      var best = st.best;
      caps = D.methods.map(function (m, i) {
        var c = st.cells[m.id], kind = m.secs_kind !== "clean";
        var cap = el("div", "cap" + (m.ours ? " ours" : "") + (m.id === "area" ? " area" : "") + (m.id === best ? " best" : ""), null, [
          el("span", "cap-m", { text: m.label }),
          el("span", "cap-ir", { title: "ImageReward of this image" + (m.id === best ? ", the best of the four" : ""), text: num(c.ir, 2) }),
          el("span", "cap-s", { text: m.secs.toFixed(1) + " s" + (kind ? "\u2021" : "") + " per image" })
        ]);
        gridEl.appendChild(cap);
        return cap;
      });
      function order() {
        var cols = +(gridEl.style.getPropertyValue("--zg-cols") || 4) || 4;
        var panelsEl = gridEl.querySelectorAll(".zg-p");
        for (var i = 0; i < panelsEl.length; i++) {
          var row = Math.floor(i / cols), col = i % cols;
          panelsEl[i].style.order = String(row * 2 * cols + col);
          if (caps[i]) caps[i].style.order = String(row * 2 * cols + cols + col);
        }
        root.classList.toggle(P + "narrow", cols <= 2);
      }
      order();
      mo = new MutationObserver(order);
      mo.observe(gridEl, { attributes: true, attributeFilter: ["style"] });
      bindGrid(g, gridEl, st);
    }

    /* zoom to native pixels at a native point, through the grid's own flight (its last tour stop) */
    function zoomAt(nx, ny) {
      if (!grid || !slot) return;
      slot[0] = nx; slot[1] = ny;
      var k = D.stops[cur].regions.length;     /* index of the slot */
      grid.flyTo(k);
    }
    function bindGrid(g, gridEl, st) {
      var down = null, swiped = false;
      gridEl.addEventListener("pointerdown", function (e) {
        down = { x: e.clientX, y: e.clientY, t: performance.now(), z: g.classList.contains("zoomed"), id: e.pointerId };
        swiped = false;
      }, true);
      gridEl.addEventListener("pointerup", function (e) {
        if (!down || down.id !== e.pointerId) return;
        var dx = e.clientX - down.x, dy = e.clientY - down.y, dt = performance.now() - down.t;
        if (!down.z && Math.abs(dx) > 48 && Math.abs(dx) > 1.6 * Math.abs(dy) && dt < 900) {
          swiped = true;
          setIdx(cur + (dx < 0 ? 1 : -1));
        }
        down = null;
      }, true);
      gridEl.addEventListener("click", function (e) {
        var box = e.target.closest ? e.target.closest(".zg-p") : null;
        if (!box || swiped || e.detail > 1 || g.classList.contains("zoomed")) { swiped = false; return; }
        if (e.target.closest(".zg-map")) return;
        var r = box.getBoundingClientRect();
        zoomAt((e.clientX - r.left) / r.width * st.size[0], (e.clientY - r.top) / r.height * st.size[1]);
      });
      gridEl.addEventListener("keydown", function (e) {
        var box = e.target.closest ? e.target.closest(".zg-p") : null;
        if (!box) return;
        if ((e.key === "Enter" || e.key === " ") && !g.classList.contains("zoomed")) {
          e.preventDefault();
          var r0 = st.regions && st.regions[0];
          zoomAt(r0 ? r0[0] : st.size[0] / 2, r0 ? r0[1] : st.size[1] / 2);
        }
      });
    }
    root.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && grid && root.querySelector(".zg.zoomed")) { e.preventDefault(); grid.reset(); }
    });

    if ("ResizeObserver" in window) { ro = new ResizeObserver(function () { fit(); }); ro.observe(root); }

    return {
      set: function (p) { if (!D) return; D.stops.forEach(function (st, i) { if (st.p === +p) setIdx(i); }); },
      destroy: function () {
        alive = false;
        if (grid) grid.destroy();
        if (mo) mo.disconnect(); if (ro) ro.disconnect(); if (io) io.disconnect();
        root.innerHTML = ""; root.classList.remove("c-" + NAME);
      }
    };
  }

  SCAComp[NAME] = mount;
})();
