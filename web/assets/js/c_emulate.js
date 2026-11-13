/* SCAComp.emulate(el, options): reach emulation of one head at 1024^2, as a toy. The weight pi blends the shrunk
   window view and the shrunk grid view (real attention of block 12 at one query, r = 4); the mix sits beside the
   head's full attention, and a dot moves along the head's real squared output error J(pi) = E - 2 pi B + pi^2 A
   (archived emulation statistics, 32 prompts); "Snap to pi*" jumps to clip(B / A, 0, 1).
   Vanilla JS, no dependencies, works from file://.

   Data: assets/data/emulate/data.js (window.SCA_DATA.emulate), written by web/build/v5_build_emulate.py from the
   paper's arrays (paper/figures/src/data/v6_attn_block12.npz, results/evidence/calib/k1_1024.npz). No number is
   typed here.

   var c = SCAComp.emulate(el, {head: 10, pi: 0.3, lazy: true, base: "<data dir url>"});  c.set(pi); c.head(h);
   options  head  block 12's head shown first (1..24; default: the method figure's head)
            pi    the weight shown first (default 0.3)
   Interaction: buttons for the paper's near-reaching, in-between and far-reaching heads (and a list of all 24);
   drag on the J chart or use the slider (arrow keys, Home/End) to move pi; "Snap to pi*". */
(function () {
  "use strict";
  var NAME = "emulate", P = "c-emulate-";
  var SCAComp = window.SCAComp = window.SCAComp || {};
  var CUR = document.currentScript && document.currentScript.src;
  var JSDIR = CUR ? CUR.replace(/[^\/]*$/, "") : "assets/js/";
  var DEF_BASE = JSDIR.replace(/js\/$/, "") + "data/" + NAME + "/";
  var REDUCED = !!(window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches);

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
  function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
  function num(x, d) { return (x < 0 ? "\u2212" : "") + Math.abs(x).toFixed(d); }
  function pct(x) { var v = x * 100; return (v >= 10 ? v.toFixed(0) : v.toFixed(1)) + "%"; }
  function fmtInt(n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
  function b64bytes(s) { var bin = atob(s), out = new Uint8Array(bin.length); for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i); return out; }
  var probe = null;
  function rgbOf(root, name, fb) {
    var v = getComputedStyle(root).getPropertyValue(name).trim() || fb;
    if (!probe) { probe = document.createElement("span"); probe.style.display = "none"; document.body.appendChild(probe); }
    probe.style.color = ""; probe.style.color = v;
    var m = getComputedStyle(probe).color.match(/[\d.]+/g) || [128, 128, 128];
    return [+m[0], +m[1], +m[2]];
  }
  function css(c, a) { return "rgba(" + Math.round(c[0]) + "," + Math.round(c[1]) + "," + Math.round(c[2]) + "," + (a === undefined ? 1 : a) + ")"; }
  function ease(t) { return t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }

  function mount(root, opts) {
    opts = opts || {};
    var base = opts.base || DEF_BASE;
    var D = null, alive = true, io = null, ro = null, mo = null, mq = null, anim = 0;
    var C = {}, H = null, pi = typeof opts.pi === "number" ? clamp(opts.pi, 0, 1) : 0.3, img = null, FQ = null;
    root.classList.add("c-" + NAME);
    root.innerHTML = "";

    /* ------------------------------------------------------------ skeleton */
    var seg = el("div", "seg", { role: "radiogroup", "aria-label": "Head of block 12" });
    var pick = el("select", "pick", { "aria-label": "Any head of block 12" });
    var top = el("div", "top", null, [seg, el("label", "pick-w", null, [el("span", "pick-l", { text: "or head" }), pick])]);
    var names = [["near", "window view"], ["far", "grid view"], ["mix", "mix"], ["full", "full attention"]];
    var panels = {}, ops = ["+", "\u2192", "vs"];
    var maps = el("div", "maps");
    names.forEach(function (n, i) {
      var cv = el("canvas", "cv", { role: "img" });
      var lab = el("span", "lab lab-" + n[0], { text: n[1] });
      var sub = el("span", "sub");
      var fr = el("div", "frame", null, [cv, lab]);
      var pn = el("div", "panel panel-" + n[0], null, [fr, sub]);
      panels[n[0]] = { cv: cv, lab: lab, sub: sub, fr: fr };
      maps.appendChild(pn);
      if (i < 3) maps.appendChild(el("span", "op", { "aria-hidden": "true", text: ops[i] }));
    });
    var chart = el("canvas", "chart", { "aria-hidden": "true" });
    var chartBox = el("div", "chartbox", null, [chart]);
    var range = el("input", "range", { type: "range", min: "0", max: "1", step: "0.01", value: String(pi), "aria-label": "Weight pi of the window view", disabled: true });
    var snap = el("button", "snap", { type: "button", disabled: true, text: "Snap to \u03c0*" });
    var ctl = el("div", "ctl", null, [el("span", "pi-l", { text: "\u03c0" }), range, snap]);
    var big = el("div", "big", { "aria-live": "polite" });
    var star = el("div", "star");
    var small = el("p", "small");
    var side = el("div", "side", null, [chartBox, ctl, big, star, small]);
    var body = el("div", "body", null, [maps, side]);
    var note = el("p", "note");
    root.appendChild(top); root.appendChild(body); root.appendChild(note);

    function fail() {
      if (!alive) return;
      var fig = root.closest ? root.closest("figure") : null;
      if (fig && fig.querySelector(".comp-fallback")) { fig.classList.remove("is-live"); root.innerHTML = ""; return; }
      root.innerHTML = "";
      root.appendChild(el("img", "fallback", { src: base + "fallback.webp", alt: "The squared output error J of three heads as a function of the weight pi, with the minimiser marked.", loading: "lazy" }));
    }
    function start() {
      if (window.SCA_DATA && window.SCA_DATA[NAME]) { ready(); return; }
      loadScript(base + "data.js", function () { if (window.SCA_DATA && window.SCA_DATA[NAME]) ready(); else fail(); }, fail);
    }
    if (opts.lazy === false) start(); else io = whenNear(root, start);

    function ready() {
      if (!alive) return;
      D = window.SCA_DATA[NAME];
      FQ = b64bytes(D.full_q);
      img = new Image(); img.decoding = "async"; img.onload = function () { drawMaps(); }; img.src = base + D.image;
      readColours();
      var F = D.featured;
      [["near", F.near, "Near-reaching"], ["mid", F.mid, "In between"], ["far", F.far, "Far-reaching"]].forEach(function (q) {
        var b = el("button", "seg-b", { type: "button", role: "radio", "aria-checked": "false", "data-h": String(q[1]) }, [
          el("span", "seg-k", { text: q[2] }), el("span", "seg-h", { text: "head " + q[1] })]);
        b.addEventListener("click", function () { setHead(q[1]); });
        seg.appendChild(b);
      });
      D.heads.forEach(function (h) { pick.appendChild(el("option", null, { value: String(h.h), text: String(h.h) + " (\u03c0* " + h.J["4"].pi.toFixed(2) + ")" })); });
      pick.addEventListener("change", function () { setHead(+pick.value); });
      range.disabled = false; snap.disabled = false;
      var g = D.grid, w = D.window[2];
      panels.near.sub.textContent = w + " \u00d7 " + w + " window keys around the query's tile";
      panels.far.sub.textContent = g + " \u00d7 " + g + " block-centre keys over the whole image";
      panels.full.sub.textContent = "all " + D.n + " \u00d7 " + D.n + " keys, the model's own attention";
      note.innerHTML = "";
      note.appendChild(document.createTextNode("FLUX.1-dev at 1024\u00b2, block " + D.block + ", one query (the leftmost sheep's face), noise level \u03c3 " + D.sigma.toFixed(2)
        + ". The views are shrunk by r = " + D.r + ", the ratio of a 4096\u00b2 image to the native window, so they cover what the native-size views cover at 4096\u00b2. "
        + "The error J is measured on the attention outputs of " + D.n_prompts + " prompts (" + fmtInt(D.samples["4"]) + " query samples, noise bin at \u03c3 " + D.bin_sigma.toFixed(2)
        + "), not on this one map, and is shown rescaled between its minimum and maximum on [0, 1]. Maps exist for r = " + D.r + " only; r = 2 (2048\u00b2) enters through J alone."));
      setHead(opts.head && opts.head >= 1 && opts.head <= D.heads.length ? opts.head : F.mid, true);
    }
    function readColours() {
      C.ink = rgbOf(root, "--ink", "#111513"); C.soft = rgbOf(root, "--ink-soft", "#4F5853"); C.faint = rgbOf(root, "--ink-faint", "#7C847F");
      C.rule = rgbOf(root, "--rule", "#DCD9CF"); C.bg = rgbOf(root, "--bg-raised", "#FFFFFF"); C.near = rgbOf(root, "--near", "#E08A2B");
      C.far = rgbOf(root, "--far", "#2F6DB3"); C.accent = rgbOf(root, "--accent", "#13865F"); C.screen = rgbOf(root, "--screen", "#0B0D0C");
      C.sink = rgbOf(root, "--screen-ink", "#ECEEE9");
    }

    /* ------------------------------------------------------------ J */
    function J(h, r, p) { var j = h.J[r]; return j.e - 2 * p * j.b + p * p * j.a; }
    function range01(h, r) {
      var j = h.J[r], lo = J(h, r, j.pi), hi = Math.max(J(h, r, 0), J(h, r, 1));
      return [lo, hi];
    }
    function setHead(hh, first) {
      H = D.heads[hh - 1];
      Array.prototype.forEach.call(seg.children, function (b) { b.setAttribute("aria-checked", String(+b.getAttribute("data-h") === hh)); });
      pick.value = String(hh);
      root.classList.toggle(P + "featured", Array.prototype.some.call(seg.children, function (b) { return +b.getAttribute("data-h") === hh; }));
      update(first);
    }
    function setPi(p, quiet) { pi = clamp(Math.round(p * 100) / 100, 0, 1); range.value = String(pi); update(quiet); }

    function update(quiet) {
      if (!D || !H) return;
      panels.mix.lab.textContent = "mix, \u03c0 = " + pi.toFixed(2);
      panels.mix.sub.textContent = pi.toFixed(2) + " \u00d7 window + " + (1 - pi).toFixed(2) + " \u00d7 grid";
      range.setAttribute("aria-valuetext", "pi " + pi.toFixed(2) + ", error " + (J(H, "4", pi) / J(H, "4", H.J["4"].pi)).toFixed(2) + " times its minimum");
      drawMaps(); drawChart(); readout(quiet);
    }
    function readout(quiet) {
      var j4 = H.J["4"], j2 = H.J["2"], jp = J(H, "4", pi), js = J(H, "4", j4.pi), ratio = jp / js;
      big.innerHTML = "";
      big.appendChild(el("span", "big-v", { text: ratio < 1.005 ? "minimum" : ratio.toFixed(2) + "\u00d7" }));
      big.appendChild(el("span", "big-l", { text: ratio < 1.005 ? "this is the head's best weight" : "the head's smallest error J(\u03c0*), at \u03c0 = " + pi.toFixed(2) }));
      var ba = j4.b / j4.a;
      star.innerHTML = "";
      star.appendChild(el("b", null, { text: "\u03c0* = clip(B/A, 0, 1) = " + j4.pi.toFixed(2) }));
      star.appendChild(document.createTextNode(ba < 0 ? " (B/A = " + num(ba, 2) + ", clipped: this head keeps only the grid view)" : ba > 1 ? " (B/A = " + num(ba, 2) + ", clipped)" : " at r = " + D.r));
      star.appendChild(el("span", "star-2", { text: "r = 2: \u03c0* = " + j2.pi.toFixed(2) }));
      small.textContent = "Error relative to the energy of the full-attention output: " + pct(jp) + " at this \u03c0, " + pct(js) + " at \u03c0*.";
    }

    /* ------------------------------------------------------------ maps */
    function prep(cv) {
      var w = cv.parentNode.clientWidth || 200, d = Math.min(2.5, window.devicePixelRatio || 1), W = Math.round(w * d);
      if (cv.width !== W) { cv.width = W; cv.height = W; }
      var ctx = cv.getContext("2d");
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = css(C.screen); ctx.fillRect(0, 0, W, W);
      if (img && img.complete && img.naturalWidth) { ctx.imageSmoothingEnabled = true; ctx.drawImage(img, 0, 0, W, W); }
      return { ctx: ctx, W: W, s: W / D.n, d: d };
    }
    var off = null;
    function heat(g, fn) {               /* fn(row, col) -> [v in 0..1, colour] or null */
      var n = D.n;
      if (!off) { off = document.createElement("canvas"); off.width = n; off.height = n; }
      var oc = off.getContext("2d"), im = oc.createImageData(n, n);
      for (var y = 0; y < n; y++) for (var x = 0; x < n; x++) {
        var r = fn(y, x); if (!r) continue;
        var a = Math.pow(r[0], 0.6), o = (y * n + x) * 4;
        im.data[o] = r[1][0]; im.data[o + 1] = r[1][1]; im.data[o + 2] = r[1][2]; im.data[o + 3] = Math.round(a * 255);
      }
      oc.putImageData(im, 0, 0);
      g.ctx.imageSmoothingEnabled = false; g.ctx.drawImage(off, 0, 0, g.W, g.W);
    }
    function lv(dens) { return dens <= 1 ? 0 : Math.min(1, Math.log(dens) / Math.LN10 / (Math.log(D.dmax) / Math.LN10)); }
    function inWin(y, x) { var w = D.window; return y >= w[0] && y < w[0] + w[2] && x >= w[1] && x < w[1] + w[2]; }
    function veil(g, amount) {
      var w = D.window, s = g.s, ctx = g.ctx;
      ctx.fillStyle = css(C.screen, amount);
      ctx.beginPath(); ctx.rect(0, 0, g.W, g.W); ctx.rect(w[1] * s, w[0] * s, w[2] * s, w[2] * s); ctx.fill("evenodd");
    }
    function frame(g) {
      var w = D.window, s = g.s, ctx = g.ctx;
      ctx.setLineDash([4 * g.d, 3 * g.d]); ctx.lineWidth = 1.3 * g.d; ctx.strokeStyle = css(C.near);
      ctx.strokeRect(w[1] * s + .5, w[0] * s + .5, w[2] * s - 1, w[2] * s - 1); ctx.setLineDash([]);
      var q = D.query, cx = (q[1] + .5) * s, cy = (q[0] + .5) * s, a = Math.max(6 * g.d, 1.6 * s), b = Math.max(3.5 * g.d, 0.9 * s);
      ctx.fillStyle = "#fff"; ctx.fillRect(cx - a / 2, cy - a / 2, a, a);
      ctx.fillStyle = css(C.screen); ctx.fillRect(cx - b / 2, cy - b / 2, b, b);
    }
    function nearHeat(g, scale, mass) {              /* window view's attention times scale, density over the mix's image mass */
      var w = D.window, n2 = D.n * D.n;
      heat(g, function (y, x) {
        if (!inWin(y, x)) return null;
        var p = H.loc[(y - w[0]) * w[2] + (x - w[1])] * scale;
        return [lv(p / mass * n2), C.near];
      });
    }
    function farDots(g, scale, mass, alpha) {
      var k = D.n / D.grid, s = g.s, ctx = g.ctx, rmax = k * s * 0.62;
      ctx.fillStyle = css(C.far, alpha);
      for (var i = 0; i < D.grid; i++) for (var j = 0; j < D.grid; j++) {
        var p = H.glob[i * D.grid + j] * scale / mass; if (p < 2e-4) continue;
        var r = Math.min(rmax, rmax * Math.sqrt(p / 0.05));
        ctx.beginPath(); ctx.arc((j * k + Math.floor(k / 2) + .5) * s, (i * k + Math.floor(k / 2) + .5) * s, Math.max(.9 * g.d, r), 0, 7); ctx.fill();
      }
    }
    function sum(a) { var t = 0; for (var i = 0; i < a.length; i++) t += a[i]; return t; }
    function drawMaps() {
      if (!D || !H) return;
      var Lm = sum(H.loc), Gm = sum(H.glob), hi = H.h - 1;
      var g = prep(panels.near.cv); nearHeat(g, 1, Lm); veil(g, .55); frame(g);
      g = prep(panels.far.cv); farDots(g, 1, Gm, .92); frame(g);
      var M = pi * Lm + (1 - pi) * Gm;
      g = prep(panels.mix.cv); if (pi > 0) nearHeat(g, pi, M); if (pi < 1) farDots(g, 1 - pi, M, .92); frame(g);
      g = prep(panels.full.cv);
      var n = D.n, offs = hi * n * n;
      heat(g, function (y, x) { var v = FQ[offs + y * n + x] / 255; return v > 0 ? [v, inWin(y, x) ? C.near : C.far] : null; });
      frame(g);
      var t = function (x) { return Math.round(x * 100) + "%"; };
      panels.near.cv.setAttribute("aria-label", "Window view of head " + H.h + ": its attention over the window around the query; " + t(H.loc_text) + " of it goes to text.");
      panels.far.cv.setAttribute("aria-label", "Grid view of head " + H.h + ": dots at the block centres, area by attention; " + t(H.glob_text) + " goes to text.");
      panels.mix.cv.setAttribute("aria-label", "The mix at pi " + pi.toFixed(2) + ".");
      panels.full.cv.setAttribute("aria-label", "Full attention of head " + H.h + ": " + t(H.m4) + " of its image attention lies inside the window (orange), the rest outside (blue).");
    }

    /* ------------------------------------------------------------ the J chart */
    var CG = { w: 0, h: 0, d: 1, L: 14, R: 14, T: 26, B: 30 };
    function px(p) { return CG.L + p * (CG.w - CG.L - CG.R); }
    function py(v) { return CG.T + (1 - v) * (CG.h - CG.T - CG.B); }
    function drawChart() {
      if (!D || !H) return;
      var w = chartBox.clientWidth || 360, h = chartBox.clientHeight || 220, d = Math.min(2.5, window.devicePixelRatio || 1);
      CG.w = w; CG.h = h; CG.d = d;
      if (chart.width !== Math.round(w * d) || chart.height !== Math.round(h * d)) { chart.width = Math.round(w * d); chart.height = Math.round(h * d); }
      var ctx = chart.getContext("2d"), fn = getComputedStyle(root).getPropertyValue("--font-sans") || "system-ui, sans-serif";
      ctx.setTransform(d, 0, 0, d, 0, 0); ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = css(C.rule); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(CG.L, py(0) + .5); ctx.lineTo(w - CG.R, py(0) + .5); ctx.stroke();
      ctx.fillStyle = css(C.faint); ctx.font = "500 11px " + fn; ctx.textBaseline = "top";
      [0, .5, 1].forEach(function (p) { ctx.textAlign = p === 0 ? "left" : p === 1 ? "right" : "center"; ctx.fillText(p === .5 ? "0.5" : String(p), px(p), py(0) + 6); });
      ctx.textAlign = "center"; ctx.fillStyle = css(C.soft); ctx.font = "500 12px " + fn;
      ctx.fillText("\u03c0: grid view \u2190 \u2192 window view", (CG.L + w - CG.R) / 2, py(0) + 6);
      ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillStyle = css(C.soft);
      ctx.fillText("J(\u03c0) = E \u2212 2\u03c0B + \u03c0\u00b2A", CG.L, 2);
      function curve(r, col, dash, lw) {
        var mm = range01(H, r);
        ctx.setLineDash(dash); ctx.strokeStyle = css(col); ctx.lineWidth = lw; ctx.beginPath();
        for (var i = 0; i <= 100; i++) { var p = i / 100, v = (J(H, r, p) - mm[0]) / (mm[1] - mm[0] || 1); if (i) ctx.lineTo(px(p), py(v)); else ctx.moveTo(px(p), py(v)); }
        ctx.stroke(); ctx.setLineDash([]);
        return mm;
      }
      curve("2", C.faint, [5, 4], 1.4);
      var mm = curve("4", C.ink, [], 2.4);
      /* pi* (r = 4) */
      var ps = H.J["4"].pi, xs = px(ps);
      ctx.strokeStyle = css(C.accent); ctx.lineWidth = 1.4; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(xs, CG.T - 6); ctx.lineTo(xs, py(0)); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = css(C.accent); ctx.font = "600 12px " + fn; ctx.textBaseline = "bottom";
      ctx.textAlign = xs > CG.w - 70 ? "right" : xs < 70 ? "left" : "center";
      ctx.fillText("\u03c0* = " + ps.toFixed(2), clamp(xs, CG.L, CG.w - CG.R), CG.T - 8);
      /* r = 2 minimum */
      var p2 = H.J["2"].pi, m2 = range01(H, "2");
      ctx.strokeStyle = css(C.faint); ctx.lineWidth = 1.4; ctx.fillStyle = css(C.bg);
      ctx.beginPath(); ctx.arc(px(p2), py((J(H, "2", p2) - m2[0]) / (m2[1] - m2[0] || 1)), 3.6, 0, 7); ctx.fill(); ctx.stroke();
      /* the dot */
      var v = (J(H, "4", pi) - mm[0]) / (mm[1] - mm[0] || 1), x = px(pi), y = py(v);
      ctx.strokeStyle = css(C.ink, .35); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, py(0)); ctx.stroke();
      ctx.fillStyle = css(C.bg); ctx.beginPath(); ctx.arc(x, y, 8.5, 0, 7); ctx.fill();
      ctx.fillStyle = css(C.accent); ctx.beginPath(); ctx.arc(x, y, 6, 0, 7); ctx.fill();
      /* legend */
      ctx.font = "500 11px " + fn; ctx.textAlign = "right"; ctx.textBaseline = "top";
      var lx = CG.w - CG.R;
      ctx.fillStyle = css(C.ink); ctx.fillText("r = 4 (4096\u00b2)", lx - 30, 2);
      ctx.strokeStyle = css(C.ink); ctx.lineWidth = 2.4; ctx.beginPath(); ctx.moveTo(lx - 24, 9); ctx.lineTo(lx, 9); ctx.stroke();
      ctx.fillStyle = css(C.faint); ctx.fillText("r = 2 (2048\u00b2)", lx - 30, 16);
      ctx.strokeStyle = css(C.faint); ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(lx - 24, 23); ctx.lineTo(lx, 23); ctx.stroke(); ctx.setLineDash([]);
    }
    var drag = false;
    function fromX(e) { var r = chart.getBoundingClientRect(); return clamp((e.clientX - r.left - CG.L) / (CG.w - CG.L - CG.R), 0, 1); }
    chartBox.addEventListener("pointerdown", function (e) { if (!D) return; cancel(); drag = true; try { chartBox.setPointerCapture(e.pointerId); } catch (x) { /* ignore */ } setPi(fromX(e), true); });
    chartBox.addEventListener("pointermove", function (e) { if (drag) setPi(fromX(e), true); });
    ["pointerup", "pointercancel"].forEach(function (t) { chartBox.addEventListener(t, function () { if (drag) { drag = false; update(); } }); });
    range.addEventListener("input", function () { cancel(); setPi(+range.value, true); });
    range.addEventListener("change", function () { update(); });
    function cancel() { if (anim) { cancelAnimationFrame(anim); anim = 0; } }
    snap.addEventListener("click", function () {
      if (!H) return;
      cancel();
      var p0 = pi, p1 = H.J["4"].pi, t0 = performance.now(), dur = REDUCED ? 0 : 520;
      if (!dur) { setPi(p1); return; }
      var step = function (now) {
        var t = clamp((now - t0) / dur, 0, 1);
        pi = p0 + (p1 - p0) * ease(t); range.value = String(pi); update(true);
        if (t < 1) anim = requestAnimationFrame(step); else { anim = 0; setPi(p1); }
      };
      anim = requestAnimationFrame(step);
    });

    /* ------------------------------------------------------------ size and theme */
    function fit() { root.classList.toggle(P + "narrow", (root.clientWidth || 800) < 760); }
    fit();
    if ("ResizeObserver" in window) {
      var rq = 0;
      ro = new ResizeObserver(function () { if (rq) return; rq = requestAnimationFrame(function () { rq = 0; fit(); if (D) { drawMaps(); drawChart(); } }); });
      ro.observe(root);
    }
    function onTheme() { requestAnimationFrame(function () { if (!D) return; readColours(); drawMaps(); drawChart(); }); }
    document.addEventListener("sca:themechange", onTheme);
    if (window.matchMedia) { mq = matchMedia("(prefers-color-scheme: dark)"); if (mq.addEventListener) mq.addEventListener("change", onTheme); }
    mo = new MutationObserver(onTheme);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class"] });

    return {
      set: function (p) { if (D) setPi(+p); },
      head: function (h) { if (D) setHead(clamp(h | 0, 1, D.heads.length)); },
      destroy: function () {
        alive = false; cancel();
        if (io) io.disconnect(); if (ro) ro.disconnect(); if (mo) mo.disconnect();
        if (mq && mq.removeEventListener) mq.removeEventListener("change", onTheme);
        document.removeEventListener("sca:themechange", onTheme);
        root.innerHTML = ""; root.classList.remove("c-" + NAME);
      }
    };
  }

  SCAComp[NAME] = mount;
})();
