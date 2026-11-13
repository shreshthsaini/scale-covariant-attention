/* SCAComp.heads(el, options): reach belongs to the head. All 1,368 near-share curves of FLUX.1-dev at 1024^2 over
   the noise level (paper Figure 2b) on a canvas, the variance split beside them, and the real attention of block 12's
   heads at one query. Vanilla JS, no dependencies, works from file://.

   Data: assets/data/heads/data.js (window.SCA_DATA.heads), written by web/build/v5_build_heads.py through the
   paper's own loader (paper/figures/src/v6_observation.py). No number is typed here.

   var c = SCAComp.heads(el, {view: "curves", head: [12, 4], sigma: 1, lazy: true, base: "<data dir url>"});
   options  view   "curves" (default) or "matrix" (heads as rows sorted by mean near share)
            head   [block, head], one-based, selected first (default: the paper's near-reaching head 4 of block 12)
            sigma  index of the attention map's noise level (0, 1, 2 for sigma 0.90, 0.61, 0.33; default 1)
   Interaction: hover or tap a curve (or a matrix row) to read it, click to select; arrow keys on the plot step
   through the heads by mean reach; chips select block 12's heads, whose real attention is shown on the image. */
(function () {
  "use strict";
  var NAME = "heads", P = "c-heads-";
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
  function f2(x) { return x.toFixed(2); }
  function fmtInt(n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
  function b64bytes(s) { var bin = atob(s), out = new Uint8Array(bin.length); for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i); return out; }

  /* colours from the page's tokens, resolved to rgb through the browser (any CSS colour syntax works) */
  var probe = null;
  function rgbOf(root, name, fb) {
    var v = getComputedStyle(root).getPropertyValue(name).trim() || fb;
    if (!probe) { probe = document.createElement("span"); probe.style.display = "none"; document.body.appendChild(probe); }
    probe.style.color = ""; probe.style.color = v;
    var m = getComputedStyle(probe).color.match(/[\d.]+/g) || [128, 128, 128];
    return [+m[0], +m[1], +m[2]];
  }
  function css(c, a) { return "rgba(" + Math.round(c[0]) + "," + Math.round(c[1]) + "," + Math.round(c[2]) + "," + (a === undefined ? 1 : a) + ")"; }
  function mix(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }

  function mount(root, opts) {
    opts = opts || {};
    var base = opts.base || DEF_BASE;
    var D = null, alive = true, io = null, ro = null, mo = null, mq = null;
    var C = {};                                   /* colours */
    var view = opts.view === "matrix" ? "matrix" : "curves";
    var sel = -1, hov = -1, sigI = typeof opts.sigma === "number" ? clamp(opts.sigma | 0, 0, 2) : 1, mapHead = -1;
    var N = 0, NB = 0, vals = null, means = null, order = null, rankOf = null, img = null, Q = null;
    root.classList.add("c-" + NAME);
    root.innerHTML = "";

    /* ------------------------------------------------------------ skeleton */
    var bCurves = el("button", "seg-b", { type: "button", role: "radio", "aria-checked": "true", text: "Curves" });
    var bMatrix = el("button", "seg-b", { type: "button", role: "radio", "aria-checked": "false", text: "Sorted by reach" });
    var seg = el("div", "seg", { role: "radiogroup", "aria-label": "View" }, [bCurves, bMatrix]);
    var cbar = el("div", "cbar", { "aria-hidden": "true" }, [el("span", "cbar-l", { text: "far" }), el("span", "cbar-g"), el("span", "cbar-r", { text: "near" })]);
    var legend = el("div", "legend", null, [el("span", "legend-t", { text: "colour: mean near share" }), cbar,
      el("span", "lg-avg", { text: "head average" })]);
    var top = el("div", "top", null, [seg, legend]);
    var cv = el("canvas", "cv", { "aria-hidden": "true" });
    var ov = el("canvas", "ov", { tabindex: "0", role: "img", "aria-label": "Near share of every attention head over the noise level" });
    var tip = el("div", "tip", { "aria-hidden": "true" });
    var plot = el("div", "plot", null, [cv, ov, tip]);
    var read = el("p", "read", { "aria-live": "polite" });
    var left = el("div", "left", null, [top, plot, read]);

    var split = el("div", "split");
    var mapTitle = el("div", "map-t");
    var mapShare = el("div", "map-s");
    var mapCv = el("canvas", "map", { role: "img", "aria-label": "Attention of the selected head at one query" });
    var mapFrame = el("div", "map-f", null, [mapCv]);
    var scrub = el("div", "scrub", { role: "radiogroup", "aria-label": "Noise level of the attention map" });
    var chips = el("div", "chips", { role: "group", "aria-label": "Heads of block 12" });
    var mapNote = el("p", "map-n");
    var mapBox = el("div", "mapbox", null, [el("div", "map-h", null, [mapTitle, mapShare]), mapFrame,
      el("div", "map-c", null, [el("div", "scrub-l", { text: "noise level, \u2192 denoising" }), scrub, chips, mapNote])]);
    var side = el("div", "side", null, [split, mapBox]);
    var body = el("div", "body", null, [left, side]);
    var note = el("p", "note");
    root.appendChild(body); root.appendChild(note);
    root.classList.add(P + "loading");          /* reserves the final heights until the data arrives (no layout shift) */

    /* ------------------------------------------------------------ data */
    function fail() {
      if (!alive) return;
      root.classList.remove(P + "loading");
      var fig = root.closest ? root.closest("figure") : null;
      if (fig && fig.querySelector(".comp-fallback")) { fig.classList.remove("is-live"); root.innerHTML = ""; return; }
      root.innerHTML = "";
      root.appendChild(el("img", "fallback", { src: base + "fallback.webp", alt: "The near share of all 1,368 attention heads of FLUX.1-dev over the noise level: flat lines at very different heights.", loading: "lazy" }));
    }
    function start() {
      if (window.SCA_DATA && window.SCA_DATA[NAME]) { ready(); return; }
      loadScript(base + "data.js", function () { if (window.SCA_DATA && window.SCA_DATA[NAME]) ready(); else fail(); }, fail);
    }
    if (opts.lazy === false) start(); else io = whenNear(root, start);

    function ready() {
      if (!alive) return;
      D = window.SCA_DATA[NAME];
      NB = D.sig.length; N = D.n_blocks * D.n_heads;
      vals = new Float32Array(N * NB);
      for (var i = 0; i < vals.length; i++) vals[i] = D.curves[i] / 1e4;
      means = new Float32Array(N);
      for (var h = 0; h < N; h++) { var s = 0; for (var b = 0; b < NB; b++) s += vals[h * NB + b]; means[h] = s / NB; }
      order = Array.prototype.slice.call(means).map(function (m, i) { return i; }).sort(function (a, b) { return means[b] - means[a] || a - b; });
      rankOf = new Int32Array(N); order.forEach(function (h, r) { rankOf[h] = r; });
      Q = b64bytes(D.maps.q);
      img = new Image(); img.decoding = "async"; img.onload = drawMap; img.src = base + D.maps.image;
      var mb = D.maps.block - 1;
      var want = opts.head && opts.head.length === 2 ? (opts.head[0] - 1) * D.n_heads + (opts.head[1] - 1) : mb * D.n_heads + D.featured.near - 1;
      sel = clamp(want, 0, N - 1);
      mapHead = Math.floor(sel / D.n_heads) === mb ? sel % D.n_heads : D.featured.near - 1;
      readColours(); buildSplit(); buildScrub(); buildChips();
      note.innerHTML = "";
      note.appendChild(el("span", "more", { text: "FLUX.1-dev at 1024\u00b2, its training size. Near share: the share of a head's image attention within " + D.radius
        + " tokens of the query, averaged over 32 prompts and 256 queries per block and noise level; " + fmtInt(N) + " heads (" + D.n_blocks + " blocks \u00d7 "
        + D.n_heads + "). Maps: one query (the leftmost sheep's face) of one image, block " + D.maps.block + "; heads " + D.featured.near + " and "
        + D.featured.far + " are the paper's near- and far-reaching examples. " }));
      var tog = el("button", "tog", { type: "button", "aria-expanded": "false", text: "About these numbers" });
      tog.addEventListener("click", function () { var o = root.classList.toggle(P + "open"); tog.setAttribute("aria-expanded", String(o)); tog.textContent = o ? "Less" : "About these numbers"; });
      note.appendChild(tog);
      readColours(); setView(view, true); select(sel, true);
      root.classList.remove(P + "loading");
    }

    function readColours() {
      C.ink = rgbOf(root, "--ink", "#111513"); C.soft = rgbOf(root, "--ink-soft", "#4F5853"); C.faint = rgbOf(root, "--ink-faint", "#7C847F");
      C.rule = rgbOf(root, "--rule", "#DCD9CF"); C.bg = rgbOf(root, "--bg-raised", "#FFFFFF"); C.near = rgbOf(root, "--near", "#E08A2B");
      C.far = rgbOf(root, "--far", "#2F6DB3"); C.neutral = rgbOf(root, "--neutral", "#8C9490"); C.accent = rgbOf(root, "--accent", "#13865F");
      C.screen = rgbOf(root, "--screen", "#0B0D0C");
      var lum = 0.2126 * C.bg[0] + 0.7152 * C.bg[1] + 0.0722 * C.bg[2];
      C.dark = lum < 110;
    }
    function ramp(v) { v = clamp(v, 0, 1); return v < .5 ? mix(C.far, C.neutral, v / .5) : mix(C.neutral, C.near, (v - .5) / .5); }

    /* ------------------------------------------------------------ variance split */
    function buildSplit() {
      var p = D.split.flux1_dev, pf = D.split_f.flux1_dev;
      split.innerHTML = "";
      split.appendChild(el("div", "kicker", { text: "What sets the near share" }));
      var nums = el("div", "nums", null, [
        el("div", "num num-h", null, [el("span", "num-v", { text: p.head + "%" }), el("span", "num-l", { text: "the head" })]),
        el("div", "num num-s", null, [el("span", "num-v", { text: p.step + "%" }), el("span", "num-l", { text: "the noise level" })]),
        el("div", "num num-a", null, [el("span", "num-v", { text: p.avg + "%" }), el("span", "num-l", { text: "a shared schedule" })])]);
      split.appendChild(nums);
      var b1 = el("div", "bar", { "aria-hidden": "true" }, [el("span", "bar-h", { style: "width:" + (100 * pf.head_f) + "%" }), el("span", "bar-s", { style: "width:" + (100 * pf.step_f) + "%" })]);
      var wedge = el("div", "wedge", { "aria-hidden": "true", style: "--c-heads-cut:" + (100 * pf.head_f) + "%" });
      var b2 = el("div", "bar bar-z", { "aria-hidden": "true" }, [el("span", "bar-a", { style: "width:" + (100 * pf.avg_f / pf.step_f) + "%" }), el("span", "bar-s2")]);
      split.appendChild(b1); split.appendChild(wedge); split.appendChild(b2);
      var k = D.split.flux1_krea_dev, f2s = D.split.flux2_dev;
      split.appendChild(el("p", "split-n more", { text: "Shares of the variance of the near share over " + fmtInt(N) + " heads \u00d7 " + NB + " noise levels. Lower bar: the noise level's part, magnified; "
        + p.avg + "% follows the head-averaged curve, a schedule all heads could share. The head's share is " + k.head + "% in FLUX.1-Krea-dev and " + f2s.head + "% in FLUX.2-dev." }));
    }

    /* ------------------------------------------------------------ plot */
    var G = { w: 0, h: 0, d: 1, L: 40, R: 86, T: 24, B: 36 };
    function sx(s) { var a = D.sig[0] + .02, b = D.sig[NB - 1] - .02; return G.L + (a - s) / (a - b) * (G.w - G.L - G.R); }
    function sy(v) { return G.T + (1 - v) * (G.h - G.T - G.B); }
    function size() {
      var w = plot.clientWidth || 600, h = plot.clientHeight || 360, d = Math.min(2.5, window.devicePixelRatio || 1);
      G.w = w; G.h = h; G.d = d; G.R = w < 480 ? 58 : 86; G.L = w < 480 ? 34 : 40;
      [cv, ov].forEach(function (c) { var W = Math.round(w * d), H = Math.round(h * d); if (c.width !== W || c.height !== H) { c.width = W; c.height = H; } });
    }
    function axes(ctx) {
      ctx.font = "500 11px " + (getComputedStyle(root).getPropertyValue("--font-sans") || "system-ui, sans-serif");
      ctx.textBaseline = "middle";
      if (view === "curves") {
        [0, .25, .5, .75, 1].forEach(function (v) {
          var y = Math.round(sy(v)) + .5;
          ctx.strokeStyle = css(C.rule); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(G.L, y); ctx.lineTo(G.w - G.R, y); ctx.stroke();
          if (v === 0 || v === .5 || v === 1) { ctx.fillStyle = css(C.faint); ctx.textAlign = "right"; ctx.fillText(v === .5 ? "0.5" : String(v), G.L - 6, y); }
        });
      }
      ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillStyle = css(C.faint);
      var ticks = view === "curves" ? [.8, .6, .4, .2].map(function (s) { return [sx(s), s.toFixed(1)]; })
        : [1, 4, 7, 10].filter(function (t) { return t < NB; }).map(function (t) { return [G.L + (t + .5) * (G.w - G.L - G.R) / NB, D.sig[t].toFixed(2)]; });
      ticks.forEach(function (q) {
        var x = Math.round(q[0]) + .5;
        ctx.strokeStyle = css(C.faint); ctx.beginPath(); ctx.moveTo(x, G.h - G.B); ctx.lineTo(x, G.h - G.B + 4); ctx.stroke();
        ctx.fillText(q[1], x, G.h - G.B + 6);
      });
      ctx.fillStyle = css(C.soft); ctx.font = "500 12px " + (getComputedStyle(root).getPropertyValue("--font-sans") || "system-ui, sans-serif");
      ctx.fillText("noise level \u03c3  \u2192  denoising", (G.L + G.w - G.R) / 2, G.h - 15);
      ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText(view === "curves" ? "near share" : "heads, sorted by mean near share", 0, 0);
    }
    function line(ctx, h, col, w, a) {
      ctx.strokeStyle = css(col, a); ctx.lineWidth = w; ctx.beginPath();
      for (var b = 0; b < NB; b++) { var x = sx(D.sig[b]), y = sy(vals[h * NB + b]); if (b) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
      ctx.stroke();
    }
    function avgLine(ctx, w) {
      ctx.strokeStyle = css(C.bg); ctx.lineWidth = w + 3; ctx.beginPath();
      D.avg.forEach(function (v, b) { var x = sx(D.sig[b]), y = sy(v); if (b) ctx.lineTo(x, y); else ctx.moveTo(x, y); }); ctx.stroke();
      ctx.strokeStyle = css(C.ink); ctx.lineWidth = w; ctx.stroke();
    }
    function endLabel(ctx, txt, y, col) {
      ctx.fillStyle = css(col); ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillText(txt, G.w - G.R + 6, y);
    }
    var mOff = null;
    function drawBase() {
      size();
      var ctx = cv.getContext("2d");
      ctx.setTransform(G.d, 0, 0, G.d, 0, 0);
      ctx.clearRect(0, 0, G.w, G.h);
      axes(ctx);
      ctx.lineJoin = "round"; ctx.lineCap = "round";
      var fn = getComputedStyle(root).getPropertyValue("--font-sans") || "system-ui, sans-serif";
      if (view === "curves") {
        var idx = order.slice().sort(function (a, b) { return Math.abs(means[b] - .5) - Math.abs(means[a] - .5); });
        var a = C.dark ? .3 : .26;
        for (var i = 0; i < idx.length; i++) line(ctx, idx[i], ramp(means[idx[i]]), 1, a);
        var fb = (D.featured.block - 1) * D.n_heads, nH = fb + D.featured.near - 1, fH = fb + D.featured.far - 1;
        [[nH, C.near], [fH, C.far]].forEach(function (q) { line(ctx, q[0], C.bg, 4.2, 1); line(ctx, q[0], q[1], 2, 1); });
        avgLine(ctx, 2.4);
        ctx.font = "600 12px " + fn;
        var labs = [[vals[nH * NB + NB - 1], "head " + D.featured.near, C.near], [D.avg[NB - 1], "average", C.ink], [vals[fH * NB + NB - 1], "head " + D.featured.far, C.far]]
          .map(function (q) { return [sy(q[0]), q[1], q[2]]; }).sort(function (a, b) { return a[0] - b[0]; });
        for (var k = 1; k < labs.length; k++) if (labs[k][0] - labs[k - 1][0] < 14) labs[k][0] = labs[k - 1][0] + 14;
        labs.forEach(function (q) { endLabel(ctx, q[1], q[0], q[2]); });
      } else {
        if (!mOff) {
          mOff = document.createElement("canvas"); mOff.width = NB; mOff.height = N;
        }
        var mc = mOff.getContext("2d"), im = mc.createImageData(NB, N);
        for (var r = 0; r < N; r++) for (var b = 0; b < NB; b++) {
          var c = ramp(vals[order[r] * NB + b]), o = (r * NB + b) * 4;
          im.data[o] = c[0]; im.data[o + 1] = c[1]; im.data[o + 2] = c[2]; im.data[o + 3] = 255;
        }
        mc.putImageData(im, 0, 0);
        var x0 = G.L, x1 = G.w - G.R, y0 = G.T, y1 = G.h - G.B - 18;
        ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
        var cw = (x1 - x0) / NB;
        ctx.drawImage(mOff, 0, 0, NB, N, x0, y0, x1 - x0, y1 - y0);
        /* head average strip */
        for (var b2 = 0; b2 < NB; b2++) { ctx.fillStyle = css(ramp(D.avg[b2])); ctx.fillRect(x0 + b2 * cw, y1 + 6, cw + .5, 10); }
        ctx.fillStyle = css(C.faint); ctx.font = "500 11px " + fn; ctx.textAlign = "right"; ctx.textBaseline = "top";
        ctx.fillText("1", x0 - 6, y0); ctx.textBaseline = "bottom"; ctx.fillText(fmtInt(N), x0 - 6, y1);
        ctx.font = "600 12px " + fn; ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillStyle = css(C.ink); ctx.fillText("average", x1 + 6, y1 + 11);
        ctx.fillStyle = css(C.soft); ctx.font = "500 11px " + fn;
        ctx.fillText("near-reaching", x1 + 6, y0 + 8); ctx.fillText("far-reaching", x1 + 6, y1 - 8);
      }
      drawOv();
    }
    function mxRow(h) { var y0 = G.T, y1 = G.h - G.B - 18; return y0 + (rankOf[h] + .5) / N * (y1 - y0); }
    function drawOv() {
      var ctx = ov.getContext("2d");
      ctx.setTransform(G.d, 0, 0, G.d, 0, 0);
      ctx.clearRect(0, 0, G.w, G.h);
      ctx.lineJoin = "round"; ctx.lineCap = "round";
      if (view === "curves") {
        if (hov >= 0 && hov !== sel) { line(ctx, hov, C.bg, 4, .9); line(ctx, hov, C.ink, 1.6, .85); }
        if (sel >= 0) { line(ctx, sel, C.bg, 5.5, 1); line(ctx, sel, C.ink, 2.6, 1); dots(ctx, sel); }
      } else {
        [[hov, .6], [sel, 1]].forEach(function (q) {
          if (q[0] < 0) return;
          var y = mxRow(q[0]);
          ctx.strokeStyle = css(C.ink, q[1]); ctx.lineWidth = q[0] === sel ? 2 : 1;
          ctx.beginPath(); ctx.moveTo(G.L - 4, y); ctx.lineTo(G.w - G.R + 4, y); ctx.stroke();
          if (q[0] === sel) { ctx.fillStyle = css(C.ink); ctx.beginPath(); ctx.moveTo(G.w - G.R + 4, y); ctx.lineTo(G.w - G.R + 12, y - 5); ctx.lineTo(G.w - G.R + 12, y + 5); ctx.fill(); }
        });
      }
    }
    function dots(ctx, h) {
      for (var b = 0; b < NB; b++) {
        ctx.fillStyle = css(C.bg); ctx.beginPath(); ctx.arc(sx(D.sig[b]), sy(vals[h * NB + b]), 3.6, 0, 7); ctx.fill();
        ctx.fillStyle = css(C.ink); ctx.beginPath(); ctx.arc(sx(D.sig[b]), sy(vals[h * NB + b]), 2.3, 0, 7); ctx.fill();
      }
    }
    function nearest(px, py) {
      if (view === "matrix") {
        var y0 = G.T, y1 = G.h - G.B - 18;
        if (py < y0 - 4 || py > y1 + 4 || px < G.L - 4 || px > G.w - G.R + 4) return -1;
        return order[clamp(Math.floor((py - y0) / (y1 - y0) * N), 0, N - 1)];
      }
      if (px < G.L - 8 || px > G.w - G.R + 8) return -1;
      var a = D.sig[0] + .02, b = D.sig[NB - 1] - .02, s = a - (px - G.L) / (G.w - G.L - G.R) * (a - b);
      var j = 0; while (j < NB - 2 && D.sig[j + 1] > s) j++;
      var t = clamp((D.sig[j] - s) / (D.sig[j] - D.sig[j + 1]), 0, 1), v = 1 - (py - G.T) / (G.h - G.T - G.B);
      var best = -1, bd = 1e9;
      for (var h = 0; h < N; h++) {
        var y = vals[h * NB + j] * (1 - t) + vals[h * NB + j + 1] * t, dd = Math.abs(y - v);
        if (dd < bd) { bd = dd; best = h; }
      }
      return bd * (G.h - G.T - G.B) < 26 ? best : -1;
    }
    function who(h) {
      var b = Math.floor(h / D.n_heads), hh = h % D.n_heads;
      return { b: b + 1, h: hh + 1, kind: b < D.n_double ? "double-stream" : "single-stream", mean: means[h], rank: rankOf[h] + 1 };
    }
    function showTip(h, px, py) {
      if (h < 0) { tip.classList.remove(P + "on"); return; }
      var w = who(h);
      tip.innerHTML = "";
      tip.appendChild(el("b", null, { text: "Block " + w.b + ", head " + w.h }));
      tip.appendChild(el("span", null, { text: w.kind + " block" }));
      tip.appendChild(el("span", null, { text: "mean near share " + f2(w.mean) }));
      tip.classList.add(P + "on");
      var tw = tip.offsetWidth || 160, th = tip.offsetHeight || 60;
      var x = px + 14 + tw > G.w ? px - tw - 14 : px + 14, y = clamp(py - th - 10, 0, G.h - th);
      tip.style.transform = "translate(" + Math.round(x) + "px," + Math.round(y) + "px)";
    }
    function readout() {
      if (sel < 0) { read.textContent = ""; return; }
      var w = who(sel);
      read.innerHTML = "";
      read.appendChild(el("b", null, { text: "Block " + w.b + ", head " + w.h }));
      read.appendChild(document.createTextNode(" \u00b7 " + w.kind + " \u00b7 mean near share " + f2(w.mean) + " \u00b7 rank " + fmtInt(w.rank) + " of " + fmtInt(N) + " by reach"));
      read.appendChild(el("span", "read-h", { text: " Hover or tap a curve; arrow keys step through the heads by reach." }));
    }
    function select(h, quiet) {
      sel = h; readout(); drawOv();
      var b = Math.floor(h / D.n_heads);
      if (b === D.maps.block - 1) { mapHead = h % D.n_heads; }
      updChips(); drawMap();
      ov.setAttribute("aria-label", "Near share of every attention head over the noise level; selected: " + read.textContent);
    }
    function pt(e) { var r = ov.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
    var down = null;
    ov.addEventListener("pointermove", function (e) {
      if (!D) return;
      var p = pt(e), h = nearest(p[0], p[1]);
      if (h !== hov) { hov = h; drawOv(); }
      showTip(h, p[0], p[1]);
      if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 6) down = null;
    });
    ov.addEventListener("pointerleave", function () { hov = -1; drawOv(); showTip(-1); });
    ov.addEventListener("pointerdown", function (e) { down = [e.clientX, e.clientY]; });
    ov.addEventListener("pointerup", function (e) {
      if (!D || !down) return;
      var p = pt(e), h = nearest(p[0], p[1]);
      down = null;
      if (h >= 0) { select(h); hov = h; drawOv(); showTip(h, p[0], p[1]); }
    });
    ov.addEventListener("keydown", function (e) {
      if (!D) return;
      var r = rankOf[sel], k = e.key, n = null;
      if (k === "ArrowDown" || k === "ArrowRight") n = r + 1; else if (k === "ArrowUp" || k === "ArrowLeft") n = r - 1;
      else if (k === "PageDown") n = r + 25; else if (k === "PageUp") n = r - 25; else if (k === "Home") n = 0; else if (k === "End") n = N - 1;
      if (n === null) return;
      e.preventDefault(); hov = -1; showTip(-1);
      select(order[clamp(n, 0, N - 1)]);
    });
    function setView(v, first) {
      view = v;
      bCurves.setAttribute("aria-checked", String(v === "curves")); bMatrix.setAttribute("aria-checked", String(v === "matrix"));
      root.classList.toggle(P + "is-matrix", v === "matrix");
      hov = -1; showTip(-1);
      if (D) drawBase();
    }
    bCurves.addEventListener("click", function () { setView("curves"); });
    bMatrix.addEventListener("click", function () { setView("matrix"); });
    seg.addEventListener("keydown", function (e) {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); setView(view === "curves" ? "matrix" : "curves"); (view === "curves" ? bCurves : bMatrix).focus(); }
    });

    /* ------------------------------------------------------------ the attention map of a block-12 head */
    function buildScrub() {
      scrub.innerHTML = "";
      D.maps.sigma.forEach(function (s, i) {
        var b = el("button", "scrub-b", { type: "button", role: "radio", "aria-checked": String(i === sigI), text: "\u03c3 " + s.toFixed(2) });
        b.addEventListener("click", function () { setSig(i); });
        scrub.appendChild(b);
      });
      scrub.addEventListener("keydown", function (e) {
        var d = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
        if (!d) return;
        e.preventDefault(); setSig(clamp(sigI + d, 0, D.maps.sigma.length - 1)); scrub.children[sigI].focus();
      });
    }
    function setSig(i) {
      sigI = i;
      Array.prototype.forEach.call(scrub.children, function (b, k) { b.setAttribute("aria-checked", String(k === i)); b.tabIndex = k === i ? 0 : -1; });
      drawMap();
    }
    function buildChips() {
      chips.innerHTML = "";
      var mb = D.maps.block - 1;
      for (var h = 0; h < D.n_heads; h++) (function (h) {
        var g = mb * D.n_heads + h, m = means[g];
        var b = el("button", "chip", { type: "button", "aria-pressed": "false", title: "Head " + (h + 1) + ", mean near share " + f2(m),
          "aria-label": "Block " + D.maps.block + " head " + (h + 1) + ", mean near share " + f2(m), text: String(h + 1) });
        b.style.setProperty("--c-heads-chip", css(ramp(m)));
        b.addEventListener("click", function () { select(g); });
        chips.appendChild(b);
      })(h);
    }
    function updChips() {
      Array.prototype.forEach.call(chips.children, function (b, h) { b.setAttribute("aria-pressed", String(h === mapHead)); });
      var b = Math.floor(sel / D.n_heads);
      mapNote.textContent = b === D.maps.block - 1 ? "" : "Maps exist for block " + D.maps.block + " only; showing head " + (mapHead + 1) + ".";
    }
    var hOff = null;
    function drawMap() {
      if (!D || mapHead < 0) return;
      var M = D.maps, n = M.n, r = M.radius || D.radius, qr = M.query[0], qc = M.query[1];
      var w = mapFrame.clientWidth || 300, d = Math.min(2.5, window.devicePixelRatio || 1), W = Math.round(w * d);
      if (mapCv.width !== W) { mapCv.width = W; mapCv.height = W; }
      var ctx = mapCv.getContext("2d");
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = css(C.screen); ctx.fillRect(0, 0, W, W);
      if (img && img.complete && img.naturalWidth) { ctx.imageSmoothingEnabled = true; ctx.drawImage(img, 0, 0, W, W); }
      if (!hOff) { hOff = document.createElement("canvas"); hOff.width = n; hOff.height = n; }
      var hc = hOff.getContext("2d"), im = hc.createImageData(n, n), off = (sigI * D.n_heads + mapHead) * n * n;
      for (var y = 0; y < n; y++) for (var x = 0; x < n; x++) {
        var v = Q[off + y * n + x] / 255, a = Math.pow(v, 0.6), near = Math.max(Math.abs(y - qr), Math.abs(x - qc)) <= r;
        var c = near ? C.near : C.far, o = (y * n + x) * 4;
        im.data[o] = c[0]; im.data[o + 1] = c[1]; im.data[o + 2] = c[2]; im.data[o + 3] = Math.round(a * 255);
      }
      hc.putImageData(im, 0, 0);
      ctx.imageSmoothingEnabled = false; ctx.drawImage(hOff, 0, 0, W, W);
      var s = W / n, R = r;
      var r0 = Math.max(0, qr - R), r1 = Math.min(n, qr + R + 1), c0 = Math.max(0, qc - R), c1 = Math.min(n, qc + R + 1);
      ctx.setLineDash([5 * d, 3.5 * d]); ctx.lineWidth = 1.4 * d; ctx.strokeStyle = css(C.near);
      ctx.strokeRect(c0 * s + .5, r0 * s + .5, (c1 - c0) * s - 1, (r1 - r0) * s - 1); ctx.setLineDash([]);
      var cx = (qc + .5) * s, cy = (qr + .5) * s, q1 = Math.max(7 * d, 1.7 * s), q2 = Math.max(4.5 * d, 1.1 * s);
      ctx.fillStyle = "#fff"; ctx.fillRect(cx - q1 / 2, cy - q1 / 2, q1, q1);
      ctx.fillStyle = css(C.screen); ctx.fillRect(cx - q2 / 2, cy - q2 / 2, q2, q2);
      var g = (M.block - 1) * D.n_heads + mapHead, share = M.share16[sigI][mapHead];
      mapTitle.innerHTML = "";
      mapTitle.appendChild(el("b", null, { text: "Block " + M.block + ", head " + (mapHead + 1) }));
      mapTitle.appendChild(document.createTextNode(" at one query, \u03c3 " + M.sigma[sigI].toFixed(2)));
      mapShare.innerHTML = "";
      mapShare.appendChild(el("span", "map-sv", { text: f2(share) }));
      mapShare.appendChild(el("span", "map-sl", { text: "near share here" }));
      mapCv.setAttribute("aria-label", "Attention of block " + M.block + " head " + (mapHead + 1) + " at the query (the leftmost sheep's face), noise level " + M.sigma[sigI].toFixed(2)
        + ": " + Math.round(share * 100) + " percent of its image attention lies within " + R + " tokens (orange), the rest farther away (blue). Its mean near share over all prompts is " + f2(means[g]) + ".");
    }

    /* ------------------------------------------------------------ size and theme */
    function redraw() { if (!D) return; readColours(); buildChips(); updChips(); drawBase(); drawMap(); }
    function fit() { root.classList.toggle(P + "narrow", (root.clientWidth || 800) < 760); }
    fit();
    if ("ResizeObserver" in window) {
      var rq = 0;
      ro = new ResizeObserver(function () { if (rq) return; rq = requestAnimationFrame(function () { rq = 0; fit(); if (D) { drawBase(); drawMap(); } }); });
      ro.observe(root);
    }
    function onTheme() { requestAnimationFrame(redraw); }
    document.addEventListener("sca:themechange", onTheme);
    if (window.matchMedia) { mq = matchMedia("(prefers-color-scheme: dark)"); if (mq.addEventListener) mq.addEventListener("change", onTheme); }
    mo = new MutationObserver(onTheme);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class"] });

    return {
      select: function (block, head) { if (D) select(clamp((block - 1) * D.n_heads + head - 1, 0, N - 1)); },
      view: function (v) { setView(v === "matrix" ? "matrix" : "curves"); },
      destroy: function () {
        alive = false;
        if (io) io.disconnect(); if (ro) ro.disconnect(); if (mo) mo.disconnect();
        if (mq && mq.removeEventListener) mq.removeEventListener("change", onTheme);
        document.removeEventListener("sca:themechange", onTheme);
        root.innerHTML = ""; root.classList.remove("c-" + NAME);
      }
    };
  }

  SCAComp[NAME] = mount;
})();
