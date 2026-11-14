/* SCAComp.formation(el, options): pick a patch, watch it form. A recorded progressive SCA run (1024^2, then 2048^2,
   then 4096^2) step by step; the visitor picks a patch and sees it enlarged at each step, the native-size window
   around it (near view) or the native-size grid over the whole image (far view), and how two example heads mix
   the two views at that stage. Vanilla JS, no dependencies, works from file://.

   Data: assets/data/formation/data.js (window.SCA_DATA.formation), written by web/build/v5_build_formation.py from
   the recorded traces (x0 previews, the trace's native window, the weights the run used, the final PNG).
   No number is typed here.

   var c = SCAComp.formation(el, {run: 0, preset: 0, view: "near", lazy: true, base: "<data dir url>"});
   options  run     which traced prompt first (0, 1, 2)
            preset  which preset patch first (0 = the trace's own window, which has its pixels at every step)
            view    "near" (window) or "far" (grid)
            stop    keyframe shown first (default: the last, the final image)
   Interaction: tabs pick the run; click or tap the image (or a numbered preset) to pick a patch; the scrubber, its
   arrow keys and Play step through the run; one switch shows the near view or the far view. */
(function () {
  "use strict";
  var NAME = "formation", P = "c-formation-";
  var SCAComp = window.SCAComp = window.SCAComp || {};
  var CUR = document.currentScript && document.currentScript.src;
  var JSDIR = CUR ? CUR.replace(/[^\/]*$/, "") : "assets/js/";
  var DEF_BASE = JSDIR.replace(/js\/$/, "") + "data/" + NAME + "/";
  var REDUCED = !!(window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches);
  var PATCH_PX = 512;                    /* side of the enlarged patch in 4096^2 pixels: the trace's own window size */

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
  var probe = null;
  function rgbOf(root, name, fb) {
    var v = getComputedStyle(root).getPropertyValue(name).trim() || fb;
    if (!probe) { probe = document.createElement("span"); probe.style.display = "none"; document.body.appendChild(probe); }
    probe.style.color = ""; probe.style.color = v;
    var m = getComputedStyle(probe).color.match(/[\d.]+/g) || [128, 128, 128];
    return [+m[0], +m[1], +m[2]];
  }
  function css(c, a) { return "rgba(" + Math.round(c[0]) + "," + Math.round(c[1]) + "," + Math.round(c[2]) + "," + (a === undefined ? 1 : a) + ")"; }
  function pct(x) { return Math.round(x * 100) + "%"; }

  function mount(root, opts) {
    opts = opts || {};
    var base = opts.base || DEF_BASE;
    var D = null, R = null, alive = true, io = null, ro = null, mo = null, mq = null, timer = 0, C = {};
    var run = opts.run | 0, stop = -1, view = opts.view === "far" ? "far" : "near", q = null, preset = typeof opts.preset === "number" ? opts.preset : 0;
    var imgs = {};                                 /* url -> Image */
    root.classList.add("c-" + NAME);
    root.innerHTML = "";

    /* ------------------------------------------------------------ skeleton */
    var tabs = el("div", "tabs", { role: "tablist", "aria-label": "Run" });
    var vNear = el("button", "vt", { type: "button", role: "radio", "aria-checked": "true", text: "Near view" });
    var vFar = el("button", "vt", { type: "button", role: "radio", "aria-checked": "false", text: "Far view" });
    var vsw = el("div", "vsw", { role: "radiogroup", "aria-label": "View" }, [vNear, vFar]);
    var top = el("div", "top", null, [tabs, vsw]);
    var main = el("canvas", "main", { tabindex: "0", role: "img", "aria-label": "The run at the chosen step" });
    var mainBox = el("div", "mainbox", null, [main]);
    var play = el("button", "play", { type: "button", "aria-pressed": "false", disabled: true });
    var range = el("input", "range", { type: "range", min: "0", max: "1", step: "1", value: "0", "aria-label": "Step of the run", disabled: true });
    var segs = el("div", "segs", { "aria-hidden": "true" });
    var track = el("div", "track", null, [range, segs]);
    var scrub = el("div", "scrub", null, [play, track]);
    var left = el("div", "left", null, [mainBox, scrub]);
    var presetsEl = el("div", "presets", { role: "group", "aria-label": "Preset patches" });
    var patch = el("canvas", "patch", { role: "img", "aria-label": "The chosen patch at this step" });
    var patchBox = el("div", "patchbox", null, [patch]);
    var pcap = el("p", "pcap");
    var bars = el("div", "bars");
    var say = el("p", "say", { "aria-live": "polite" });
    var side = el("div", "side", null, [el("div", "ph", null, [el("span", "ph-t", { text: "Patch" }), presetsEl]), patchBox, pcap, bars, say]);
    var body = el("div", "body", null, [left, side]);
    var small = el("p", "small");
    root.appendChild(top); root.appendChild(body); root.appendChild(small);
    if (REDUCED) play.hidden = true;
    function fit() { root.classList.toggle(P + "narrow", (root.clientWidth || 800) < 700); }
    fit();
    root.classList.add(P + "loading");

    function fail() {
      if (!alive) return;
      root.classList.remove(P + "loading");
      var fig = root.closest ? root.closest("figure") : null;
      if (fig && fig.querySelector(".comp-fallback")) { fig.classList.remove("is-live"); root.innerHTML = ""; return; }
      root.innerHTML = "";
      root.appendChild(el("p", "small", { text: "The run could not load." }));
    }
    function start() {
      if (window.SCA_DATA && window.SCA_DATA[NAME]) { ready(); return; }
      loadScript(base + "data.js", function () { if (window.SCA_DATA && window.SCA_DATA[NAME]) ready(); else fail(); }, fail);
    }
    if (opts.lazy === false) start(); else io = whenNear(root, start);

    function ready() {
      if (!alive) return;
      D = window.SCA_DATA[NAME];
      readColours();
      D.runs.forEach(function (r, i) {
        var b = el("button", "tab", { type: "button", role: "tab", "aria-selected": "false", text: r.title });
        b.addEventListener("click", function () { setRun(i); });
        tabs.appendChild(b);
      });
      tabs.addEventListener("keydown", function (e) {
        var d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0; if (!d) return;
        e.preventDefault(); setRun((run + d + D.runs.length) % D.runs.length); tabs.children[run].focus();
      });
      small.textContent = "Boxes and dots: exact positions. Bars: the weights this run used. No attention maps exist at 4096\u00b2. Earlier steps are previews.";
      range.disabled = false; play.disabled = false; setPlay(false);
      setRun(clamp(run, 0, D.runs.length - 1), true);
      root.classList.remove(P + "loading");
    }
    function readColours() {
      C.near = rgbOf(root, "--near", "#E08A2B"); C.far = rgbOf(root, "--far", "#2F6DB3"); C.screen = rgbOf(root, "--screen", "#0B0D0C");
      C.accent = rgbOf(root, "--accent", "#13865F");
    }

    /* ------------------------------------------------------------ run, stop, patch */
    function stops() { return R.frames.length + 1; }        /* keyframes, then the final image */
    function frame(k) { return k < R.frames.length ? R.frames[k] : null; }
    function stageOf(k) { var f = frame(k); return f ? f.stage : D.stages.length - 1; }
    function sizeOf(k) { var f = frame(k); return f ? f.size : R.size; }
    function setRun(i, first) {
      run = i; R = D.runs[i];
      Array.prototype.forEach.call(tabs.children, function (b, k) { b.setAttribute("aria-selected", String(k === i)); b.tabIndex = k === i ? 0 : -1; });
      presetsEl.innerHTML = "";
      R.presets.forEach(function (p, k) {
        var b = el("button", "pre", { type: "button", "aria-pressed": "false", "aria-label": "Preset patch " + (k + 1) + (k === 0 ? ", recorded at every step" : ""), text: String(k + 1) });
        b.addEventListener("click", function () { setPreset(k); });
        presetsEl.appendChild(b);
      });
      range.max = String(stops() - 1);
      segs.innerHTML = "";
      var n = stops(), counts = D.stages.map(function (s, si) { return R.frames.filter(function (f) { return f.stage === si; }).length + (si === D.stages.length - 1 ? 1 : 0); });
      counts.forEach(function (c, si) { segs.appendChild(el("span", "seg seg-" + si, { style: "flex:" + c }, [el("i"), el("b", null, { text: D.stages[si].size + "\u00b2" })])); });
      if (first) stop = typeof opts.stop === "number" ? clamp(opts.stop | 0, 0, n - 1) : n - 1;
      stop = clamp(stop, 0, n - 1); range.value = String(stop);
      setPreset(clamp(preset < 0 ? 0 : preset, 0, R.presets.length - 1));
      preloadAll();
    }
    function setPreset(k) {
      preset = k;
      var p = R.presets[k];
      q = [p[1] / R.size, p[0] / R.size];
      Array.prototype.forEach.call(presetsEl.children, function (b, i) { b.setAttribute("aria-pressed", String(i === k)); });
      draw();
    }
    function setStop(k) { stop = clamp(k, 0, stops() - 1); range.value = String(stop); draw(); }
    function img(url, cb) {
      var im = imgs[url];
      if (!im) { im = imgs[url] = new Image(); im.decoding = "async"; im.onload = function () { if (alive) draw(); }; im.src = base + url; }
      return im.complete && im.naturalWidth ? im : null;
    }
    function preloadAll() { R.frames.forEach(function (f) { img(f.view); }); img(R.final.view); }

    /* ------------------------------------------------------------ geometry (sca/attn.py window_index, pool_tokens) */
    function geo(S) {
      var n = S / D.patch, n0 = D.n0, t = D.tile, r = n / n0, w = Math.min(n0, n);
      var row = clamp(Math.floor(q[0] * n), 0, n - 1), col = clamp(Math.floor(q[1] * n), 0, n - 1);
      function st(p) { return clamp(Math.floor(p / t) * t + Math.floor(t / 2) - Math.floor(w / 2), 0, n - w); }
      return { n: n, n0: n0, t: t, r: r, w: w, row: row, col: col, r0: st(row), c0: st(col), tr: Math.floor(row / t) * t, tc: Math.floor(col / t) * t };
    }
    function patchRect() {        /* the enlarged patch in 4096^2 pixels, centred on the query, clamped */
      var S = R.size, h = PATCH_PX / 2, cx = clamp(q[1] * S, h, S - h), cy = clamp(q[0] * S, h, S - h);
      if (preset === 0) return [R.box[0], R.box[1], R.box[2]];
      return [Math.round(cx - h), Math.round(cy - h), PATCH_PX];
    }
    function fitCv(cv) {
      var w = cv.parentNode.clientWidth || 300, d = Math.min(2.5, window.devicePixelRatio || 1), W = Math.round(w * d);
      if (cv.width !== W) { cv.width = W; cv.height = W; }
      return { ctx: cv.getContext("2d"), W: W, d: d };
    }
    function reticle(ctx, x, y, s, col, lw) {
      var a = Math.max(6, s * .22);
      ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.beginPath();
      ctx.moveTo(x, y + a); ctx.lineTo(x, y); ctx.lineTo(x + a, y);
      ctx.moveTo(x + s - a, y); ctx.lineTo(x + s, y); ctx.lineTo(x + s, y + a);
      ctx.moveTo(x + s, y + s - a); ctx.lineTo(x + s, y + s); ctx.lineTo(x + s - a, y + s);
      ctx.moveTo(x + a, y + s); ctx.lineTo(x, y + s); ctx.lineTo(x, y + s - a);
      ctx.stroke();
    }

    /* ------------------------------------------------------------ drawing */
    function draw() {
      if (!D || !R || !q) return;
      var f = frame(stop), S = sizeOf(stop), g = geo(S), m = fitCv(main), ctx = m.ctx, W = m.W;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = css(C.screen); ctx.fillRect(0, 0, W, W);
      var im = img(f ? f.view : R.final.view);
      if (im) { ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high"; ctx.drawImage(im, 0, 0, W, W); }
      var s = W / g.n, x0 = g.c0 * s, y0 = g.r0 * s, ws = g.w * s;
      if (view === "near") {
        ctx.fillStyle = "rgba(11,13,12,.42)"; ctx.beginPath(); ctx.rect(0, 0, W, W); ctx.rect(x0, y0, ws, ws); ctx.fill("evenodd");
        ctx.lineWidth = 2.5 * m.d; ctx.strokeStyle = css(C.near); ctx.strokeRect(x0 + ctx.lineWidth / 2, y0 + ctx.lineWidth / 2, ws - ctx.lineWidth, ws - ctx.lineWidth);
      } else {
        ctx.fillStyle = "rgba(11,13,12,.18)"; ctx.fillRect(0, 0, W, W);
        var rr = Math.max(.9 * m.d, Math.min(s * g.r * .16, 2.8 * m.d));
        ctx.fillStyle = css(C.far, .92);
        for (var bi = 0; bi < g.n0; bi++) for (var bj = 0; bj < g.n0; bj++)
          ctx.fillRect((g.r * bj + Math.floor(g.r / 2) + .5) * s - rr, (g.r * bi + Math.floor(g.r / 2) + .5) * s - rr, 2 * rr, 2 * rr);
        ctx.lineWidth = 1.5 * m.d; ctx.strokeStyle = css(C.near, .75); ctx.setLineDash([5 * m.d, 4 * m.d]);
        ctx.strokeRect(x0 + 1, y0 + 1, ws - 2, ws - 2); ctx.setLineDash([]);
      }
      var pr = patchRect(), k = W / R.size;
      reticle(ctx, pr[0] * k, pr[1] * k, pr[2] * k, css(C.accent), 2.2 * m.d);
      ctx.lineWidth = 1.5 * m.d; ctx.strokeStyle = "#fff"; ctx.strokeRect(g.tc * s, g.tr * s, g.t * s, g.t * s);
      drawPatch(f, pr);
      text(f, S, g);
    }
    function drawPatch(f, pr) {
      var m = fitCv(patch), ctx = m.ctx, W = m.W, src = null, sx = 0, sy = 0, sw = 0, native = false, own = false;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = css(C.screen); ctx.fillRect(0, 0, W, W);
      if (!f) {                                     /* the final image: native tiles */
        var T = R.final.tiles, c0 = Math.floor(pr[0] / T.size), c1 = Math.floor((pr[0] + pr[2] - 1) / T.size);
        var r0 = Math.floor(pr[1] / T.size), r1 = Math.floor((pr[1] + pr[2] - 1) / T.size), all = true;
        ctx.imageSmoothingEnabled = W < pr[2] * .95;
        for (var r = r0; r <= r1; r++) for (var c = c0; c <= c1; c++) {
          var ti = img(T.path.replace("{c}", c).replace("{r}", r));
          if (!ti) { all = false; continue; }
          var k = W / pr[2];
          ctx.drawImage(ti, (c * T.size - pr[0]) * k, (r * T.size - pr[1]) * k, T.size * k, T.size * k);
        }
        if (!all) { var ov = img(R.final.view); if (ov) { var kk = ov.naturalWidth / R.size; ctx.globalCompositeOperation = "destination-over"; ctx.drawImage(ov, pr[0] * kk, pr[1] * kk, pr[2] * kk, pr[2] * kk, 0, 0, W, W); ctx.globalCompositeOperation = "source-over"; } }
        native = true;
      } else if (preset === 0) {                    /* the trace's own window, decoded at this step */
        src = img(f.crop); own = true;
        if (src) { ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high"; ctx.drawImage(src, 0, 0, W, W); }
      } else {                                      /* any other patch: the step's preview */
        src = img(f.view);
        if (src) { var k2 = src.naturalWidth / R.size; ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high"; ctx.drawImage(src, pr[0] * k2, pr[1] * k2, pr[2] * k2, pr[2] * k2, 0, 0, W, W); }
      }
      var stepTxt = f ? "Step " + f.step + " of " + D.steps_total : "Final image";
      var res = f ? (own ? (f.crop_px >= pr[2] ? "native pixels" : "decoded at " + f.size + "\u00b2") : "preview") : "native pixels";
      pcap.textContent = stepTxt + " \u00b7 " + res;
      patch.setAttribute("aria-label", "The chosen " + pr[2] + "-pixel patch, " + stepTxt.toLowerCase() + ", " + res + ".");
    }
    function bar(label, near) {
      var row = el("div", "bar"), w = clamp(near, 0, 1);
      row.appendChild(el("span", "bar-l", { text: label }));
      var t = el("span", "bar-t", { role: "img", "aria-label": label + ": window " + pct(w) + ", grid " + pct(1 - w) });
      if (w > 0) t.appendChild(el("i", "bar-n", { style: "width:" + (w * 100) + "%" }));
      if (w < 1) t.appendChild(el("i", "bar-f", { style: "width:" + ((1 - w) * 100) + "%" }));
      row.appendChild(t);
      row.appendChild(el("span", "bar-v", null, [w > 0 ? el("b", "bv-n", { text: "window " + pct(w) }) : null, w > 0 && w < 1 ? document.createTextNode(" \u00b7 ") : null,
        w < 1 ? el("b", "bv-f", { text: "grid " + pct(1 - w) }) : null]));
      return row;
    }
    function text(f, S, g) {
      var st = stageOf(stop);
      bars.innerHTML = "";
      var fr = f || R.frames[R.frames.length - 1];
      if (fr && fr.near !== undefined) {
        bars.appendChild(bar("Head " + D.near, fr.near));
        bars.appendChild(bar("Head " + D.far, fr.far));
        bars.classList.remove(P + "off");
      } else {
        bars.appendChild(el("div", "bar-o", { text: "Every head attends to the whole image." }));
        bars.classList.add(P + "off");
      }
      say.textContent = st === 0
        ? S + "\u00b2, the training size: the layout forms with the model's own attention."
        : S + "\u00b2: each head mixes a native-size window and a native-size grid; near-looking heads read the window, far-looking heads the grid.";
      range.setAttribute("aria-valuetext", (f ? "step " + f.step + " of " + D.steps_total + ", " + S + " squared" : "final image"));
      main.setAttribute("aria-label", R.title + " at " + (f ? "step " + f.step : "the end") + " (" + S + " squared): the " + g.w + " by " + g.w + "-token window around the patch"
        + (view === "far" ? " and the " + g.n0 + " by " + g.n0 + " grid over the whole image" : "") + ".");
    }

    /* ------------------------------------------------------------ input */
    function pick(e) {
      var r = main.getBoundingClientRect();
      q = [clamp((e.clientY - r.top) / r.height, 0, .9999), clamp((e.clientX - r.left) / r.width, 0, .9999)];
      preset = -1;
      Array.prototype.forEach.call(presetsEl.children, function (b) { b.setAttribute("aria-pressed", "false"); });
      draw();
    }
    var drag = false;
    main.addEventListener("pointerdown", function (e) { if (!D) return; drag = true; try { main.setPointerCapture(e.pointerId); } catch (x) { /* ignore */ } pick(e); });
    main.addEventListener("pointermove", function (e) { if (drag && D) pick(e); });
    ["pointerup", "pointercancel"].forEach(function (t) { main.addEventListener(t, function () { drag = false; }); });
    main.addEventListener("keydown", function (e) {
      if (!D) return;
      var d = 1 / 16, dy = e.key === "ArrowUp" ? -d : e.key === "ArrowDown" ? d : 0, dx = e.key === "ArrowLeft" ? -d : e.key === "ArrowRight" ? d : 0;
      if (!dx && !dy) return;
      e.preventDefault(); preset = -1; q = [clamp(q[0] + dy, 0, .9999), clamp(q[1] + dx, 0, .9999)];
      Array.prototype.forEach.call(presetsEl.children, function (b) { b.setAttribute("aria-pressed", "false"); });
      draw();
    });
    range.addEventListener("input", function () { if (D) { setPlay(false); setStop(+range.value); } });
    function setView(v) {
      view = v; vNear.setAttribute("aria-checked", String(v === "near")); vFar.setAttribute("aria-checked", String(v === "far"));
      draw();
    }
    vNear.addEventListener("click", function () { setView("near"); });
    vFar.addEventListener("click", function () { setView("far"); });
    vsw.addEventListener("keydown", function (e) {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      e.preventDefault(); setView(view === "near" ? "far" : "near"); (view === "near" ? vNear : vFar).focus();
    });
    function setPlay(on) {
      if (timer) { clearInterval(timer); timer = 0; }
      play.setAttribute("aria-pressed", String(on));
      play.setAttribute("aria-label", on ? "Pause" : "Play the run");
      play.innerHTML = on ? '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h3.6v14H7zM13.4 5H17v14h-3.6z" fill="currentColor"/></svg>'
        : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13l10.5-6.5z" fill="currentColor"/></svg>';
      if (on) timer = setInterval(function () {
        if (!alive || !D) return;
        if (stop >= stops() - 1) { setPlay(false); return; }
        setStop(stop + 1);
      }, 650);
    }
    play.addEventListener("click", function () {
      if (!D) return;
      var on = play.getAttribute("aria-pressed") !== "true";
      if (on && stop >= stops() - 1) setStop(0);
      setPlay(on);
    });
    vNear.setAttribute("aria-checked", String(view === "near")); vFar.setAttribute("aria-checked", String(view === "far"));

    if ("ResizeObserver" in window) {
      var rq = 0;
      ro = new ResizeObserver(function () { if (rq) return; rq = requestAnimationFrame(function () { rq = 0; fit(); if (D) draw(); }); });
      ro.observe(root);
    }
    function onTheme() { requestAnimationFrame(function () { if (!D) return; readColours(); draw(); }); }
    document.addEventListener("sca:themechange", onTheme);
    if (window.matchMedia) { mq = matchMedia("(prefers-color-scheme: dark)"); if (mq.addEventListener) mq.addEventListener("change", onTheme); }
    mo = new MutationObserver(onTheme);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class"] });
    var vis = null;
    if ("IntersectionObserver" in window) { vis = new IntersectionObserver(function (es) { if (!es[es.length - 1].isIntersecting && timer) setPlay(false); }); vis.observe(root); }

    return {
      run: function (i) { if (D) setRun(clamp(i | 0, 0, D.runs.length - 1)); },
      stop: function (k) { if (D) { setPlay(false); setStop(k | 0); } },
      destroy: function () {
        alive = false; if (timer) clearInterval(timer);
        if (io) io.disconnect(); if (ro) ro.disconnect(); if (mo) mo.disconnect(); if (vis) vis.disconnect();
        if (mq && mq.removeEventListener) mq.removeEventListener("change", onTheme);
        document.removeEventListener("sca:themechange", onTheme);
        root.innerHTML = ""; root.classList.remove("c-" + NAME);
      }
    };
  }

  SCAComp[NAME] = mount;
})();
