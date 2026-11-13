/* SCAComp.views(el, options): method step 1, the two native-size views of every head. Move the query over a
   4096^2 image: the window view is the native-size key window around the query's tile (orange), the grid view is
   one key per r x r block over the whole image (blue). Both hold n0 x n0 keys at every size, the geometry the model
   was trained on. Vanilla JS, no dependencies, works from file://.

   Data: assets/data/views/data.js (window.SCA_DATA.views), written by web/build/v5_build_views.py; the geometry
   constants (n0, tile, patch) come from the code through that script. No number is typed here.

   var c = SCAComp.views(el, {size: 4096, lazy: true, base: "<data dir url>"});  c.size(2048); c.destroy();
   options  size   image side shown first (1024, 2048 or 4096; default 4096)
   Interaction: drag or tap on the image to move the query; arrow keys move it one tile; buttons switch the size. */
(function () {
  "use strict";
  var NAME = "views", P = "c-views-";
  var SCAComp = window.SCAComp = window.SCAComp || {};
  var CUR = document.currentScript && document.currentScript.src;
  var JSDIR = CUR ? CUR.replace(/[^\/]*$/, "") : "assets/js/";
  var DEF_BASE = JSDIR.replace(/js\/$/, "") + "data/" + NAME + "/";

  function el(tag, cls, attrs, kids) {
    var e = document.createElement(tag);
    if (cls) e.className = cls.split(" ").map(function (c) { return P + c; }).join(" ");
    if (attrs) Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (v === null || v === undefined || v === false) return;
      if (k === "text") e.textContent = v; else e.setAttribute(k, v === true ? "" : v);
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
  function fmtInt(n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
  var probe = null;
  function rgbOf(root, name, fb) {
    var v = getComputedStyle(root).getPropertyValue(name).trim() || fb;
    if (!probe) { probe = document.createElement("span"); probe.style.display = "none"; document.body.appendChild(probe); }
    probe.style.color = ""; probe.style.color = v;
    var m = getComputedStyle(probe).color.match(/[\d.]+/g) || [128, 128, 128];
    return [+m[0], +m[1], +m[2]];
  }
  function css(c, a) { return "rgba(" + Math.round(c[0]) + "," + Math.round(c[1]) + "," + Math.round(c[2]) + "," + (a === undefined ? 1 : a) + ")"; }

  function mount(root, opts) {
    opts = opts || {};
    var base = opts.base || DEF_BASE;
    var D = null, alive = true, io = null, ro = null, mo = null, mq = null;
    var C = {}, S = opts.size || 4096, q = null, img = null, grids = {}, drag = false;
    root.classList.add("c-" + NAME);
    root.innerHTML = "";

    var seg = el("div", "seg", { role: "radiogroup", "aria-label": "Image size" });
    var hint = el("span", "hint", null, [el("span", "hint-m", { text: "Drag or click the image to move the query; arrow keys move it one tile." }),
      el("span", "hint-t", { text: "Tap the image to move the query." })]);
    var top = el("div", "top", null, [seg, hint]);
    var main = el("canvas", "main", { tabindex: "0", role: "img", "aria-label": "The image with the two views of one query" });
    var mainBox = el("div", "mainbox", null, [main]);
    var winCv = el("canvas", "inset", { role: "img" });
    var gridCv = el("canvas", "inset", { role: "img" });
    var winLab = el("div", "il il-near"), gridLab = el("div", "il il-far");
    var winBox = el("figure", "ib", null, [el("div", "if", null, [winCv]), winLab]);
    var gridBox = el("figure", "ib", null, [el("div", "if", null, [gridCv]), gridLab]);
    var insets = el("div", "insets", null, [winBox, gridBox]);
    var stats = el("div", "stats");
    var side = el("div", "side", null, [insets, stats]);
    var body = el("div", "body", null, [mainBox, side]);
    var note = el("p", "note");
    root.appendChild(top); root.appendChild(body); root.appendChild(note);

    function fail() {
      if (!alive) return;
      var fig = root.closest ? root.closest("figure") : null;
      if (fig && fig.querySelector(".comp-fallback")) { fig.classList.remove("is-live"); root.innerHTML = ""; return; }
      root.innerHTML = "";
      root.appendChild(el("img", "fallback", { src: base + "fallback.webp", alt: "A 4096 by 4096 image with the window view (orange box) and the grid view (blue dots) of one query.", loading: "lazy" }));
    }
    function start() {
      if (window.SCA_DATA && window.SCA_DATA[NAME]) { ready(); return; }
      loadScript(base + "data.js", function () { if (window.SCA_DATA && window.SCA_DATA[NAME]) ready(); else fail(); }, fail);
    }
    if (opts.lazy === false) start(); else io = whenNear(root, start);

    function ready() {
      if (!alive) return;
      D = window.SCA_DATA[NAME];
      if (D.sizes.indexOf(S) < 0) S = D.sizes[D.sizes.length - 1];
      q = D.query.slice();
      readColours();
      img = new Image(); img.decoding = "async"; img.onload = draw; img.src = base + D.image;
      D.sizes.forEach(function (s) {
        var g = new Image(); g.decoding = "async"; g.onload = function () { if (s === S) draw(); }; g.src = base + D.grids[s]; grids[s] = g;
        var b = el("button", "seg-b", { type: "button", role: "radio", "aria-checked": String(s === S) }, [
          el("span", "seg-k", { text: s + "\u00b2" }), el("span", "seg-h", { text: "r = " + (s / D.patch / D.n0) })]);
        b.addEventListener("click", function () { setSize(s); });
        seg.appendChild(b);
      });
      seg.addEventListener("keydown", function (e) {
        var d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0; if (!d) return;
        e.preventDefault(); var i = clamp(D.sizes.indexOf(S) + d, 0, D.sizes.length - 1); setSize(D.sizes[i]); seg.children[i].focus();
      });
      note.innerHTML = "";
      var tog = el("button", "tog", { type: "button", "aria-expanded": "false", text: "About this picture" });
      tog.addEventListener("click", function () { var o = root.classList.toggle(P + "open"); tog.setAttribute("aria-expanded", String(o)); tog.textContent = o ? "Less" : "About this picture"; });
      var more = el("span", "more");
      note.appendChild(more); note.appendChild(tog);
      more.textContent = "The picture is progressive SCA's 4096\u00b2 output, resized for the smaller sizes; the boxes and dots are the exact key positions at each size. "
        + "One token is " + D.patch + " \u00d7 " + D.patch + " pixels; queries share a window per " + D.tile + " \u00d7 " + D.tile + "-token tile. "
        + "The grid inset lays the " + D.patch + "-pixel patch of every block-centre token side by side.";
      setSize(S, true);
    }
    function readColours() {
      C.near = rgbOf(root, "--near", "#E08A2B"); C.far = rgbOf(root, "--far", "#2F6DB3"); C.screen = rgbOf(root, "--screen", "#0B0D0C");
      C.ink = rgbOf(root, "--ink", "#111513");
    }
    function geo() {
      var n = S / D.patch, n0 = D.n0, t = D.tile, r = n / n0, w = Math.min(n0, n);
      var row = clamp(Math.floor(q[0] * n), 0, n - 1), col = clamp(Math.floor(q[1] * n), 0, n - 1);
      function st(p) { return clamp(Math.floor(p / t) * t + Math.floor(t / 2) - Math.floor(w / 2), 0, n - w); }
      return { n: n, n0: n0, t: t, r: r, w: w, row: row, col: col, r0: st(row), c0: st(col), tr: Math.floor(row / t) * t, tc: Math.floor(col / t) * t };
    }
    function setSize(s, first) {
      S = s;
      Array.prototype.forEach.call(seg.children, function (b, i) { var on = D.sizes[i] === s; b.setAttribute("aria-checked", String(on)); b.tabIndex = on ? 0 : -1; });
      draw();
    }
    function fit(cv) {
      var w = cv.parentNode.clientWidth || 300, d = Math.min(2.5, window.devicePixelRatio || 1), W = Math.round(w * d);
      if (cv.width !== W) { cv.width = W; cv.height = W; }
      return { ctx: cv.getContext("2d"), W: W, d: d };
    }
    function draw() {
      if (!D || !q) return;
      var g = geo(), m = fit(main), ctx = m.ctx, W = m.W, s = W / g.n;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = css(C.screen); ctx.fillRect(0, 0, W, W);
      if (img && img.complete && img.naturalWidth) { ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high"; ctx.drawImage(img, 0, 0, W, W); }
      ctx.fillStyle = "rgba(11,13,12,.28)"; ctx.fillRect(0, 0, W, W);
      /* grid view: one key per r x r block */
      var rr = Math.max(.9 * m.d, Math.min(s * g.r * .17, 3 * m.d));
      ctx.fillStyle = css(C.far, .88);
      for (var bi = 0; bi < g.n0; bi++) for (var bj = 0; bj < g.n0; bj++) {
        var cy = (g.r * bi + Math.floor(g.r / 2) + .5) * s, cx = (g.r * bj + Math.floor(g.r / 2) + .5) * s;
        ctx.fillRect(cx - rr, cy - rr, 2 * rr, 2 * rr);
      }
      /* window view */
      var x0 = g.c0 * s, y0 = g.r0 * s, ws = g.w * s;
      ctx.fillStyle = css(C.near, .16); ctx.fillRect(x0, y0, ws, ws);
      ctx.lineWidth = 2.5 * m.d; ctx.strokeStyle = css(C.near); ctx.strokeRect(x0 + ctx.lineWidth / 2, y0 + ctx.lineWidth / 2, ws - ctx.lineWidth, ws - ctx.lineWidth);
      /* the query's tile and the query */
      ctx.lineWidth = 1.5 * m.d; ctx.strokeStyle = "#fff"; ctx.strokeRect(g.tc * s, g.tr * s, g.t * s, g.t * s);
      var qx = (g.col + .5) * s, qy = (g.row + .5) * s, qa = Math.max(7 * m.d, 1.5 * s);
      ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(qx, qy, qa * .62, 0, 7); ctx.fill();
      ctx.fillStyle = css(C.screen); ctx.beginPath(); ctx.arc(qx, qy, qa * .34, 0, 7); ctx.fill();
      insetsDraw(g);
      statsDraw(g);
      main.setAttribute("aria-label", "Image at " + S + " by " + S + ": the query at token row " + (g.row + 1) + ", column " + (g.col + 1) + " of " + g.n
        + "; its window view is the " + g.w + " by " + g.w + " token box around its tile; its grid view is one key in every " + g.r + " by " + g.r + " block.");
    }
    function insetsDraw(g) {
      /* window view at native pixels: the window's pixels, cropped from the picture */
      var a = fit(winCv), ctx = a.ctx, W = a.W, k = (img && img.naturalWidth || 2048) / g.n;
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = css(C.screen); ctx.fillRect(0, 0, W, W);
      if (img && img.complete && img.naturalWidth) { ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high"; ctx.drawImage(img, g.c0 * k, g.r0 * k, g.w * k, g.w * k, 0, 0, W, W); }
      var s = W / g.w;
      ctx.lineWidth = 1.5 * a.d; ctx.strokeStyle = "#fff"; ctx.strokeRect((g.tc - g.c0) * s, (g.tr - g.r0) * s, g.t * s, g.t * s);
      /* grid view: the block-centre patches of the whole image, with the window's place on it */
      var b = fit(gridCv), c2 = b.ctx, W2 = b.W, gi = grids[S];
      c2.setTransform(1, 0, 0, 1, 0, 0); c2.fillStyle = css(C.screen); c2.fillRect(0, 0, W2, W2);
      if (gi && gi.complete && gi.naturalWidth) { c2.imageSmoothingEnabled = false; c2.drawImage(gi, 0, 0, W2, W2); }
      var s2 = W2 / g.n;
      c2.lineWidth = 2 * b.d; c2.strokeStyle = css(C.near); c2.strokeRect(g.c0 * s2 + c2.lineWidth / 2, g.r0 * s2 + c2.lineWidth / 2, g.w * s2 - c2.lineWidth, g.w * s2 - c2.lineWidth);
      var n0 = g.n0;
      winLab.innerHTML = ""; gridLab.innerHTML = "";
      winLab.appendChild(el("b", null, { text: "Window view" }));
      winLab.appendChild(el("span", null, { text: g.w + " \u00d7 " + g.w + " keys at full resolution: the " + (g.w * D.patch) + "-pixel square around the query's tile" }));
      gridLab.appendChild(el("b", null, { text: "Grid view" }));
      gridLab.appendChild(el("span", null, { text: n0 + " \u00d7 " + n0 + " keys over the whole image, one per " + g.r + " \u00d7 " + g.r + " block" }));
      winCv.setAttribute("aria-label", "The window view's pixels: a " + (g.w * D.patch) + "-pixel square around the query.");
      gridCv.setAttribute("aria-label", "The grid view: one 16-pixel patch per block, side by side; the orange box marks the window's place.");
    }
    function statsDraw(g) {
      var full = g.n * g.n, keys = g.n0 * g.n0;
      stats.innerHTML = "";
      function row(cls, big, lab) { stats.appendChild(el("div", "st " + cls, null, [el("span", "st-v", { text: big }), el("span", "st-l", { text: lab })])); }
      row("st-full", fmtInt(full), "keys in full attention at " + S + "\u00b2 (" + g.n + " \u00d7 " + g.n + " tokens)");
      row("st-near", fmtInt(g.w * g.w), "keys in the window view");
      row("st-far", fmtInt(keys), "keys in the grid view");
      stats.appendChild(el("p", "st-m", { text: g.r === 1
        ? "At the training size both views are the whole image: the head's own full attention."
        : "Each view keeps the " + g.n0 + " \u00d7 " + g.n0 + " geometry of the training size, so a head's reach applies unchanged; r = " + g.r + " is the factor reach emulation shrinks both views by." }));
    }

    /* ------------------------------------------------------------ input */
    function setFrom(e) {
      var r = main.getBoundingClientRect();
      q = [clamp((e.clientY - r.top) / r.height, 0, .9999), clamp((e.clientX - r.left) / r.width, 0, .9999)];
      draw();
    }
    main.addEventListener("pointerdown", function (e) { if (!D) return; drag = true; try { main.setPointerCapture(e.pointerId); } catch (x) { /* ignore */ } setFrom(e); });
    main.addEventListener("pointermove", function (e) { if (drag && D) setFrom(e); });
    ["pointerup", "pointercancel"].forEach(function (t) { main.addEventListener(t, function () { drag = false; }); });
    main.addEventListener("keydown", function (e) {
      if (!D) return;
      var g = geo(), step = g.t / g.n, dy = 0, dx = 0;
      if (e.key === "ArrowUp") dy = -1; else if (e.key === "ArrowDown") dy = 1; else if (e.key === "ArrowLeft") dx = -1; else if (e.key === "ArrowRight") dx = 1; else return;
      e.preventDefault();
      q = [clamp(q[0] + dy * step, 0, .9999), clamp(q[1] + dx * step, 0, .9999)];
      draw();
    });

    if ("ResizeObserver" in window) {
      var rq = 0;
      ro = new ResizeObserver(function () { if (rq) return; rq = requestAnimationFrame(function () { rq = 0; root.classList.toggle(P + "narrow", (root.clientWidth || 800) < 700); if (D) draw(); }); });
      ro.observe(root);
    }
    root.classList.toggle(P + "narrow", (root.clientWidth || 800) < 700);
    function onTheme() { requestAnimationFrame(function () { if (!D) return; readColours(); draw(); }); }
    document.addEventListener("sca:themechange", onTheme);
    if (window.matchMedia) { mq = matchMedia("(prefers-color-scheme: dark)"); if (mq.addEventListener) mq.addEventListener("change", onTheme); }
    mo = new MutationObserver(onTheme);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class"] });

    return {
      size: function (s) { if (D && D.sizes.indexOf(s) >= 0) setSize(s); },
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
