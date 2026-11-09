/* Synced deep-zoom grids for the SCA pages (index.html and review/). Vanilla JS, no dependencies, works from file://.

   var g = SCAZoom.grid(spec); host.appendChild(g.el);
   spec  size     [W, H], the native size every panel shares
         panels   [{label, ours, view, vs: [w, h], tiles: {path "...{c}_{r}.webp", size, cols, rows} | rt: [[x, y, w, h, src]]}]
         regions  [[cx, cy, span]], the auto-zoom stops in native pixels (built per grid from an edge-energy map)
         alt      text for the panels' labels; minW, maxCols: panel layout; eager: load the overviews at once
   Scroll or pinch to zoom and drag to pan on any panel: every panel shows the same region. Double-click or
   double-tap resets. Auto zoom flies through the stops; any input stops it and leaves the view where it is.
   At the whole-image view each panel is a plain <img>; canvases exist only while a grid is zoomed, and
   native-pixel tiles load only when the view needs more pixels than the overview holds. */
(function () {
  "use strict";
  var REDUCED = !!(window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches);
  var HOLD = 2300;                       /* ms at each stop of the tour */

  function el(tag, attrs, kids) {
    var e = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (v === null || v === undefined || v === false) return;
      if (k === "class") e.className = v; else if (k === "text") e.textContent = v; else if (k === "html") e.innerHTML = v;
      else e.setAttribute(k, v === true ? "" : v);
    });
    (kids || []).forEach(function (c) { if (c) e.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return e;
  }
  function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
  function dpr() { return Math.min(2.5, window.devicePixelRatio || 1); }
  function ease(t) { return t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }
  function cosh(x) { x = Math.exp(x); return (x + 1 / x) / 2; }
  function sinh(x) { x = Math.exp(x); return (x - 1 / x) / 2; }
  function tanh(x) { x = Math.exp(2 * x); return (x - 1) / (x + 1); }

  /* Smooth zoom and pan along the path of van Wijk and Nuij (2003), as d3's interpolateZoom: p = [cx, cy, width]. */
  function flight(p0, p1) {
    var rho = Math.SQRT2, ux0 = p0[0], uy0 = p0[1], w0 = p0[2], ux1 = p1[0], uy1 = p1[1], w1 = p1[2];
    var dx = ux1 - ux0, dy = uy1 - uy0, d2 = dx * dx + dy * dy, S, f;
    if (d2 < 1e-6) {
      S = Math.log(w1 / w0) / rho;
      f = function (t) { return [ux0 + t * dx, uy0 + t * dy, w0 * Math.exp(rho * t * S)]; };
    } else {
      var d1 = Math.sqrt(d2), b0 = (w1 * w1 - w0 * w0 + 4 * d2) / (4 * w0 * d1), b1 = (w1 * w1 - w0 * w0 - 4 * d2) / (4 * w1 * d1);
      var r0 = Math.log(Math.sqrt(b0 * b0 + 1) - b0), r1 = Math.log(Math.sqrt(b1 * b1 + 1) - b1);
      S = (r1 - r0) / rho;
      f = function (t) {
        var s = t * S, c0 = cosh(r0), u = w0 / (2 * d1) * (c0 * tanh(rho * s + r0) - sinh(r0));
        return [ux0 + u * dx, uy0 + u * dy, w0 * c0 / cosh(rho * s + r0)];
      };
    }
    f.duration = Math.abs(S) * 1000;
    return f;
  }

  var ICON_PLAY = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13l10.5-6.5z" fill="currentColor"/></svg>';
  var ICON_PAUSE = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h3.6v14H7zM13.4 5H17v14h-3.6z" fill="currentColor"/></svg>';
  var ICON_RESET = '<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9V4h5M20 15v5h-5M4 4l6 6M20 20l-6-6"/></svg>';

  function grid(spec) {
    var W = spec.size[0], H = spec.size[1], base = spec.base || "";
    var regions = (spec.regions || []).filter(function (r) { return r && r.length >= 2; });
    var st = { cx: W / 2, cy: H / 2, z: 1 };
    var root = el("div", { class: "zg" + (spec.cls ? " " + spec.cls : ""), role: "group", "aria-label": spec.label || "Zoomable comparison" });
    var bar = el("div", { class: "zg-bar" });
    var autoB = el("button", { type: "button", class: "zg-btn zg-auto", "aria-pressed": "false", title: "Fly through the most detailed regions; any input takes over" });
    var resetB = el("button", { type: "button", class: "zg-btn zg-reset", html: ICON_RESET + "<span>Whole image</span>", title: "Back to the whole image (or double-click a panel)" });
    var read = el("span", { class: "zg-read", "aria-live": "off" });
    var hint = el("span", { class: "zg-hint", html: '<span class="zg-h-m">Scroll to zoom, drag to pan, double-click to reset</span><span class="zg-h-t">Pinch to zoom, drag to pan, double-tap to reset</span>' });
    if (regions.length) bar.appendChild(autoB);
    bar.appendChild(resetB); bar.appendChild(read); bar.appendChild(hint);
    var gridEl = el("div", { class: "zg-grid" });
    gridEl.style.setProperty("--zg-cols", Math.min(spec.panels.length, spec.maxCols || 4));   /* until layout() measures */
    if (spec.barBelow) { root.appendChild(gridEl); root.appendChild(bar); } else { root.appendChild(bar); root.appendChild(gridEl); }
    var panels = spec.panels.map(makePanel);
    var map = el("div", { class: "zg-map", "aria-hidden": "true" }), mapR = el("span", { class: "zg-map-r" });
    map.appendChild(el("img", { src: base + spec.panels[0].view, alt: "", loading: spec.eager ? "eager" : "lazy", decoding: "async", draggable: "false" }));
    map.appendChild(mapR);
    panels[0].box.appendChild(map);

    var zoomed = false, visible = true, raf = 0, anim = null, inflight = 0, queue = [];
    var tour = { on: false, next: 0, timer: 0 };
    var ptrs = {}, gest = null, seq = null, lastTap = null, lastDbl = 0, ro = null, io = null, bg = "#888";

    function makePanel(p, i) {
      var box = el("div", { class: "zg-p" + (p.ours ? " ours" : ""), tabindex: "0", role: "img",
                            "aria-label": p.label + (spec.alt ? ": " + spec.alt : "") + ". Zoom and pan move every panel of this grid together." });
      box.style.aspectRatio = W + " / " + H;
      var im = el("img", { class: "zg-img", src: base + p.view, alt: "", loading: spec.eager ? "eager" : "lazy", decoding: "async",
                           draggable: "false", width: p.vs ? p.vs[0] : null, height: p.vs ? p.vs[1] : null });
      box.appendChild(im);
      box.appendChild(el("span", { class: "zg-tag", text: p.label }));
      var dot = el("span", { class: "zg-load", "aria-hidden": "true" });
      box.appendChild(dot);
      var P = { i: i, p: p, box: box, img: im, dot: dot, cv: null, ctx: null, tiles: {}, pending: 0, pw: 0, ph: 0, ok: !!(im.complete && im.naturalWidth) };
      im.addEventListener("load", function () { P.ok = true; dirty(); });
      bind(P);
      gridEl.appendChild(box);
      return P;
    }

    /* ---------------------------------------------------------------- geometry */
    function measure() {
      panels.forEach(function (P) {
        var r = P.box.getBoundingClientRect();
        P.pw = r.width; P.ph = r.height;
        if (P.cv) {
          var d = dpr(), w = Math.max(1, Math.round(r.width * d)), h = Math.max(1, Math.round(r.height * d));
          if (P.cv.width !== w || P.cv.height !== h) { P.cv.width = w; P.cv.height = h; }
        }
      });
    }
    function s0(P) { return Math.min(P.pw / W, P.ph / H) || 1e-6; }
    function zmax() { var P = panels[0]; return Math.max(2, (6 / dpr()) / s0(P)); }
    function clampSt() {
      var P = panels[0];
      st.z = clamp(st.z, 1, zmax());
      var s = s0(P) * st.z, hw = P.pw / (2 * s), hh = P.ph / (2 * s);
      st.cx = 2 * hw >= W ? W / 2 : clamp(st.cx, hw, W - hw);
      st.cy = 2 * hh >= H ? H / 2 : clamp(st.cy, hh, H - hh);
    }
    function nativeAt(P, r, X, Y) { var s = s0(P) * st.z; return [st.cx + (X - r.left - P.pw / 2) / s, st.cy + (Y - r.top - P.ph / 2) / s]; }
    function place(P, r, n, X, Y) {       /* put native point n under screen point (X, Y) */
      var s = s0(P) * st.z;
      st.cx = n[0] - (X - r.left - P.pw / 2) / s; st.cy = n[1] - (Y - r.top - P.ph / 2) / s;
    }
    function layout() {
      var w = gridEl.clientWidth || root.clientWidth || 600, n = panels.length, gap = 6;
      var minW = spec.minW ? (w < 560 ? Math.min(spec.minW, 140) : spec.minW) : (w < 560 ? 140 : 210);
      var cols = Math.max(1, Math.min(n, spec.maxCols || n, Math.floor((w + gap) / (minW + gap))));
      cols = Math.ceil(n / Math.ceil(n / cols));
      gridEl.style.setProperty("--zg-cols", cols);
    }

    /* ---------------------------------------------------------------- drawing */
    function setZoomed(on) {
      if (on === zoomed) return;
      zoomed = on;
      if (on) {
        measure();
        bg = getComputedStyle(panels[0].box).backgroundColor || "#888";
        panels.forEach(function (P) {
          if (!P.cv) {
            P.cv = el("canvas", { class: "zg-cv", "aria-hidden": "true" });
            P.box.insertBefore(P.cv, P.img.nextSibling);
            P.ctx = P.cv.getContext("2d");
          }
        });
        measure();
        drawAll();
      } else {
        panels.forEach(function (P) { if (P.cv) { P.cv.width = P.cv.height = 0; P.box.removeChild(P.cv); P.cv = P.ctx = null; } });
      }
      root.classList.toggle("zoomed", on);
    }
    function blit(ctx, im, x0, y0, vw, vh, tx, ty, tw, th, sd, snap) {
      var ix0 = Math.max(x0, tx), iy0 = Math.max(y0, ty), ix1 = Math.min(x0 + vw, tx + tw), iy1 = Math.min(y0 + vh, ty + th);
      if (ix1 <= ix0 || iy1 <= iy0) return;
      var kx = im.naturalWidth / tw, ky = im.naturalHeight / th;
      var dx0 = (ix0 - x0) * sd, dy0 = (iy0 - y0) * sd, dx1 = (ix1 - x0) * sd, dy1 = (iy1 - y0) * sd;
      if (snap) { dx0 = Math.floor(dx0); dy0 = Math.floor(dy0); dx1 = Math.ceil(dx1); dy1 = Math.ceil(dy1); }
      var sx = clamp((ix0 - tx) * kx, 0, im.naturalWidth), sy = clamp((iy0 - ty) * ky, 0, im.naturalHeight);
      var sw = Math.min(im.naturalWidth - sx, (ix1 - ix0) * kx), sh = Math.min(im.naturalHeight - sy, (iy1 - iy0) * ky);
      if (sw > 0 && sh > 0) ctx.drawImage(im, sx, sy, sw, sh, dx0, dy0, dx1 - dx0, dy1 - dy0);
    }
    function tilesIn(P, x0, y0, vw, vh) {
      var out = [], p = P.p;
      if (p.tiles) {
        var g = p.tiles, T = g.size;
        var c0 = Math.max(0, Math.floor(x0 / T)), c1 = Math.min(g.cols - 1, Math.floor((x0 + vw - 1e-6) / T));
        var r0 = Math.max(0, Math.floor(y0 / T)), r1 = Math.min(g.rows - 1, Math.floor((y0 + vh - 1e-6) / T));
        for (var r = r0; r <= r1; r++) for (var c = c0; c <= c1; c++)
          out.push({ key: c + "_" + r, x: c * T, y: r * T, w: Math.min(T, W - c * T), h: Math.min(T, H - r * T), src: g.path.replace("{c}", c).replace("{r}", r) });
      } else if (p.rt) {
        p.rt.forEach(function (q, k) {
          if (q[0] < x0 + vw && q[0] + q[2] > x0 && q[1] < y0 + vh && q[1] + q[3] > y0) out.push({ key: "r" + k, x: q[0], y: q[1], w: q[2], h: q[3], src: q[4] });
        });
      }
      return out;
    }
    function tile(P, t) {
      var T = P.tiles[t.key];
      if (!T) {
        T = P.tiles[t.key] = { img: new Image(), ok: false, started: false, src: base + t.src, P: P };
        queue.push(T); P.pending++; P.dot.classList.add("on");
        pump();
      }
      return T;
    }
    function pump() {
      while (inflight < 6 && queue.length) {
        var T = queue.shift();
        if (T.started) continue;
        T.started = true; inflight++;
        (function (T) {
          T.img.decoding = "async";
          T.img.onload = function () { T.ok = true; done(T); dirty(); };
          T.img.onerror = function () { done(T); };
          T.img.src = T.src;
        })(T);
      }
    }
    function done(T) { inflight--; T.P.pending--; if (T.P.pending <= 0) { T.P.pending = 0; T.P.dot.classList.remove("on"); } pump(); }
    function drawPanel(P) {
      if (!P.cv || !P.pw) return;
      var ctx = P.ctx, d = P.cv.width / P.pw, s = s0(P) * st.z, sd = s * d;
      var vw = P.pw / s, vh = P.ph / s, x0 = st.cx - vw / 2, y0 = st.cy - vh / 2;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = bg; ctx.fillRect(0, 0, P.cv.width, P.cv.height);
      var k = 0;
      if (P.ok && P.img.naturalWidth) {
        k = P.img.naturalWidth / W;
        ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
        blit(ctx, P.img, x0, y0, vw, vh, 0, 0, W, H, sd, false);
      }
      if (sd > k * 1.12) {
        var list = tilesIn(P, x0, y0, vw, vh);
        ctx.imageSmoothingEnabled = sd < 2.2;
        list.forEach(function (t) { var T = tile(P, t); if (T.ok) blit(ctx, T.img, x0, y0, vw, vh, t.x, t.y, t.w, t.h, sd, !!P.p.tiles); });
      }
    }
    function drawAll() {
      if (!zoomed || !visible) return;
      panels.forEach(drawPanel);
    }
    function ui() {
      var P = panels[0], s = s0(P) * st.z, vw = P.pw / s, vh = P.ph / s, m = s * dpr();
      mapR.style.left = ((st.cx - vw / 2) / W * 100).toFixed(2) + "%"; mapR.style.top = ((st.cy - vh / 2) / H * 100).toFixed(2) + "%";
      mapR.style.width = (vw / W * 100).toFixed(2) + "%"; mapR.style.height = (vh / H * 100).toFixed(2) + "%";
      read.textContent = st.z < 1.005 ? "" : (m > .92 ? (m < 1.45 ? "native pixels, 1:1" : "native pixels, " + Math.round(m) + ":1") : "zoom " + st.z.toFixed(1) + "×");
    }
    function dirty() { if (!raf) raf = requestAnimationFrame(function () { raf = 0; drawAll(); }); }
    function changed() { setZoomed(st.z > 1.004 || !!anim); ui(); dirty(); }

    /* ---------------------------------------------------------------- the tour and animated moves */
    function stopView(k) {
      var P = panels[0], cap = Math.min(W, H), p = P.p;
      if (p.rt && p.rt.length) cap = Math.min(cap, p.rt[k % p.rt.length][2]);
      else if (p.tiles) cap = Math.min(cap, 1024);
      var span = clamp(P.pw * dpr(), Math.min(240, cap), cap), r = regions[k];
      return [r[0], r[1], span];
    }
    function prefetch(v) {
      var P0 = panels[0], vw = v[2], vh = vw * P0.ph / (P0.pw || 1);
      panels.forEach(function (P) { tilesIn(P, v[0] - vw / 2, v[1] - vh / 2, vw, vh).forEach(function (t) { tile(P, t); }); });
    }
    function cancelAnim() { if (anim) { cancelAnimationFrame(anim.raf); anim = null; } }
    function animateTo(v, then) {          /* v = [cx, cy, visible width in native px] */
      cancelAnim();
      if (v[2] < W) prefetch(v);
      var f = flight([st.cx, st.cy, W / st.z], v), dur = REDUCED ? 1 : clamp(f.duration * .85, 650, 2400), t0 = performance.now();
      anim = { raf: 0 };
      setZoomed(true);
      var step = function (now) {
        if (!anim) return;
        var t = clamp((now - t0) / dur, 0, 1), q = f(ease(t));
        st.cx = q[0]; st.cy = q[1]; st.z = W / q[2]; clampSt(); ui(); drawAll();
        if (t < 1) anim.raf = requestAnimationFrame(step);
        else { anim = null; changed(); if (then) then(); }
      };
      anim.raf = requestAnimationFrame(step);
    }
    function setAuto() {
      autoB.setAttribute("aria-pressed", String(tour.on));
      autoB.innerHTML = tour.on ? ICON_PAUSE + "<span>Pause</span>" : ICON_PLAY + "<span>Auto zoom</span>";
      root.classList.toggle("touring", tour.on);
    }
    function legOf(k) {
      if (!tour.on) return;
      if (k >= regions.length) { animateTo([W / 2, H / 2, W], function () { tour.on = false; tour.next = 0; setAuto(); }); return; }
      animateTo(stopView(k), function () {
        tour.next = k + 1;
        tour.timer = setTimeout(function () { legOf(tour.next); }, HOLD);
      });
    }
    function play() { if (!regions.length) return; measure(); tour.on = true; setAuto(); legOf(tour.next % (regions.length + 1)); }
    function pause() { if (!tour.on && !anim) return; tour.on = false; clearTimeout(tour.timer); cancelAnim(); setAuto(); changed(); }
    function takeOver() { if (tour.on || anim) pause(); }
    function resetView() { takeOver(); tour.next = 0; if (st.z > 1.004) animateTo([W / 2, H / 2, W]); }
    autoB.addEventListener("click", function () { if (tour.on) pause(); else play(); });
    resetB.addEventListener("click", resetView);
    setAuto(); ui();

    /* ---------------------------------------------------------------- input */
    function bind(P) {
      var box = P.box;
      box.addEventListener("pointerdown", function (e) {
        if (e.button) return;
        if (map.contains(e.target)) return;
        if (!Object.keys(ptrs).length) seq = { multi: false, moved: false };
        try { box.setPointerCapture(e.pointerId); } catch (x) { /* old browsers */ }
        ptrs[e.pointerId] = { x: e.clientX, y: e.clientY };
        if (Object.keys(ptrs).length > 1) seq.multi = true;
        takeOver(); measure(); begin(P);
        if (e.pointerType === "mouse") e.preventDefault();
      });
      box.addEventListener("pointermove", function (e) {
        var q = ptrs[e.pointerId]; if (!q) return;
        q.x = e.clientX; q.y = e.clientY; move(P);
      });
      var end = function (e) {
        var q = ptrs[e.pointerId]; if (!q) return;
        delete ptrs[e.pointerId];
        var left = Object.keys(ptrs).length;
        if (!left && e.type === "pointerup" && e.pointerType !== "mouse" && seq && !seq.multi && !seq.moved) {
          var now = performance.now();
          if (lastTap && now - lastTap.t < 330 && Math.hypot(q.x - lastTap.x, q.y - lastTap.y) < 32) { lastTap = null; dbl(P, q.x, q.y); }
          else lastTap = { t: now, x: q.x, y: q.y };
        }
        if (left) begin(P); else gest = null;
      };
      box.addEventListener("pointerup", end);
      box.addEventListener("pointercancel", end);
      box.addEventListener("lostpointercapture", end);
      box.addEventListener("dblclick", function (e) { e.preventDefault(); if (performance.now() - lastDbl > 600) dbl(P, e.clientX, e.clientY); });
      box.addEventListener("wheel", function (e) {
        var dy = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1);
        if (!dy) return;
        measure();
        var z1 = clamp(st.z * Math.pow(2, -dy * (e.ctrlKey ? .01 : .002)), 1, zmax());
        if (Math.abs(z1 - st.z) < 1e-4) return;          /* at a limit: the page scrolls */
        e.preventDefault(); takeOver();
        var r = box.getBoundingClientRect(), n = nativeAt(P, r, e.clientX, e.clientY);
        st.z = z1; place(P, r, n, e.clientX, e.clientY); clampSt(); changed();
      }, { passive: false });
      box.addEventListener("keydown", function (e) {
        var k = e.key, s = s0(P) * st.z, stp = P.pw / s * .12;
        measure();
        if (k === "+" || k === "=") st.z *= 1.5; else if (k === "-" || k === "_") st.z /= 1.5;
        else if (k === "ArrowLeft") st.cx -= stp; else if (k === "ArrowRight") st.cx += stp;
        else if (k === "ArrowUp") st.cy -= stp; else if (k === "ArrowDown") st.cy += stp;
        else if (k === "0") { e.preventDefault(); resetView(); return; } else return;
        if (k.indexOf("Arrow") === 0 && st.z < 1.004) return;     /* at the whole image the arrows scroll the page */
        e.preventDefault(); takeOver(); clampSt(); changed();
      });
    }
    function begin(P) {
      var ids = Object.keys(ptrs), a = ptrs[ids[0]], b = ptrs[ids[1]], r = P.box.getBoundingClientRect();
      var c = b ? [(a.x + b.x) / 2, (a.y + b.y) / 2] : [a.x, a.y];
      gest = { P: P, r: r, c: c, d: b ? Math.hypot(a.x - b.x, a.y - b.y) : 0, z: st.z, n: nativeAt(P, r, c[0], c[1]) };
    }
    function move(P) {
      if (!gest) return;
      var ids = Object.keys(ptrs), a = ptrs[ids[0]], b = ptrs[ids[1]], g = gest;
      var c = b ? [(a.x + b.x) / 2, (a.y + b.y) / 2] : [a.x, a.y];
      if (!b && !seq.moved && Math.hypot(c[0] - g.c[0], c[1] - g.c[1]) < 4) return;
      seq.moved = true;
      if (b && g.d > 0) st.z = clamp(g.z * Math.hypot(a.x - b.x, a.y - b.y) / g.d, 1, zmax());
      place(g.P, g.r, g.n, c[0], c[1]); clampSt(); changed();
    }
    function dbl(P, X, Y) {
      lastDbl = performance.now(); takeOver(); measure();
      if (st.z > 1.05) { tour.next = 0; animateTo([W / 2, H / 2, W]); return; }
      var r = P.box.getBoundingClientRect(), n = nativeAt(P, r, X, Y);
      animateTo([n[0], n[1], W / Math.min(4, zmax())]);
    }
    function mapAt(e) {
      var r = map.getBoundingClientRect();
      st.cx = clamp((e.clientX - r.left) / r.width, 0, 1) * W; st.cy = clamp((e.clientY - r.top) / r.height, 0, 1) * H;
      clampSt(); changed();
    }
    var mdrag = false;
    map.addEventListener("pointerdown", function (e) { e.stopPropagation(); e.preventDefault(); takeOver(); mdrag = true; try { map.setPointerCapture(e.pointerId); } catch (x) { /* ignore */ } mapAt(e); });
    map.addEventListener("pointermove", function (e) { if (mdrag) mapAt(e); });
    ["pointerup", "pointercancel"].forEach(function (t) { map.addEventListener(t, function () { mdrag = false; }); });
    map.addEventListener("dblclick", function (e) { e.stopPropagation(); });

    /* ---------------------------------------------------------------- size and visibility */
    function onResize() { layout(); measure(); clampSt(); ui(); drawAll(); }
    if ("ResizeObserver" in window) { ro = new ResizeObserver(function () { onResize(); }); ro.observe(gridEl); }
    else window.addEventListener("resize", onResize);
    if ("IntersectionObserver" in window) {
      io = new IntersectionObserver(function (es) {
        visible = es[es.length - 1].isIntersecting;
        if (!visible) { if (tour.on) pause(); if (zoomed) panels.forEach(function (P) { if (P.cv) { P.cv.width = P.cv.height = 0; } }); }
        else if (zoomed) { measure(); drawAll(); }
      }, { rootMargin: "400px 0px" });
      io.observe(root);
    }
    requestAnimationFrame(function () { layout(); measure(); ui(); });

    return {
      el: root,
      play: play,
      pause: pause,
      reset: resetView,
      flyTo: function (k) { takeOver(); measure(); if (regions[k]) animateTo(stopView(k)); },
      destroy: function () {
        pause(); cancelAnim(); if (ro) ro.disconnect(); if (io) io.disconnect();
        panels.forEach(function (P) { if (P.cv) { P.cv.width = P.cv.height = 0; } });
        queue.length = 0;
      }
    };
  }

  window.SCAZoom = { grid: grid };
})();
