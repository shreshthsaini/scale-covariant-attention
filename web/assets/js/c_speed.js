/* SCAComp.speed(el, options): seconds per image against image side (log-log), single-pass SCA, AReA and full
   attention at 2048^2 to 8192^2, progressive SCA and HRDiT at 4096^2, with hover values and single-pass SCA's
   speed-up over AReA at every size. Vanilla JS and SVG, no dependencies, works from file://.

   Data: assets/data/speed/data.js (window.SCA_DATA.speed), written by web/build/v5_build_speed.py through the paper's
   Figure 7 loader and checks (paper/figures/src/v6_scaling.py). No number is typed here.

   var c = SCAComp.speed(el, {side: 4096, lazy: true, base: "<data dir url>"});  c.show(8192); c.destroy();
   options  side  image side whose values are shown first (default 4096; null shows none)
   Interaction: hover or tap the chart to read every method at the nearest size; arrow keys step through sizes. */
(function () {
  "use strict";
  var NAME = "speed", P = "c-speed-", NS = "http://www.w3.org/2000/svg";
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
  function sv(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === "text") e.textContent = attrs[k]; else if (k === "cls") e.setAttribute("class", attrs[k].split(" ").map(function (c) { return P + c; }).join(" "));
      else e.setAttribute(k, attrs[k]);
    });
    if (parent) parent.appendChild(e);
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

  function mount(root, opts) {
    opts = opts || {};
    var base = opts.base || DEF_BASE;
    var D = null, alive = true, io = null, ro = null, cur = -1;
    var G = { w: 0, h: 0, L: 46, R: 150, T: 16, B: 34 };
    root.classList.add("c-" + NAME);
    root.innerHTML = "";
    var svg = sv("svg", { cls: "svg", role: "img", "aria-label": "Seconds per image against image side" });
    var plot = el("div", "plot", { tabindex: "0" }, [svg]);
    var tip = el("div", "tip", { "aria-hidden": "true" });
    plot.appendChild(tip);
    var ratios = el("div", "ratios");
    var read = el("p", "sr", { "aria-live": "polite" });
    var note = el("p", "note");
    root.appendChild(plot); root.appendChild(ratios); root.appendChild(note); root.appendChild(read);

    function fail() {
      if (!alive) return;
      var fig = root.closest ? root.closest("figure") : null;
      if (fig && fig.querySelector(".comp-fallback")) { fig.classList.remove("is-live"); root.innerHTML = ""; return; }
      root.innerHTML = "";
      root.appendChild(el("p", "note", { text: "The speed chart could not load." }));
    }
    function start() {
      if (window.SCA_DATA && window.SCA_DATA[NAME]) { ready(); return; }
      loadScript(base + "data.js", function () { if (window.SCA_DATA && window.SCA_DATA[NAME]) ready(); else fail(); }, fail);
    }
    if (opts.lazy === false) start(); else io = whenNear(root, start);

    function ready() {
      if (!alive) return;
      D = window.SCA_DATA[NAME];
      note.textContent = "Clean batch-1 median of three timed images on one GH200, text encoding and VAE decoding included; whiskers: fastest and slowest of the three. "
        + "b: fitted exponent of seconds \u221d N^b (N image tokens). Full attention was not run at 8192\u00b2. Progressive SCA and HRDiT were timed at 4096\u00b2 only.";
      render();
      var s0 = opts.side === null ? -1 : D.sides.indexOf(opts.side || 4096);
      if (s0 >= 0) show(s0, true);
    }
    var x0 = 0, x1 = 0, y0 = 0, y1 = 0;
    function X(s) { return G.L + (Math.log(s) - x0) / (x1 - x0) * (G.w - G.L - G.R); }
    function Y(v) { return G.T + (y1 - Math.log(v)) / (y1 - y0) * (G.h - G.T - G.B); }
    function render() {
      var w = plot.clientWidth || 700, h = plot.clientHeight || 380, narrow = w < 560;
      G.w = w; G.h = h; G.R = narrow ? 92 : 150; G.L = narrow ? 38 : 46;
      root.classList.toggle(P + "narrow", narrow);
      if (narrow && tip.parentNode !== root) root.insertBefore(tip, note);
      if (!narrow && tip.parentNode !== plot) plot.appendChild(tip);
      x0 = Math.log(D.sides[0] / 1.13); x1 = Math.log(D.sides[D.sides.length - 1] * 1.09);
      y0 = Math.log(11); y1 = Math.log(1500);
      svg.setAttribute("viewBox", "0 0 " + w + " " + h); svg.setAttribute("width", w); svg.setAttribute("height", h);
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var g = sv("g", { cls: "axes" }, svg);
      [20, 50, 100, 200, 500, 1000].forEach(function (v) {
        sv("line", { x1: G.L, x2: G.w - G.R, y1: Y(v), y2: Y(v), cls: "grid" }, g);
        sv("text", { x: G.L - 7, y: Y(v), cls: "yt", text: String(v) }, g);
      });
      sv("text", { x: G.L - 7, y: G.T - 4, cls: "yl", text: "s" }, g);
      D.sides.forEach(function (s, i) {
        sv("line", { x1: X(s), x2: X(s), y1: G.h - G.B, y2: G.h - G.B + 4, cls: "tick" }, g);
        if (!narrow || i % 2 === 0) sv("text", { x: X(s), y: G.h - G.B + 18, cls: "xt", text: s + "\u00b2" }, g);
      });
      sv("line", { x1: G.L, x2: G.w - G.R, y1: G.h - G.B, y2: G.h - G.B, cls: "base" }, g);
      var guide = sv("line", { x1: 0, x2: 0, y1: G.T, y2: G.h - G.B, cls: "guide" }, svg);
      guide.style.display = "none";
      var labels = [];
      ["full", "area", "sca_v2"].forEach(function (id) {
        var ln = D.lines.filter(function (q) { return q.id === id; })[0];
        var gg = sv("g", { cls: "ln ln-" + id }, svg);
        sv("polyline", { points: ln.pts.map(function (p) { return X(p.s) + "," + Y(p.med); }).join(" "), cls: "line" }, gg);
        ln.pts.forEach(function (p) {
          sv("line", { x1: X(p.s), x2: X(p.s), y1: Y(p.lo), y2: Y(p.hi), cls: "wh" }, gg);
          if (id === "full") sv("path", { d: "M" + X(p.s) + " " + (Y(p.med) - 5) + "l4.6 8h-9.2z", cls: "mk" }, gg);
          else sv("circle", { cx: X(p.s), cy: Y(p.med), r: id === "sca_v2" ? 4.2 : 3.8, cls: "mk" }, gg);
        });
        var last = ln.pts[ln.pts.length - 1];
        var early = last.s !== D.sides[D.sides.length - 1];      /* a line that stops early is labelled above its end */
        labels.push({ id: id, y: Y(last.med) - (early ? 22 : 0), x: X(last.s) + (early ? -10 : 10), end: early, name: ln.label, b: "b = " + ln.expo });
      });
      /* points at 4096^2 */
      D.points.forEach(function (p) {
        var gg = sv("g", { cls: "pt pt-" + p.id }, svg), x = X(p.s), y = Y(p.med);
        if (p.id === "sca_v2_prog") sv("path", { d: star(x, y, 9.5, 4.1), cls: "mk" }, gg);
        else sv("path", { d: "M" + x + " " + (y - 5.5) + "l5.5 5.5l-5.5 5.5l-5.5-5.5z", cls: "mk" }, gg);
        if (narrow) sv("text", { x: x, y: y + (p.id === "sca_v2_prog" ? 22 : -11), cls: "pl pl-c", text: p.label + " " + p.t + " s" }, gg);
        else sv("text", { x: x - 12, y: y + (p.id === "sca_v2_prog" ? 4 : -9), cls: "pl", text: p.label + " " + p.t + " s" }, gg);
      });
      /* direct labels at the line ends, nudged apart */
      var right = labels.filter(function (q) { return !q.end; }).sort(function (a, b) { return a.y - b.y; });
      for (var k = 1; k < right.length; k++) if (right[k].y - right[k - 1].y < 30) right[k].y = right[k - 1].y + 30;
      labels.forEach(function (q) {
        var t = sv("text", { x: q.x, y: q.y - 3, cls: "ll ll-" + q.id + (q.end ? " ll-end" : "") }, svg);
        sv("tspan", { x: q.x, dy: 0, text: q.name }, t);
        sv("tspan", { x: q.x, dy: 15, cls: "llb", text: q.b }, t);
      });
      /* ratio row: single-pass SCA's speed-up over AReA at every size */
      ratios.innerHTML = "";
      ratios.appendChild(el("span", "rl", { text: narrow ? "Single-pass SCA's speed-up over AReA, per size" : "Single-pass SCA vs. AReA" }));
      D.sides.forEach(function (s, i) {
        var b = el("button", "rc", { type: "button", style: narrow ? null : "left:" + X(s) + "px", "aria-label": s + " squared: single-pass SCA " + D.ratio_t[s] + " times faster than AReA" },
          [narrow ? el("span", "rc-s", { text: s + "\u00b2" }) : null, document.createTextNode(D.ratio_t[s] + "\u00d7")]);
        b.addEventListener("click", function () { show(i); });
        b.addEventListener("mouseenter", function () { show(i); });
        ratios.appendChild(b);
      });
      ratios.style.setProperty("--c-speed-rw", (G.R - 12) + "px");
      if (cur >= 0) show(cur, true);
    }
    function star(cx, cy, R, r) {
      var d = "";
      for (var i = 0; i < 10; i++) { var a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r : R; d += (i ? "L" : "M") + (cx + rr * Math.cos(a)).toFixed(2) + " " + (cy + rr * Math.sin(a)).toFixed(2); }
      return d + "Z";
    }
    function valuesAt(s) {
      var out = [];
      D.lines.forEach(function (ln) { var p = ln.pts.filter(function (q) { return q.s === s; })[0]; out.push([ln.id, ln.label, p ? p.t + " s" : "not run"]); });
      D.points.forEach(function (p) { if (p.s === s) out.push([p.id, p.label, p.t + " s"]); });
      var order = { sca_v2_prog: 0, sca_v2: 1, hrdit_dev_author: 2, full: 3, area: 4 };
      return out.sort(function (a, b) { return order[a[0]] - order[b[0]]; });
    }
    function show(i, quiet) {
      cur = i;
      var s = D.sides[i], x = X(s), guide = svg.querySelector("." + P + "guide");
      if (guide) { guide.setAttribute("x1", x); guide.setAttribute("x2", x); guide.style.display = ""; }
      Array.prototype.forEach.call(ratios.querySelectorAll("." + P + "rc"), function (b, k) { b.classList.toggle(P + "on", k === i); });
      tip.innerHTML = "";
      tip.appendChild(el("b", "tip-h", { text: s + " \u00d7 " + s + " \u00b7 " + fmtInt(D.tokens[s]) + " tokens" }));
      valuesAt(s).forEach(function (v) { tip.appendChild(el("span", "tip-r tip-" + v[0], null, [el("i"), document.createTextNode(v[1]), el("em", null, { text: v[2] })])); });
      tip.appendChild(el("span", "tip-x", { text: "Single-pass SCA " + D.ratio_t[s] + "\u00d7 faster than AReA" }));
      if (s === 4096) tip.appendChild(el("span", "tip-x tip-p", { text: "Progressive SCA " + D.prog.area + "\u00d7 faster than AReA, " + D.prog.hrdit + "\u00d7 than HRDiT" }));
      tip.classList.add(P + "on");
      tip.style.transform = root.classList.contains(P + "narrow") ? "" : "translate(" + Math.round(G.L + 10) + "px," + Math.round(G.T) + "px)";
      if (!quiet) read.textContent = tip.textContent;
    }
    function hide() {
      tip.classList.remove(P + "on"); cur = -1;
      var guide = svg.querySelector("." + P + "guide"); if (guide) guide.style.display = "none";
      Array.prototype.forEach.call(ratios.querySelectorAll("." + P + "rc"), function (b) { b.classList.remove(P + "on"); });
    }
    function nearest(e) {
      var r = plot.getBoundingClientRect(), x = e.clientX - r.left, best = 0, bd = 1e9;
      D.sides.forEach(function (s, i) { var d = Math.abs(X(s) - x); if (d < bd) { bd = d; best = i; } });
      return best;
    }
    plot.addEventListener("pointermove", function (e) { if (D) show(nearest(e), true); });
    plot.addEventListener("pointerdown", function (e) { if (D) show(nearest(e)); });
    plot.addEventListener("keydown", function (e) {
      if (!D) return;
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); show(clamp((cur < 0 ? 2 : cur) + (e.key === "ArrowRight" ? 1 : -1), 0, D.sides.length - 1)); }
      else if (e.key === "Escape") hide();
    });
    if ("ResizeObserver" in window) {
      var rq = 0;
      ro = new ResizeObserver(function () { if (rq) return; rq = requestAnimationFrame(function () { rq = 0; if (D) render(); }); });
      ro.observe(plot);
    }
    return {
      show: function (side) { if (D) { var i = D.sides.indexOf(side); if (i >= 0) show(i); else hide(); } },
      destroy: function () { alive = false; if (io) io.disconnect(); if (ro) ro.disconnect(); root.innerHTML = ""; root.classList.remove("c-" + NAME); }
    };
  }

  SCAComp[NAME] = mount;
})();
