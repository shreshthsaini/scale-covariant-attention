/* SCAComp.stages(el, options): method step 3, one table for every size and sampler stage. A real progressive SCA run
   (1024^2, then 2048^2, then 4096^2) step by step: the noise level of each step, the stage's ratio r, and block 12's
   row of the measured weight table that the step reads, with every head's weight at that step.
   Vanilla JS and SVG, no dependencies, works from file://.

   Data: assets/data/stages/data.js (window.SCA_DATA.stages), written by web/build/v5_build_stages.py from the run's
   own record and the table it read, through the sampler's own lookup (sca.attn.PiTable). No number is typed here.

   var c = SCAComp.stages(el, {step: 42, lazy: true, base: "<data dir url>"});  c.step(i); c.destroy();
   options  step   step shown first (0-based over all stages; default: the first 4096^2 step)
   Interaction: slider, arrow keys, Play (absent under reduced motion), click on the timeline. */
(function () {
  "use strict";
  var NAME = "stages", P = "c-stages-", NS = "http://www.w3.org/2000/svg";
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
  var probe = null;
  function rgbOf(root, name, fb) {
    var v = getComputedStyle(root).getPropertyValue(name).trim() || fb;
    if (!probe) { probe = document.createElement("span"); probe.style.display = "none"; document.body.appendChild(probe); }
    probe.style.color = ""; probe.style.color = v;
    var m = getComputedStyle(probe).color.match(/[\d.]+/g) || [128, 128, 128];
    return [+m[0], +m[1], +m[2]];
  }
  function mix(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }
  function css(c, a) { return "rgba(" + Math.round(c[0]) + "," + Math.round(c[1]) + "," + Math.round(c[2]) + "," + (a === undefined ? 1 : a) + ")"; }

  function mount(root, opts) {
    opts = opts || {};
    var base = opts.base || DEF_BASE;
    var D = null, alive = true, io = null, ro = null, mo = null, mq = null, cur = 0, timer = 0, C = {};
    root.classList.add("c-" + NAME);
    root.innerHTML = "";

    var play = el("button", "play", { type: "button", "aria-pressed": "false", disabled: true });
    var range = el("input", "range", { type: "range", min: "0", max: "1", step: "1", value: "0", "aria-label": "Sampler step", disabled: true });
    var readout = el("div", "readout", { "aria-live": "polite" });
    var top = el("div", "top", null, [play, range]);
    var tl = sv("svg", { cls: "tl", role: "img", height: "0", "aria-label": "Noise level and mean weight at every step of the three stages" });
    var tlBox = el("div", "tlbox", null, [tl]);
    var hm = sv("svg", { cls: "hm", role: "img", height: "0" });
    var hmBox = el("div", "hmbox", null, [hm]);
    var tip = el("div", "tip", { "aria-hidden": "true" });
    hmBox.appendChild(tip);
    var hmHead = el("div", "hmh");
    var note = el("p", "note");
    root.appendChild(top); root.appendChild(readout); root.appendChild(tlBox); root.appendChild(hmHead); root.appendChild(hmBox); root.appendChild(note);
    if (REDUCED) play.hidden = true;
    root.classList.toggle(P + "narrow", (root.clientWidth || 800) < 560);
    root.classList.add(P + "loading");          /* reserves the final heights until the data arrives (no layout shift) */

    function fail() {
      if (!alive) return;
      root.classList.remove(P + "loading");
      var fig = root.closest ? root.closest("figure") : null;
      if (fig && fig.querySelector(".comp-fallback")) { fig.classList.remove("is-live"); root.innerHTML = ""; return; }
      root.innerHTML = "";
      root.appendChild(el("p", "note", { text: "The sampler-stage view could not load." }));
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
      range.max = String(D.steps.length - 1); range.disabled = false; play.disabled = false;
      var first = 0; for (var i = 0; i < D.steps.length; i++) if (D.steps[i].stage === D.stages.length - 1) { first = i; break; }
      cur = typeof opts.step === "number" ? clamp(opts.step | 0, 0, D.steps.length - 1) : first;
      range.value = String(cur);
      setPlay(false);
      var st = D.stages;
      note.innerHTML = "";
      var more = el("span", "more"), tog = el("button", "tog", { type: "button", "aria-expanded": "false", text: "About this run" });
      tog.addEventListener("click", function () { var o = root.classList.toggle(P + "open"); tog.setAttribute("aria-expanded", String(o)); tog.textContent = o ? "Less" : "About this run"; });
      note.appendChild(more); note.appendChild(tog);
      more.textContent = "One progressive SCA run (the sheep image of the heads and emulation views): " + st.map(function (s) { return s.n + " steps at " + s.size + "\u00b2"; }).join(", ")
        + ", each later stage re-noised to its first noise level. The " + st[0].size + "\u00b2 stage uses the model's own attention; the others read one table measured once at "
        + st[0].size + "\u00b2, at r = " + st.filter(function (s) { return s.r; }).map(function (s) { return s.r; }).join(" and r = ")
        + ", interpolated in \u03c3 between its bins exactly as the sampler does. Rows: block " + D.block + "'s 24 heads by their mean r = 4 weight; the mean line covers all 1,368 heads.";
      render(); show(cur, true);
      root.classList.remove(P + "loading");
    }
    function readColours() {
      C.ink = rgbOf(root, "--ink", "#111513"); C.soft = rgbOf(root, "--ink-soft", "#4F5853"); C.faint = rgbOf(root, "--ink-faint", "#7C847F");
      C.rule = rgbOf(root, "--rule", "#DCD9CF"); C.near = rgbOf(root, "--near", "#E08A2B"); C.far = rgbOf(root, "--far", "#2F6DB3");
      C.neutral = rgbOf(root, "--neutral", "#8C9490"); C.accent = rgbOf(root, "--accent", "#13865F"); C.bg = rgbOf(root, "--bg-raised", "#FFFFFF");
      C.sunk = rgbOf(root, "--bg-sunk", "#EAE8E1");
    }
    function ramp(v) { v = clamp(v, 0, 1); return v < .5 ? mix(C.far, C.neutral, v / .5) : mix(C.neutral, C.near, (v - .5) / .5); }

    /* ------------------------------------------------------------ timeline */
    var T = { w: 0, h: 0, L: 34, R: 12, T: 22, B: 20 };
    function tx(i) { return T.L + (i + .5) / D.steps.length * (T.w - T.L - T.R); }
    function ty(v) { return T.T + (1 - v) * (T.h - T.T - T.B); }
    function render() {
      var w = tlBox.clientWidth || 700, narrow = w < 560;
      root.classList.toggle(P + "narrow", narrow);
      T.w = w; T.h = narrow ? 120 : 132;
      tl.setAttribute("viewBox", "0 0 " + w + " " + T.h); tl.setAttribute("width", w); tl.setAttribute("height", T.h);
      while (tl.firstChild) tl.removeChild(tl.firstChild);
      var i0 = 0;
      D.stages.forEach(function (s, k) {
        var a = tx(i0) - (tx(1) - tx(0)) / 2, b = tx(i0 + s.n - 1) + (tx(1) - tx(0)) / 2;
        sv("rect", { x: a + 1, y: T.T - 4, width: b - a - 2, height: T.h - T.T - T.B + 8, rx: 6, cls: "band band-" + k }, tl);
        sv("text", { x: a + (narrow ? 4 : 8), y: 12, cls: "bl", text: s.size + "\u00b2" + (narrow ? "" : " \u00b7 " + s.n + " steps" + (s.r ? " \u00b7 r = " + s.r : " \u00b7 own attention")) }, tl);
        i0 += s.n;
      });
      [0, .5, 1].forEach(function (v) { sv("text", { x: T.L - 6, y: ty(v), cls: "yt", text: v === .5 ? "0.5" : String(v) }, tl); });
      var sp = D.steps.map(function (s, i) { return tx(i) + "," + ty(s.sigma); });
      var cuts = [], j = 0; D.stages.forEach(function (s) { cuts.push(sp.slice(j, j + s.n)); j += s.n; });
      cuts.forEach(function (c) { sv("polyline", { points: c.join(" "), cls: "sig" }, tl); });
      var pm = [], cur2 = [];
      D.steps.forEach(function (s, i) { if (s.pi_mean === undefined) { if (cur2.length) { pm.push(cur2); cur2 = []; } return; } if (i && D.steps[i - 1].stage !== s.stage && cur2.length) { pm.push(cur2); cur2 = []; } cur2.push(tx(i) + "," + ty(s.pi_mean)); });
      if (cur2.length) pm.push(cur2);
      pm.forEach(function (c) { sv("polyline", { points: c.join(" "), cls: "pim" }, tl); });
      var lg = sv("text", { x: T.w - T.R, y: T.h - 4, cls: "lg" }, tl);
      sv("tspan", { cls: "lg-s", text: "\u2501 noise level \u03c3" }, lg);
      sv("tspan", { cls: "lg-p", dx: 12, text: "\u2501 mean weight \u03c0 of all heads" }, lg);
      sv("line", { x1: 0, x2: 0, y1: T.T - 6, y2: T.h - T.B + 6, cls: "cur" }, tl);
      sv("circle", { cx: 0, cy: 0, r: 5, cls: "dot dot-s" }, tl);
      sv("circle", { cx: 0, cy: 0, r: 5, cls: "dot dot-p" }, tl);
      hmRender();
    }
    tlBox.addEventListener("pointerdown", function (e) {
      if (!D) return;
      var r = tlBox.getBoundingClientRect(), f = (e.clientX - r.left - T.L) / (T.w - T.L - T.R);
      setPlay(false); go(clamp(Math.floor(f * D.steps.length), 0, D.steps.length - 1));
    });

    /* ------------------------------------------------------------ heatmap of block 12's table row */
    var H = { w: 0, rows: 24, L: 58, R: 92, T: 18, B: 22, rh: 12 };
    function hx(c) { return H.L + c * (H.w - H.L - H.R) / D.bins.length; }
    function hmRender() {
      var w = hmBox.clientWidth || 700, narrow = w < 560;
      H.w = w; H.rows = D.order.length; H.rh = narrow ? 10 : 12; H.L = narrow ? 46 : 58; H.R = narrow ? 64 : 92;
      var h = H.T + H.rows * H.rh + H.B;
      hm.setAttribute("viewBox", "0 0 " + w + " " + h); hm.setAttribute("width", w); hm.setAttribute("height", h);
      while (hm.firstChild) hm.removeChild(hm.firstChild);
      var g = sv("g", { cls: "cells" }, hm);
      D.order.forEach(function (head, ri) {
        D.bins.forEach(function (b, c) {
          sv("rect", { x: hx(c) + .5, y: H.T + ri * H.rh + .5, width: hx(c + 1) - hx(c) - 1, height: H.rh - 1, "data-r": ri, "data-c": c }, g);
        });
        if (head === 4 || head === 7 || head === 10 || ri === 0 || ri === H.rows - 1 || !narrow)
          sv("text", { x: H.L - 6, y: H.T + (ri + .5) * H.rh, cls: "rl" + (head === 4 ? " rl-n" : head === 7 ? " rl-f" : ""), text: (narrow ? "" : "head ") + head }, hm);
      });
      [0, 3, 6, 9, 12].filter(function (c) { return c < D.bins.length; }).forEach(function (c) {
        sv("text", { x: (hx(c) + hx(c + 1)) / 2, y: H.T + H.rows * H.rh + 14, cls: "ct", text: D.bins[c].toFixed(2) }, hm);
      });
      sv("text", { x: H.L, y: 11, cls: "ht", text: "noise bins, \u03c3 falling \u2192" }, hm);
      var nx = w - H.R + 16, nw = Math.min(26, H.R - 40);
      sv("text", { x: nx + nw / 2, y: 11, cls: "ht ht-c", text: "now" }, hm);
      var ng = sv("g", { cls: "now" }, hm);
      D.order.forEach(function (head, ri) { sv("rect", { x: nx, y: H.T + ri * H.rh + .5, width: nw, height: H.rh - 1, rx: 2, "data-r": ri }, ng); });
      var nv = sv("g", { cls: "nowv" }, hm);
      D.order.forEach(function (head, ri) { sv("text", { x: nx + nw + 5, y: H.T + (ri + .5) * H.rh, "data-r": ri, text: "" }, nv); });
      sv("line", { x1: 0, x2: 0, y1: H.T - 4, y2: H.T + H.rows * H.rh + 4, cls: "hcur" }, hm);
      H.nx = nx; H.nw = nw;
    }
    function hmPaint(s) {
      var r = s.r || 2, tab = D.table[String(r)], off = !s.r;
      hm.classList.toggle(P + "off", off);
      Array.prototype.forEach.call(hm.querySelectorAll("." + P + "cells rect"), function (rc) {
        var ri = +rc.getAttribute("data-r"), c = +rc.getAttribute("data-c"), head = D.order[ri];
        rc.setAttribute("fill", css(ramp(tab[head - 1][c])));
      });
      var pis = D.b12[cur];
      Array.prototype.forEach.call(hm.querySelectorAll("." + P + "now rect"), function (rc) {
        var head = D.order[+rc.getAttribute("data-r")];
        rc.setAttribute("fill", pis ? css(ramp(pis[head - 1])) : css(C.sunk));
      });
      Array.prototype.forEach.call(hm.querySelectorAll("." + P + "nowv text"), function (t) {
        var ri = +t.getAttribute("data-r"), head = D.order[ri];
        t.textContent = pis && (H.rh >= 12 || ri % 2 === 0) ? pis[head - 1].toFixed(2) : "";
      });
      /* the cursor: the step's sigma placed between the bin centres it falls between */
      var b = D.bins, x;
      if (s.sigma >= b[0]) x = (hx(0) + hx(1)) / 2; else if (s.sigma <= b[b.length - 1]) x = (hx(b.length - 1) + hx(b.length)) / 2;
      else { var j = 0; while (j < b.length - 1 && !(s.sigma <= b[j] && s.sigma >= b[j + 1])) j++; var a = (b[j] - s.sigma) / (b[j] - b[j + 1]); x = (hx(j) + hx(j + 1)) / 2 + a * (hx(1) - hx(0)); }
      var cl = hm.querySelector("." + P + "hcur"); cl.setAttribute("x1", x); cl.setAttribute("x2", x); cl.style.display = off ? "none" : "";
      hmHead.innerHTML = "";
      hmHead.appendChild(el("b", null, { text: "Block " + D.block + "'s row of the table" + (off ? "" : " at r = " + s.r) }));
      hmHead.appendChild(document.createTextNode(off ? ": not read at " + s.size + "\u00b2, where every head attends to the whole image" : ": each head's weight \u03c0 per noise bin; the line marks this step's \u03c3, the right column the weight read now"));
      hm.setAttribute("aria-label", off ? "Table not read at the native size." : "Block " + D.block + " weights at r = " + s.r + "; at this step they range from " + Math.min.apply(null, pis).toFixed(2) + " to " + Math.max.apply(null, pis).toFixed(2) + ".");
    }
    hmBox.addEventListener("pointermove", function (e) {
      if (!D) return;
      var t = e.target, ri = t.getAttribute && t.getAttribute("data-r");
      if (ri === null || ri === undefined) { tip.classList.remove(P + "on"); return; }
      var s = D.steps[cur], head = D.order[+ri], c = t.getAttribute("data-c"), r = s.r || 2;
      var v = c !== null ? D.table[String(r)][head - 1][+c] : (D.b12[cur] ? D.b12[cur][head - 1] : null);
      tip.textContent = "Head " + head + (c !== null ? " \u00b7 bin \u03c3 " + D.bins[+c].toFixed(2) + " \u00b7 \u03c0 " + v.toFixed(2) + " (r = " + r + ")" : v === null ? " \u00b7 no table at this step" : " \u00b7 \u03c0 now " + v.toFixed(2));
      var b = hmBox.getBoundingClientRect();
      tip.style.transform = "translate(" + Math.round(clamp(e.clientX - b.left + 12, 0, b.width - 220)) + "px," + Math.round(e.clientY - b.top - 34) + "px)";
      tip.classList.add(P + "on");
    });
    hmBox.addEventListener("pointerleave", function () { tip.classList.remove(P + "on"); });

    /* ------------------------------------------------------------ a step */
    function show(i, quiet) {
      cur = i; range.value = String(i);
      var s = D.steps[i], x = tx(i);
      var cl = tl.querySelector("." + P + "cur"); if (cl) { cl.setAttribute("x1", x); cl.setAttribute("x2", x); }
      var ds = tl.querySelector("." + P + "dot-s"), dp = tl.querySelector("." + P + "dot-p");
      if (ds) { ds.setAttribute("cx", x); ds.setAttribute("cy", ty(s.sigma)); }
      if (dp) { dp.style.display = s.pi_mean === undefined ? "none" : ""; if (s.pi_mean !== undefined) { dp.setAttribute("cx", x); dp.setAttribute("cy", ty(s.pi_mean)); } }
      var k = 0, j = 0; for (var q = 0; q < s.stage; q++) j += D.stages[q].n; k = i - j + 1;
      readout.innerHTML = "";
      readout.appendChild(el("span", "ro-s", { text: "Step " + (i + 1) + " of " + D.steps.length }));
      readout.appendChild(el("span", "ro-z", { text: s.size + "\u00b2, step " + k + " of " + D.stages[s.stage].n }));
      readout.appendChild(el("span", "ro-g", { text: "\u03c3 " + s.sigma.toFixed(2) }));
      readout.appendChild(el("span", "ro-r", { text: s.r ? "reads the table at r = " + s.r + "; mean \u03c0 over all heads " + s.pi_mean.toFixed(2) : "the model's own attention, no table" }));
      range.setAttribute("aria-valuetext", "step " + (i + 1) + ", " + s.size + " squared, sigma " + s.sigma.toFixed(2) + (s.r ? ", table at r = " + s.r : ", native attention"));
      hmPaint(s);
    }
    function go(i) { show(clamp(i, 0, D.steps.length - 1)); }
    range.addEventListener("input", function () { if (D) { setPlay(false); go(+range.value); } });
    function setPlay(on) {
      if (timer) { clearInterval(timer); timer = 0; }
      play.setAttribute("aria-pressed", String(on));
      play.innerHTML = on ? '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h3.6v14H7zM13.4 5H17v14h-3.6z" fill="currentColor"/></svg><span>Pause</span>'
        : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13l10.5-6.5z" fill="currentColor"/></svg><span>Play the run</span>';
      if (on) timer = setInterval(function () {
        if (!alive || !D) return;
        if (cur >= D.steps.length - 1) { setPlay(false); return; }
        go(cur + 1);
      }, 380);
    }
    play.addEventListener("click", function () {
      if (!D) return;
      var on = play.getAttribute("aria-pressed") !== "true";
      if (on && cur >= D.steps.length - 1) go(0);
      setPlay(on);
    });

    if ("ResizeObserver" in window) {
      var rq = 0;
      ro = new ResizeObserver(function () { if (rq) return; rq = requestAnimationFrame(function () { rq = 0; if (D) { render(); show(cur, true); } }); });
      ro.observe(root);
    }
    function onTheme() { requestAnimationFrame(function () { if (!D) return; readColours(); show(cur, true); }); }
    document.addEventListener("sca:themechange", onTheme);
    if (window.matchMedia) { mq = matchMedia("(prefers-color-scheme: dark)"); if (mq.addEventListener) mq.addEventListener("change", onTheme); }
    mo = new MutationObserver(onTheme);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class"] });
    var vis = null;
    if ("IntersectionObserver" in window) {
      vis = new IntersectionObserver(function (es) { if (!es[es.length - 1].isIntersecting && timer) setPlay(false); });
      vis.observe(root);
    }

    return {
      step: function (i) { if (D) { setPlay(false); go(i | 0); } },
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
