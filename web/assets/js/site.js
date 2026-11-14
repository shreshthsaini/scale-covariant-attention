/* SCA project page. Vanilla JS, no build step, works from file://.
   Data (generated, see web/README.md): numbers.js SCA_NUMBERS, images.js SCA_IMAGES, media.js SCA_MEDIA,
   showcase.js SCA_SHOWCASE, process.js SCA_PROCESS, traces.js SCA_TRACES, gallery.js SCA_GALLERY,
   zoomsets.js SCA_ZOOMSETS, method_data.js SCA_METHOD, ideogram_manifest.js SCA_IDEOGRAM.
   Every multi-method comparison is a synced zoom grid from zoomgrid.js (window.SCAZoom). */
(function () {
  "use strict";
  var NUM = window.SCA_NUMBERS || {}, IM = window.SCA_IMAGES || {}, MEDIA = window.SCA_MEDIA || {};
  var SHOW = window.SCA_SHOWCASE || {}, SVGNS = "http://www.w3.org/2000/svg";
  var REDUCED = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------------------------------------------------------------- helpers */
  function $(s, r) { return (r || document).querySelector(s); }
  function $$(s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); }
  function el(tag, attrs, kids) {
    var e = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (v === null || v === undefined || v === false) return;
      if (k === "class") e.className = v; else if (k === "text") e.textContent = v; else if (k === "html") e.innerHTML = v;
      else e.setAttribute(k, v === true ? "" : v);
    });
    (kids || []).forEach(function (c) { if (c !== null && c !== undefined) e.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return e;
  }
  function img(src, alt, attrs) {
    var a = { src: src, alt: alt || "", loading: "lazy", decoding: "async" };
    if (attrs) Object.keys(attrs).forEach(function (k) { a[k] = attrs[k]; });
    return el("img", a);
  }
  function svg(tag, attrs, text) {
    var e = document.createElementNS(SVGNS, tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function num(key) { return Object.prototype.hasOwnProperty.call(NUM, key) ? NUM[key] : null; }
  function val(key) {
    var v = num(key); if (v === null) return NaN;
    var m = String(v).replace(/,/g, "").replace(/−/g, "-").match(/[-+]?\d*\.?\d+(?:e[-+]?\d+)?/i);
    return m ? parseFloat(m[0]) : NaN;
  }
  function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
  function isDark() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t) return t === "dark";
    return !!(window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches);
  }
  function short(p, n) { p = String(p || ""); if (p.length <= n) return p; var c = p.slice(0, n), k = c.lastIndexOf(" "); return (k > n * .6 ? c.slice(0, k) : c).replace(/[,;:.]$/, "") + "…"; }
  function sizeLabel(wh) { return wh[0] + " × " + wh[1]; }   /* width x height */
  var PERSON = /\b(man|men|woman|women|boy|boys|girl|girls|child|children|kid|kids|person|people|persons|climber|player|chef|lady|ladies|gentleman|couple|bride|groom|king|queen|soldier|monk|dancer|surfer|rider|farmer|fisherman|worker|musician|guitarist|portrait|face|he|she|his|her|him)\b/i;
  function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
  function ph(title, body) { return el("div", { class: "ph" }, [el("div", {}, [el("strong", { text: title }), body ? el("span", { html: body }) : null])]); }
  function hexRgb(hex) { hex = hex.replace("#", ""); if (hex.length === 3) hex = hex.replace(/(.)/g, "$1$1"); return [0, 2, 4].map(function (i) { return parseInt(hex.substr(i, 2), 16); }); }
  function piStops() {
    return [[0, hexRgb(cssVar("--pi-0"))], [.22, hexRgb(cssVar("--pi-g"))], [.5, hexRgb(cssVar("--pi-mid"))], [.78, hexRgb(cssVar("--pi-l"))], [1, hexRgb(cssVar("--pi-1"))]];
  }
  function piRgb(v, stops) {
    v = clamp(v, 0, 1);
    for (var i = 1; i < stops.length; i++) if (v <= stops[i][0]) {
      var u = (v - stops[i - 1][0]) / (stops[i][0] - stops[i - 1][0]), a = stops[i - 1][1], b = stops[i][1];
      return [0, 1, 2].map(function (k) { return Math.round(a[k] + u * (b[k] - a[k])); });
    }
    return stops[stops.length - 1][1];
  }
  function rgbStr(c) { return "rgb(" + c.join(",") + ")"; }
  var store = {
    get: function (k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } }
  };
  var METHOD = { prog: "Progressive SCA", sca: "Single-pass SCA", area: "AReA", hrdit: "HRDiT", ultra: "UltraImage", full: "Full attention" };
  var STEPS = { prog: num("main.prog.steps") || "30+17+10", sca: num("main.sca.steps") || "28", area: num("main.area.steps") || "28",
                hrdit: num("main.hrdit.steps") || "30+17+10", ultra: num("main.ultraimage.steps") || "30+30", full: num("main.full.steps") || "28" };

  /* ---------------------------------------------------------------- theme */
  var themeHooks = [];
  function themeEvent() {   /* components listen for this (design.md): document "sca:themechange", detail.dark */
    var ev; try { ev = new CustomEvent("sca:themechange", { detail: { dark: isDark() } }); } catch (e) { ev = document.createEvent("CustomEvent"); ev.initCustomEvent("sca:themechange", false, false, { dark: isDark() }); }
    document.dispatchEvent(ev);
  }
  (function theme() {
    var btn = $("#theme-btn"), root = document.documentElement;
    if (btn) btn.addEventListener("click", function () {
      var next = isDark() ? "light" : "dark";
      root.setAttribute("data-theme", next); store.set("sca-theme", next);
      themeHooks.forEach(function (f) { f(); });
      themeEvent();
    });
    if (window.matchMedia) {
      var mq = matchMedia("(prefers-color-scheme: dark)");
      var on = function () { if (!root.getAttribute("data-theme")) { themeHooks.forEach(function (f) { f(); }); themeEvent(); } };
      if (mq.addEventListener) mq.addEventListener("change", on); else if (mq.addListener) mq.addListener(on);
    }
  })();

  /* ---------------------------------------------------------------- numbers (also written into the HTML by build_numbers.py) */
  var MANAGED = ["best", "second", "sig-up", "sig-dn"];
  $$("[data-num]").forEach(function (e) {
    var k = e.getAttribute("data-num"), v = num(k);
    if (v !== null) e.textContent = v;
    var c = num(k + ".cls");
    if (c !== null) { MANAGED.forEach(function (x) { e.classList.remove(x); }); String(c).split(/\s+/).forEach(function (x) { if (x) e.classList.add(x); }); }
  });

  /* ================================================================ v5: hero, top bar, component mounts, v5 videos, race, figures */
  var V5 = window.SCA_V5 || { comps: {}, videos: {} };
  function v5video(role) { var v = (V5.videos || {})[role]; return v && v.file ? v : null; }
  var ICON_PAUSE = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>';
  var ICON_PLAY = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.5-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5z"/></svg>';

  /* the top bar is transparent over the hero and solid below it */
  (function topbar() {
    var bar = $(".topbar[data-hero]"), hero = $(".hero5"); if (!bar || !hero) return;
    if (!("IntersectionObserver" in window)) { bar.classList.add("solid"); return; }
    new IntersectionObserver(function (es) { es.forEach(function (en) { bar.classList.toggle("solid", !en.isIntersecting); }); },
      { rootMargin: "-60px 0px 0px 0px", threshold: 0 }).observe(hero);
  })();

  /* the hero loop: v5 deep zoom (16:9, or 9:16 on a portrait screen), else the v4 zoom loop; a pause button; posters
     only under reduced motion until the visitor presses play */
  (function heroMedia() {
    var box = $("#hero-media"); if (!box) return;
    var vb = $("#video-btn"); if (vb && !v5video("explainer")) vb.hidden = true;   /* shown in the HTML; hidden only if the explainer is missing */
    var m = v5video("hero16x9") || v5video("hero9x16"), src = null, poster = null;   /* the box is 16:10 on phones too */
    if (m) { src = m.file; poster = m.poster; }
    else if (MEDIA.teaser && MEDIA.teaser.zoom) { src = MEDIA.teaser.zoom.mp4; poster = MEDIA.teaser.zoom.poster; }
    var pim = $("img.poster", box);
    if (poster && pim && !pim.closest("picture") && pim.getAttribute("src") !== poster) pim.src = poster;
    var btn = $("#hero-pause"); if (!src) { if (btn) btn.hidden = true; return; }
    var v = null, userPaused = REDUCED, visible = true;
    function make() {
      v = document.createElement("video");
      v.muted = true; v.defaultMuted = true; v.loop = true; v.playsInline = true;
      v.setAttribute("muted", ""); v.setAttribute("playsinline", ""); v.setAttribute("aria-hidden", "true");
      v.preload = "auto"; if (poster) v.poster = poster; v.src = src;
      v.addEventListener("playing", function () { v.classList.add("ready"); });
      box.insertBefore(v, box.querySelector(".hero5-hud"));
      if (m && m.readout) readout(v, m.readout);
    }
    function play() { if (!v) make(); var p = v.play(); if (p && p.catch) p.catch(function () {}); }
    function sync() {
      if (btn) { btn.innerHTML = userPaused ? ICON_PLAY : ICON_PAUSE; btn.setAttribute("aria-label", userPaused ? "Play the deep zoom" : "Pause the deep zoom"); btn.setAttribute("aria-pressed", String(userPaused)); }
      if (!userPaused && visible) play(); else if (v) v.pause();
    }
    if (btn) btn.addEventListener("click", function () { userPaused = !userPaused; sync(); });
    if ("IntersectionObserver" in window) new IntersectionObserver(function (es) { es.forEach(function (en) { visible = en.isIntersecting; sync(); }); }, { threshold: 0.05 }).observe(box);
    document.addEventListener("visibilitychange", function () { if (document.visibilityState === "hidden" && v) v.pause(); else sync(); });
    sync();
  })();

  /* the hero readout: magnification against the whole image and the native-pixel hold, from the loop's own zoom curve */
  function readout(v, R) {
    var tEl = $("#hud-title"), zEl = $("#hud-zoom"), curve = R.zoom_curve || [], segs = R.segments || [], whole = R.whole_scale || (curve[0] && curve[0][1]) || 1;
    if (!tEl || !zEl || !curve.length) return;
    var last = "";
    function tick() {
      var t = v.currentTime || 0, i = Math.max(0, Math.min(curve.length - 1, Math.round(t * 10))), s = curve[i][1];
      var seg = segs.filter(function (g) { return t >= g.t0 && t < g.t1; })[0];
      var nat = seg && seg.native && t >= seg.native[0] && t <= seg.native[1];
      var txt = (seg ? seg.title + " · " : "") + (R.label || "4096 × 4096").replace(" x ", " × ") + "|" + (nat ? (R.native_label || "1:1 native pixels") : "zoom ×" + (s / whole).toFixed(1));
      if (txt !== last) { last = txt; var p = txt.split("|"); tEl.textContent = p[0]; zEl.textContent = p[1]; zEl.className = nat ? "nat" : ""; }
    }
    v.addEventListener("timeupdate", tick);
    var raf = 0; function loop() { tick(); if (!v.paused) raf = requestAnimationFrame(loop); }
    v.addEventListener("playing", function () { cancelAnimationFrame(raf); raf = requestAnimationFrame(loop); });
  }

  /* ================================================================ everything below the fold */
  function rest() {
    IM = window.SCA_IMAGES || {}; SHOW = window.SCA_SHOWCASE || {}; NUM = window.SCA_NUMBERS || NUM;   /* deferred on the main page */
    STEPS = { prog: num("main.prog.steps") || STEPS.prog, sca: num("main.sca.steps") || STEPS.sca, area: num("main.area.steps") || STEPS.area,
              hrdit: num("main.hrdit.steps") || STEPS.hrdit, ultra: num("main.ultraimage.steps") || STEPS.ultra, full: num("main.full.steps") || STEPS.full };
  /* ---------------------------------------------------------------- nav highlight */
  (function nav() {
    var links = $$(".nav a"); if (!links.length || !("IntersectionObserver" in window)) return;
    var map = {}; links.forEach(function (a) { map[a.getAttribute("href").slice(1)] = a; });
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (en) {
        if (en.isIntersecting && map[en.target.id]) { links.forEach(function (a) { a.classList.remove("on"); }); map[en.target.id].classList.add("on"); }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    Object.keys(map).forEach(function (id) { var s = document.getElementById(id); if (s) io.observe(s); });
  })();

  /* ---------------------------------------------------------------- videos that play while on screen */
  var vio = ("IntersectionObserver" in window) ? new IntersectionObserver(function (es) {
    es.forEach(function (en) {
      var v = en.target;
      if (en.isIntersecting && !v._userPaused) { var p = v.play(); if (p && p.catch) p.catch(function () {}); } else if (!en.isIntersecting) v.pause();
    });
  }, { threshold: .3 }) : null;
  function loopVideo(src, poster, label, controls) {
    var v = el("video", { muted: true, loop: true, playsinline: true, preload: "none", poster: poster || null, "aria-label": label, controls: controls || REDUCED || null });
    v.muted = true; v.defaultMuted = true;
    v.src = src;
    v.addEventListener("pause", function () { if (document.visibilityState === "visible" && v._seen) v._userPaused = !!v._manual; });
    if (vio && !REDUCED) vio.observe(v); else if (!REDUCED) v.autoplay = true;
    return v;
  }

  /* ---------------------------------------------------------------- synced zoom grids (assets/js/zoomgrid.js, data zoomsets.js) */
  var ZS = window.SCA_ZOOMSETS || {}, ZG = window.SCAZoom || null;
  function zgrid(spec) { return ZG.grid(spec); }
  function zset(name, sid) { return (ZS[name] || []).filter(function (z) { return z.sid === sid; })[0] || null; }

  /* ---------------------------------------------------------------- teaser loops */
  (function loops() {
    var host = $("#loops"); if (!host) return;
    var T = MEDIA.teaser || {};
    var slots = [
      ["zoom", "<b>Deep zoom.</b> Progressive SCA at 4096&sup2;, down to native pixels", "Zoom from the whole image to native pixels"],
      ["stages", "<b>Three stages.</b> Layout at 1024&sup2;, refined at 2048&sup2; and 4096&sup2;", "One sample through the three stages"]
    ];
    var ready = slots.filter(function (s) { return T[s[0]] && T[s[0]].mp4; });
    (ready.length ? ready : slots.slice(0, 1)).forEach(function (s) {
      var m = T[s[0]], box = el("div", { class: "lbox" });
      if (m && m.mp4) box.appendChild(loopVideo(m.mp4, m.poster, s[2]));
      else box.appendChild(ph("Video in preparation", s[2] + "."));
      host.appendChild(el("figure", { class: "loop" }, [box, el("figcaption", { html: s[1] })]));
    });
    host.classList.add("n" + Math.max(1, ready.length));
  })();

  /* ---------------------------------------------------------------- hero: very large images, zoomable to native pixels */
  function metaRow(bits, oursFirst) {
    return el("div", { class: "meta" }, bits.filter(Boolean).map(function (b, i) { return el("span", { class: i === 0 && oursFirst ? "m-ours" : null, text: b }); }));
  }
  function itemMeta(it) {
    var steps = it.config === "Progressive SCA" ? STEPS.prog : it.steps;
    return metaRow([it.config || "Progressive SCA", it.model || "FLUX.1-dev", sizeLabel(it.size), steps ? steps + " steps" : null, "seed " + (it.seed === undefined ? 0 : it.seed)], true);
  }
  (function hero() {
    var host = $("#hero-figure"); if (!host) return;
    var H = (SHOW.hero || []).filter(function (it) { return it.panels && it.panels.length; }).slice(0, 2);
    if (!H.length || !ZG) { host.appendChild(ph("Hero images in preparation")); return; }
    var wrap = el("div", { class: "hshow" + (H.length > 1 ? " two" : "") });
    wrap.style.setProperty("--cols", H.map(function (it) { return (it.size[0] / it.size[1]).toFixed(3) + "fr"; }).join(" "));
    H.forEach(function (it, i) {
      var g = zgrid({ size: it.size, panels: it.panels, regions: it.regions, alt: it.prompt, eager: i === 0, cls: "compact", minW: 260 });
      wrap.appendChild(el("figure", { class: "sc-card hero-card" }, [g.el,
        el("figcaption", {}, [el("span", { class: "sc-title", text: it.title || "" }), itemMeta(it), el("p", { class: "prompt" }, [el("q", { text: it.prompt })])])]));
    });
    host.appendChild(wrap);
    host.appendChild(el("p", { class: "caption hcap", html: "Scroll or pinch on an image to zoom down to native pixels and drag to pan; double-click goes back to the whole image. <b>Auto zoom</b> flies through its most detailed regions. More images in the <a href=\"#showcase\">showcase</a>." }));
  })();

  /* ---------------------------------------------------------------- the failure row: full attention and the fixes at 4096^2, zoomed together */
  (function failRow() {
    var host = $("#fail-row"); if (!host || !ZG) return;
    var z = zset("compare", "m8343371271909676658") || (ZS.compare || [])[0]; if (!z) return;
    var panels = ["full", "area", "hrdit", "prog"].map(function (m) { return z.panels.filter(function (p) { return p.m === m; })[0]; }).filter(Boolean);
    host.appendChild(zgrid({ size: z.size, panels: panels, regions: z.regions, alt: z.prompt, cls: "compact", minW: 150 }).el);
    host.appendChild(el("p", { class: "caption fail-cap", html: "FLUX.1-dev at 4096&nbsp;&times;&nbsp;4096, one prompt and seed: <q>" + z.prompt + "</q> With full attention at the target the model loses the layout. Zoom any panel and the others follow; <a href=\"#compare\">Comparisons</a> has every method on 15 prompts." }));
  })();

  /* ---------------------------------------------------------------- method 1: attention mass rings of three heads */
  var MD = window.SCA_METHOD || null;
  (function rings() {
    var host = $("#rings"); if (!host || !MD || !MD.rings) return;
    var roles = [["local", "Local head"], ["mid", "Mixed head"], ["global", "Global head"]];
    var ex = {}; (MD.examples || []).forEach(function (e) { ex[e.role] = e; });
    var cell = 132, gap = 22, W = 3 * cell + 2 * gap + 8, top = 30, H = top + cell + 84;
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Attention mass around one query for a local, a mixed and a global head of FLUX.1-dev at 1024 squared" });
    roles.forEach(function (ro, i) {
      var R = MD.rings[ro[0]]; if (!R) return;
      var x0 = 4 + i * (cell + gap), y0 = top, c = cell / 33;   /* 33 x 33 tokens: Chebyshev radius 16 around the query */
      var col = ro[0] === "local" ? "var(--loc)" : ro[0] === "global" ? "var(--glob)" : "var(--ink)";
      s.appendChild(svg("rect", { x: x0, y: y0, width: cell, height: cell, fill: "var(--bg)" }));
      R.rings.forEach(function (q) {
        var so = (2 * q.cheb_to + 1) * c, si = q.cheb_from > 0 ? (2 * (q.cheb_from - 1) + 1) * c : 0;
        var d = "M" + (x0 + (cell - so) / 2) + " " + (y0 + (cell - so) / 2) + "h" + so + "v" + so + "h" + (-so) + "z";
        if (si > 0) d += "M" + (x0 + (cell - si) / 2) + " " + (y0 + (cell - si) / 2) + "v" + si + "h" + si + "v" + (-si) + "z";
        s.appendChild(svg("path", { d: d, fill: col, "fill-opacity": clamp(q.opacity, .035, .92), "fill-rule": "evenodd" }));
      });
      s.appendChild(svg("rect", { x: x0 + cell / 2 - c / 2, y: y0 + cell / 2 - c / 2, width: c, height: c, fill: "var(--accent)", stroke: "var(--bg)", "stroke-width": .6 }));
      s.appendChild(svg("rect", { x: x0, y: y0, width: cell, height: cell, fill: "none", stroke: "var(--ink)", "stroke-width": 1.2 }));
      var share = R.rings.reduce(function (a, q) { return a + q.share; }, 0), e = ex[ro[0]];
      s.appendChild(svg("text", { x: x0 + cell / 2, y: y0 - 11, "text-anchor": "middle", "font-size": 14, "font-weight": 600, fill: "var(--ink)" }, ro[1]));
      s.appendChild(svg("text", { x: x0 + cell / 2, y: y0 + cell + 22, "text-anchor": "middle", "font-size": 14, fill: "var(--ink)" }, "m = " + share.toFixed(2)));
      if (e) s.appendChild(svg("text", { x: x0 + cell / 2, y: y0 + cell + 42, "text-anchor": "middle", "font-size": 13, fill: "var(--accent-ink)" }, "\u03c0 = " + e.r2.pistar.toFixed(2) + " (ratio 2)"));
      s.appendChild(svg("text", { x: x0 + cell / 2, y: y0 + cell + 62, "text-anchor": "middle", "font-size": 12, fill: "var(--faint)" }, "block " + R.block + ", head " + R.head));
      if (e) { var se = $("#rings-sigma"); if (se) se.textContent = e.sigma.toFixed(2); }
    });
    host.appendChild(s);
  })();

  /* ---------------------------------------------------------------- method 2: the two views at 4096^2, to scale */
  (function views() {
    var host = $("#views"); if (!host) return;
    var S = 300, N = 256, k = S / N, pad = 14;
    var s = svg("svg", { viewBox: "0 0 " + (S + 2 * pad + 190) + " " + (S + 2 * pad), role: "img", "aria-label": "The 256 by 256 token grid at 4096 squared with a 64-token local window and the 64 by 64 grid of block-centre keys" });
    var defs = svg("defs", {});
    var pat = svg("pattern", { id: "gridpat", width: 4 * k, height: 4 * k, patternUnits: "userSpaceOnUse", x: pad, y: pad });
    pat.appendChild(svg("circle", { cx: 2 * k, cy: 2 * k, r: .62, fill: "var(--glob)" }));
    defs.appendChild(pat); s.appendChild(defs);
    s.appendChild(svg("rect", { x: pad, y: pad, width: S, height: S, fill: "var(--surface)", stroke: "var(--ink)", "stroke-width": 1.2 }));
    s.appendChild(svg("rect", { x: pad, y: pad, width: S, height: S, fill: "url(#gridpat)" }));
    var qx = 150, qy = 96;  /* query tile (tokens) */
    var wx = qx + 4 - 32, wy = qy + 4 - 32;
    s.appendChild(svg("rect", { x: pad + wx * k, y: pad + wy * k, width: 64 * k, height: 64 * k, fill: "var(--loc)", "fill-opacity": .22, stroke: "var(--loc)", "stroke-width": 1.8 }));
    s.appendChild(svg("rect", { x: pad + qx * k, y: pad + qy * k, width: 8 * k, height: 8 * k, fill: "var(--accent)" }));
    function lab(x, y, t, c, anchor, w) { s.appendChild(svg("text", { x: x, y: y, "font-size": 14.5, fill: c || "var(--ink)", "text-anchor": anchor || "start", "font-weight": w || 400 }, t)); }
    var lx = pad + S + 16;
    lab(lx, pad + 16, "256 × 256 tokens", "var(--muted)");
    lab(lx, pad + 56, "local view", "var(--loc)", "start", 600);
    lab(lx, pad + 74, "64 × 64-token window", "var(--muted)");
    lab(lx, pad + 92, "+ text, 1,024 bridge tokens", "var(--muted)");
    lab(lx, pad + 132, "global view", "var(--glob)", "start", 600);
    lab(lx, pad + 150, "1 token per 4 × 4 block", "var(--muted)");
    lab(lx, pad + 168, "64 × 64 keys", "var(--muted)");
    lab(lx, pad + 208, "query tile", "var(--accent-ink)", "start", 600);
    lab(lx, pad + 226, "8 × 8 tokens", "var(--muted)");
    s.appendChild(svg("line", { x1: pad + (wx + 64) * k, y1: pad + (wy + 20) * k, x2: lx - 6, y2: pad + 52, stroke: "var(--loc)", "stroke-width": .8 }));
    host.appendChild(s);
  })();

  /* ---------------------------------------------------------------- method 3: emulation, same ratio at two sizes */
  (function emu() {
    var host = $("#emu"); if (!host) return;
    var s = svg("svg", { viewBox: "0 0 540 330", role: "img", "aria-label": "A 64-token window on a 256-token side at 4096 squared has the same image-to-window ratio as a 16-token window on a 64-token side at 1024 squared" });
    function panel(x0, y0, side, nTok, win, title, sub, ok) {
      var k = side / nTok, g = svg("g", {});
      var step = nTok / 16;   /* draw the 16 x 16 global keys of the emulated ratio-4 grid (one per r x r block at 1024^2) */
      g.appendChild(svg("rect", { x: x0, y: y0, width: side, height: side, fill: "var(--surface)", stroke: "var(--ink)", "stroke-width": 1.2 }));
      for (var i = 0; i < 16; i++) for (var j = 0; j < 16; j++) g.appendChild(svg("circle", { cx: x0 + (i + .5) * step * k, cy: y0 + (j + .5) * step * k, r: Math.max(.7, side / 230), fill: "var(--glob)", "fill-opacity": .75 }));
      var wx = x0 + side * .47 - win * k / 2, wy = y0 + side * .34 - win * k / 2;
      g.appendChild(svg("rect", { x: wx, y: wy, width: win * k, height: win * k, fill: "var(--loc)", "fill-opacity": .25, stroke: "var(--loc)", "stroke-width": 1.8 }));
      g.appendChild(svg("text", { x: x0 + side / 2, y: y0 - 32, "text-anchor": "middle", "font-size": 15, "font-weight": 600, fill: "var(--ink)" }, title));
      g.appendChild(svg("text", { x: x0 + side / 2, y: y0 - 13, "text-anchor": "middle", "font-size": 13, fill: "var(--muted)" }, sub));
      g.appendChild(svg("text", { x: x0 + side / 2, y: y0 + side + 23, "text-anchor": "middle", "font-size": 14, fill: "var(--ink)" }, "r = " + nTok + " / " + win + " = " + (nTok / win)));
      g.appendChild(svg("text", { x: x0 + side / 2, y: y0 + side + 44, "text-anchor": "middle", "font-size": 13.5, fill: ok ? "var(--accent-ink)" : "var(--loss)", "font-weight": 600 }, ok ? "full attention: valid reference" : "full attention fails here"));
      s.appendChild(g);
    }
    panel(28, 56, 224, 256, 64, "Target, 4096²", "window 64 tokens on a 256-token side", false);
    panel(400, 112, 112, 64, 16, "At 1024²", "window 16 on a 64-token side", true);
    s.appendChild(svg("text", { x: 326, y: 176, "text-anchor": "middle", "font-size": 30, fill: "var(--muted)", "font-family": "var(--serif)" }, "="));
    s.appendChild(svg("text", { x: 326, y: 200, "text-anchor": "middle", "font-size": 12, fill: "var(--muted)" }, "same ratio"));
    host.appendChild(s);
  })();

  /* ---------------------------------------------------------------- method 4: the closed-form fit for three real heads */
  (function parab() {
    var host = $("#parab"); if (!host || !MD || !MD.examples) return;
    var exs = MD.examples, cur = Math.min(1, exs.length - 1), ratio = "r4", pi = .3, drag = false;
    var W = 500, H = 320, L = 50, R = 14, T = 22, B = 48;
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Relative squared error of a head's output as a function of its weight, a parabola with its minimum at the closed-form weight" });
    var ctl = el("div", { class: "seg-mini" });
    var headSeg = el("div", { class: "seg", role: "group", "aria-label": "Head" }), ratioSeg = el("div", { class: "seg", role: "group", "aria-label": "Ratio" });
    var names = { local: "Local head", mid: "Mixed head", global: "Global head" };
    exs.forEach(function (e, i) {
      var b = el("button", { type: "button", "aria-pressed": String(i === cur), text: names[e.role] || e.role });
      b.addEventListener("click", function () { cur = i; $$("button", headSeg).forEach(function (x, j) { x.setAttribute("aria-pressed", String(j === cur)); }); draw(); });
      headSeg.appendChild(b);
    });
    [["r2", "ratio 2"], ["r4", "ratio 4"]].forEach(function (r) {
      var b = el("button", { type: "button", "aria-pressed": String(r[0] === ratio), text: r[1] });
      b.addEventListener("click", function () { ratio = r[0]; $$("button", ratioSeg).forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); }); draw(); });
      ratioSeg.appendChild(b);
    });
    ctl.appendChild(headSeg); ctl.appendChild(ratioSeg);
    var readout = el("p", { class: "small muted", style: "text-align:center;margin:8px 0 0;min-height:1.5em", "aria-live": "polite" });
    host.appendChild(s); host.appendChild(ctl); host.appendChild(readout);
    function J(st, p) { return (st.E - 2 * p * st.B + p * p * st.A) / st.F; }
    function draw() {
      var e = exs[cur], st = e[ratio];
      var ys = []; for (var i = 0; i <= 100; i++) ys.push(J(st, i / 100));
      var ymax = Math.max.apply(null, ys) * 1.08, ymin = 0;
      function X(p) { return L + p * (W - L - R); }
      function Y(v) { return T + (1 - (v - ymin) / (ymax - ymin)) * (H - T - B); }
      s.innerHTML = "";
      var ticks = [0, ymax / 2 / 1.08, ymax / 1.08];
      ticks.forEach(function (t) {
        s.appendChild(svg("line", { x1: L, x2: W - R, y1: Y(t), y2: Y(t), stroke: "var(--chart-grid)" }));
        s.appendChild(svg("text", { x: L - 8, y: Y(t) + 4, "text-anchor": "end", "font-size": 13, fill: "var(--muted)" }, t.toFixed(2)));
      });
      [0, .25, .5, .75, 1].forEach(function (p) { s.appendChild(svg("text", { x: X(p), y: H - B + 19, "text-anchor": "middle", "font-size": 13, fill: "var(--muted)" }, String(p))); });
      s.appendChild(svg("line", { x1: L, x2: W - R, y1: Y(0), y2: Y(0), stroke: "var(--chart-base)" }));
      s.appendChild(svg("text", { x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle", "font-size": 12, fill: "var(--muted)" }, "weight on the window π  (0: global view, 1: window)"));
      var ay = (T + H - B) / 2;
      s.appendChild(svg("text", { x: 14, y: ay, "text-anchor": "middle", "font-size": 12, fill: "var(--muted)", transform: "rotate(-90 14 " + ay + ")" }, "relative error J(π)"));
      var d = ys.map(function (y, i) { return (i ? "L" : "M") + X(i / 100).toFixed(1) + " " + Y(y).toFixed(1); }).join(" ");
      s.appendChild(svg("path", { d: d, fill: "none", stroke: "var(--ink)", "stroke-width": 2 }));
      var ps = st.pistar;
      s.appendChild(svg("line", { x1: X(ps), x2: X(ps), y1: Y(J(st, ps)), y2: Y(0), stroke: "var(--accent)", "stroke-dasharray": "3 3" }));
      s.appendChild(svg("circle", { cx: X(ps), cy: Y(J(st, ps)), r: 5.5, fill: "var(--accent)" }));
      s.appendChild(svg("text", { x: X(ps), y: Y(J(st, ps)) - 12, "text-anchor": "middle", "font-size": 12.5, "font-weight": 600, fill: "var(--accent-ink)", style: "paint-order:stroke;stroke:var(--bg);stroke-width:4px" }, "π* = B/A = " + ps.toFixed(2)));
      var g = svg("g", { style: "cursor:ew-resize" });
      g.appendChild(svg("line", { x1: X(pi), x2: X(pi), y1: T, y2: Y(0), stroke: "var(--muted)", "stroke-width": 1 }));
      g.appendChild(svg("circle", { cx: X(pi), cy: Y(J(st, pi)), r: 7, fill: "var(--bg)", stroke: "var(--ink)", "stroke-width": 2 }));
      s.appendChild(g);
      var excess = J(st, pi) - J(st, ps);
      readout.innerHTML = "Block " + e.block + ", head " + e.head + ", σ = " + e.sigma.toFixed(2) + ": J(" + pi.toFixed(2) + ") = " + J(st, pi).toFixed(3) + ", which is " + excess.toFixed(3) + " above the optimum J(π*) = " + J(st, ps).toFixed(3) + ".";
      s._X = X;
    }
    function setFrom(ev) { var r = s.getBoundingClientRect(); var x = (ev.clientX - r.left) / r.width * W; pi = clamp((x - L) / (W - L - R), 0, 1); draw(); }
    s.addEventListener("pointerdown", function (ev) { drag = true; try { s.setPointerCapture(ev.pointerId); } catch (x) { /* ignore */ } setFrom(ev); });
    s.addEventListener("pointermove", function (ev) { if (drag) setFrom(ev); });
    ["pointerup", "pointercancel"].forEach(function (t) { s.addEventListener(t, function () { drag = false; }); });
    s.setAttribute("tabindex", "0");
    s.addEventListener("keydown", function (ev) { if (ev.key === "ArrowLeft") { pi = clamp(pi - .02, 0, 1); draw(); ev.preventDefault(); } if (ev.key === "ArrowRight") { pi = clamp(pi + .02, 0, 1); draw(); ev.preventDefault(); } });
    themeHooks.push(draw);
    draw();
  })();

  /* ---------------------------------------------------------------- the per-head table, 57 blocks x 24 heads */
  (function ptable() {
    var host = $("#ptable"); if (!host || !MD || !MD.table) return;
    var nb = MD.sigma.length, B = MD.blocks, Hh = MD.heads, ratio = "r4", bin = MD.mid;
    var T = {};
    ["r2", "r4"].forEach(function (r) { var raw = atob(MD.table[r]), a = new Uint8Array(raw.length); for (var i = 0; i < raw.length; i++) a[i] = raw.charCodeAt(i); T[r] = a; });
    var ctl = el("div", { class: "pt-ctl" });
    var seg = el("div", { class: "seg", role: "group", "aria-label": "Ratio" });
    [["r2", "ratio 2 (2048²)"], ["r4", "ratio 4 (4096²)"]].forEach(function (r) {
      var b = el("button", { type: "button", "aria-pressed": String(r[0] === ratio), text: r[1] });
      b.addEventListener("click", function () { ratio = r[0]; $$("button", seg).forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); }); draw(); });
      seg.appendChild(b);
    });
    var out = el("output", {}), range = el("input", { type: "range", min: "0", max: String(nb - 1), step: "1", value: String(bin), "aria-label": "Noise level" });
    range.addEventListener("input", function () { bin = +range.value; draw(); });
    ctl.appendChild(seg);
    ctl.appendChild(el("label", {}, ["noise level σ = ", out, " ", range]));
    var cv = el("canvas", { width: B * 12, height: Hh * 12, role: "img", "aria-label": "Heat map of the fitted window weight of every head" });
    var stops = el("div", { class: "pt-legend" }, [el("span", { text: "0 global" }), el("span", { class: "bar-g" }), el("span", { text: "1 local" })]);
    host.appendChild(ctl);
    host.appendChild(el("div", { class: "pt-maps" }, [cv, el("div", { class: "pt-axis" }, [el("span", { text: "block 0" }), el("span", { text: "blocks left to right; heads of each block sorted top to bottom" }), el("span", { text: "block " + (B - 1) })])]));
    host.appendChild(stops);
    function draw() {
      out.textContent = MD.sigma[bin].toFixed(2);
      var st = piStops(), ctx = cv.getContext("2d"), a = T[ratio], cw = 12;
      ctx.fillStyle = cssVar("--bg"); ctx.fillRect(0, 0, cv.width, cv.height);
      for (var l = 0; l < B; l++) {
        var ord = MD.order[l];
        for (var j = 0; j < Hh; j++) {
          var h = ord[j], v = a[(l * Hh + h) * nb + bin] / 255;
          ctx.fillStyle = rgbStr(piRgb(v, st));
          ctx.fillRect(l * cw, (Hh - 1 - j) * cw, cw - 1, cw - 1);
        }
      }
      var g = st.map(function (s0) { return rgbStr(s0[1]) + " " + (s0[0] * 100) + "%"; }).join(", ");
      $(".bar-g", stops).style.background = "linear-gradient(90deg, " + g + ")";
    }
    themeHooks.push(draw);
    draw();
  })();

  /* ---------------------------------------------------------------- schedule bars */
  function legend(host, items) {
    host.parentNode.insertBefore(el("div", { class: "sched-legend" }, items.map(function (i) { return el("span", { class: "k " + i[0], text: i[1] }); })), host.nextSibling);
  }
  (function schedBar() {
    var host = $("#sched-bar"); if (!host) return;
    var open = val("config.opening_steps"), nat = val("config.native_steps"), w0 = val("config.window");
    var wins = String(num("config.win_sched") || "").split(",").map(function (x) { return parseInt(x, 10); }).filter(function (x) { return !isNaN(x); });
    if (isNaN(open) || isNaN(nat) || !wins.length) return;
    var i;
    for (i = 0; i < open; i++) host.appendChild(el("span", { class: "c glob" }));
    wins.forEach(function (w) { host.appendChild(el("span", { class: "c wide", text: String(w) })); });
    for (i = 0; i < nat; i++) host.appendChild(el("span", { class: "c nat", text: i === 0 ? String(w0) : "" }));
    legend(host, [["glob", "global view only (" + open + ")"], ["wide", "shrinking windows (" + wins.length + ")"], ["nat", "native " + w0 + "-token windows (" + nat + ")"]]);
  })();
  (function progBar() {
    var host = $("#sched-prog"); if (!host) return;
    var b = val("prog.cfg.base_steps"), n2 = val("prog.cfg.steps_2048"), n4 = val("prog.cfg.steps_4096");
    if (isNaN(b) || isNaN(n2) || isNaN(n4)) return;
    var i;
    for (i = 0; i < b; i++) host.appendChild(el("span", { class: "c full" }));
    for (i = 0; i < n2; i++) host.appendChild(el("span", { class: "c r2" }));
    for (i = 0; i < n4; i++) host.appendChild(el("span", { class: "c r4" }));
    legend(host, [["full", "1024², full attention (" + b + ")"], ["r2", "2048², SCA at ratio " + num("prog.cfg.ratio_2048") + " (" + n2 + ")"], ["r4", "4096², SCA at ratio " + num("prog.cfg.ratio_4096") + " (" + n4 + ")"]]);
  })();

  /* ---------------------------------------------------------------- explainer video */
  (function explainer() {
    var host = $("#explainer-box"); if (!host) return;
    var m = MEDIA.explainer;
    if (m && m.mp4) {
      var v = el("video", { controls: true, playsinline: true, preload: "metadata", poster: m.poster || null, "aria-describedby": "explainer-cap" });
      v.appendChild(el("source", { src: m.mp4, type: "video/mp4" }));
      if (m.vtt) v.appendChild(el("track", { kind: "captions", src: m.vtt, srclang: "en", label: "English" }));
      host.appendChild(v);
    } else host.appendChild(ph("Explainer video in preparation", "About two minutes: near and far fields, the reach of every head, the emulation at 1024&sup2;, the closed-form fit and progressive sampling."));
  })();

  /* ---------------------------------------------------------------- lightbox */
  var lbHooks = [];
  function lbOnClose(f) { lbHooks.push(f); }
  function lbClean() { lbHooks.splice(0).forEach(function (f) { try { f(); } catch (e) { /* ignore */ } }); }
  function lightbox(render) {
    var lb = $("#lightbox"), body = $("#lb-body"); if (!lb) return;
    lbClean(); body.innerHTML = ""; lb.hidden = false; document.body.style.overflow = "hidden"; render(body);
    var close = $(".lb-close", lb); close.focus();
  }
  (function lightboxInit() {
    var lb = $("#lightbox"); if (!lb) return;
    function shut() { lbClean(); lb.hidden = true; document.body.style.overflow = ""; $("#lb-body").innerHTML = ""; }
    $(".lb-close", lb).addEventListener("click", shut);
    lb.addEventListener("click", function (e) { if (e.target === lb) shut(); });
    document.addEventListener("keydown", function (e) { if (!lb.hidden && e.key === "Escape") shut(); });
  })();

  /* ---------------------------------------------------------------- showcase: one zoom grid per image (and per comparison when it exists) */
  function showCard(it) {
    var g = zgrid({ size: it.size, panels: it.panels, regions: it.regions, alt: it.prompt, cls: "compact", minW: 150 });
    var crops = el("div", { class: "sc-crops" }, (it.crops || []).map(function (c, k) {
      var b = el("button", { type: "button", class: "tile", "aria-label": "Zoom every panel to window " + (k + 1) }, [img(c.file, "Native pixels, window " + (k + 1)), el("span", { class: "n", text: String(k + 1) })]);
      b.addEventListener("click", function () { g.flyTo(k); });
      return b;
    }));
    return el("figure", { class: "sc-card" + (it.panels.length > 1 ? " multi" : "") }, [g.el, crops,
      el("figcaption", {}, [el("span", { class: "sc-title", text: it.title || "" }), itemMeta(it), el("p", { class: "prompt" }, [el("q", { text: it.prompt })])])]);
  }
  (function showcase() {
    var host = $("#showcase-body"); if (!host) return;
    var groups = (SHOW.groups || []).map(function (g) { return { id: g.id, title: g.title, items: (g.items || []).filter(function (it) { return it.panels && it.panels.length; }) }; })
                                    .filter(function (g) { return g.items.length; });
    if (!groups.length || !ZG) {
      host.appendChild(el("div", { class: "sc-group" }, [ph("Showcase images in preparation", "5120&nbsp;&times;&nbsp;5120 and 6144&nbsp;&times;&nbsp;4096 images, product shots, landscapes, architecture, portraits and animals, zoomable to native pixels beside AReA on the same prompt and seed.")]));
    } else {
      var all = [], multi = false, chips = el("div", { class: "seg sc-filter", role: "group", "aria-label": "Category" });
      groups.forEach(function (g) { g.items.forEach(function (it) { if (it.panels.length > 1) multi = true; all.push([g.id, it]); }); });
      var limit = parseInt(host.getAttribute("data-limit") || "0", 10);
      if (limit > 0) {   /* main page: one card per category first, in manifest order, up to the limit */
        var pick = [], seen = {};
        all.forEach(function (a) { if (!seen[a[0]] && pick.length < limit) { seen[a[0]] = true; pick.push(a); } });
        all.forEach(function (a) { if (pick.indexOf(a) < 0 && pick.length < limit) pick.push(a); });
        all = pick; groups = [];
      }
      var grid = el("div", { class: "sc-grid " + (multi ? "two" : "three") });
      all = all.map(function (a) { return [a[0], showCard(a[1])]; });
      var cats = [["all", "All"]].concat(groups.map(function (g) { return [g.id, g.title]; }));
      cats.forEach(function (c) {
        var b = el("button", { type: "button", "aria-pressed": String(c[0] === "all"), text: c[1] });
        b.addEventListener("click", function () {
          Array.prototype.forEach.call(chips.querySelectorAll("button"), function (x) { x.setAttribute("aria-pressed", String(x === b)); });
          all.forEach(function (a) { a[1].hidden = !(c[0] === "all" || a[0] === c[0]); });
        });
        chips.appendChild(b);
      });
      all.forEach(function (a) { grid.appendChild(a[1]); });
      if (cats.length > 2) host.appendChild(el("div", { class: "sc-filter-row" }, [chips]));
      host.appendChild(el("div", { class: "sc-group" }, [grid]));
    }
    /* one prompt at five sizes, single pass (from the timing runs) */
    var sizes = host.hasAttribute("data-no-sizes") ? [] : (IM.scale || []).filter(function (d) { return d.group === "sizes"; });
    if (sizes.length) {
      var sec = el("div", { class: "sc-group", id: "sc-sizes" });
      sec.appendChild(el("h3", { text: "One prompt from 2048² to 8192²" }));
      sec.appendChild(el("p", { html: "Single-pass SCA, one prompt and seed at five sizes; each size starts from its own noise. Below each image, a 384&nbsp;&times;&nbsp;384 native crop and the seconds per image on one GH200. At 8192&sup2; blur and block artifacts remain." }));
      var row = el("div", { class: "sc-grid", style: "grid-template-columns:repeat(auto-fit,minmax(170px,1fr))" });
      sizes.forEach(function (d) {
        var z = d.size[0], t = num("latency.sca." + z + "x" + z) || num("latency.sca." + z);
        row.appendChild(el("figure", { class: "sc-card" }, [
          el("div", { class: "tile", style: "aspect-ratio:1" }, [img(d.full, "Single-pass SCA at " + sizeLabel(d.size) + ": " + d.prompt)]),
          d.crop_files && d.crop_files[0] ? el("div", { class: "tile", style: "aspect-ratio:1;margin-top:6px" }, [img(d.crop_files[0], "Native 1:1 crop at " + sizeLabel(d.size))]) : null,
          el("figcaption", { html: "<b>" + z + "&sup2;</b>" + (t ? " &middot; " + t + "&nbsp;s" : "") })]));
      });
      sec.appendChild(row);
      sec.appendChild(el("p", { class: "prompt", style: "text-align:center" }, [el("q", { text: sizes[0].prompt })]));
      host.appendChild(sec);
    }
  })();

  /* ---------------------------------------------------------------- other backbones: SCA beside the same model sampled directly at the target size */
  (function backbones() {
    var host = $("#backbones-body"); if (!host) return;
    var B = (SHOW.backbones || []).filter(function (it) { return it.panels && it.panels.length; });
    if (!B.length || !ZG) {
      host.appendChild(ph("FLUX.2-dev and Qwen-Image comparisons in preparation", "Each: SCA at the target size beside the same model sampled directly at that size with full attention, zoomable together to native pixels."));
      return;
    }
    var grid = el("div", { class: "sc-grid two" });
    B.forEach(function (it) { grid.appendChild(showCard(it)); });
    host.appendChild(grid);
  })();

  /* ---------------------------------------------------------------- Qwen-Image: SCA beside AReA's released code (and full attention where it ran) */
  (function qwenArea() {
    var block = $("#qa-block"), host = $("#qa-zoom"), thumbs = $("#qa-thumbs"), chips = $("#qa-methods"), meta = $("#qa-meta"), sum = $("#qa-summary");
    var Q = SHOW.qwen_area || {}, Z = (Q.items || []).filter(function (it) { return it.panels && it.panels.length > 1; });
    if (!block || !host || !Z.length || !ZG) return;      /* the block stays hidden until its images exist */
    block.hidden = false;
    var ORDER = [], seen = {}, on = {}, cur = null, at = 0;
    Z.forEach(function (z) { z.panels.forEach(function (p) { if (!seen[p.m]) { seen[p.m] = 1; ORDER.push([p.m, p.label, !!p.ours]); } }); });
    ORDER.forEach(function (o) {
      on[o[0]] = true;
      var b = el("button", { type: "button", "aria-pressed": "true", text: o[1], class: o[2] ? "ours" : null });
      b.addEventListener("click", function () {
        if (on[o[0]] && ORDER.filter(function (k) { return on[k[0]]; }).length <= 2) return;   /* keep two methods at least */
        on[o[0]] = !on[o[0]]; b.setAttribute("aria-pressed", String(on[o[0]])); render();
      });
      chips.appendChild(b);
    });
    Z.forEach(function (z, i) {
      var b = el("button", { type: "button", "aria-label": "Prompt " + (i + 1) + ": " + (z.title || short(z.prompt, 80)) }, [img(z.thumb || z.panels[0].view, "")]);
      b.addEventListener("click", function () { if (at !== i) { at = i; render(); } });
      thumbs.appendChild(b);
    });
    function secs(z, p) {
      var t = (z.timing || {})[p.m];
      return t && t.seconds ? p.label + ": " + Math.round(t.seconds) + " s" + (t.shared ? " (GPU shared)" : "") : null;
    }
    function render() {
      var z = Z[at], ps = z.panels.filter(function (p) { return on[p.m]; });
      if (ps.length < 2) ps = z.panels;
      if (cur) { cur.destroy(); cur = null; }
      host.innerHTML = ""; meta.innerHTML = "";
      $$("button", thumbs).forEach(function (b, i) { b.setAttribute("aria-pressed", String(i === at)); });
      cur = zgrid({ size: z.size, panels: ps, regions: z.regions, alt: z.prompt, minW: 220, label: "Qwen-Image methods on one prompt; zoom moves every panel" });
      host.appendChild(cur.el);
      meta.appendChild(metaRow(["Prompt " + (at + 1) + " of " + Z.length, z.title, "Qwen-Image", sizeLabel(z.size), "seed " + (z.seed || 0)], false));
      meta.appendChild(metaRow(z.panels.map(function (p) { return secs(z, p); }), false));
      meta.appendChild(el("p", { class: "prompt" }, [el("q", { text: z.prompt })]));
    }
    render();
    var R = Q.results;
    if (!sum || !R || !R.metrics || !R.metrics.length) return;
    function fx(x) { return x === null || x === undefined ? "" : Math.abs(x) >= 10 ? x.toFixed(2) : x.toFixed(3); }
    function sx(x) { return (x > 0 ? "+" : x < 0 ? "−" : "") + fx(Math.abs(x)); }
    function dcell(d) {
      if (!d || d.delta === null || d.delta === undefined) return el("td", { text: "" });
      var sig = d.lo > 0 || d.hi < 0;
      return el("td", {}, [el(sig ? "b" : "span", { text: sx(d.delta) }), el("span", { class: "faint ci", text: "[" + fx(d.lo) + ", " + fx(d.hi) + "]" })]);
    }
    var has40 = R.metrics.some(function (m) { return m.d40; });
    var rows = R.metrics.map(function (m) {
      var cells = [el("td", { text: m.name }), el("td", { text: fx(m.area) })];
      if (has40) cells.push(el("td", { text: fx(m.sca40) }), dcell(m.d40));
      cells.push(el("td", { text: fx(m.sca) }), dcell(m.d50));
      return el("tr", {}, cells);
    });
    var head = ["Metric", "AReA, 40 steps"].concat(has40 ? ["SCA, 40 steps", "SCA 40 minus AReA"] : []).concat(["SCA, 50 steps", "SCA 50 minus AReA"]);
    var T = R.timing || {}, tl = [["qwen_area", "AReA"], ["qwen_sca40", "SCA"], ["qwen_sca", "SCA"], ["qwen_full", "full attention"]].filter(function (k) { return T[k[0]] && T[k[0]].median_clean; })
      .map(function (k) { var t = T[k[0]]; return k[1] + " at " + t.steps + " steps " + Math.round(t.median_clean) + " s (" + t.n_clean + " of " + t.n + " images)"; });
    var M = R.step_matched, ms = M && M.sca40 && M.area ? "Step-matched, both at 40 steps, over the " + M.n + " prompts where both ran alone: median " + Math.round(M.sca40) + " s per image for SCA and " + Math.round(M.area) + " s for AReA." : null;
    sum.appendChild(el("div", { class: "qa-sum" }, [
      el("p", { class: "muted", html: "Means over the " + R.n_prompts + " prompts at 5312&nbsp;&times;&nbsp;5312, and SCA minus AReA with a 95% bootstrap interval over prompts (bold where the interval excludes zero), for SCA at AReA&rsquo;s 40 steps and at Qwen-Image&rsquo;s default 50. Higher is better for every metric; each scores the prompt as written." }),
      el("div", { class: "qa-scroll" }, [el("table", { class: "mini" }, [el("thead", {}, [el("tr", {}, head.map(function (h) { return el("th", { text: h }); }))]), el("tbody", {}, rows)])]),
      tl.length ? el("p", { class: "muted", text: "Median seconds per image on one GH200 with no other process on it: " + tl.join("; ") + "." }) : null,
      ms ? el("p", { class: "muted", text: ms }) : null]));
  })();

  /* ---------------------------------------------------------------- comparisons at 4096^2: every method in one synced zoom grid */
  (function compare() {
    var host = $("#cmp-zoom"), thumbs = $("#cmp-thumbs"), chips = $("#cmp-methods"), meta = $("#cmp-meta");
    var Z = ZS.compare || []; if (!host || !Z.length || !ZG) return;
    var thumbOf = {}; ((IM.compare || {}).items || []).forEach(function (r) { if (r.thumb) thumbOf[r.sid] = r.thumb; });
    var ORDER = Z[0].panels.map(function (p) { return [p.m, p.label, !!p.ours]; }), on = {}, cur = null, at = 0;
    ORDER.forEach(function (o) {
      on[o[0]] = true;
      var b = el("button", { type: "button", "aria-pressed": "true", text: o[1], class: o[2] ? "ours" : null });
      b.addEventListener("click", function () {
        if (on[o[0]] && ORDER.filter(function (k) { return on[k[0]]; }).length <= 2) return;   /* keep two methods at least */
        on[o[0]] = !on[o[0]]; b.setAttribute("aria-pressed", String(on[o[0]])); render();
      });
      chips.appendChild(b);
    });
    Z.forEach(function (z, i) {
      var b = el("button", { type: "button", "aria-label": "Prompt " + (i + 1) + ": " + short(z.prompt, 80) }, [img(z.thumb || thumbOf[z.sid] || z.panels[0].view, "")]);
      b.addEventListener("click", function () { if (at !== i) { at = i; render(); } });
      thumbs.appendChild(b);
    });
    function render() {
      var z = Z[at];
      if (cur) { cur.destroy(); cur = null; }
      host.innerHTML = ""; meta.innerHTML = "";
      $$("button", thumbs).forEach(function (b, i) { b.setAttribute("aria-pressed", String(i === at)); });
      /* single-pass SCA is shown only for prompts that name no person (the page's rule; the progressive configuration
         is the one the paper evaluates on people) */
      var person = PERSON.test(z.prompt || "");
      cur = zgrid({ size: z.size, panels: z.panels.filter(function (p) { return on[p.m] && !(person && p.m === "sca"); }), regions: z.regions, alt: z.prompt, minW: 200, label: "Every method on one prompt; zoom moves every panel" });
      host.appendChild(cur.el);
      meta.appendChild(metaRow(["Prompt " + (at + 1) + " of " + Z.length, "FLUX.1-dev", sizeLabel(z.size), "seed 0"].concat(person ? ["single pass not shown: the prompt names a person"] : []), false));
      meta.appendChild(el("p", { class: "prompt" }, [el("q", { text: z.prompt })]));
    }
    render();
  })();

  /* ---------------------------------------------------------------- per-example videos */
  (function process() {
    var host = $("#process-body"); if (!host) return;
    var P = window.SCA_PROCESS, list = [];
    if (P) {
      var raw = P.items || P.examples || (Array.isArray(P) ? P : Object.keys(P).filter(function (k) { return P[k] && P[k].mp4; }).map(function (k) { var o = P[k]; o.sid = o.sid || k; return o; }));
      list = (raw || []).filter(function (x) { return x && x.mp4; }).map(function (x) {
        return { sid: x.sid, prompt: x.prompt, mp4: x.mp4, poster: x.poster, still: x.final || x.still || null, size: x.size || [4096, 4096], model: x.model || "FLUX.1-dev",
                 seed: x.seed === undefined ? 0 : x.seed, label: "Progressive SCA beside AReA", title: x.title || "" };
      });
    }
    var legacy = false;
    if (!list.length) {   /* v5 process videos (webmedia manifest roles process_<n>): final image beside the run */
      var VV = (window.SCA_V5 || {}).videos || {};
      var keys = Object.keys(VV).filter(function (k) { return /^formation/.test(k) && VV[k].file; });
      if (!keys.length) keys = Object.keys(VV).filter(function (k) { return /^process/.test(k) && VV[k].file; });
      list = keys.sort().map(function (k) {
        var x = VV[k];
        return { sid: k, prompt: x.prompt || "", cap: x.prompt ? "" : pretty(x.caption || x.alt || ""), mp4: x.file, poster: x.poster, still: x.still || null, size: x.size || [4096, 4096],
                 model: x.model || "FLUX.1-dev", seed: x.seed === undefined ? 0 : x.seed, label: x.label || "Progressive SCA beside AReA", title: x.title || "", caption: x.caption || "" };
      });
    }
    if (!list.length) {
      var TR = window.SCA_TRACES || {}, byId = {};
      (TR.available || []).forEach(function (v) { if (v.kind === "vs_area") byId[v.sid] = v; });
      var thumbs = {}; ((IM.compare || {}).items || []).forEach(function (it) { thumbs[it.sid] = it; });
      list = Object.keys(byId).map(function (k) { var v = byId[k], it = thumbs[k]; return { sid: k, prompt: v.prompt, mp4: v.mp4, poster: v.poster, still: it ? it.img.sca.full : null, size: [4096, 4096], model: "FLUX.1-dev", seed: 0, label: "Single-pass SCA beside AReA" }; });
      legacy = true;
      var lede = $("#process-lede");
      if (lede && list.length) lede.innerHTML = "Each video replays one sample of <b>single-pass SCA</b> beside AReA: the decoded estimate of the clean image at each of the " + (num("config.steps") || "28") + " steps, the phase, the weights of all " + (num("calib.n_heads_c") || "1,368") + " heads and a native-pixel crop.";
    }
    if (legacy && host.hasAttribute("data-no-person")) list = list.filter(function (x) { return !PERSON.test(x.prompt || ""); });
    if (!list.length) { host.appendChild(ph("Videos in preparation")); return; }
    var at = 0, stage = el("div", { class: "proc-stage" }), head = el("div", { class: "proc-head" }), nav = el("div", { class: "proc-nav" });
    var prev = el("button", { type: "button", "aria-label": "Previous example", html: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 5l-7 7 7 7"/></svg>' });
    var next = el("button", { type: "button", "aria-label": "Next example", html: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 5l7 7-7 7"/></svg>' });
    var count = el("span", { class: "proc-count", "aria-live": "polite" });
    nav.appendChild(prev); nav.appendChild(count); nav.appendChild(next);
    var dots = el("div", { class: "proc-dots" });
    list.forEach(function (x, i) { var d = el("button", { type: "button", "aria-label": "Example " + (i + 1) }); d.addEventListener("click", function () { show(i); }); dots.appendChild(d); });
    host.appendChild(head); host.appendChild(stage); host.appendChild(nav); host.appendChild(dots);
    function show(i) {
      at = (i + list.length) % list.length;
      var x = list[at];
      $$("video", stage).forEach(function (v) { v.pause(); v.removeAttribute("src"); v.load(); });
      stage.innerHTML = ""; head.innerHTML = "";
      head.appendChild(metaRow([x.label, x.model, sizeLabel(x.size), "seed " + x.seed], true));
      head.appendChild(x.cap ? el("p", { class: "prompt", text: x.cap }) : el("p", { class: "prompt" }, [el("q", { text: x.prompt })]));
      var left = x.still ? el("div", { class: "still tile" }, [img(x.still, "Final image: " + x.prompt)]) : null;
      var v = loopVideo(x.mp4, x.poster, (legacy ? "Single-pass SCA" : "Progressive SCA") + " beside AReA, step by step: " + short(x.prompt, 80), true);
      if (left) stage.appendChild(left);
      stage.appendChild(v);
      if (!left) stage.style.gridTemplateColumns = "1fr";
      count.textContent = (at + 1) + " / " + list.length;
      $$("button", dots).forEach(function (d, j) { d.setAttribute("aria-pressed", String(j === at)); });
    }
    prev.addEventListener("click", function () { show(at - 1); });
    next.addEventListener("click", function () { show(at + 1); });
    show(0);
  })();

  /* ---------------------------------------------------------------- results: time bars */
  (function timebars() {
    var host = $("#timebars"); if (!host) return;
    var rows = [["Progressive SCA", "prog.clean.prog", true], ["HRDiT", "prog.clean.hrdit"], ["AReA", "prog.clean.area"]];
    var vals = rows.map(function (r) { return val(r[1]); }), mx = Math.max.apply(null, vals.filter(function (v) { return !isNaN(v); }));
    var fills = [];
    rows.forEach(function (r, i) {
      if (isNaN(vals[i])) return;
      var fill = el("div", { class: "fill" }), v = el("span", { class: "v", text: num(r[1]) + " s" });
      var track = el("div", { class: "track" }, [fill, v]);
      host.appendChild(el("div", { class: "tb" + (r[2] ? " ours" : "") }, [el("span", { class: "nm", text: r[0] }), track]));
      fills.push([fill, v, vals[i] / mx]);
    });
    function go() { fills.forEach(function (f) { f[0].style.width = (f[2] * 82) + "%"; f[1].style.left = (f[2] * 82) + "%"; }); }
    if ("IntersectionObserver" in window && !REDUCED) {
      var io = new IntersectionObserver(function (es) { if (es[0].isIntersecting) { go(); io.disconnect(); } }, { threshold: .4 });
      io.observe(host);
    } else go();
  })();

  /* ---------------------------------------------------------------- results: both modes minus AReA */
  function signed(v, d) { var t = Math.abs(v).toFixed(d); return (v > 0 ? "+" : v < 0 ? "−" : "") + t; }
  (function pairedChart() {
    var host = $("#chart-paired"); if (!host) return;
    var spec = window.SCA_CHART_PAIRED || [], SER = [["prog", "Progressive SCA", "circle"], ["single", "Single-pass SCA", "diamond"]];
    var rows = [], gap = 0;
    spec.forEach(function (r) {
      if (!r) { gap += 1; return; }
      var row = { key: r[0], label: r[1], gap: gap, s: {} }; gap = 0;
      SER.forEach(function (se) {
        var p = "chart.pa." + se[0] + "." + r[0], v = val(p + ".pct");
        if (!isNaN(v)) row.s[se[0]] = { v: v, lo: val(p + ".pct_lo"), hi: val(p + ".pct_hi"), vd: String(num(p + ".verdict") || "") };
      });
      if (Object.keys(row.s).length) rows.push(row);
    });
    if (!rows.length) { host.textContent = "Numbers not built yet."; return; }
    var W = 520, rh = 24, gh = 10, top = 34, left = 104, right = 18;
    var H = top + rows.length * rh + rows.reduce(function (a, r) { return a + r.gap * gh; }, 0) + 40;
    var ends = []; rows.forEach(function (r) { Object.keys(r.s).forEach(function (k) { ends.push(r.s[k].lo, r.s[k].hi); }); });
    var lo = Math.min(0, Math.min.apply(null, ends)), hi = Math.max(0, Math.max.apply(null, ends)), pad = (hi - lo) * .04; lo -= pad; hi += pad;
    function X(v) { return left + (v - lo) / (hi - lo) * (W - left - right); }
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Progressive and single-pass SCA minus AReA at 4096 squared, percent of AReA's mean, 95 percent paired intervals; right is better" });
    var step = (hi - lo) > 120 ? 50 : (hi - lo) > 60 ? 25 : (hi - lo) > 30 ? 10 : 5;
    for (var t = Math.ceil(lo / step) * step; t <= hi; t += step) {
      s.appendChild(svg("line", { x1: X(t), x2: X(t), y1: top - 6, y2: H - 30, stroke: t === 0 ? "var(--chart-base)" : "var(--chart-grid)", "stroke-width": t === 0 ? 1.2 : 1 }));
      s.appendChild(svg("text", { x: X(t), y: H - 16, "text-anchor": "middle" }, t === 0 ? "0" : signed(t, 0) + "%"));
    }
    function marker(kind, x, y, fill, stroke) {
      if (kind === "circle") return svg("circle", { cx: x, cy: y, r: 3.9, fill: fill, stroke: stroke, "stroke-width": 1.4 });
      var d = 4.4; return svg("polygon", { points: [x, y - d, x + d, y, x, y + d, x - d, y].join(" "), fill: fill, stroke: stroke, "stroke-width": 1.4 });
    }
    var lx = left;
    SER.forEach(function (se) { s.appendChild(marker(se[2], lx + 5, 12, "var(--accent)", "var(--accent)")); s.appendChild(svg("text", { x: lx + 15, y: 16, class: "lbl" }, se[1])); lx += 140; });
    var y = top;
    rows.forEach(function (r) {
      y += r.gap * gh; var yc = y + rh / 2;
      s.appendChild(svg("text", { x: left - 10, y: yc + 4, "text-anchor": "end", class: "lbl" }, r.label));
      SER.forEach(function (se, j) {
        var p = r.s[se[0]]; if (!p) return;
        var yy = yc + (j ? 4.6 : -4.6), col = p.vd === "better" ? "var(--accent)" : p.vd === "worse" ? "var(--loss)" : "var(--chart-light)", filled = p.vd === "better" || p.vd === "worse";
        var g = svg("g", {});
        g.appendChild(svg("title", {}, se[1] + ", " + r.label + ": " + signed(p.v, 1) + "% [" + signed(p.lo, 1) + "%, " + signed(p.hi, 1) + "%] of AReA's mean"));
        g.appendChild(svg("line", { x1: X(p.lo), x2: X(p.hi), y1: yy, y2: yy, stroke: col, "stroke-width": 1.7 }));
        g.appendChild(marker(se[2], X(p.v), yy, filled ? col : "var(--bg)", col));
        s.appendChild(g);
      });
      y += rh;
    });
    s.appendChild(svg("text", { x: (left + W - right) / 2, y: H - 2, "text-anchor": "middle" }, "gain over AReA (% of AReA's mean) →"));
    host.appendChild(s);
  })();

  /* ---------------------------------------------------------------- gallery: a fixed rule, progressive SCA beside AReA in a synced zoom grid */
  (function gallery() {
    var host = $("#gal"), G = window.SCA_GALLERY; if (!host) return;
    if (!G || !G.items || !G.items.length) { host.appendChild(ph("Gallery in preparation")); return; }
    function open(i) {
      lightbox(function (body) {
        var it = G.items[i], z = zset("gallery", it.sid);
        if (z && ZG) {
          var g = zgrid({ size: z.size, panels: z.panels, regions: z.regions, alt: it.prompt, eager: true, minW: 160 });
          lbOnClose(g.destroy);
          body.appendChild(el("div", { class: "lb-zoom", style: "--n:" + z.panels.length }, [g.el]));
        } else {
          body.appendChild(el("div", { class: "lb-zoom", style: "--n:2" }, [el("div", { class: "lb-pair" }, [img(it.img.prog.view, "Progressive SCA: " + it.prompt), img(it.img.area.view, "AReA: " + it.prompt)])]));
        }
        body.appendChild(metaRow(["Prompt " + it.index + " of " + (num("main.n_prompts") || "195"), "FLUX.1-dev", sizeLabel(it.img.prog.size), "seed 0"], false));
        body.appendChild(el("p", { class: "prompt" }, [el("q", { text: it.prompt })]));
        var nav = el("div", { class: "lb-nav" });
        var pv = el("button", { type: "button", text: "Previous" }), nx = el("button", { type: "button", text: "Next" });
        pv.addEventListener("click", function () { open((i - 1 + G.items.length) % G.items.length); });
        nx.addEventListener("click", function () { open((i + 1) % G.items.length); });
        nav.appendChild(pv); nav.appendChild(nx); body.appendChild(nav);
      });
    }
    G.items.forEach(function (it, i) {
      var b = el("button", { type: "button", "aria-label": "Open prompt " + it.index + ": " + short(it.prompt, 90) }, [img(it.img.prog.thumb || it.img.prog.view, "Progressive SCA: " + it.prompt), el("span", { class: "n", text: String(it.index) })]);
      b.addEventListener("click", function () { open(i); });
      host.appendChild(b);
    });
  })();

  /* ---------------------------------------------------------------- against Ideogram 4.0 (placeholders until the files exist) */
  (function ideogram() {
    var host = $("#vs-grid"); if (!host) return;
    function build(man) {
      host.innerHTML = "";
      var byId = {}; ((IM.compare || {}).items || []).forEach(function (it) { byId[it.sid] = it; });
      (man.items || []).forEach(function (e) {
        var it = byId[e.prompt_id]; if (!it) return;
        var ours = it.img.prog || it.img.sca, oursName = it.img.prog ? "Progressive SCA" : "Single-pass SCA";
        var file = e.file || ("assets/ideogram/" + e.prompt_id + ".webp");
        var left = el("div", { class: "col" }, [el("div", {}, [img(ours.full, oursName + " on FLUX.1-dev: " + it.prompt)]), el("div", {}, [img(ours.crop, oursName + ", native 1:1 crop")])]);
        var main = el("div", {}, [ph("Ideogram 4.0", "<code>" + file + "</code>")]), crop = el("div", {}, [ph("1:1 crop")]);
        var right = el("div", { class: "col" }, [main, crop]);
        var rName = el("span", { text: "Ideogram 4.0" });
        host.appendChild(el("article", { class: "vs-card" }, [el("p", { class: "name" }, [el("b", { text: oursName + " · 4096²" }), rName]),
          el("div", { class: "vs-pair" }, [left, right]), el("p", { class: "prompt" }, [el("q", { text: it.prompt })]),
          e.verdict ? el("p", { class: "verdict muted", text: e.verdict }) : null]));
        var probe = new Image();
        probe.onload = function () {
          var w = probe.naturalWidth, h = probe.naturalHeight;
          main.innerHTML = ""; main.appendChild(img(file, "Ideogram 4.0: " + it.prompt));
          rName.textContent = "Ideogram 4.0 · " + w + " × " + h + (e.upscaled ? " (upscaled)" : "");
          var c = e.crop_center || [(it.box[0] + it.box[2] / 2) / it.size[0], (it.box[1] + it.box[2] / 2) / it.size[1]], n = it.box[2];
          crop.innerHTML = "";
          var cv = el("div", { style: "width:100%;height:100%;background-repeat:no-repeat;background-image:url('" + file + "')", role: "img", "aria-label": "Ideogram 1:1 crop" });
          crop.appendChild(cv);
          function fit() {
            var D = cv.clientWidth || 200, k = D / n;
            var cx = clamp(c[0] * w, n / 2, w - n / 2), cy = clamp(c[1] * h, n / 2, h - n / 2);
            cv.style.backgroundSize = (w * k) + "px " + (h * k) + "px";
            cv.style.backgroundPosition = (-(cx - n / 2) * k) + "px " + (-(cy - n / 2) * k) + "px";
          }
          fit(); window.addEventListener("resize", fit);
        };
        if (e.resolution) probe.src = file;   /* build_ideogram.py records the resolution once the file exists */
      });
    }
    build(window.SCA_IDEOGRAM || { items: [] });
  })();

  /* ---------------------------------------------------------------- paper figures open full size */
  $$(".paper-fig .pf").forEach(function (b) {
    b.addEventListener("click", function () {
      var im = $("img", b), cap = b.parentNode.querySelector("figcaption");
      lightbox(function (body) {
        body.appendChild(el("div", { class: "lb-fig" }, [el("img", { src: im.getAttribute("src"), alt: im.getAttribute("alt") })]));
        if (cap) body.appendChild(el("p", { class: "prompt", html: cap.innerHTML }));
      });
    });
  });

  /* ---------------------------------------------------------------- BibTeX copy */
  $$(".copy").forEach(function (b) {
    b.addEventListener("click", function () {
      var t = $(b.getAttribute("data-target")).textContent;
      var ok = function () { b.textContent = "Copied"; setTimeout(function () { b.textContent = "Copy"; }, 1400); };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(t).then(ok, function () {});
      else { var r = document.createRange(); r.selectNodeContents($(b.getAttribute("data-target"))); var sel = getSelection(); sel.removeAllRanges(); sel.addRange(r); }
    });
  });
  /* inline theme-aware SVG figures (assets/js/v6_figs.js from web/build/v5_figs.py) */
  (function figs() {
    var F = window.SCA_FIGS || {};
    $$("[data-fig]").forEach(function (box) { var f = F[box.getAttribute("data-fig")]; if (f && f.svg) { box.innerHTML = f.svg; box.classList.add("vfig"); } });
  })();

  /* v5 videos from assets/video/v5/manifest.json (through assets/js/v5_assets.js): a figure with data-vrole="<role>"
     swaps its static fallback for the video once the file exists */
  function pretty(t) {   /* manifest captions are plain ASCII: typographic times, pi and squares */
    return String(t || "").replace(/(\d) x (\d)/g, "$1 × $2").replace(/\bpi\*/g, "π*").replace(/\bpi\b/g, "π")
      .replace(/(\d)x faster/g, "$1× faster").replace(/ - /g, ", ");
  }
  (function vslots() {
    var portrait = window.matchMedia && matchMedia("(max-aspect-ratio: 4/5)").matches;
    $$("[data-vrole]").forEach(function (fig) {
      var m = (portrait && fig.getAttribute("data-vrole-portrait") && v5video(fig.getAttribute("data-vrole-portrait"))) || v5video(fig.getAttribute("data-vrole"));
      if (!m) return;
      var box = $(".vslot-media", fig); if (!box) return;
      var v;
      if (m.loop) v = loopVideo(m.file, m.poster, m.alt || "", false);
      else {
        v = el("video", { controls: true, playsinline: true, preload: "none", poster: m.poster || null, "aria-label": m.alt || "" });
        v.appendChild(el("source", { src: m.file, type: "video/mp4" }));
        if (m.vtt) v.appendChild(el("track", { kind: "captions", src: m.vtt, srclang: "en", label: "English" }));
      }
      if (m.width && m.height) { v.setAttribute("width", m.width); v.setAttribute("height", m.height); v.style.aspectRatio = m.width + " / " + m.height; }
      box.innerHTML = ""; box.appendChild(v);
      fig.classList.add("has-video"); fig.hidden = false;
      var row = fig.closest(".vrow"); if (row) row.hidden = false;
      if (fig.id === "explainer") { var vb = $("#video-btn"); if (vb) vb.hidden = false; }
      if (m.loop && REDUCED) v.removeAttribute("autoplay");
      var cap = $("[data-vcap]", fig); if (cap && m.caption && !cap.hasAttribute("data-keep")) cap.textContent = pretty(m.caption);
    });
  })();

  /* the speed race, drawn from the measured clean times (numbers.js) when the race video is not there */
  (function race() {
    var host = $("#race"); if (!host || host.closest(".has-video")) return;
    var rows = [["prog", "Progressive SCA", "prog.clean.prog", "headline"], ["single", "Single-pass SCA", "main.sca.time_clean", "fast mode"],
                ["hrdit", "HRDiT", "prog.clean.hrdit", ""], ["area", "AReA", "prog.clean.area", ""]]
      .map(function (r) { return { k: r[0], name: r[1], t: val(r[2]), note: r[3] }; }).filter(function (r) { return !isNaN(r.t) && r.t > 0; });
    if (rows.length < 2) return;
    var T = Math.max.apply(null, rows.map(function (r) { return r.t; })), slow = rows.filter(function (r) { return r.t === T; })[0];
    var SPEED = Math.max(1, Math.round(T / 10)), HOLD = 2600;
    host.innerHTML = "";
    var clock = el("b", { text: "0 s" });
    host.appendChild(el("div", { class: "race-top" }, [el("span", { text: "Seconds per 4096² image, one GH200" }), clock]));
    rows.forEach(function (r) {
      r.fl = el("span", { class: "fl" }); r.v = el("span", { class: "v", text: "0" });
      r.row = el("div", { class: "race-row r-" + r.k }, [el("span", { class: "nm" }, [document.createTextNode(r.name), el("small", { text: num(r.k === "single" ? "main.sca.time_clean" : r.k === "prog" ? "prog.clean.prog" : "prog.clean." + r.k) + " s per image" })]),
        el("span", { class: "tr" }, [r.fl]), r.v]);
      host.appendChild(r.row);
    });
    var done = el("span"), foot = el("div", { class: "race-foot" }, [el("span", { html: "Images finished in " + slow.name + "&rsquo;s <b>" + num("prog.clean.area") + " s</b>" }), el("span", { text: "Replayed " + SPEED + "× faster" })]);
    host.appendChild(foot);
    function draw(tsim) {
      clock.textContent = Math.min(T, tsim).toFixed(0) + " s";
      rows.forEach(function (r) {
        var n = Math.floor(Math.min(tsim, T) / r.t + 1e-9), f = Math.min(tsim, T) / r.t - n;
        if (tsim >= T && r === slow) { n = 1; f = 0; }
        r.fl.style.width = (r === slow && tsim >= T ? 100 : f * 100).toFixed(2) + "%";
        r.v.textContent = n + (n === 1 ? " image" : " images");
        r.row.classList.toggle("done", n > 0);
      });
    }
    if (REDUCED || !("requestAnimationFrame" in window)) { draw(T); return; }
    var t0 = null, running = false, raf = 0;
    function frame(ts) {
      if (t0 === null) t0 = ts;
      var el_ = ts - t0, tsim = el_ / 1000 * SPEED;
      draw(tsim);
      if (tsim * 1000 / SPEED > T * 1000 / SPEED + HOLD) t0 = ts;
      if (running) raf = requestAnimationFrame(frame);
    }
    draw(0);
    if ("IntersectionObserver" in window) new IntersectionObserver(function (es) {
      es.forEach(function (en) { if (en.isIntersecting && !running) { running = true; t0 = null; raf = requestAnimationFrame(frame); } else if (!en.isIntersecting && running) { running = false; cancelAnimationFrame(raf); } });
    }, { threshold: 0.25 }).observe(host);
    else { draw(T); }
  })();

  /* interactive components (webinteractive): load assets/{js,css}/c_<name>.* when they exist (v5_assets.js) and mount
     them near the viewport; the static figure stays until SCAComp.<name>(el, options) returns without throwing */
  (function comps() {
    var mounts = $$(".comp-mount[data-comp]"), C = V5.comps || {}; if (!mounts.length) return;
    var state = {};
    function fallback(m) {   /* static images held back by data-src load only when their component is absent or fails */
      var fig = m.closest(".comp-figure"); if (!fig) return;
      $$("img[data-src]", fig).forEach(function (im) { if (!im.getAttribute("src")) im.src = im.getAttribute("data-src"); });
    }
    function ensure(name, cb) {
      var c = C[name]; if (!c || !c.js) return;
      var st = state[name];
      if (st && st.ready) { cb(); return; }
      if (st) { st.q.push(cb); return; }
      st = state[name] = { q: [cb], ready: false };
      if (c.css) document.head.appendChild(el("link", { rel: "stylesheet", href: c.css }));
      var s = document.createElement("script"); s.src = c.js; s.async = true;
      s.onload = function () { st.ready = true; st.q.splice(0).forEach(function (f) { f(); }); };
      s.onerror = function () { st.q = []; $$('.comp-mount[data-comp="' + name + '"]').forEach(fallback); };
      document.head.appendChild(s);
    }
    function mount(m) {
      var name = m.getAttribute("data-comp"), opts = {};
      try { opts = JSON.parse(m.getAttribute("data-options") || "{}"); } catch (e) { opts = {}; }
      ensure(name, function () {
        var f = window.SCAComp && window.SCAComp[name]; if (typeof f !== "function") return;
        try { m._comp = f(m, opts); var fig = m.closest(".comp-figure"); if (fig && m.children.length) fig.classList.add("is-live"); else fallback(m); }
        catch (e) { m.innerHTML = ""; fallback(m); }
      });
    }
    /* observe the figure, not the mount: an empty mount is display:none and never intersects */
    var io = ("IntersectionObserver" in window) ? new IntersectionObserver(function (es) {
      es.forEach(function (en) { if (en.isIntersecting) { io.unobserve(en.target); mount(en.target._mount); } });
    }, { rootMargin: "900px 0px 900px 0px" }) : null;
    mounts.forEach(function (m) {
      if (!C[m.getAttribute("data-comp")]) { fallback(m); return; }
      var t = m.closest(".comp-figure") || m.parentNode; t._mount = m;
      if (io) io.observe(t); else mount(m);
    });
  })();

  /* "How it works", step 3: each bar's window share comes from the number printed beside it (numbers.js) */
  $$(".mix-w[data-pct]").forEach(function (b) {
    var v = val(b.getAttribute("data-pct"));
    if (!isNaN(v)) b.style.width = clamp(v, 0, 100) + "%";
  });

  /* paper figures in .figcard open full size */
  $$(".figcard[data-zoomable]").forEach(function (b) {
    b.setAttribute("tabindex", "0"); b.setAttribute("role", "button");
    var open = function () {
      var im = $("img", b), cap = b.parentNode.querySelector("figcaption");
      lightbox(function (body) {
        body.appendChild(el("div", { class: "lb-fig" }, [el("img", { src: im.getAttribute("src"), alt: im.getAttribute("alt") })]));
        if (cap) body.appendChild(el("p", { class: "prompt", html: cap.innerHTML }));
      });
    };
    b.addEventListener("click", open);
    b.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
  });
  }

  /* below the fold: the main page lists its heavy data scripts in window.SCA_DEFER; they load (in order, in parallel)
     after the first scroll, tap or key, or two seconds after load, so the first screen is the hero and little else.
     Pages that include the scripts directly run rest() at once. */
  (function deferred() {
    var D = window.SCA_DEFER || [], started = false;
    if (!D.length) { rest(); return; }
    function go() {
      if (started) return; started = true;
      ["scroll", "pointerdown", "keydown", "touchstart"].forEach(function (t) { window.removeEventListener(t, go); });
      var left = D.length;
      D.forEach(function (src) {
        var sc = document.createElement("script"); sc.src = src; sc.async = false;
        sc.onload = sc.onerror = function () { if (--left === 0) rest(); };
        document.body.appendChild(sc);
      });
    }
    ["scroll", "pointerdown", "keydown", "touchstart"].forEach(function (t) { window.addEventListener(t, go, { passive: true }); });
    if (location.hash && location.hash.length > 1) go();
    else if (document.readyState === "complete") setTimeout(go, 2000);
    else window.addEventListener("load", function () { setTimeout(go, 2000); });
  })();
})();
