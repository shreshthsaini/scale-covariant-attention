/* SCA project page. Vanilla JS, no dependencies. Data comes from the generated files
   images.js (SCA_IMAGES), numbers.js (SCA_NUMBERS), traces.js (SCA_TRACES), concept.js (SCA_CONCEPT) and
   ideogram_manifest.js (SCA_IDEOGRAM); see web/README.md. */
(function () {
  "use strict";
  var IM = window.SCA_IMAGES || {}, NUM = window.SCA_NUMBERS || {}, TR = window.SCA_TRACES || {};
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  function el(tag, attrs, kids) {
    var e = document.createElement(tag);
    for (var k in attrs || {}) {
      if (k === "text") e.textContent = attrs[k];
      else if (k === "html") e.innerHTML = attrs[k];
      else if (k === "style") e.setAttribute("style", attrs[k]);
      else if (attrs[k] !== null && attrs[k] !== undefined) e.setAttribute(k, attrs[k]);
    }
    (kids || []).forEach(function (c) { if (c) e.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return e;
  }
  function img(src, alt, extra) {
    var a = { src: src, alt: alt || "", loading: "lazy", decoding: "async" };
    for (var k in extra || {}) a[k] = extra[k];
    return el("img", a);
  }
  function num(key) { return Object.prototype.hasOwnProperty.call(NUM, key) ? NUM[key] : null; }
  function val(key) {
    var s = num(key);
    if (s === null) return NaN;
    return parseFloat(String(s).replace(/−/g, "-").replace(/[^0-9.+\-eE]/g, ""));
  }
  function short(p, n) { p = String(p || ""); if (p.length <= n) return p; var c = p.slice(0, n), k = c.lastIndexOf(" "); return (k > n * .6 ? c.slice(0, k) : c).replace(/[,;:.]$/, "") + "…"; }
  function sizeLabel(wh) { return wh[1] + " × " + wh[0]; }   /* height x width, as in the paper */
  var store = {
    get: function (k) { try { return window.localStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { window.localStorage.setItem(k, v); } catch (e) { /* private mode */ } }
  };
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function isDark() {
    var t = document.documentElement.getAttribute("data-theme");
    return t ? t === "dark" : !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  }
  var themeHooks = [];   /* redraw functions of the canvas charts */

  /* Theme */
  (function theme() {
    var root = document.documentElement, btn = $("#theme-btn");
    var saved = store.get("sca-theme");
    if (saved === "light" || saved === "dark") root.setAttribute("data-theme", saved);
    if (!btn) return;
    btn.addEventListener("click", function () {
      var next = isDark() ? "light" : "dark";
      root.setAttribute("data-theme", next);
      store.set("sca-theme", next);
      themeHooks.forEach(function (f) { f(); });
    });
    if (window.matchMedia) {
      var mq = window.matchMedia("(prefers-color-scheme: dark)");
      var on = function () { if (!root.getAttribute("data-theme")) themeHooks.forEach(function (f) { f(); }); };
      if (mq.addEventListener) mq.addEventListener("change", on); else if (mq.addListener) mq.addListener(on);
    }
  })();

  /* Numbers already filled into the HTML by build_numbers.py --fill; refresh from numbers.js in case it is newer.
     "<key>.cls" entries carry Table 1's rank and significance classes; "pending" values are greyed. */
  var MANAGED = ["best", "second", "sig-up", "sig-dn"];
  $$("[data-num]").forEach(function (e) {
    var k = e.getAttribute("data-num"), v = num(k), c = num(k + ".cls");
    if (v !== null) e.textContent = v;
    if (c !== null) {
      MANAGED.forEach(function (x) { e.classList.remove(x); });
      String(c).split(/\s+/).forEach(function (x) { if (x) e.classList.add(x); });
    }
    e.classList.toggle("pend", String(e.textContent).trim() === "pending");
  });

  /* Nav highlight */
  (function nav() {
    var links = $$(".nav a");
    if (!("IntersectionObserver" in window) || !links.length) return;
    var map = {};
    links.forEach(function (a) { map[a.getAttribute("href").slice(1)] = a; });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting && map[en.target.id]) {
          links.forEach(function (a) { a.classList.remove("on"); });
          map[en.target.id].classList.add("on");
        }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    Object.keys(map).forEach(function (id) { var s = document.getElementById(id); if (s) io.observe(s); });
  })();

  /* Teaser lens: native pixels of the panorama, loaded on first use */
  (function lens() {
    var box = $("#pano"); if (!box) return;
    var pic = $("img", box), lensEl = $(".lens", box), hint = $(".hint", box);
    var nativeSrc = box.getAttribute("data-native"), W = +box.getAttribute("data-w"), H = +box.getAttribute("data-h");
    var ready = false, loading = false, active = false;
    function load() {
      if (ready || loading) return; loading = true;
      lensEl.innerHTML = '<div class="ring">loading native pixels</div>';
      var im = new Image();
      im.onload = function () { ready = true; lensEl.innerHTML = ""; lensEl.style.backgroundImage = "url('" + nativeSrc + "')"; lensEl.style.backgroundSize = W + "px " + H + "px"; };
      im.src = nativeSrc;
    }
    function place(clientX, clientY) {
      var r = pic.getBoundingClientRect();
      /* object-fit: cover maps the image into the box; undo it to find the image coordinate */
      var s = Math.max(r.width / W, r.height / H), dw = W * s, dh = H * s;
      var pos = getComputedStyle(pic).objectPosition.split(" ");
      var px = parseFloat(pos[0]) / 100, py = parseFloat(pos[1] || "50%") / 100;
      var ox = (r.width - dw) * (isNaN(px) ? .5 : px), oy = (r.height - dh) * (isNaN(py) ? .5 : py);
      var x = clientX - r.left, y = clientY - r.top;
      var ix = (x - ox) / s, iy = (y - oy) / s;
      if (ix < 0 || iy < 0 || ix > W || iy > H) { show(false); return; }
      var d = lensEl.offsetWidth / 2;
      lensEl.style.left = x + "px"; lensEl.style.top = y + "px";
      lensEl.style.backgroundPosition = (d - ix) + "px " + (d - iy) + "px";
      show(true);
    }
    function show(on) { active = on; lensEl.classList.toggle("on", on); if (on && hint) hint.style.opacity = 0; }
    box.addEventListener("pointerenter", load);
    box.addEventListener("pointerdown", function (e) { load(); place(e.clientX, e.clientY); });
    box.addEventListener("pointermove", function (e) { if (e.pointerType === "mouse" || active) place(e.clientX, e.clientY); });
    box.addEventListener("pointerleave", function () { show(false); });
    box.addEventListener("keydown", function (e) {
      var r = pic.getBoundingClientRect(), k = e.key, step = e.shiftKey ? 60 : 20;
      if (!box._kx) { box._kx = r.left + r.width / 2; box._ky = r.top + r.height / 2; }
      if (k === "ArrowLeft") box._kx -= step; else if (k === "ArrowRight") box._kx += step;
      else if (k === "ArrowUp") box._ky -= step; else if (k === "ArrowDown") box._ky += step;
      else if (k === "Escape") { show(false); return; } else return;
      e.preventDefault(); load(); place(box._kx, box._ky);
    });
    box.addEventListener("blur", function () { show(false); box._kx = null; });
  })();

  /* Before/after slider (clip-path plus a hidden native range for keyboard and screen readers) */
  function Slider(root, group) {
    this.root = root; this.group = group || [this];
    var self = this, input = $("input", root), dragging = false;
    function fromEvent(e) { var r = root.getBoundingClientRect(); return (e.clientX - r.left) / r.width * 100; }
    root.addEventListener("pointerdown", function (e) {
      if (e.button !== undefined && e.button !== 0) return;
      dragging = true; try { root.setPointerCapture(e.pointerId); } catch (x) { /* ignore */ }
      self.set(fromEvent(e)); input.focus({ preventScroll: true });
    });
    root.addEventListener("pointermove", function (e) { if (dragging) self.set(fromEvent(e)); });
    ["pointerup", "pointercancel", "lostpointercapture"].forEach(function (t) { root.addEventListener(t, function () { dragging = false; }); });
    input.addEventListener("input", function () { self.set(+input.value); });
  }
  Slider.prototype.set = function (p) {
    p = Math.max(0, Math.min(100, p));
    this.group.forEach(function (s) {
      s.root.style.setProperty("--pos", p + "%");
      var i = $("input", s.root); i.value = p; i.setAttribute("aria-valuetext", Math.round(p) + " percent SCA");
    });
  };
  function sliderMarkup(cls, label) {
    var knob = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 6l-6 6 6 6M15 6l6 6-6 6"/></svg>';
    var d = el("div", { class: "slider " + (cls || "") });
    d.innerHTML = '<img class="base" alt=""><img class="top" alt=""><div class="bar"></div><div class="knob">' + knob + '</div>' +
      '<span class="tag l sca"></span><span class="tag r"></span>' +
      '<input type="range" min="0" max="100" step="0.5" value="50" aria-label="' + label + '">';
    return d;
  }

  /* Comparison section: progressive or single-pass SCA against AReA, HRDiT, full attention or the other SCA mode */
  (function compare() {
    var host = $("#cmp"); var data = IM.compare; if (!host || !data) return;
    var items = data.items;
    var NAMES = { prog: "Progressive SCA", sca: "Single-pass SCA", area: "AReA", hrdit: "HRDiT", full: "Full attention" };
    function have(m) { return items.every(function (it) { return it.img && it.img[m]; }); }
    var state = { i: 0, sca: have("prog") ? "prog" : "sca", base: "area" };
    if (!have("prog")) {
      [$("#cmp-sca"), $("#cmp-sca-label"), $('#cmp-base [data-base="other"]')].forEach(function (x) { if (x) x.hidden = true; });
    }
    if (!have("hrdit")) { var hb = $('#cmp-base [data-base="hrdit"]'); if (hb) hb.hidden = true; }
    var main = sliderMarkup("", "Divider between SCA and the compared method, full image");
    var side = sliderMarkup("", "Divider between SCA and the compared method, 1:1 crop");
    var boxEl = el("div", { class: "box", "aria-hidden": "true" }); main.appendChild(boxEl);
    var group = []; var s1 = new Slider(main, group), s2 = new Slider(side, group); group.push(s1, s2);
    var thumbs = $("#cmp-thumbs"), meta = $("#cmp-meta");
    $("#cmp-main").appendChild(main); $("#cmp-side").appendChild(side);
    items.forEach(function (it, i) {
      var b = el("button", { type: "button", "aria-pressed": i === 0 ? "true" : "false", "aria-label": (it.label === "random" ? "Random draw: " : "Curated: ") + it.prompt },
        [img(it.thumb, ""), it.label === "random" ? el("span", { class: "rnd", text: "random" }) : null]);
      b.addEventListener("click", function () { state.i = i; render(); });
      thumbs.appendChild(b);
    });
    function right() { return state.base === "other" ? (state.sca === "prog" ? "sca" : "prog") : state.base; }
    function sync() {
      $$("#cmp-sca button").forEach(function (x) { x.setAttribute("aria-pressed", x.getAttribute("data-sca") === state.sca ? "true" : "false"); });
      $$("#cmp-base button").forEach(function (x) { x.setAttribute("aria-pressed", x.getAttribute("data-base") === state.base ? "true" : "false"); });
      var ob = $('#cmp-base [data-base="other"]'); if (ob) ob.textContent = state.sca === "prog" ? NAMES.sca : NAMES.prog;
    }
    $$("#cmp-sca button").forEach(function (b) {
      b.addEventListener("click", function () { state.sca = b.getAttribute("data-sca"); sync(); render(); });
    });
    $$("#cmp-base button").forEach(function (b) {
      b.addEventListener("click", function () { state.base = b.getAttribute("data-base"); sync(); render(); });
    });
    function swap(sl, topSrc, baseSrc, alt, lname, rname) {
      var t = $(".top", sl), b = $(".base", sl), left = 2;
      sl.classList.add("loading");
      function done() { if (--left === 0) sl.classList.remove("loading"); }
      function setSrc(im, src) {
        if (im.getAttribute("src") === src && im.complete) { done(); return; }
        im.onload = im.onerror = done; im.src = src;
      }
      setSrc(t, topSrc); setSrc(b, baseSrc); t.alt = lname + ": " + alt; b.alt = rname + ": " + alt;
      $(".tag.l", sl).textContent = lname; $(".tag.r", sl).textContent = rname;
    }
    function render() {
      var it = items[state.i], m = it.img, L = state.sca, R = right();
      swap(main, m[L].full, m[R].full, it.prompt, NAMES[L], NAMES[R]);
      swap(side, m[L].crop, m[R].crop, it.prompt + " (1:1 crop)", NAMES[L], NAMES[R]);
      var W = it.size[0], H = it.size[1], x = it.box[0], y = it.box[1], s = it.box[2];
      boxEl.style.left = (x / W * 100) + "%"; boxEl.style.top = (y / H * 100) + "%";
      boxEl.style.width = (s / W * 100) + "%"; boxEl.style.height = (s / H * 100) + "%";
      $$("button", thumbs).forEach(function (b, i) { b.setAttribute("aria-pressed", i === state.i ? "true" : "false"); });
      meta.innerHTML = "";
      meta.appendChild(el("p", { class: "prompt" }, [
        el("span", { class: "chip " + (it.label === "random" ? "" : "curated"), text: it.label === "random" ? "random draw" : "curated" }),
        el("q", { text: it.prompt })]));
      var steps = num("config.steps") || "28", prog = num("main.prog.steps") || "30+17+10";
      meta.appendChild(el("p", { class: "small muted", text: "FLUX.1-dev, " + sizeLabel(it.size) + ", seed label 0 for every method. Progressive SCA runs HRDiT's " + prog + "-step schedule with SCA attention; single-pass SCA is the final operator, " + steps + " steps at the target. Crop: " + s + " × " + s + " pixels at (" + x + ", " + y + "), shown at native resolution; the window was fixed on the previous operator's image." }));
      /* warm the next prompt */
      var nx = items[(state.i + 1) % items.length];
      [nx.img[L].full, nx.img[R].full].forEach(function (u) { var p = new Image(); p.src = u; });
    }
    sync(); render();
  })();

  /* Traces: real videos when rendered, labelled placeholders otherwise */
  (function traces() {
    var host = $("#trace-grid"); if (!host) return;
    var thumbs = {}; ((IM.compare || {}).items || []).forEach(function (it) { thumbs[it.sid] = it; });
    var avail = {}; (TR.available || []).forEach(function (v) { (avail[v.sid] = avail[v.sid] || {})[v.kind] = v; });
    var expected = TR.expected || [];
    var nRendered = Object.keys(avail).length;
    var status = $("#trace-status");
    var nPrev = (TR.available || []).filter(function (v) { return v.operator !== "final"; }).length;
    if (status) status.textContent = nRendered ? (nRendered + " of " + expected.length + " prompts rendered." +
      (nPrev ? " Videos marked “previous operator” show SCA without the window schedule; the final operator's traces replace them once their GPU runs finish." : "")) :
      "None rendered yet: the 16 GPU runs these videos need are queued but have not run (see web/README.md).";
    var io = ("IntersectionObserver" in window) ? new IntersectionObserver(function (es) {
      es.forEach(function (en) { var v = en.target; if (en.isIntersecting) { var p = v.play(); if (p && p.catch) p.catch(function () {}); } else v.pause(); });
    }, { threshold: .5 }) : null;
    expected.forEach(function (ex) {
      var a = avail[ex.sid] || {}, it = thumbs[ex.sid] || {};
      var media = el("div", { class: "media" }), body = el("div", { class: "body" });
      var card = el("article", { class: "trace-card" }, [media, body]);
      var kinds = ["sca", "vs_area"].filter(function (k) { return a[k]; });
      function show(kind) {
        media.innerHTML = "";
        if (!kind) {
          if (it.thumb) media.appendChild(img(it.img ? it.img.sca.full : it.thumb, "", { class: "ph-img" }));
          media.appendChild(el("div", { class: "placeholder" }, [el("div", {}, [el("strong", { text: "Trajectory video not rendered yet" }),
            el("span", { text: "Placeholder. The run that decodes every step at 4096² is queued for a GPU." })])]));
          return;
        }
        var v = a[kind], vid = el("video", { muted: "", loop: "", playsinline: "", preload: "none", controls: "", poster: v.poster, "aria-label": "Denoising trajectory of single-pass SCA: " + ex.prompt });
        vid.muted = true; vid.appendChild(el("source", { src: v.mp4, type: "video/mp4" }));
        media.appendChild(vid); if (io) io.observe(vid);
      }
      var lab = ex.label || it.label;
      var prev = kinds.some(function (k) { return a[k].operator !== "final"; });
      var label = el("p", { class: "prompt" }, [el("span", { class: "chip " + (lab === "random" ? "" : "curated"), text: lab === "random" ? "random draw" : "curated" }),
        prev ? el("span", { class: "chip prev", text: "previous operator" }) : null, ex.prompt || it.prompt || ""]);
      body.appendChild(label);
      if (kinds.length > 1) {
        var seg = el("div", { class: "seg", role: "group", "aria-label": "Video" });
        kinds.forEach(function (k, i) {
          var b = el("button", { type: "button", "aria-pressed": i === 0 ? "true" : "false", text: k === "sca" ? "Single-pass SCA" : "Beside AReA" });
          b.addEventListener("click", function () { $$("button", seg).forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); }); show(k); });
          seg.appendChild(b);
        });
        body.appendChild(seg);
      }
      show(kinds[0]);
      host.appendChild(card);
    });
  })();

  /* Scale strip */
  (function scale() {
    var data = IM.scale || []; var sizes = $("#scale-sizes"), shapes = $("#scale-shapes"); if (!sizes) return;
    data.filter(function (d) { return d.group === "sizes"; }).forEach(function (d) {
      var side = d.size[0], t = num("latency.sca." + side + "x" + side) || num("latency.sca." + side);
      sizes.appendChild(el("figure", { class: "cell" }, [
        el("div", { class: "img" }, [img(d.full, "Single-pass SCA at " + sizeLabel(d.size) + ": " + d.prompt), el("span", { class: "size-tag", text: sizeLabel(d.size) })]),
        el("div", { class: "crop" }, [img(d.crop_files[0], "1:1 crop at " + sizeLabel(d.size))]),
        el("figcaption", { class: "meta", html: "<b>" + sizeLabel(d.size) + "</b>" + (t ? " · " + t + " s" : "") + (side >= 8192 ? "<br>exploratory size; blur and blocks remain" : "") })]));
    });
    data.filter(function (d) { return d.group === "shapes"; }).forEach(function (d) {
      shapes.appendChild(el("figure", {}, [
        el("div", { class: "img" }, [img(d.full, (d.model || "FLUX.1-dev") + " with SCA at " + sizeLabel(d.size) + ": " + d.prompt), el("span", { class: "size-tag", text: sizeLabel(d.size) })]),
        el("div", { class: "crop" }, [img(d.crop_files[0], "1:1 crop")]),
        el("figcaption", { class: "meta strip-meta", html: "<b>" + (d.model || "FLUX.1-dev") + "</b>, " + sizeLabel(d.size) + (d.operator === "previous" ? '<br><span class="prev-tag">previous operator</span>' : "") })]));
    });
  })();

  /* Local errors of the previous operator: SCA and AReA crops in the same window */
  (function failures() {
    var data = IM.failures || []; var local = $("#fail-local"); if (!local) return;
    var byNote = {};
    data.filter(function (d) { return d.group === "local"; }).forEach(function (d) { var k = d.sid; (byNote[k] = byNote[k] || {})[d.src] = d; });
    Object.keys(byNote).forEach(function (k) {
      var p = byNote[k];
      if (!p.sca || !p.area) return;
      local.appendChild(el("figure", {}, [el("div", { class: "two" }, [img(p.sca.crop_files[0], "SCA 1:1 crop: " + p.sca.note), img(p.area.crop_files[0], "AReA 1:1 crop, same window")]),
        el("figcaption", { class: "meta", html: "<b>" + p.sca.note.charAt(0).toUpperCase() + p.sca.note.slice(1) + "</b> (SCA, left); AReA right" })]));
    });
  })();

  /* Charts drawn as inline SVG from SCA_NUMBERS */
  var SVGNS = "http://www.w3.org/2000/svg";
  function s(tag, attrs, text) { var e = document.createElementNS(SVGNS, tag); for (var k in attrs) e.setAttribute(k, attrs[k]); if (text !== undefined) e.textContent = text; return e; }
  function signed(v, d) { var t = Math.abs(v).toFixed(d); return (v > 0 ? "+" : v < 0 ? "−" : "") + t; }

  /* Both configurations minus AReA, the paper's paired-difference figure: circles progressive, diamonds single pass */
  (function pairedChart() {
    var host = $("#chart-paired"); if (!host) return;
    var spec = window.SCA_CHART_PAIRED || []; /* [[key, label] or null for a group gap, ...] set in index.html */
    var SER = [["prog", "Progressive SCA", "circle"], ["single", "Single-pass SCA", "diamond"]];
    var rows = [], gap = 0;
    spec.forEach(function (r) {
      if (!r) { gap += 1; return; }
      var row = { key: r[0], label: r[1], gap: gap, s: {} }; gap = 0;
      SER.forEach(function (se) {
        var p = "chart.pa." + se[0] + "." + r[0];
        var v = val(p + ".pct");
        if (!isNaN(v)) row.s[se[0]] = { v: v, lo: val(p + ".pct_lo"), hi: val(p + ".pct_hi"), vd: String(num(p + ".verdict") || "") };
      });
      if (Object.keys(row.s).length) rows.push(row);
    });
    if (!rows.length) { host.textContent = "Numbers not built yet."; return; }
    var W = 520, rh = 24, gh = 10, top = 34, left = 104, right = 18;
    var H = top + rows.length * rh + rows.reduce(function (a, r) { return a + r.gap * gh; }, 0) + 40;
    var ends = []; rows.forEach(function (r) { Object.keys(r.s).forEach(function (k) { ends.push(r.s[k].lo, r.s[k].hi); }); });
    var lo = Math.min(0, Math.min.apply(null, ends)), hi = Math.max(0, Math.max.apply(null, ends));
    var pad = (hi - lo) * .04; lo -= pad; hi += pad;
    function X(v) { return left + (v - lo) / (hi - lo) * (W - left - right); }
    var svg = s("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Progressive and single-pass SCA minus AReA at 4096 squared, percent of AReA's mean, 95 percent paired intervals; right is better" });
    var step = (hi - lo) > 120 ? 50 : (hi - lo) > 60 ? 25 : (hi - lo) > 30 ? 10 : 5;
    for (var t = Math.ceil(lo / step) * step; t <= hi; t += step) {
      svg.appendChild(s("line", { x1: X(t), x2: X(t), y1: top - 6, y2: H - 30, stroke: t === 0 ? "var(--chart-base)" : "var(--chart-grid)", "stroke-width": t === 0 ? 1.2 : 1 }));
      svg.appendChild(s("text", { x: X(t), y: H - 16, "text-anchor": "middle" }, t === 0 ? "0" : signed(t, 0) + "%"));
    }
    function marker(kind, x, y, fill, stroke) {
      if (kind === "circle") return s("circle", { cx: x, cy: y, r: 3.9, fill: fill, stroke: stroke, "stroke-width": 1.4 });
      var d = 4.4;
      return s("polygon", { points: [x, y - d, x + d, y, x, y + d, x - d, y].join(" "), fill: fill, stroke: stroke, "stroke-width": 1.4 });
    }
    /* legend */
    var lx = left;
    SER.forEach(function (se) {
      svg.appendChild(marker(se[2], lx + 5, 12, "var(--accent)", "var(--accent)"));
      svg.appendChild(s("text", { x: lx + 15, y: 16, class: "lbl" }, se[1]));
      lx += 140;
    });
    var y = top;
    rows.forEach(function (r) {
      y += r.gap * gh;
      var yc = y + rh / 2;
      svg.appendChild(s("text", { x: left - 10, y: yc + 4, "text-anchor": "end", class: "lbl" }, r.label));
      SER.forEach(function (se, j) {
        var p = r.s[se[0]]; if (!p) return;
        var yy = yc + (j ? 4.6 : -4.6);
        var col = p.vd === "better" ? "var(--accent)" : p.vd === "worse" ? "var(--loss)" : "var(--chart-light)";
        var filled = p.vd === "better" || p.vd === "worse";
        var g = s("g", {});
        g.appendChild(s("title", {}, se[1] + ", " + r.label + ": " + signed(p.v, 1) + "% [" + signed(p.lo, 1) + "%, " + signed(p.hi, 1) + "%] of AReA's mean (" + (p.vd || "no verdict") + ")"));
        g.appendChild(s("line", { x1: X(p.lo), x2: X(p.hi), y1: yy, y2: yy, stroke: col, "stroke-width": 1.7 }));
        g.appendChild(marker(se[2], X(p.v), yy, filled ? col : "var(--surface)", col));
        svg.appendChild(g);
      });
      y += rh;
    });
    svg.appendChild(s("text", { x: (left + W - right) / 2, y: H - 2, "text-anchor": "middle" }, "gain over AReA (% of AReA's mean) →"));
    host.appendChild(svg);
  })();

  /* Step bars: single-pass SCA (global opening, shrinking windows, native windows) and progressive SCA (HRDiT's stages) */
  function legend(host, items) {
    var leg = el("div", { class: "sched-legend" }, items.map(function (i) { return el("span", { class: "k " + i[0], text: i[1] }); }));
    host.parentNode.insertBefore(leg, host.nextSibling);
  }
  (function schedBar() {
    var host = $("#sched-bar"); if (!host) return;
    var open = val("config.opening_steps"), nat = val("config.native_steps"), wins = String(num("config.win_sched") || "").split(",").map(function (x) { return parseInt(x, 10); }).filter(function (x) { return !isNaN(x); });
    var w0 = val("config.window"); if (isNaN(open) || isNaN(nat) || !wins.length) return;
    var cells = [];
    for (var i = 0; i < open; i++) cells.push(["glob", ""]);
    wins.forEach(function (w) { cells.push(["wide", String(w)]); });
    for (var j = 0; j < nat; j++) cells.push(["nat", j === 0 ? String(w0) : ""]);
    cells.forEach(function (c) { host.appendChild(el("span", { class: "c " + c[0], text: c[1] })); });
    legend(host, [["glob", "global view only (" + open + " steps)"], ["wide", "shrinking windows (" + wins.length + ")"], ["nat", "native " + w0 + "-token windows (" + nat + ")"]]);
  })();
  (function progBar() {
    var host = $("#sched-prog"); if (!host) return;
    var b = val("prog.cfg.base_steps"), n2 = val("prog.cfg.steps_2048"), n4 = val("prog.cfg.steps_4096");
    var r2 = num("prog.cfg.ratio_2048"), r4 = num("prog.cfg.ratio_4096");
    if (isNaN(b) || isNaN(n2) || isNaN(n4)) return;
    var i;
    for (i = 0; i < b; i++) host.appendChild(el("span", { class: "c full", text: "" }));
    for (i = 0; i < n2; i++) host.appendChild(el("span", { class: "c r2", text: "" }));
    for (i = 0; i < n4; i++) host.appendChild(el("span", { class: "c r4", text: "" }));
    legend(host, [["full", "full attention at 1024² (" + b + " steps)"], ["r2", "SCA at 2048², ratio " + r2 + " (" + n2 + ")"], ["r4", "SCA at 4096², ratio " + r4 + " (" + n4 + ")"]]);
  })();

  /* Seconds per image against size, single pass */
  (function latencyChart() {
    var host = $("#chart-latency"); if (!host) return;
    var sizes = [2048, 3072, 4096, 6144, 8192], methods = [["full", "Full attention", "var(--chart-light)"], ["area", "AReA", "var(--chart-base)"], ["sca", "Single-pass SCA", "var(--accent)"]];
    function get(m, z) { var k = "latency." + m + "." + z + "x" + z; var v = val(k); return isNaN(v) ? val("latency." + m + "." + z) : v; }
    var series = methods.map(function (m) { return { m: m, pts: sizes.map(function (z) { return [z, get(m[0], z)]; }).filter(function (p) { return !isNaN(p[1]); }) }; }).filter(function (sr) { return sr.pts.length; });
    if (!series.length) { host.textContent = "Numbers not built yet."; return; }
    var W = 520, H = 300, L = 46, R = 110, T = 12, B = 34;
    var ys = [].concat.apply([], series.map(function (sr) { return sr.pts.map(function (p) { return p[1]; }); }));
    var y0 = Math.log10(Math.min.apply(null, ys) * .8), y1 = Math.log10(Math.max.apply(null, ys) * 1.15);
    var x0 = Math.log10(2048 * 2048), x1 = Math.log10(8192 * 8192);
    function X(z) { return L + (Math.log10(z * z) - x0) / (x1 - x0) * (W - L - R); }
    function Y(v) { return T + (1 - (Math.log10(v) - y0) / (y1 - y0)) * (H - T - B); }
    var svg = s("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Seconds per image against image size, batch 1, one GH200, single pass" });
    [10, 20, 50, 100, 200, 500, 1000, 2000].forEach(function (v) {
      if (Math.log10(v) < y0 || Math.log10(v) > y1) return;
      svg.appendChild(s("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: "var(--chart-grid)" }));
      svg.appendChild(s("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end" }, v + " s"));
    });
    sizes.forEach(function (z) { svg.appendChild(s("text", { x: X(z), y: H - 12, "text-anchor": "middle" }, z + "²")); });
    series.forEach(function (sr) {
      var d = sr.pts.map(function (p, i) { return (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1); }).join(" ");
      svg.appendChild(s("path", { d: d, fill: "none", stroke: sr.m[2], "stroke-width": sr.m[0] === "sca" ? 2.4 : 1.8 }));
      sr.pts.forEach(function (p) { svg.appendChild(s("circle", { cx: X(p[0]), cy: Y(p[1]), r: 3.2, fill: sr.m[2] })); });
      var last = sr.pts[sr.pts.length - 1];
      svg.appendChild(s("text", { x: X(last[0]) + 7, y: Y(last[1]) + 4, class: sr.m[0] === "sca" ? "acc" : "lbl" }, sr.m[1]));
    });
    host.appendChild(svg);
  })();

  /* Concept chart: every head's calibrated window weight against noise level, beside AReA's one switch (canvas for
     the 1,368 curves, SVG for axes, median, AReA and the picked head) */
  (function conceptChart() {
    var box = $("#concept"), D = window.SCA_CONCEPT; if (!box || !D) return;
    var nH = D.n_heads, nS = D.n_sigma, sig = D.sigma, gate = D.gate, med = D.median, mean = D.mean;
    var raw = atob(D.pi_b64), PI = new Uint8Array(raw.length);
    for (var q = 0; q < raw.length; q++) PI[q] = raw.charCodeAt(q);
    var order = []; for (var h0 = 0; h0 < nH; h0++) order.push(h0);
    order.sort(function (a, b) { return Math.abs(mean[b] - .5) - Math.abs(mean[a] - .5); });  /* extreme heads first */
    var canvas = el("canvas", { "aria-hidden": "true" }), svg = s("svg", { "aria-hidden": "true" });
    box.appendChild(canvas); box.appendChild(svg);
    var readout = $("#concept-readout"), hint = readout ? readout.innerHTML : "";
    var M = { l: 50, r: 14, t: 24, b: 42 }, X0 = 1.0, X1 = 0.08, Y0 = -0.04, Y1 = 1.16;
    var G = { w: 0, h: 0 }, colors = [], selEl = null;
    function px(x) { return M.l + (X0 - x) / (X0 - X1) * (G.w - M.l - M.r); }
    function py(y) { return M.t + (Y1 - y) / (Y1 - Y0) * (G.h - M.t - M.b); }
    function rgb(hex) { hex = hex.replace("#", ""); if (hex.length === 3) hex = hex.replace(/(.)/g, "$1$1"); return [0, 2, 4].map(function (i) { return parseInt(hex.substr(i, 2), 16); }); }
    function piColor(v, stops) {
      for (var i = 1; i < stops.length; i++) {
        if (v <= stops[i][0]) {
          var u = (v - stops[i - 1][0]) / (stops[i][0] - stops[i - 1][0]), a = stops[i - 1][1], b = stops[i][1];
          return "rgb(" + [0, 1, 2].map(function (k) { return Math.round(a[k] + u * (b[k] - a[k])); }).join(",") + ")";
        }
      }
      var e = stops[stops.length - 1][1]; return "rgb(" + e.join(",") + ")";
    }
    function curve(k, ys) {
      var pts = [[px(1), py(0)], [px(gate), py(0)]];
      for (var j = 0; j < nS; j++) pts.push([px(sig[j]), py(ys ? ys[j] : PI[k * nS + j] / 255)]);
      return pts;
    }
    function pathD(pts) { return pts.map(function (p, i) { return (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1); }).join(" "); }
    function draw() {
      var w = box.clientWidth; if (!w) return;
      var h = Math.round(Math.max(250, Math.min(430, w * .68)));
      box.style.height = h + "px"; G.w = w; G.h = h;
      var dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      var ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
      var stops = [[0, rgb(cssVar("--pi-0"))], [.22, rgb(cssVar("--pi-g"))], [.5, rgb(cssVar("--pi-mid"))],
                   [.78, rgb(cssVar("--pi-l"))], [1, rgb(cssVar("--pi-1"))]];   /* the paper's pi colour map */
      colors = mean.map(function (m) { return piColor(m, stops); });
      ctx.fillStyle = cssVar("--surface-2");
      ctx.fillRect(px(1), py(Y1), px(gate) - px(1), py(Y0) - py(Y1));
      ctx.globalAlpha = isDark() ? .22 : .16; ctx.lineWidth = .9; ctx.lineJoin = "round";
      order.forEach(function (k) {
        var pts = curve(k);
        ctx.strokeStyle = colors[k]; ctx.beginPath();
        pts.forEach(function (p, i) { if (i) ctx.lineTo(p[0], p[1]); else ctx.moveTo(p[0], p[1]); });
        ctx.stroke();
      });
      ctx.globalAlpha = 1;
      svg.setAttribute("viewBox", "0 0 " + w + " " + h); svg.innerHTML = "";
      [0, .5, 1].forEach(function (v) {
        svg.appendChild(s("line", { x1: px(X0), x2: px(X1), y1: py(v), y2: py(v), stroke: "var(--chart-grid)", "stroke-width": 1 }));
        svg.appendChild(s("text", { x: M.l - 8, y: py(v) + 4, "text-anchor": "end" }, String(v)));
      });
      [[1, "1"], [gate, String(num("config.gate_sigma") || gate)], [.6, "0.6"], [.3, "0.3"], [.1, "0.1"]].forEach(function (t) {
        svg.appendChild(s("line", { x1: px(t[0]), x2: px(t[0]), y1: py(Y0), y2: py(Y0) + 4, stroke: "var(--chart-base)" }));
        svg.appendChild(s("text", { x: px(t[0]), y: py(Y0) + 17, "text-anchor": "middle" }, t[1]));
      });
      svg.appendChild(s("line", { x1: px(X0), x2: px(X1), y1: py(Y0), y2: py(Y0), stroke: "var(--chart-base)" }));
      svg.appendChild(s("line", { x1: px(X0), x2: px(X0), y1: py(Y1), y2: py(Y0), stroke: "var(--chart-base)" }));
      var gx = (px(1) + px(gate)) / 2, gy = (py(Y1) + py(Y0)) / 2;
      svg.appendChild(s("text", { x: gx, y: gy, "text-anchor": "middle", class: "lbl-glob", transform: "rotate(-90 " + gx + " " + gy + ")" }, "global view"));
      svg.appendChild(s("text", { x: (px(X0) + px(X1)) / 2, y: h - 6, "text-anchor": "middle" }, "noise level σ  (sampling runs left to right)"));
      var ay = (py(Y1) + py(Y0)) / 2;
      var yt = s("text", { x: 13, y: ay, "text-anchor": "middle", transform: "rotate(-90 13 " + ay + ")" }, "weight on the window, π");
      yt.appendChild(s("tspan", { "baseline-shift": "sub", "font-size": "9" }, "h"));
      svg.appendChild(yt);
      svg.appendChild(s("path", { d: pathD(curve(-1, med)), fill: "none", stroke: "var(--accent)", "stroke-width": 2.6, "stroke-linejoin": "round" }));
      svg.appendChild(s("path", { d: pathD([[px(1), py(0)], [px(gate), py(0)], [px(gate), py(1)], [px(.1), py(1)]]), fill: "none", stroke: "var(--area)", "stroke-width": 2.2, "stroke-dasharray": "7 4" }));
      var halo = "paint-order: stroke; stroke: var(--surface); stroke-width: 4px; stroke-linejoin: round;";
      svg.appendChild(s("text", { x: px(.1), y: py(1) - 9, "text-anchor": "end", class: "lbl-area", style: halo }, "AReA: one switch for every head"));
      svg.appendChild(s("text", { x: px(.1), y: py(.36), "text-anchor": "end", class: "lbl-sca", style: halo }, "SCA: one mix per head"));
      svg.appendChild(s("text", { x: px(.1), y: py(.36) + 15, "text-anchor": "end", class: "lbl-sca", style: halo + "font-weight:500" }, "green: median of " + nH + " heads"));
      selEl = s("g", {}); svg.appendChild(selEl);
    }
    function pick(clientX, clientY) {
      var r = box.getBoundingClientRect(), x = clientX - r.left, y = clientY - r.top;
      var sx = X0 - (x - M.l) / (G.w - M.l - M.r) * (X0 - X1), sy = Y1 - (y - M.t) / (G.h - M.t - M.b) * (Y1 - Y0);
      if (sx > gate + .005 || sx < sig[nS - 1] - .02 || sy < Y0 || sy > Y1) { clear(); return; }
      var j = Math.max(0, Math.min(nS - 1, Math.round((sig[0] - sx) / (sig[0] - sig[nS - 1]) * (nS - 1))));
      var best = -1, bd = 1e9;
      for (var k = 0; k < nH; k++) { var dd = Math.abs(PI[k * nS + j] / 255 - sy); if (dd < bd) { bd = dd; best = k; } }
      if (best < 0 || bd > .06) { clear(); return; }
      selEl.innerHTML = "";
      var pts = curve(best);
      selEl.appendChild(s("path", { d: pathD(pts), fill: "none", stroke: "var(--surface)", "stroke-width": 5, "stroke-linejoin": "round" }));
      selEl.appendChild(s("path", { d: pathD(pts), fill: "none", stroke: colors[best], "stroke-width": 2.4, "stroke-linejoin": "round" }));
      selEl.appendChild(s("circle", { cx: px(sig[j]), cy: py(PI[best * nS + j] / 255), r: 4, fill: colors[best], stroke: "var(--ink)", "stroke-width": 1.2 }));
      var b = Math.floor(best / D.heads_per_block), hh = best % D.heads_per_block;
      if (readout) readout.innerHTML = "<b>Block " + b + ", head " + hh + "</b> (counting from 0): mean weight " + mean[best].toFixed(2) + " on the window after the gate.";
    }
    function clear() { if (selEl) selEl.innerHTML = ""; if (readout) readout.innerHTML = hint; }
    box.addEventListener("pointermove", function (e) { pick(e.clientX, e.clientY); });
    box.addEventListener("pointerdown", function (e) { pick(e.clientX, e.clientY); });
    box.addEventListener("pointerleave", clear);
    var raf = 0;
    window.addEventListener("resize", function () { cancelAnimationFrame(raf); raf = requestAnimationFrame(function () { if (box.clientWidth !== G.w) draw(); }); });
    themeHooks.push(draw);
    draw();
  })();

  /* Against a commercial model: single-pass SCA beside Ideogram 4.0 when the files exist */
  (function ideogram() {
    var host = $("#vs-grid"); if (!host) return;
    function build(man) {
      host.innerHTML = "";
      var byId = {}; ((IM.compare || {}).items || []).forEach(function (it) { byId[it.sid] = it; });
      (man.items || []).forEach(function (e) {
        var it = byId[e.prompt_id]; if (!it) return;
        var file = e.file || ("assets/ideogram/" + e.prompt_id + ".webp");
        var right = el("div", { class: "col" }), left = el("div", { class: "col" });
        left.appendChild(el("div", {}, [img(it.img.sca.full, "Single-pass SCA on FLUX.1-dev: " + it.prompt)]));
        left.appendChild(el("div", { class: "crop1" }, [img(it.img.sca.crop, "Single-pass SCA 1:1 crop")]));
        var main = el("div", {}, [el("div", { class: "placeholder" }, [el("div", {}, [el("strong", { text: "Placeholder: Ideogram 4.0 image not added yet" }), el("code", { text: file })])])]);
        var crop = el("div", { class: "crop1" }, [el("div", { class: "placeholder", text: "1:1 crop appears here" })]);
        right.appendChild(main); right.appendChild(crop);
        var rName = el("span", { text: "Ideogram 4.0" });
        var card = el("article", { class: "vs-card" }, [
          el("p", { class: "name" }, [el("b", { text: "Single-pass SCA, FLUX.1-dev · 4096²" }), rName]),
          el("div", { class: "vs-pair" }, [left, right]),
          el("p", { class: "prompt small" }, [el("q", { text: it.prompt })]),
          el("p", { class: "verdict muted", text: e.verdict ? e.verdict : "Verdict pending: written after the Ideogram image is added." })]);
        host.appendChild(card);
        var probe = new Image();
        probe.onload = function () {
          var w = probe.naturalWidth, h = probe.naturalHeight;
          main.innerHTML = ""; main.appendChild(img(file, "Ideogram 4.0: " + it.prompt));
          rName.textContent = "Ideogram 4.0 · " + w + " × " + h + (e.upscaled ? " (upscaled)" : "");
          var c = e.crop_center || [(it.box[0] + it.box[2] / 2) / it.size[0], (it.box[1] + it.box[2] / 2) / it.size[1]];
          var n = it.box[2];                          /* same native pixel count as the SCA crop */
          crop.innerHTML = "";
          var cv = el("div", { style: "width:100%;height:100%;background-repeat:no-repeat;background-image:url('" + file + "')", role: "img", "aria-label": "Ideogram 1:1 crop" });
          crop.appendChild(cv);
          function fit() {
            var D = cv.clientWidth || 200, k = D / n;
            var cx = Math.min(Math.max(c[0] * w, n / 2), w - n / 2), cy = Math.min(Math.max(c[1] * h, n / 2), h - n / 2);
            cv.style.backgroundSize = (w * k) + "px " + (h * k) + "px";
            cv.style.backgroundPosition = (-(cx - n / 2) * k) + "px " + (-(cy - n / 2) * k) + "px";
          }
          fit(); window.addEventListener("resize", fit);
        };
        probe.src = file;
      });
    }
    var fallback = window.SCA_IDEOGRAM || { items: [] };
    if (location.protocol.indexOf("http") === 0 && window.fetch) {
      fetch("build/ideogram_manifest.json", { cache: "no-store" }).then(function (r) { if (!r.ok) throw 0; return r.json(); })
        .then(build).catch(function () { build(fallback); });
    } else build(fallback);
  })();

  /* Method video: a note if the file is missing */
  (function methodVideo() {
    var v = $("#method-video-el"); if (!v) return;
    v.addEventListener("error", function () { var p = $("#method-video-missing"); if (p) p.hidden = false; }, true);
  })();

  /* BibTeX copy */
  $$(".copy").forEach(function (b) {
    b.addEventListener("click", function () {
      var t = $(b.getAttribute("data-target")).textContent;
      var ok = function () { b.textContent = "Copied"; setTimeout(function () { b.textContent = "Copy"; }, 1400); };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(t).then(ok, function () {});
      else { var r = document.createRange(); r.selectNodeContents($(b.getAttribute("data-target"))); var sel = getSelection(); sel.removeAllRanges(); sel.addRange(r); }
    });
  });
})();
