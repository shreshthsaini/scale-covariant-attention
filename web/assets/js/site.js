/* SCA project page. Vanilla JS, no dependencies. Data comes from the generated files
   images.js (SCA_IMAGES), numbers.js (SCA_NUMBERS), traces.js (SCA_TRACES) and
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
  function sizeLabel(wh) { return wh[1] + " × " + wh[0]; }   /* height x width, as in the paper */
  var store = {
    get: function (k) { try { return window.localStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { window.localStorage.setItem(k, v); } catch (e) { /* private mode */ } }
  };

  /* Theme */
  (function theme() {
    var root = document.documentElement, btn = $("#theme-btn");
    var saved = store.get("sca-theme");
    if (saved === "light" || saved === "dark") root.setAttribute("data-theme", saved);
    if (!btn) return;
    btn.addEventListener("click", function () {
      var dark = root.getAttribute("data-theme") ? root.getAttribute("data-theme") === "dark"
        : window.matchMedia("(prefers-color-scheme: dark)").matches;
      var next = dark ? "light" : "dark";
      root.setAttribute("data-theme", next);
      store.set("sca-theme", next);
    });
  })();

  /* Numbers already filled into the HTML by build_numbers.py --fill; refresh from numbers.js in case it is newer */
  $$("[data-num]").forEach(function (e) { var v = num(e.getAttribute("data-num")); if (v !== null) e.textContent = v; });

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
      '<span class="tag l sca">SCA (ours)</span><span class="tag r"></span>' +
      '<input type="range" min="0" max="100" step="0.5" value="50" aria-label="' + label + '">';
    return d;
  }

  /* Comparison section */
  (function compare() {
    var host = $("#cmp"); var data = IM.compare; if (!host || !data) return;
    var items = data.items, NAMES = { area: "AReA", full: "Full attention" };
    var state = { i: 0, base: "area" };
    var main = sliderMarkup("", "Divider between SCA and the baseline, full image");
    var side = sliderMarkup("", "Divider between SCA and the baseline, 1:1 crop");
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
    $$("#cmp-base button").forEach(function (b) {
      b.addEventListener("click", function () {
        state.base = b.getAttribute("data-base");
        $$("#cmp-base button").forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
        render();
      });
    });
    function swap(sl, topSrc, baseSrc, alt) {
      var t = $(".top", sl), b = $(".base", sl), left = 2;
      sl.classList.add("loading");
      function done() { if (--left === 0) sl.classList.remove("loading"); }
      function setSrc(im, src) {
        if (im.getAttribute("src") === src && im.complete) { done(); return; }
        im.onload = im.onerror = done; im.src = src;
      }
      setSrc(t, topSrc); setSrc(b, baseSrc); t.alt = "SCA: " + alt; b.alt = NAMES[state.base] + ": " + alt;
      $(".tag.r", sl).textContent = NAMES[state.base];
    }
    function render() {
      var it = items[state.i], m = it.img;
      swap(main, m.sca.full, m[state.base].full, it.prompt);
      swap(side, m.sca.crop, m[state.base].crop, it.prompt + " (1:1 crop)");
      var W = it.size[0], H = it.size[1], x = it.box[0], y = it.box[1], s = it.box[2];
      boxEl.style.left = (x / W * 100) + "%"; boxEl.style.top = (y / H * 100) + "%";
      boxEl.style.width = (s / W * 100) + "%"; boxEl.style.height = (s / H * 100) + "%";
      $$("button", thumbs).forEach(function (b, i) { b.setAttribute("aria-pressed", i === state.i ? "true" : "false"); });
      meta.innerHTML = "";
      meta.appendChild(el("p", { class: "prompt" }, [
        el("span", { class: "chip " + (it.label === "random" ? "" : "curated"), text: it.label === "random" ? "random draw" : "curated" }),
        el("q", { text: it.prompt })]));
      meta.appendChild(el("p", { class: "small muted", text: "FLUX.1-dev, " + sizeLabel(it.size) + ", seed label 0 for every method. SCA: " + (it.operator === "final" ? "final operator (window schedule and bridge)" : "previous operator, without the schedule") + ". Crop: " + s + " × " + s + " pixels at (" + x + ", " + y + "), shown at native resolution; the window was fixed on the previous operator's image." }));
      /* warm the next prompt */
      var nx = items[(state.i + 1) % items.length];
      [nx.img.sca.full, nx.img[state.base].full].forEach(function (u) { var p = new Image(); p.src = u; });
    }
    render();
  })();

  /* Traces: real videos when rendered, labelled placeholders otherwise */
  (function traces() {
    var host = $("#trace-grid"); if (!host) return;
    var thumbs = {}; ((IM.compare || {}).items || []).forEach(function (it) { thumbs[it.sid] = it; });
    var avail = {}; (TR.available || []).forEach(function (v) { (avail[v.sid] = avail[v.sid] || {})[v.kind] = v; });
    var expected = TR.expected || [];
    var nRendered = Object.keys(avail).length;
    var status = $("#trace-status");
    if (status) status.textContent = nRendered ? (nRendered + " of " + expected.length + " prompts rendered.") :
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
        var v = a[kind], vid = el("video", { muted: "", loop: "", playsinline: "", preload: "none", controls: "", poster: v.poster, "aria-label": "Denoising trajectory: " + ex.prompt });
        vid.muted = true; vid.appendChild(el("source", { src: v.mp4, type: "video/mp4" }));
        media.appendChild(vid); if (io) io.observe(vid);
      }
      var lab = ex.label || it.label;
      var label = el("p", { class: "prompt" }, [el("span", { class: "chip " + (lab === "random" ? "" : "curated"), text: lab === "random" ? "random draw" : "curated" }), ex.prompt || it.prompt || ""]);
      body.appendChild(label);
      if (kinds.length > 1) {
        var seg = el("div", { class: "seg", role: "group", "aria-label": "Video" });
        kinds.forEach(function (k, i) {
          var b = el("button", { type: "button", "aria-pressed": i === 0 ? "true" : "false", text: k === "sca" ? "SCA" : "SCA vs AReA" });
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
        el("div", { class: "img" }, [img(d.full, "SCA at " + sizeLabel(d.size) + ": " + d.prompt), el("span", { class: "size-tag", text: sizeLabel(d.size) })]),
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

  /* Failure cases */
  (function failures() {
    var data = IM.failures || []; var late = $("#fail-late"), gate = $("#fail-gate"), local = $("#fail-local"); if (!late) return;
    (IM.failures_v2 || []).forEach(function (d) {
      late.appendChild(el("figure", {}, [
        el("div", { class: "img" }, [img(d.full, "SCA, final operator, uncurated: " + d.prompt)]),
        el("div", { class: "crops" }, d.crop_files.map(function (c) { return img(c, "1:1 crop: " + d.note); })),
        el("figcaption", { class: "meta", html: "<b>" + d.note.charAt(0).toUpperCase() + d.note.slice(1) + "</b> (detector). " + d.prompt.split(",")[0] + "." })]));
    });
    var gl = { gate_v0: "Mix from step 1", gate_w16: "Mix from step 1, 16-token windows", gate_sca: "Global-only opening, no schedule" };
    data.filter(function (d) { return d.group === "gate"; }).forEach(function (d) {
      gate.appendChild(el("figure", {}, [el("div", { class: "img" }, [img(d.full, gl[d.id] + ": " + d.prompt)]), el("figcaption", { class: "meta", text: gl[d.id] })]));
    });
    var byNote = {};
    data.filter(function (d) { return d.group === "local"; }).forEach(function (d) { var k = d.sid; (byNote[k] = byNote[k] || {})[d.src] = d; });
    Object.keys(byNote).forEach(function (k) {
      var p = byNote[k];
      local.appendChild(el("figure", {}, [el("div", { class: "two" }, [img(p.sca.crop_files[0], "SCA 1:1 crop: " + p.sca.note), img(p.area.crop_files[0], "AReA 1:1 crop, same window")]),
        el("figcaption", { class: "meta", html: "<b>" + p.sca.note.charAt(0).toUpperCase() + p.sca.note.slice(1) + "</b> (SCA, left); AReA right" })]));
    });
  })();

  /* Charts drawn as inline SVG from SCA_NUMBERS */
  var SVGNS = "http://www.w3.org/2000/svg";
  function s(tag, attrs, text) { var e = document.createElementNS(SVGNS, tag); for (var k in attrs) e.setAttribute(k, attrs[k]); if (text !== undefined) e.textContent = text; return e; }
  (function pairedChart() {
    var host = $("#chart-paired"); if (!host) return;
    var spec = window.SCA_CHART_PAIRED || []; /* [[key, label], ...] set in index.html */
    var rows = spec.map(function (r) {
      return { label: r[1], v: val("paired." + r[0] + ".pct"), lo: val("paired." + r[0] + ".pct_lo"), hi: val("paired." + r[0] + ".pct_hi"), sig: String(num("paired." + r[0] + ".sig")) === "true" || String(num("paired." + r[0] + ".sig")) === "yes" };
    }).filter(function (r) { return !isNaN(r.v); });
    if (!rows.length) { host.textContent = "Numbers not built yet."; return; }
    var W = 520, rh = 22, top = 8, left = 112, right = 46, H = top + rows.length * rh + 30;
    var lo = Math.min(0, Math.min.apply(null, rows.map(function (r) { return r.lo; }))), hi = Math.max.apply(null, rows.map(function (r) { return r.hi; }));
    var pad = (hi - lo) * .06; lo -= pad; hi += pad;
    function X(v) { return left + (v - lo) / (hi - lo) * (W - left - right); }
    var svg = s("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Paired differences of SCA minus AReA at 4096 squared, percent of AReA's mean, 95 percent intervals" });
    var step = (hi - lo) > 60 ? 20 : (hi - lo) > 30 ? 10 : 5;
    for (var t = Math.ceil(lo / step) * step; t <= hi; t += step) {
      svg.appendChild(s("line", { x1: X(t), x2: X(t), y1: top - 4, y2: H - 26, stroke: t === 0 ? "var(--chart-base)" : "var(--chart-grid)", "stroke-width": t === 0 ? 1 : 1 }));
      svg.appendChild(s("text", { x: X(t), y: H - 12, "text-anchor": "middle" }, (t > 0 ? "+" : "") + t + "%"));
    }
    rows.forEach(function (r, i) {
      var y = top + i * rh + rh / 2, c = r.sig ? "var(--accent)" : "var(--chart-light)";
      svg.appendChild(s("text", { x: left - 8, y: y + 4, "text-anchor": "end", class: "lbl" }, r.label));
      svg.appendChild(s("line", { x1: X(r.lo), x2: X(r.hi), y1: y, y2: y, stroke: c, "stroke-width": 2, "stroke-linecap": "round" }));
      svg.appendChild(s("circle", { cx: X(r.v), cy: y, r: 4.2, fill: c }));
      svg.appendChild(s("text", { x: X(r.hi) + 6, y: y + 4, class: r.sig ? "acc" : "" }, (r.v > 0 ? "+" : "") + r.v.toFixed(1) + "%"));
    });
    svg.appendChild(s("text", { x: W - right, y: H - 1, "text-anchor": "end" }, "SCA better →"));
    host.appendChild(svg);
  })();
  (function schedBar() {
    var host = $("#sched-bar"); if (!host) return;
    var open = val("config.opening_steps"), nat = val("config.native_steps"), wins = String(num("config.win_sched") || "").split(",").map(function (x) { return parseInt(x, 10); }).filter(function (x) { return !isNaN(x); });
    var w0 = val("config.window"); if (isNaN(open) || isNaN(nat) || !wins.length) return;
    var cells = [];
    for (var i = 0; i < open; i++) cells.push(["glob", ""]);
    wins.forEach(function (w) { cells.push(["wide", String(w)]); });
    for (var j = 0; j < nat; j++) cells.push(["nat", j === 0 ? String(w0) : ""]);
    cells.forEach(function (c) { host.appendChild(el("span", { class: "c " + c[0], text: c[1] })); });
    var leg = el("div", { class: "sched-legend" }, [
      el("span", { class: "k glob", text: "global view only (" + open + " steps)" }),
      el("span", { class: "k wide", text: "shrinking windows (" + wins.length + ")" }),
      el("span", { class: "k nat", text: "native " + w0 + "-token windows (" + nat + ")" })]);
    host.parentNode.insertBefore(leg, host.nextSibling);
  })();
  function hbarChart(hostSel, rows, opts) {
    var host = $(hostSel); if (!host) return;
    rows = rows.filter(function (r) { return !isNaN(r.v); });
    if (!rows.length) { host.textContent = "Numbers not built yet."; return; }
    var W = 520, rh = 24, top = 6, left = opts.left || 150, right = 64, H = top + rows.length * rh + 30;
    var hi = opts.max || 100;
    function X(v) { return left + v / hi * (W - left - right); }
    var svg = s("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": opts.aria });
    for (var t = 0; t <= hi; t += opts.step || 20) {
      svg.appendChild(s("line", { x1: X(t), x2: X(t), y1: top - 2, y2: H - 26, stroke: t === 0 ? "var(--chart-base)" : "var(--chart-grid)", "stroke-width": 1 }));
      svg.appendChild(s("text", { x: X(t), y: H - 12, "text-anchor": "middle" }, t + "%"));
    }
    rows.forEach(function (r, i) {
      var y = top + i * rh + rh / 2, c = r.ours ? "var(--accent)" : (r.old ? "var(--warn)" : "var(--chart-light)");
      svg.appendChild(s("text", { x: left - 8, y: y + 4, "text-anchor": "end", class: r.ours ? "acc" : "lbl" }, r.label));
      svg.appendChild(s("rect", { x: X(0), y: y - 7, width: Math.max(1.5, X(r.v) - X(0)), height: 14, rx: 2, fill: c, opacity: r.ours || r.old ? 1 : .85 }));
      if (!isNaN(r.lo) && !isNaN(r.hi)) svg.appendChild(s("line", { x1: X(r.lo), x2: X(r.hi), y1: y, y2: y, stroke: "var(--ink)", "stroke-width": 1.2, opacity: .55 }));
      svg.appendChild(s("text", { x: X(isNaN(r.hi) ? r.v : Math.max(r.v, r.hi)) + 6, y: y + 4, class: r.ours ? "acc" : "" }, r.text));
    });
    host.appendChild(svg);
  }
  (function gateChart() {
    var spec = [["a46_sca_g93", "Gate σ = 0.93", 0, 1], ["a46_sca_nb", "Gate σ = " + (num("config.gate_sigma") || "0.89") + " (old SCA)", 0, 1], ["a46_sca_g85", "Gate σ = 0.85", 0, 0], ["a46_abl_globalonly", "Global view only", 0, 0], ["a46_dup_i_shrink", "SCA, final operator", 1, 0]];
    hbarChart("#chart-gate", spec.map(function (r) {
      var k = val("dup." + r[0] + ".k"), n = val("dup." + r[0] + ".n");
      return { label: r[1], v: 100 * k / n, lo: NaN, hi: NaN, ours: r[2] === 1, old: r[3] === 1, text: k + " of " + n };
    }), { aria: "Prompts with a repeated person or animal on a 46-prompt subset, by gate", left: 160 });
  })();
  (function dupChart() {
    var spec = [["sca_v2", "SCA (ours)", 1, 0], ["sca_nb", "SCA, no schedule", 0, 1], ["sigma", "SigMa", 0, 0], ["dype", "DyPE", 0, 0], ["sega", "SEGA", 0, 0], ["hrdit", "HRDiT", 0, 0], ["full", "Full attention", 0, 0], ["sca_v2_prog", "Progressive SCA", 0, 0], ["ultraimage", "UltraImage", 0, 0], ["area", "AReA", 0, 0]];
    hbarChart("#chart-dup", spec.map(function (r) {
      var v = val("dup." + r[0] + ".rate");
      return { label: r[1], v: v, lo: val("dup." + r[0] + ".lo"), hi: val("dup." + r[0] + ".hi"), ours: r[2] === 1, old: r[3] === 1, text: Math.round(v) + "%" };
    }), { aria: "Share of people or animal prompts with a repeated subject at 4096 squared, all methods, Wilson 95 percent intervals", left: 120 });
  })();
  (function dupPairs() {
    var host = $("#dup-pairs"); if (!host) return;
    var cap = function (c) { return String(c || "").split(";").map(function (t) { var a = t.split(":"), b = (a[1] || "").split("/"); return a[0] + " " + b[0] + " of " + b[1]; }).join(", "); };
    (IM.dup_pairs || []).forEach(function (d) {
      host.appendChild(el("figure", {}, [
        el("div", { class: "two" }, [
          el("div", { class: "pi" }, [img(d.old, "Old operator, 1:1 crop: " + d.prompt), el("span", { class: "tag", text: "old" })]),
          el("div", { class: "pi" }, [img(d["new"], "Final operator, same prompt, seed and window: " + d.prompt), el("span", { class: "tag new", text: "final" })])]),
        el("figcaption", { class: "meta", html: "<b>Detector:</b> " + cap(d.counts_old) + " &rarr; " + cap(d.counts_new) + ". <span class=\"q\">" + d.prompt.split(",")[0] + ".</span>" })]));
    });
  })();
  (function latencyChart() {
    var host = $("#chart-latency"); if (!host) return;
    var sizes = [2048, 3072, 4096, 6144, 8192], methods = [["full", "Full attention", "var(--chart-light)"], ["area", "AReA", "var(--chart-base)"], ["sca", "SCA (ours)", "var(--accent)"]];
    function get(m, z) { var k = "latency." + m + "." + z + "x" + z; var v = val(k); return isNaN(v) ? val("latency." + m + "." + z) : v; }
    var series = methods.map(function (m) { return { m: m, pts: sizes.map(function (z) { return [z, get(m[0], z)]; }).filter(function (p) { return !isNaN(p[1]); }) }; }).filter(function (sr) { return sr.pts.length; });
    if (!series.length) { host.textContent = "Numbers not built yet."; return; }
    var W = 520, H = 300, L = 46, R = 92, T = 12, B = 34;
    var ys = [].concat.apply([], series.map(function (sr) { return sr.pts.map(function (p) { return p[1]; }); }));
    var y0 = Math.log10(Math.min.apply(null, ys) * .8), y1 = Math.log10(Math.max.apply(null, ys) * 1.15);
    var x0 = Math.log10(2048 * 2048), x1 = Math.log10(8192 * 8192);
    function X(z) { return L + (Math.log10(z * z) - x0) / (x1 - x0) * (W - L - R); }
    function Y(v) { return T + (1 - (Math.log10(v) - y0) / (y1 - y0)) * (H - T - B); }
    var svg = s("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Seconds per image against image size, batch 1, one GH200" });
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

  /* Against a commercial model: SCA beside Ideogram 4.0 when the files exist */
  (function ideogram() {
    var host = $("#vs-grid"); if (!host) return;
    function build(man) {
      host.innerHTML = "";
      var byId = {}; ((IM.compare || {}).items || []).forEach(function (it) { byId[it.sid] = it; });
      (man.items || []).forEach(function (e) {
        var it = byId[e.prompt_id]; if (!it) return;
        var file = e.file || ("assets/ideogram/" + e.prompt_id + ".webp");
        var right = el("div", { class: "col" }), left = el("div", { class: "col" });
        left.appendChild(el("div", {}, [img(it.img.sca.full, "SCA on FLUX.1-dev: " + it.prompt)]));
        left.appendChild(el("div", { class: "crop1" }, [img(it.img.sca.crop, "SCA 1:1 crop")]));
        var main = el("div", {}, [el("div", { class: "placeholder" }, [el("div", {}, [el("strong", { text: "Placeholder: Ideogram 4.0 image not added yet" }), el("code", { text: file })])])]);
        var crop = el("div", { class: "crop1" }, [el("div", { class: "placeholder", text: "1:1 crop appears here" })]);
        right.appendChild(main); right.appendChild(crop);
        var rName = el("span", { text: "Ideogram 4.0" });
        var card = el("article", { class: "vs-card" }, [
          el("p", { class: "name" }, [el("b", { text: "SCA, FLUX.1-dev · 4096²" }), rName]),
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

  /* Method video: caption track text size and a note if the file is missing */
  (function methodVideo() {
    var v = $("#method-video"); if (!v) return;
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
