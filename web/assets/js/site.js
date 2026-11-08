/* SCA project page. Vanilla JS, no build step, works from file://.
   Data (generated, see web/README.md): numbers.js SCA_NUMBERS, images.js SCA_IMAGES, media.js SCA_MEDIA,
   showcase.js SCA_SHOWCASE, process.js SCA_PROCESS, traces.js SCA_TRACES, gallery.js SCA_GALLERY,
   concept.js SCA_CONCEPT, method_data.js SCA_METHOD, ideogram_manifest.js SCA_IDEOGRAM. */
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
  (function theme() {
    var btn = $("#theme-btn"), root = document.documentElement;
    if (btn) btn.addEventListener("click", function () {
      var next = isDark() ? "light" : "dark";
      root.setAttribute("data-theme", next); store.set("sca-theme", next);
      themeHooks.forEach(function (f) { f(); });
    });
    if (window.matchMedia) {
      var mq = matchMedia("(prefers-color-scheme: dark)");
      var on = function () { if (!root.getAttribute("data-theme")) themeHooks.forEach(function (f) { f(); }); };
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
    var v = el("video", { muted: true, loop: true, playsinline: true, preload: "metadata", poster: poster || null, "aria-label": label, controls: controls || REDUCED || null });
    v.muted = true; v.defaultMuted = true;
    v.src = src;
    v.addEventListener("pause", function () { if (document.visibilityState === "visible" && v._seen) v._userPaused = !!v._manual; });
    if (vio && !REDUCED) vio.observe(v); else if (!REDUCED) v.autoplay = true;
    return v;
  }

  /* ---------------------------------------------------------------- native-pixel lens */
  /* spec: {view, vw, vh, w, h, alt, badge, native (one file) or tiles {path "…{c}_{r}.webp", size, cols, rows}, insets} */
  function ZoomFrame(spec) {
    var W = spec.w, H = spec.h;
    var fr = el("div", { class: "zf", tabindex: "0", role: "group", "aria-label": (spec.alt || "Image") + ". Move the pointer, touch, or use the arrow keys for a native-pixel lens; plus and minus change its power." });
    var view = img(spec.view, spec.alt || "", { class: "view", width: spec.vw, height: spec.vh, loading: spec.eager ? "eager" : "lazy", fetchpriority: spec.eager ? "high" : null });
    fr.appendChild(view);
    (spec.insets || []).forEach(function (s) {
      fr.appendChild(el("span", { class: "win", style: "left:" + (s.window[0] / W * 100).toFixed(3) + "%;top:" + (s.window[1] / H * 100).toFixed(3) + "%;width:" + (s.window[2] / W * 100).toFixed(3) + "%;height:" + (s.window[2] / H * 100).toFixed(3) + "%", "aria-hidden": "true" }));
      fr.appendChild(el("span", { class: "inset " + (s.corner || "bl") }, [img(s.file, "Native 1:1 crop" + (s.content ? ": " + s.content : ""))]));
    });
    var lens = el("div", { class: "lens", "aria-hidden": "true" }), cv = el("canvas"), tag = el("div", { class: "ltag", "aria-hidden": "true" });
    lens.appendChild(cv); fr.appendChild(lens); fr.appendChild(tag);
    if (spec.badge) fr.appendChild(el("span", { class: "badge", text: spec.badge }));
    fr.appendChild(el("span", { class: "hint", "aria-hidden": "true", html: '<span class="h-hover">Hover for native pixels</span><span class="h-touch">Touch for native pixels</span>' }));
    var POW = [1, 2], P = 1;
    var pw = el("div", { class: "powers", role: "group", "aria-label": "Lens power" });
    var pbs = POW.map(function (p) {
      var b = el("button", { type: "button", "aria-pressed": String(p === P), text: p + ":1", title: p === 1 ? "one image pixel per screen pixel" : "each image pixel twice as large" });
      b.addEventListener("click", function (e) { e.stopPropagation(); setP(p); });
      b.addEventListener("pointerdown", function (e) { e.stopPropagation(); });
      pw.appendChild(b); return b;
    });
    fr.appendChild(pw);
    function setP(p) { P = p; pbs.forEach(function (b, i) { b.setAttribute("aria-pressed", String(POW[i] === P)); }); if (st.on) draw(); }

    var st = { on: false, x: .5, y: .5 }, cache = {}, single = null;
    function srcTile(c, r) {
      var key = c + "_" + r;
      if (cache[key]) return cache[key];
      var t = { img: new Image(), ok: false };
      t.img.decoding = "async";
      t.img.onload = function () { t.ok = true; if (st.on) draw(); };
      t.img.src = spec.tiles.path.replace("{c}", c).replace("{r}", r);
      cache[key] = t; return t;
    }
    function nativeOne() {
      if (!single) { single = { img: new Image(), ok: false }; single.img.onload = function () { single.ok = true; tag.textContent = label(); if (st.on) draw(); }; single.img.src = spec.native; }
      return single;
    }
    function label() { return P + ":1 native pixels" + ((spec.native && !(single && single.ok)) ? " (loading)" : ""); }
    function draw() {
      var fw = fr.clientWidth, fh = fr.clientHeight; if (!fw || !fh) return;
      var D = Math.round(clamp(fw * .3, 130, 300));
      var dpr = Math.min(2, window.devicePixelRatio || 1);
      if (cv.width !== Math.round(D * dpr)) { cv.width = Math.round(D * dpr); cv.height = Math.round(D * dpr); }
      lens.style.width = lens.style.height = D + "px";
      var lx = st.x * fw, ly = st.y * fh;
      lens.style.transform = "translate(" + (lx - D / 2) + "px," + (ly - D / 2) + "px)";
      tag.style.transform = "translate(" + (lx - 0) + "px," + (ly + D / 2 + 8) + "px) translate(-50%,0)";
      tag.textContent = label();
      var nx = st.x * W, ny = st.y * H, half = D / 2 / P;                    /* native pixels around the centre */
      var x0 = nx - half, y0 = ny - half, s = (D * dpr) / (2 * half);       /* device pixels per native pixel */
      var ctx = cv.getContext("2d");
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = "#222"; ctx.fillRect(0, 0, cv.width, cv.height);
      /* base: the display image, upscaled */
      if (view.complete && view.naturalWidth) {
        var kx = view.naturalWidth / W, ky = view.naturalHeight / H;
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(view, x0 * kx, y0 * ky, 2 * half * kx, 2 * half * ky, 0, 0, cv.width, cv.height);
      }
      ctx.imageSmoothingEnabled = s < 1.5;
      if (spec.tiles) {
        var T = spec.tiles.size;
        var c0 = Math.max(0, Math.floor(x0 / T)), c1 = Math.min(spec.tiles.cols - 1, Math.floor((x0 + 2 * half) / T));
        var r0 = Math.max(0, Math.floor(y0 / T)), r1 = Math.min(spec.tiles.rows - 1, Math.floor((y0 + 2 * half) / T));
        for (var r = r0; r <= r1; r++) for (var c = c0; c <= c1; c++) {
          var t = srcTile(c, r); if (!t.ok) continue;
          var tx = c * T, ty = r * T, tw = t.img.naturalWidth, th = t.img.naturalHeight;
          var ix0 = Math.max(x0, tx), iy0 = Math.max(y0, ty), ix1 = Math.min(x0 + 2 * half, tx + tw), iy1 = Math.min(y0 + 2 * half, ty + th);
          if (ix1 <= ix0 || iy1 <= iy0) continue;
          ctx.drawImage(t.img, ix0 - tx, iy0 - ty, ix1 - ix0, iy1 - iy0, (ix0 - x0) * s, (iy0 - y0) * s, (ix1 - ix0) * s, (iy1 - iy0) * s);
        }
      } else if (spec.native) {
        var n = nativeOne();
        if (n.ok) {
          var jx0 = Math.max(0, x0), jy0 = Math.max(0, y0), jx1 = Math.min(W, x0 + 2 * half), jy1 = Math.min(H, y0 + 2 * half);
          if (jx1 > jx0 && jy1 > jy0) ctx.drawImage(n.img, jx0, jy0, jx1 - jx0, jy1 - jy0, (jx0 - x0) * s, (jy0 - y0) * s, (jx1 - jx0) * s, (jy1 - jy0) * s);
        }
      }
    }
    function place(cx, cy) {
      var r = fr.getBoundingClientRect();
      st.x = clamp((cx - r.left) / r.width, 0, 1); st.y = clamp((cy - r.top) / r.height, 0, 1);
      draw();
    }
    function show(on) { st.on = on; fr.classList.toggle("on", on); if (on) { fr.classList.add("used"); draw(); } }
    fr.addEventListener("pointermove", function (e) { if (e.pointerType === "mouse" || st.on) { if (!st.on) show(true); place(e.clientX, e.clientY); } });
    fr.addEventListener("pointerdown", function (e) { show(true); place(e.clientX, e.clientY); });
    fr.addEventListener("pointerleave", function (e) { if (e.pointerType === "mouse") show(false); });
    document.addEventListener("pointerdown", function (e) { if (st.on && !fr.contains(e.target)) show(false); });
    fr.addEventListener("keydown", function (e) {
      var d = .02, k = e.key;
      if (k === "ArrowLeft") st.x -= d; else if (k === "ArrowRight") st.x += d; else if (k === "ArrowUp") st.y -= d; else if (k === "ArrowDown") st.y += d;
      else if (k === "+" || k === "=") { setP(2); return; } else if (k === "-") { setP(1); return; } else if (k === "Escape") { show(false); return; } else return;
      e.preventDefault(); st.x = clamp(st.x, 0, 1); st.y = clamp(st.y, 0, 1); show(true);
    });
    fr.addEventListener("blur", function () { show(false); });
    window.addEventListener("resize", function () { if (st.on) draw(); });
    return fr;
  }

  /* ---------------------------------------------------------------- before/after slider, synced within a group */
  function BA(o, group) {
    /* o: {left: {src, label, ours}, right: {src, label}, w, h, alt, win: [x, y, s] in native px of a w x h image, crop} */
    var box = el("div", { class: "ba" + (o.crop ? " crop" : ""), role: "slider", tabindex: "0", "aria-valuemin": "0", "aria-valuemax": "100",
                          "aria-label": "Divider between " + o.left.label + " (left) and " + o.right.label + " (right)" });
    var base = img(o.right.src, o.right.label + (o.alt ? ": " + o.alt : ""), { width: o.vw || null, height: o.vh || null });
    var topWrap = el("div", { class: "top" }, [img(o.left.src, o.left.label + (o.alt ? ": " + o.alt : ""))]);
    box.appendChild(base); box.appendChild(topWrap);
    if (o.win) box.appendChild(el("span", { class: "win", style: "left:" + (o.win[0] / o.w * 100).toFixed(3) + "%;top:" + (o.win[1] / o.h * 100).toFixed(3) + "%;width:" + (o.win[2] / o.w * 100).toFixed(3) + "%;height:" + (o.win[2] / o.h * 100).toFixed(3) + "%", "aria-hidden": "true" }));
    box.appendChild(el("span", { class: "handle", "aria-hidden": "true" }));
    box.appendChild(el("span", { class: "tag l" + (o.left.ours ? " ours" : ""), text: o.left.label }));
    box.appendChild(el("span", { class: "tag r", text: o.right.label }));
    group = group || { p: 50, members: [] };
    group.members.push(box);
    function setAll(p) {
      group.p = clamp(p, 0, 100);
      group.members.forEach(function (m) { m.style.setProperty("--p", group.p + "%"); m.setAttribute("aria-valuenow", Math.round(group.p)); });
    }
    setAll(group.p);
    var drag = false;
    function at(e) { var r = box.getBoundingClientRect(); return (e.clientX - r.left) / r.width * 100; }
    box.addEventListener("pointerdown", function (e) { drag = true; try { box.setPointerCapture(e.pointerId); } catch (x) { /* ignore */ } setAll(at(e)); });
    box.addEventListener("pointermove", function (e) { if (drag || e.pointerType === "mouse" && e.buttons === 1) setAll(at(e)); });
    ["pointerup", "pointercancel", "lostpointercapture"].forEach(function (t) { box.addEventListener(t, function () { drag = false; }); });
    box.addEventListener("keydown", function (e) {
      if (e.key === "ArrowLeft") { setAll(group.p - 4); e.preventDefault(); } else if (e.key === "ArrowRight") { setAll(group.p + 4); e.preventDefault(); }
    });
    box._group = group;
    return box;
  }

  /* ---------------------------------------------------------------- teaser loops */
  (function loops() {
    var host = $("#loops"); if (!host) return;
    var T = MEDIA.teaser || {};
    var slots = [
      ["zoom", "<b>Deep zoom.</b> Progressive SCA at 4096&sup2;, down to native pixels", "Zoom from the whole image to native pixels"],
      ["stages", "<b>Three stages.</b> Layout at 1024&sup2;, refined at 2048&sup2; and 4096&sup2;", "One sample through the three stages"],
      ["heads", "<b>" + (num("calib.n_heads_c") || "1,368") + " heads.</b> Each with its own weight on the window", "The per-head weights of FLUX.1-dev"]
    ];
    var ready = slots.filter(function (s) { return T[s[0]] && T[s[0]].mp4; });
    (ready.length ? ready : slots).forEach(function (s) {
      var m = T[s[0]], box = el("div", { class: "lbox" });
      if (m && m.mp4) box.appendChild(loopVideo(m.mp4, m.poster, s[2]));
      else box.appendChild(ph("Video in preparation", s[2] + "."));
      host.appendChild(el("figure", { class: "loop" }, [box, el("figcaption", { html: s[1] })]));
    });
    if (ready.length && ready.length < 3) host.classList.add("n" + ready.length);
  })();

  /* ---------------------------------------------------------------- hero */
  function heroSpec(it, eager) {
    return { view: it.view, vw: it.view_size[0], vh: it.view_size[1], w: it.size[0], h: it.size[1], tiles: it.tiles, native: it.native,
             alt: (it.config || "Progressive SCA") + ", " + sizeLabel(it.size) + ": " + it.prompt, badge: sizeLabel(it.size) + (it.model && it.model !== "FLUX.1-dev" ? " · " + it.model : ""), eager: eager };
  }
  function metaRow(bits, oursFirst) {
    return el("div", { class: "meta" }, bits.filter(Boolean).map(function (b, i) { return el("span", { class: i === 0 && oursFirst ? "m-ours" : null, text: b }); }));
  }
  function itemMeta(it) {
    var steps = it.config === "Progressive SCA" ? STEPS.prog : it.steps;
    return metaRow([it.config || "Progressive SCA", it.model || "FLUX.1-dev", sizeLabel(it.size), steps ? steps + " steps" : null, "seed " + (it.seed === undefined ? 0 : it.seed)], true);
  }
  (function hero() {
    var host = $("#hero-figure"); if (!host) return;
    var H = SHOW.hero || [];
    if (H.length) {
      var wrap = el("div", { class: "hshow" + (H.length > 1 ? " two" : "") });
      wrap.style.setProperty("--cols", H.slice(0, 2).map(function (it) { return (it.size[0] / it.size[1]).toFixed(3) + "fr"; }).join(" "));
      H.slice(0, 2).forEach(function (it, i) {
        var btn = it.compare && it.compare.length ? el("button", { type: "button", class: "pill sc-cmp", text: "Compare with " + it.compare.map(function (c) { return c.label; }).join(" and ") }) : null;
        if (btn) btn.addEventListener("click", function () { openCompare(it, 0); });
        wrap.appendChild(el("figure", { class: "sc-card" }, [ZoomFrame(heroSpec(it, i === 0)),
          el("figcaption", {}, [el("span", { class: "sc-title", text: it.title || "" }), itemMeta(it), el("p", { class: "prompt" }, [el("q", { text: it.prompt })]), btn])]));
      });
      host.appendChild(wrap);
      host.appendChild(el("p", { class: "caption hcap", html: "Move the pointer over either image for a lens at native pixels (1:1) or twice that (2:1); on a phone, touch the image. More images, with progressive SCA at 4096&sup2;, are in the <a href=\"#showcase\">showcase</a>." }));
      return;
    }
    var L = IM.hero || []; if (!L.length) { host.appendChild(ph("Hero images in preparation")); return; }
    var big = L.filter(function (h) { return h.slot === "big"; })[0], tiles = L.filter(function (h) { return h.slot !== "big"; });
    var grid = el("div", { class: "hshow legacy" });
    var spec = { view: big.full, vw: 1600, vh: 1600, w: big.size[0], h: big.size[1], native: big.native_file, eager: true,
                 alt: "Progressive SCA, " + sizeLabel(big.size) + ": " + big.prompt, badge: sizeLabel(big.size),
                 insets: big.inset ? [big.inset] : [] };
    grid.appendChild(ZoomFrame(spec));
    var tg = el("div", { class: "tiles" });
    tiles.forEach(function (t) {
      var tile = el("div", { class: "tile" }, [img(t.full, "Progressive SCA, " + sizeLabel(t.size) + ": " + t.prompt, { width: 800, height: 800 })]);
      if (t.inset) {
        var w = t.inset.window;
        tile.appendChild(el("span", { class: "win", style: "left:" + (w[0] / t.size[0] * 100).toFixed(3) + "%;top:" + (w[1] / t.size[1] * 100).toFixed(3) + "%;width:" + (w[2] / t.size[0] * 100).toFixed(3) + "%;height:" + (w[2] / t.size[1] * 100).toFixed(3) + "%", "aria-hidden": "true" }));
        tile.appendChild(el("span", { class: "inset " + (t.inset.corner || "tr") }, [img(t.inset.file, "Native 1:1 crop: " + (t.inset.content || ""))]));
      }
      tg.appendChild(tile);
    });
    grid.appendChild(tg);
    host.appendChild(grid);
    host.appendChild(el("p", { class: "caption hcap", html: "Progressive SCA at 4096&nbsp;&times;&nbsp;4096 on FLUX.1-dev, seed 0, evaluation prompts. Framed insets are native pixels; the lens on the large image shows native pixels too. <a href=\"#showcase\">More in the showcase.</a>" }));
  })();

  /* ---------------------------------------------------------------- the failure row: full attention and the fixes at 4096^2 */
  (function failRow() {
    var host = $("#fail-row"); if (!host) return;
    var items = (IM.compare || {}).items || [];
    var it = items.filter(function (x) { return x.sid === "m8343371271909676658"; })[0] || items[0];
    if (!it) return;
    [["full", "Full attention"], ["area", "AReA"], ["hrdit", "HRDiT"], ["prog", "Progressive SCA"]].forEach(function (m) {
      var r = it.img[m[0]]; if (!r) return;
      host.appendChild(el("figure", { class: m[0] === "prog" ? "ours" : null }, [el("div", { class: "fi" }, [img(r.full, m[1] + ": " + it.prompt)]), el("figcaption", {}, [el("b", { text: m[1] })])]));
    });
    host.appendChild(el("p", { class: "caption fail-cap", html: "FLUX.1-dev at 4096&nbsp;&times;&nbsp;4096, one prompt and seed: <q>" + it.prompt + "</q> With full attention at the target the model loses the layout. See <a href=\"#compare\">Comparisons</a> for every method at native pixels." }));
  })();

  /* ---------------------------------------------------------------- concept chart: every head's window weight against noise level */
  (function conceptChart() {
    var box = $("#concept"), D = window.SCA_CONCEPT; if (!box || !D) return;
    var nH = D.n_heads, nS = D.n_sigma, sig = D.sigma, gate = D.gate, med = D.median, mean = D.mean;
    var raw = atob(D.pi_b64), PI = new Uint8Array(raw.length);
    for (var q = 0; q < raw.length; q++) PI[q] = raw.charCodeAt(q);
    var order = []; for (var h0 = 0; h0 < nH; h0++) order.push(h0);
    order.sort(function (a, b) { return Math.abs(mean[b] - .5) - Math.abs(mean[a] - .5); });
    var canvas = el("canvas", { "aria-hidden": "true" }), sv = svg("svg", { "aria-hidden": "true" });
    box.appendChild(canvas); box.appendChild(sv);
    var readout = $("#concept-readout"), hint = readout ? readout.innerHTML : "";
    var M = { l: 46, r: 12, t: 18, b: 40 }, X0 = 1.0, X1 = 0.08, Y0 = -0.04, Y1 = 1.12;
    var G = { w: 0, h: 0 }, colors = [], selEl = null;
    function px(x) { return M.l + (X0 - x) / (X0 - X1) * (G.w - M.l - M.r); }
    function py(y) { return M.t + (Y1 - y) / (Y1 - Y0) * (G.h - M.t - M.b); }
    function curve(k, ys) {
      var pts = [[px(1), py(0)], [px(gate), py(0)]];
      for (var j = 0; j < nS; j++) pts.push([px(sig[j]), py(ys ? ys[j] : PI[k * nS + j] / 255)]);
      return pts;
    }
    function pathD(pts) { return pts.map(function (p, i) { return (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1); }).join(" "); }
    function draw() {
      var w = box.clientWidth; if (!w) return;
      var h = Math.round(clamp(w * .66, 250, 420));
      box.style.height = h + "px"; G.w = w; G.h = h;
      var dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      var ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
      var stops = piStops();
      colors = mean.map(function (m) { return rgbStr(piRgb(m, stops)); });
      ctx.fillStyle = cssVar("--surface"); ctx.fillRect(px(1), py(Y1), px(gate) - px(1), py(Y0) - py(Y1));
      ctx.globalAlpha = isDark() ? .24 : .17; ctx.lineWidth = .9; ctx.lineJoin = "round";
      order.forEach(function (k) {
        var pts = curve(k); ctx.strokeStyle = colors[k]; ctx.beginPath();
        pts.forEach(function (p, i) { if (i) ctx.lineTo(p[0], p[1]); else ctx.moveTo(p[0], p[1]); });
        ctx.stroke();
      });
      ctx.globalAlpha = 1;
      sv.setAttribute("viewBox", "0 0 " + w + " " + h); sv.innerHTML = "";
      [0, .5, 1].forEach(function (v) {
        sv.appendChild(svg("line", { x1: px(X0), x2: px(X1), y1: py(v), y2: py(v), stroke: "var(--chart-grid)" }));
        sv.appendChild(svg("text", { x: M.l - 8, y: py(v) + 4, "text-anchor": "end" }, String(v)));
      });
      [[1, "1"], [gate, String(num("config.gate_sigma") || gate)], [.6, "0.6"], [.3, "0.3"], [.1, "0.1"]].forEach(function (t) {
        sv.appendChild(svg("line", { x1: px(t[0]), x2: px(t[0]), y1: py(Y0), y2: py(Y0) + 4, stroke: "var(--chart-base)" }));
        sv.appendChild(svg("text", { x: px(t[0]), y: py(Y0) + 17, "text-anchor": "middle" }, t[1]));
      });
      sv.appendChild(svg("line", { x1: px(X0), x2: px(X1), y1: py(Y0), y2: py(Y0), stroke: "var(--chart-base)" }));
      var gx = (px(1) + px(gate)) / 2, gy = (py(Y1) + py(Y0)) / 2;
      sv.appendChild(svg("text", { x: gx, y: gy, "text-anchor": "middle", class: "lbl-glob", transform: "rotate(-90 " + gx + " " + gy + ")" }, "global view only"));
      sv.appendChild(svg("text", { x: (px(X0) + px(X1)) / 2, y: h - 6, "text-anchor": "middle" }, "noise level σ (sampling runs left to right)"));
      var ay = (py(Y1) + py(Y0)) / 2;
      sv.appendChild(svg("text", { x: 12, y: ay, "text-anchor": "middle", transform: "rotate(-90 12 " + ay + ")" }, "weight on the window π"));
      sv.appendChild(svg("path", { d: pathD(curve(-1, med)), fill: "none", stroke: "var(--accent)", "stroke-width": 2.6, "stroke-linejoin": "round" }));
      sv.appendChild(svg("path", { d: pathD([[px(1), py(0)], [px(gate), py(0)], [px(gate), py(1)], [px(.1), py(1)]]), fill: "none", stroke: "var(--area)", "stroke-width": 2.2, "stroke-dasharray": "7 4" }));
      var halo = "paint-order: stroke; stroke: var(--bg); stroke-width: 4px; stroke-linejoin: round;";
      sv.appendChild(svg("text", { x: px(.1), y: py(1) - 9, "text-anchor": "end", class: "lbl-area", style: halo }, "AReA: one switch for every head"));
      sv.appendChild(svg("text", { x: px(.1), y: py(.33), "text-anchor": "end", class: "lbl-sca", style: halo }, "SCA: one mix per head"));
      sv.appendChild(svg("text", { x: px(.1), y: py(.33) + 15, "text-anchor": "end", class: "lbl-sca", style: halo + "font-weight:500" }, "green: median of " + nH + " heads"));
      selEl = svg("g", {}); sv.appendChild(selEl);
    }
    function pick(cx, cy) {
      var r = box.getBoundingClientRect(), x = cx - r.left, y = cy - r.top;
      var sx = X0 - (x - M.l) / (G.w - M.l - M.r) * (X0 - X1), sy = Y1 - (y - M.t) / (G.h - M.t - M.b) * (Y1 - Y0);
      if (sx > gate + .005 || sx < sig[nS - 1] - .02 || sy < Y0 || sy > Y1) { clear(); return; }
      var j = clamp(Math.round((sig[0] - sx) / (sig[0] - sig[nS - 1]) * (nS - 1)), 0, nS - 1);
      var best = -1, bd = 1e9;
      for (var k = 0; k < nH; k++) { var dd = Math.abs(PI[k * nS + j] / 255 - sy); if (dd < bd) { bd = dd; best = k; } }
      if (best < 0 || bd > .06) { clear(); return; }
      selEl.innerHTML = "";
      var pts = curve(best);
      selEl.appendChild(svg("path", { d: pathD(pts), fill: "none", stroke: "var(--bg)", "stroke-width": 5, "stroke-linejoin": "round" }));
      selEl.appendChild(svg("path", { d: pathD(pts), fill: "none", stroke: colors[best], "stroke-width": 2.4, "stroke-linejoin": "round" }));
      selEl.appendChild(svg("circle", { cx: px(sig[j]), cy: py(PI[best * nS + j] / 255), r: 4, fill: colors[best], stroke: "var(--ink)", "stroke-width": 1.2 }));
      var b = Math.floor(best / D.heads_per_block), hh = best % D.heads_per_block;
      if (readout) readout.innerHTML = "<b>Block " + b + ", head " + hh + "</b> (from 0): mean weight " + mean[best].toFixed(2) + " on the window after the opening.";
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

  /* ---------------------------------------------------------------- showcase groups */
  function lightbox(render) {
    var lb = $("#lightbox"), body = $("#lb-body"); if (!lb) return;
    body.innerHTML = ""; render(body); lb.hidden = false; document.body.style.overflow = "hidden";
    var close = $(".lb-close", lb); close.focus();
  }
  (function lightboxInit() {
    var lb = $("#lightbox"); if (!lb) return;
    function shut() { lb.hidden = true; document.body.style.overflow = ""; $("#lb-body").innerHTML = ""; }
    $(".lb-close", lb).addEventListener("click", shut);
    lb.addEventListener("click", function (e) { if (e.target === lb) shut(); });
    document.addEventListener("keydown", function (e) { if (!lb.hidden && e.key === "Escape") shut(); });
  })();
  (function showcase() {
    var host = $("#showcase-body"); if (!host) return;
    var groups = SHOW.groups || [];
    if (!groups.length) {
      host.appendChild(el("div", { class: "sc-group" }, [ph("Showcase images in preparation", "5120&nbsp;&times;&nbsp;5120 and 6144&nbsp;&times;&nbsp;4096 images, product shots, landscapes, architecture, portraits and animals, each with native-pixel crops and the same windows of AReA and of the native image upscaled.")]));
    }
    if (groups.length) {
      var all = [], chips = el("div", { class: "seg sc-filter", role: "group", "aria-label": "Category" }), grid = el("div", { class: "sc-grid three" });
      groups.forEach(function (g) { (g.items || []).forEach(function (it) { all.push([g.id, showCard(it)]); }); });
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
    var sizes = (IM.scale || []).filter(function (d) { return d.group === "sizes"; });
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

  /* a showcase card: the overview with its crop windows, the native crops, a button to compare */
  function boxStyle(b, size) {
    return "left:" + (b[0] / size[0] * 100).toFixed(3) + "%;top:" + (b[1] / size[1] * 100).toFixed(3) + "%;width:" + (b[2] / size[0] * 100).toFixed(3) + "%;height:" + (b[2] / size[1] * 100).toFixed(3) + "%";
  }
  function showCard(it) {
    var fr = el("div", { class: "sc-frame tile" }, [img(it.view, it.config + ", " + sizeLabel(it.size) + ": " + it.prompt, { width: it.view_size[0], height: it.view_size[1] })]);
    (it.crops || []).forEach(function (c, k) { fr.appendChild(el("span", { class: "win", style: boxStyle(c.box, it.size), "aria-hidden": "true" }, [el("i", { text: String(k + 1) })])); });
    fr.appendChild(el("span", { class: "badge", text: sizeLabel(it.size) }));
    var crops = el("div", { class: "sc-crops" }, (it.crops || []).map(function (c, k) { return el("div", { class: "tile" }, [img(c.file, "Native pixels, window " + (k + 1)), el("span", { class: "n", text: String(k + 1) })]); }));
    var btn = it.compare && it.compare.length ? el("button", { type: "button", class: "pill sc-cmp", text: "Compare with " + it.compare.map(function (c) { return c.label; }).join(" and ") }) : null;
    var open = function () { openCompare(it, 0); };
    if (btn) btn.addEventListener("click", open);
    fr.addEventListener("click", open); fr.style.cursor = btn ? "zoom-in" : "default";
    return el("figure", { class: "sc-card" }, [fr, crops, el("figcaption", {}, [el("span", { class: "sc-title", text: it.title || "" }), itemMeta(it), el("p", { class: "prompt" }, [el("q", { text: it.prompt })]), btn])]);
  }
  function openCompare(it, k) {
    if (!it.compare || !it.compare.length) return;
    lightbox(function (body) {
      var c = it.compare[k], grp = { p: 50, members: [] };
      var seg = el("div", { class: "seg lb-seg", role: "group", "aria-label": "Compare with" });
      it.compare.forEach(function (o, j) {
        var b = el("button", { type: "button", "aria-pressed": String(j === k), text: o.label });
        b.addEventListener("click", function () { openCompare(it, j); });
        seg.appendChild(b);
      });
      var crops = el("div", { class: "lb-crops" });
      (it.crops || []).forEach(function (cr, j) {
        if (!c.crops[j]) return;
        crops.appendChild(BA({ left: { src: cr.file, label: it.config, ours: true }, right: { src: c.crops[j], label: c.label }, w: cr.box[2], h: cr.box[2], crop: true, alt: "native window " + (j + 1) }, grp));
      });
      body.appendChild(el("div", { class: "lb-grid" }, [
        BA({ left: { src: it.view, label: it.config, ours: true }, right: { src: c.view, label: c.label }, w: it.size[0], h: it.size[1], vw: it.view_size[0], vh: it.view_size[1], alt: it.prompt }, grp),
        el("div", {}, [seg, crops, metaRow([it.model, sizeLabel(it.size), "seed " + it.seed, c.note], false), el("p", { class: "prompt" }, [el("q", { text: it.prompt })])])]));
    });
  }

  /* ---------------------------------------------------------------- other backbones: SCA beside full attention or the native image */
  (function backbones() {
    var host = $("#backbones-body"); if (!host) return;
    var B = SHOW.backbones || [];
    if (!B.length) {
      host.appendChild(ph("FLUX.2-dev and Qwen-Image comparisons in preparation", "Each pair: SCA at the target size beside the same model with full attention at the target, or at its native resolution upscaled with Lanczos, with synced native crops."));
      return;
    }
    B.forEach(function (it) {
      var cmp = (it.compare || []).filter(function (c) { return c.method !== "lanczos"; })[0] || (it.compare || [])[0];
      if (!cmp) { host.appendChild(showCard(it)); return; }
      var grp = { p: 50, members: [] };
      var main = BA({ left: { src: it.view, label: it.config, ours: true }, right: { src: cmp.view, label: cmp.label }, w: it.size[0], h: it.size[1], vw: it.view_size[0], vh: it.view_size[1], alt: it.prompt }, grp);
      var crops = el("div", { class: "lb-crops" });
      (it.crops || []).slice(0, 2).forEach(function (cr, j) {
        if (cmp.crops[j]) crops.appendChild(BA({ left: { src: cr.file, label: it.config, ours: true }, right: { src: cmp.crops[j], label: cmp.label }, w: cr.box[2], h: cr.box[2], crop: true, alt: "native window " + (j + 1) }, grp));
      });
      var more = it.compare.length > 1 ? el("button", { type: "button", class: "pill", text: "All comparisons" }) : null;
      if (more) more.addEventListener("click", function () { openCompare(it, 0); });
      host.appendChild(el("div", { class: "bb" }, [main, el("div", {}, [crops, el("div", { class: "cmp-meta" }, [metaRow([it.model, sizeLabel(it.size), "seed " + it.seed, it.steps ? it.steps + " steps" : null], false), el("p", { class: "prompt" }, [el("q", { text: it.prompt })]), it.note ? el("p", { class: "small muted", text: it.note }) : null, more])])]));
    });
  })();

  /* ---------------------------------------------------------------- comparisons at 4096^2 */
  (function compare() {
    var main = $("#cmp-main"), side = $("#cmp-side"), meta = $("#cmp-meta"), thumbs = $("#cmp-thumbs"); if (!main) return;
    var items = [];
    ((IM.qual || {}).rows || []).forEach(function (r) { items.push({ sid: r.sid, prompt: r.prompt, box: r.box, size: r.size, img: r.img, thumb: r.img.prog.full }); });
    ((IM.compare || {}).items || []).forEach(function (r) {
      if (items.some(function (x) { return x.sid === r.sid; })) return;
      items.push({ sid: r.sid, prompt: r.prompt, box: r.box, size: r.size, img: r.img, thumb: r.thumb || (r.img.prog || r.img.sca).full });
    });
    if (!items.length) return;
    var LEFT = ["prog", "sca"], RIGHT = ["area", "hrdit", "ultra", "full", "sca", "prog"];
    var st = { i: 0, l: "prog", r: "area" }, grp = { p: 50, members: [] };
    var lseg = $("#cmp-left"), rseg = $("#cmp-right");
    function mk(segEl, keys, field) {
      keys.forEach(function (k) {
        var b = el("button", { type: "button", "data-k": k, text: METHOD[k] });
        b.addEventListener("click", function () { if (b.disabled) return; st[field] = k; if (st.l === st.r) st.r = st.l === "prog" ? "area" : "prog"; render(); });
        segEl.appendChild(b);
      });
    }
    mk(lseg, LEFT, "l"); mk(rseg, RIGHT, "r");
    items.forEach(function (it, i) {
      var b = el("button", { type: "button", "aria-label": "Prompt " + (i + 1) + ": " + short(it.prompt, 80) }, [img(it.thumb, "")]);
      b.addEventListener("click", function () { st.i = i; render(); });
      thumbs.appendChild(b);
    });
    function render() {
      var it = items[st.i];
      if (!it.img[st.r] || st.r === st.l) st.r = ["area", "hrdit", "ultra", "full"].filter(function (k) { return it.img[k]; })[0];
      if (!it.img[st.l]) st.l = "prog";
      $$("button", lseg).forEach(function (b) { var k = b.getAttribute("data-k"); b.disabled = !it.img[k]; b.setAttribute("aria-pressed", String(k === st.l)); });
      $$("button", rseg).forEach(function (b) { var k = b.getAttribute("data-k"); b.disabled = !it.img[k] || k === st.l; b.setAttribute("aria-pressed", String(k === st.r)); });
      $$("button", thumbs).forEach(function (b, i) { b.setAttribute("aria-pressed", String(i === st.i)); });
      grp.members = [];
      var L = it.img[st.l], R = it.img[st.r];
      main.innerHTML = ""; side.innerHTML = ""; meta.innerHTML = "";
      main.appendChild(BA({ left: { src: L.full, label: METHOD[st.l], ours: true }, right: { src: R.full, label: METHOD[st.r] }, w: it.size[0], h: it.size[1], vw: 1600, vh: 1600, win: it.box, alt: it.prompt }, grp));
      side.appendChild(BA({ left: { src: L.crop, label: METHOD[st.l], ours: true }, right: { src: R.crop, label: METHOD[st.r] }, w: it.box[2], h: it.box[2], crop: true, alt: "native 1:1 crop of the boxed window" }, grp));
      meta.appendChild(metaRow(["FLUX.1-dev", sizeLabel(it.size), "seed 0", METHOD[st.l] + ": " + STEPS[st.l] + " steps", METHOD[st.r] + ": " + STEPS[st.r] + " steps"], false));
      meta.appendChild(el("p", { class: "prompt" }, [el("q", { text: it.prompt })]));
      meta.appendChild(el("p", { class: "small muted", html: "The crop is " + it.box[2] + "&nbsp;&times;&nbsp;" + it.box[2] + " native pixels, shown at its own size or larger; the yellow box marks it on the full view." }));
      var nx = items[(st.i + 1) % items.length];
      [nx.img[st.l], nx.img[st.r]].forEach(function (x) { if (x) { var p = new Image(); p.src = x.full; } });
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
    if (!list.length) {
      var TR = window.SCA_TRACES || {}, byId = {};
      (TR.available || []).forEach(function (v) { if (v.kind === "vs_area") byId[v.sid] = v; });
      var thumbs = {}; ((IM.compare || {}).items || []).forEach(function (it) { thumbs[it.sid] = it; });
      list = Object.keys(byId).map(function (k) { var v = byId[k], it = thumbs[k]; return { sid: k, prompt: v.prompt, mp4: v.mp4, poster: v.poster, still: it ? it.img.sca.full : null, size: [4096, 4096], model: "FLUX.1-dev", seed: 0, label: "Single-pass SCA beside AReA" }; });
      legacy = true;
      var lede = $("#process-lede");
      if (lede && list.length) lede.innerHTML = "Each video replays one sample of <b>single-pass SCA</b> beside AReA: the decoded estimate of the clean image at each of the " + (num("config.steps") || "28") + " steps, the phase, the weights of all " + (num("calib.n_heads_c") || "1,368") + " heads and a native-pixel crop.";
    }
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
      head.appendChild(el("p", { class: "prompt" }, [el("q", { text: x.prompt })]));
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

  /* ---------------------------------------------------------------- results: seconds per image against size */
  (function latencyChart() {
    var host = $("#chart-latency"); if (!host) return;
    var sizes = [2048, 3072, 4096, 6144, 8192], methods = [["full", "Full attention", "var(--chart-light)"], ["area", "AReA", "var(--chart-base)"], ["sca", "Single-pass SCA", "var(--accent)"]];
    function get(m, z) { var v = val("latency." + m + "." + z + "x" + z); return isNaN(v) ? val("latency." + m + "." + z) : v; }
    var series = methods.map(function (m) { return { m: m, pts: sizes.map(function (z) { return [z, get(m[0], z)]; }).filter(function (p) { return !isNaN(p[1]); }) }; }).filter(function (sr) { return sr.pts.length; });
    if (!series.length) { host.textContent = "Numbers not built yet."; return; }
    var W = 520, H = 300, L = 46, R = 110, T = 12, B = 34;
    var ys = [].concat.apply([], series.map(function (sr) { return sr.pts.map(function (p) { return p[1]; }); }));
    var y0 = Math.log10(Math.min.apply(null, ys) * .8), y1 = Math.log10(Math.max.apply(null, ys) * 1.15), x0 = Math.log10(2048 * 2048), x1 = Math.log10(8192 * 8192);
    function X(z) { return L + (Math.log10(z * z) - x0) / (x1 - x0) * (W - L - R); }
    function Y(v) { return T + (1 - (Math.log10(v) - y0) / (y1 - y0)) * (H - T - B); }
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Seconds per image against image size, batch 1, one GH200, single pass" });
    [10, 20, 50, 100, 200, 500, 1000, 2000].forEach(function (v) {
      if (Math.log10(v) < y0 || Math.log10(v) > y1) return;
      s.appendChild(svg("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: "var(--chart-grid)" }));
      s.appendChild(svg("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end" }, v + " s"));
    });
    sizes.forEach(function (z) { s.appendChild(svg("text", { x: X(z), y: H - 12, "text-anchor": "middle" }, z + "²")); });
    series.forEach(function (sr) {
      var d = sr.pts.map(function (p, i) { return (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1); }).join(" ");
      s.appendChild(svg("path", { d: d, fill: "none", stroke: sr.m[2], "stroke-width": sr.m[0] === "sca" ? 2.4 : 1.8 }));
      sr.pts.forEach(function (p) { s.appendChild(svg("circle", { cx: X(p[0]), cy: Y(p[1]), r: 3.2, fill: sr.m[2] })); });
      var last = sr.pts[sr.pts.length - 1];
      s.appendChild(svg("text", { x: X(last[0]) + 7, y: Y(last[1]) + 4, class: sr.m[0] === "sca" ? "acc" : "lbl" }, sr.m[1]));
    });
    host.appendChild(s);
  })();

  /* ---------------------------------------------------------------- gallery: a fixed rule, progressive SCA beside AReA */
  (function gallery() {
    var host = $("#gal"), G = window.SCA_GALLERY; if (!host) return;
    if (!G || !G.items || !G.items.length) { host.appendChild(ph("Gallery in preparation")); return; }
    function open(i) {
      lightbox(function (body) {
        var it = G.items[i], p = it.img.prog, a = it.img.area, grp = { p: 50, members: [] };
        body.appendChild(el("div", { class: "lb-grid" }, [
          BA({ left: { src: p.view, label: "Progressive SCA", ours: true }, right: { src: a.view, label: "AReA" }, w: p.size[0], h: p.size[1], win: p.box, alt: it.prompt }, grp),
          el("div", {}, [BA({ left: { src: p.crop, label: "Progressive SCA", ours: true }, right: { src: a.crop, label: "AReA" }, w: p.box[2], h: p.box[2], crop: true, alt: "central native crop" }, grp),
            metaRow(["Prompt " + it.index + " of " + (num("main.n_prompts") || "195"), "FLUX.1-dev", sizeLabel(p.size), "seed 0"], false),
            el("p", { class: "prompt" }, [el("q", { text: it.prompt })])])]));
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
})();
