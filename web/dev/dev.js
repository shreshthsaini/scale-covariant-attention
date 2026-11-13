/* Demo pages only: theme toggle (data-theme on <html>, as the page does) and mounting like site.js does. */
(function () {
  "use strict";
  var q = new URLSearchParams(location.search), th = q.get("theme");
  if (th === "dark" || th === "light") document.documentElement.setAttribute("data-theme", th);
  document.addEventListener("DOMContentLoaded", function () {
    var b = document.querySelector(".dev-theme");
    if (b) b.addEventListener("click", function () {
      var dark = document.documentElement.getAttribute("data-theme") === "dark" ||
        (!document.documentElement.hasAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
      document.documentElement.setAttribute("data-theme", dark ? "light" : "dark");
      document.dispatchEvent(new CustomEvent("sca:themechange"));
    });
    Array.prototype.forEach.call(document.querySelectorAll(".comp-mount"), function (m) {
      var name = m.getAttribute("data-comp"), opts = {};
      try { opts = JSON.parse(m.getAttribute("data-options") || "{}"); } catch (e) { opts = {}; }
      Object.keys(Object.fromEntries(q)).forEach(function (k) { if (k.indexOf("o_") === 0) opts[k.slice(2)] = isNaN(+q.get(k)) ? q.get(k) : +q.get(k); });
      try {
        window["sca_" + name] = window.SCAComp[name](m, opts);
        var fig = m.closest("figure"); if (fig) fig.classList.add("is-live");
      } catch (e) { /* the fallback stays */ }
    });
  });
})();
