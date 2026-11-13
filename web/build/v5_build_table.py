"""Data of the results table (SCAComp.table, web/assets/js/c_table.js): the paper's Table 1 (main comparison at
4096^2) and Table 3 (generality), read from the paper's own generated snippets, so the page shows exactly the
printed cells (values, bold = best and underline = second per group, the campaign-time mark, significance colour,
the Holm record), plus numeric values for sorting.

Sources: paper/figures/snippets/main_tables/tab_main_compact.tex and tab_generality.tex (written by
paper/figures/src/gen_main_tables.py from results/), the sample-size macros of paper/main.tex (\\nmain, \\nseed,
\\nhrgen, \\nkrea). Cross-checks against results/ (the build stops on a mismatch): every time cell against
fig_speed.load() (clean batch-1 medians; campaign means where marked), and every ImageReward cell against the mean of
the per-image ImageReward of the 195 prompts at seed 0 (v6_common.pool195, the evaluation root the paper reads).

Run (repository root, CPU, seconds):
  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python web/build/v5_build_table.py
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.dirname(HERE)
REPO = os.path.dirname(WEB)
sys.path.insert(0, os.path.join(REPO, "paper", "figures", "src"))

import fig_speed  # noqa: E402
import v6_common as vc  # noqa: E402

SNIP = os.path.join(REPO, "paper", "figures", "snippets", "main_tables")
OUT = os.path.join(WEB, "assets", "data", "table")
MACROS = ("nmain", "nseed", "nhrgen", "nkrea")
# Table 1 columns: key, label, column group, direction (1 higher is better, -1 lower), unit note
MAIN_COLS = [("time", "s", "time", -1), ("vs", "vs. ours", "time", -1), ("ir", "IR", "pref", 1),
             ("hps", "HPSv2.1", "pref", 1), ("pick", "Pick", "pref", 1), ("musiqc", "M_c", "detail", 1),
             ("fid", "FID", "set", -1), ("fidp", "FID_p\u2020", "set", -1), ("holm", "+\u2009/\u2009\u2212", "holm", 0)]
MAIN_GROUPS = [("time", "Time per image"), ("pref", "Preference \u2191"), ("detail", "Detail \u2191"),
               ("set", "Set level \u2193"), ("holm", "Ten metrics")]
GEN_COLS = [("n", "n", "n", 0), ("ir", "IR", "diff", 1), ("hps", "HPS", "diff", 1), ("pick", "Pick", "diff", 1),
            ("holm", "+\u2009/\u2009\u2212", "holm", 0)]
GEN_GROUPS = [("n", ""), ("diff", "Paired difference \u2191"), ("holm", "Holm")]
METHOD_ID = {"Full attention": "full", "DyPE": "dype", "SEGA": "sega_official", "AReA": "area",
             "Single-pass SCA (ours)": "sca_v2", "UltraImage": "ultraimage", "HRDiT": "hrdit_dev_author",
             "Progressive SCA (ours)": "sca_v2_prog"}


def macros():
    text = open(os.path.join(REPO, "paper", "main.tex")).read()
    out = {}
    for m in MACROS:
        hit = re.search(r"\\newcommand\{\\%s\}\{([^}]*)\}" % m, text)
        assert hit, m
        out[m] = hit.group(1)
    return out


def plain(s, mac):
    """LaTeX cell text -> plain text (Unicode), keeping only what a reader sees."""
    s = s.replace("\\sca{}", "SCA").replace("\\,", "\u2009").replace("$\\times$", "\u00d7")
    for m, v in mac.items():
        s = s.replace("$\\%s$" % m, v).replace("\\%s{}" % m, v).replace("\\%s" % m, v)
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = s.replace("$-$", "\u2212").replace("$+$", "+")
    s = re.sub(r"\$([^$]*)\$", lambda m: m.group(1).replace("-", "\u2212"), s)
    s = s.replace("\\camp", "").replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def cell(raw, mac):
    raw = raw.strip()
    c = {}
    if "\\camp" in raw:
        c["camp"] = True
    m = re.search(r"\\(up|dn)\{", raw)
    if m:
        c["sig"] = m.group(1)
        raw = re.sub(r"\\(?:up|dn)\{(.*)\}$", r"\1", raw)
    if raw.startswith("\\textbf{"):
        c["b"] = True
        raw = raw[len("\\textbf{"):-1]
    elif raw.startswith("\\underline{"):
        c["u"] = True
        raw = raw[len("\\underline{"):-1]
    t = plain(raw, mac)
    c["t"] = t
    num = re.match(r"^([\u2212+]?)(\d+(?:\.\d+)?)", t)
    if num and "/" not in t:
        c["v"] = float(num.group(2)) * (-1 if num.group(1) == "\u2212" else 1)
    rec = re.match(r"^(\d+)\s*/\s*(\d+)$", t)
    if rec:
        c["plus"], c["minus"] = int(rec.group(1)), int(rec.group(2))
    return c


def body(path):
    text = open(path).read()
    rows = text[text.index("\\midrule"):text.index("\\bottomrule")]
    return [ln.strip() for ln in rows.splitlines() if ln.strip() and not ln.strip().startswith("\\midrule")]


def parse(path, cols, mac, name_bold_ours=True):
    groups, rows, cur = [], [], None
    for ln in body(path):
        ln = ln.rstrip("\\").strip()
        g = re.match(r"\\multicolumn\{\d+\}\{l\}\{(.*)\}$", ln)
        if g:
            cur = plain(g.group(1), mac)
            groups.append(cur)
            continue
        ours = ln.startswith("\\scaRowOurs")
        ln = ln.replace("\\scaRowOurs", "").strip()
        parts = [p for p in ln.split("&")]
        assert len(parts) == len(cols) + 1, (path, ln)
        name = parts[0].strip()
        nb = name.startswith("\\textbf{")
        if nb:
            name = name[len("\\textbf{"):-1]
        r = dict(name=plain(name, mac), group=cur, ours=ours, cells={})
        if nb:
            r["name_b"] = True
        for (k, *_), raw in zip(cols, parts[1:]):
            r["cells"][k] = cell(raw, mac)
        rows.append(r)
    return groups, rows


def check_main(rows):
    d = fig_speed.load()
    pool = vc.pool195()
    for r in rows:
        m = METHOD_ID[r["name"]]
        t = r["cells"]["time"]
        src = d["camp"] if t.get("camp") else d["clean"]
        assert fig_speed.r1(src[m]) == t["v"], (m, t, src[m])
        ir = sum(e["ir"] for e in pool[m].values()) / len(pool[m])
        assert abs(round(ir, 2) - r["cells"]["ir"]["v"]) < 1e-9, (m, ir, r["cells"]["ir"])
        r["id"] = m


def main():
    os.makedirs(OUT, exist_ok=True)
    mac = macros()
    mg, mrows = parse(os.path.join(SNIP, "tab_main_compact.tex"), MAIN_COLS, mac)
    check_main(mrows)
    gg, grows = parse(os.path.join(SNIP, "tab_generality.tex"), GEN_COLS, mac)
    rec = dict(
        main=dict(title="Main comparison at 4096\u00b2", groups=mg, colgroups=[dict(k=k, label=l) for k, l in MAIN_GROUPS],
                  cols=[dict(k=k, label=l, g=g, dir=d) for k, l, g, d in MAIN_COLS], rows=mrows,
                  note=("FLUX.1-dev, " + mac["nmain"] + " Aesthetic-4K prompts, seed 0; HPSv2.1 \u00d7100; M_c: MUSIQ on 1024\u00b2 "
                        "crops; FID_p: patch FID (our definition). Time: clean batch-1 median on one GH200 (\u2021 campaign mean) "
                        "and its ratio to progressive SCA's of the same kind. +\u2009/\u2009\u2212: of the ten per-image metrics, those "
                        "where progressive SCA is significantly better / worse than the row after Holm's correction. Bold: best, "
                        "underline: second per group.")),
        gen=dict(title="Generality", groups=gg, colgroups=[dict(k=k, label=l) for k, l in GEN_GROUPS],
                 cols=[dict(k=k, label=l, g=g, dir=d) for k, l, g, d in GEN_COLS], rows=grows,
                 note=("Paired differences at 4096\u00b2, first-named method minus the second (HPS: HPSv2.1 \u00d7100). "
                       "Marked: the 95% paired interval excludes zero in the row's favour. Holm +\u2009/\u2009\u2212: per-image "
                       "metrics, of ten, on which the first-named method is significantly better / worse after Holm's correction.")),
        source=dict(script="web/build/v5_build_table.py",
                    tables=["paper/figures/snippets/main_tables/tab_main_compact.tex",
                            "paper/figures/snippets/main_tables/tab_generality.tex"],
                    generator="paper/figures/src/gen_main_tables.py", macros="paper/main.tex"))
    with open(os.path.join(OUT, "data.js"), "w") as f:
        f.write("/* Generated by web/build/v5_build_table.py; do not edit. */\n")
        f.write("window.SCA_DATA = window.SCA_DATA || {};\n")
        f.write("window.SCA_DATA.table = " + json.dumps(rec, separators=(",", ":"), ensure_ascii=True) + ";\n")
    for r in mrows:
        print(r["name"], r["group"][:20], {k: c["t"] + ("*" if c.get("b") else "") + ("_" if c.get("u") else "") + ("\u2021" if c.get("camp") else "") for k, c in r["cells"].items()})
    for r in grows:
        print(r["name"], r["group"][:30], {k: c["t"] + ("^" if c.get("sig") else "") for k, c in r["cells"].items()})
    print("data", os.path.getsize(os.path.join(OUT, "data.js")), "bytes")


if __name__ == "__main__":
    main()
