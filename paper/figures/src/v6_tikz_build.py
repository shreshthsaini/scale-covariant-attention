"""Compile the v6 TikZ figures (method candidates, method zoom) in a scratch build directory and install the kept ones.

Sources (paper/figures/src): v6_method_common.tex, v6_method_a.tex, v6_method_b.tex, v6_method_zoom.tex,
sca_colors.tex; assets: paper/figures/v6/method_assets (v6_method_assets.py). The build directory holds symlinks to
them, so no auxiliary file lands in the repository. Kept outputs:
  paper/figures/v6/method.pdf        <- KEEP["method"]
  paper/figures/v6/method_zoom.pdf   <- v6_method_zoom.tex (when it exists)
Candidates stay in the build directory (BUILD/out/*.pdf).

  /work/09032/saini_2/vista/library/uv-envs/sca/bin/python -I v6_tikz_build.py [--build DIR] [names ...]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent
V6 = FIGS / "v6"
TECTONIC = shutil.which("tectonic") or str(Path.home() / ".local" / "bin" / "tectonic")
DEFAULT_BUILD = Path("/scratch/09032/saini_2/sca/paper_builds/v6_methodfigs_tikz")
SOURCES = ("sca_colors.tex", "v6_method_common.tex", "v6_method_a.tex", "v6_method_b.tex", "v6_method_zoom.tex")
KEEP = {"method": "v6_method_b", "method_zoom": "v6_method_zoom"}


def setup(build: Path) -> Path:
    src = build / "src"
    (build / "out").mkdir(parents=True, exist_ok=True)
    src.mkdir(parents=True, exist_ok=True)
    links = {name: HERE / name for name in SOURCES if (HERE / name).exists()}
    links["method_assets"] = V6 / "method_assets"
    links["method_zoom"] = FIGS / "method_zoom"        # fig_method_zoom.py rasters and macros (read only)
    for name, target in links.items():
        dst = src / name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(target)
    return src


def compile_tex(src: Path, build: Path, stem: str) -> Path:
    env = dict(os.environ, SOURCE_DATE_EPOCH="0")
    res = subprocess.run([TECTONIC, "--outdir", str(build / "out"), f"{stem}.tex"], cwd=src, env=env,
                         capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-3000:], res.stderr[-3000:])
        raise SystemExit(f"tectonic failed on {stem}")
    return build / "out" / f"{stem}.pdf"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    ap.add_argument("--install", action="store_true", help="copy the kept outputs into paper/figures/v6")
    ap.add_argument("names", nargs="*", default=["v6_method_a", "v6_method_b", "v6_method_zoom"])
    a = ap.parse_args()
    src = setup(a.build)
    built = {}
    for stem in a.names:
        if (HERE / f"{stem}.tex").exists():
            built[stem] = compile_tex(src, a.build, stem)
            print("built", built[stem])
    if a.install:
        V6.mkdir(parents=True, exist_ok=True)
        for out, stem in KEEP.items():
            if stem in built:
                shutil.copyfile(built[stem], V6 / f"{out}.pdf")
                print("installed", V6 / f"{out}.pdf", "from", stem)


if __name__ == "__main__":
    main()
