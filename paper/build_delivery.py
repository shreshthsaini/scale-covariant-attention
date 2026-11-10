#!/usr/bin/env python3
"""Build the paper and a self-contained source ZIP; verify an extracted copy.

Run on a compute node: uv run --no-project python paper/build_delivery.py
Requires tectonic on PATH. Build logs and extracted verification files stay on scratch.

The default builds the full edition (main.tex) into main.pdf, sca_overleaf.zip and delivery_receipt.json.
--main main_short.tex --stem short builds the short review edition into main_short.pdf, sca_overleaf_short.zip and
delivery_receipt_short.json; its ZIP holds only the files that build reads (main.tex included, which main_short.tex
inputs) and names main_short.tex as the main document.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_paper(source, output, root="main.tex"):
    output.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["tectonic", "--keep-logs", "--outdir", str(output), "--makefile-rules",
         str(output / "deps.mk"), root],
        cwd=source, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    (output / "build_stdout.txt").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Paper build failed: {output / 'build_stdout.txt'}\n{result.stdout[-4000:]}")
    log = (output / (Path(root).stem + ".log")).read_text(errors="replace")
    problems = [line for line in log.splitlines() if
                "undefined references" in line or "undefined citations" in line
                or ("LaTeX Warning:" in line and "undefined" in line)]
    if problems:
        raise RuntimeError("Unresolved paper references: " + "\n".join(problems))
    return [line for line in log.splitlines() if "Overfull" in line]


def used_files(output):
    """Source paths (relative to the paper) that a tectonic run read, from --makefile-rules.
    Tectonic prefixes each relative input name with the output directory, so strip it."""
    text = (output / "deps.mk").read_text().replace("\\\n", " ")
    names = text.split(" : ", 1)[1].split() if " : " in text else []
    prefix = str(output) + "/"
    return {n[len(prefix):] if n.startswith(prefix) else n for n in names}


def page_count(output, root="main.tex"):
    """Pages reported by the engine in the build log."""
    log = (output / (Path(root).stem + ".log")).read_text(errors="replace")
    match = re.search(r"Output written on \S+ \((\d+) pages?", log)
    return int(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, default=Path("/scratch/09032/saini_2/sca/paper_builds"))
    parser.add_argument("--main", default="main.tex", help="root file: main.tex (full edition, the default) or "
                        "main_short.tex (short review edition)")
    parser.add_argument("--stem", default="", help="output suffix: empty writes main.pdf, sca_overleaf.zip and "
                        "delivery_receipt.json; 'short' writes <root>.pdf, sca_overleaf_short.zip and "
                        "delivery_receipt_short.json")
    args = parser.parse_args()
    paper = Path(__file__).resolve().parent
    root = args.main
    suffix = f"_{args.stem}" if args.stem else ""
    pdf_name = "main.pdf" if root == "main.tex" else Path(root).stem + ".pdf"
    zip_name, receipt_name = f"sca_overleaf{suffix}.zip", f"delivery_receipt{suffix}.json"
    full_pages = None
    if root != "main.tex":
        # The short edition names the full edition's page count (\fullversionpages in its root file); it must
        # match the delivered full PDF, as recorded by the full build's receipt.
        match = re.search(r"\\newcommand\{\\fullversionpages\}\{(\d+)\}", (paper / root).read_text())
        if match:
            full = json.loads((paper / "delivery_receipt.json").read_text())
            if full.get("paper_sha256") != sha256(paper / "main.pdf"):
                raise RuntimeError("delivery_receipt.json does not describe main.pdf; run the full build first")
            full_pages = full["primary_pages"]
            if int(match.group(1)) != full_pages:
                raise RuntimeError(f"{root} names {match.group(1)} pages for the full version; main.pdf has "
                                   f"{full_pages}")
    args.scratch.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix="delivery_", dir=args.scratch))
    # Ship rendered figures and the sources needed by LaTeX. Figure-generation scripts
    # and provenance remain in the repository; the ZIP is a portable LaTeX build.
    allowed = {".tex", ".bib", ".bst", ".sty", ".pdf", ".png", ".jpg", ".jpeg", ".svg"}
    inputs = []
    for path in sorted(paper.rglob("*")):
        rel = path.relative_to(paper)
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        if path.name.startswith("_test") or "src" in rel.parts or "__pycache__" in rel.parts:
            continue
        if len(rel.parts) == 1 and path.suffix == ".pdf":
            continue
        inputs.append(path)
    manifest = {str(p.relative_to(paper)): sha256(p) for p in inputs}
    warnings = compile_paper(paper, build / "primary", root)
    # Ship only the figure assets and LaTeX files the build actually read: unused renders, previews (.svg),
    # superseded image sets and unused snippets stay in the repository but not in the portable ZIP.
    used = used_files(build / "primary")
    if root not in used:
        raise RuntimeError(f"deps.mk does not list {root}; cannot tell which sources the build read")
    read_only_if_used = {".pdf", ".png", ".jpg", ".jpeg", ".svg", ".tex"}
    inputs = [p for p in inputs if p.suffix.lower() not in read_only_if_used or str(p.relative_to(paper)) in used]
    manifest = {str(p.relative_to(paper)): sha256(p) for p in inputs}
    if any(sha256(p) != manifest[str(p.relative_to(paper))] for p in inputs):
        raise RuntimeError("Paper sources changed during the primary build; rerun after edits finish")
    archive = build / zip_name
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in inputs:
            output.write(path, path.relative_to(paper))
        output.writestr("SOURCE_SHA256.json", json.dumps(manifest, indent=2) + "\n")
        if root != "main.tex":
            output.writestr("README.txt", f"Main document: {root} (tectonic {root}; on Overleaf, Menu > Main document "
                            f"> {root}). It inputs main.tex; main.tex alone builds the full edition, whose other "
                            "sources are not in this archive.\n")
    extracted = build / "extracted"
    with zipfile.ZipFile(archive) as source:
        source.extractall(extracted)
    standalone_warnings = compile_paper(extracted, build / "standalone", root)
    # Publish only after both builds succeed, and reject concurrent source edits.
    if any(sha256(p) != manifest[str(p.relative_to(paper))] for p in inputs):
        raise RuntimeError("Paper sources changed during build; rerun after edits finish")
    if page_count(build / "primary", root) != page_count(build / "standalone", root):
        raise RuntimeError("Standalone build page count differs from the primary build")
    shutil.copy2(build / "primary" / (Path(root).stem + ".pdf"), paper / pdf_name)
    shutil.copy2(archive, paper / zip_name)
    receipt = {
        "paper_sha256": sha256(paper / pdf_name),
        "source_zip_sha256": sha256(paper / zip_name),
        "source_files": len(inputs), "standalone_build_passed": True,
        "primary_pages": page_count(build / "primary", root),
        "standalone_pages": page_count(build / "standalone", root),
        "primary_overfull_warnings": warnings,
        "standalone_overfull_warnings": standalone_warnings,
        "build_directory": str(build),
    }
    if root != "main.tex":
        receipt = {"main": root, "paper": pdf_name, **receipt, "full_version_pages": full_pages}
    (paper / receipt_name).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
