#!/usr/bin/env python3
"""Build the paper and a self-contained source ZIP; verify an extracted copy.

Run on a compute node: uv run --no-project python paper/build_delivery.py
Requires tectonic on PATH. Build logs and extracted verification files stay on scratch.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_paper(source, output):
    output.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["tectonic", "--keep-logs", "--outdir", str(output), "main.tex"],
        cwd=source, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    (output / "build_stdout.txt").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Paper build failed: {output / 'build_stdout.txt'}\n{result.stdout[-4000:]}")
    log = (output / "main.log").read_text(errors="replace")
    problems = [line for line in log.splitlines() if
                "undefined references" in line or "undefined citations" in line
                or ("LaTeX Warning:" in line and "undefined" in line)]
    if problems:
        raise RuntimeError("Unresolved paper references: " + "\n".join(problems))
    return [line for line in log.splitlines() if "Overfull" in line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, default=Path("/scratch/09032/saini_2/sca/paper_builds"))
    args = parser.parse_args()
    paper = Path(__file__).resolve().parent
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
    warnings = compile_paper(paper, build / "primary")
    if any(sha256(p) != manifest[str(p.relative_to(paper))] for p in inputs):
        raise RuntimeError("Paper sources changed during the primary build; rerun after edits finish")
    archive = build / "sca_overleaf.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in inputs:
            output.write(path, path.relative_to(paper))
        output.writestr("SOURCE_SHA256.json", json.dumps(manifest, indent=2) + "\n")
    extracted = build / "extracted"
    with zipfile.ZipFile(archive) as source:
        source.extractall(extracted)
    standalone_warnings = compile_paper(extracted, build / "standalone")
    # Publish only after both builds succeed, and reject concurrent source edits.
    if any(sha256(p) != manifest[str(p.relative_to(paper))] for p in inputs):
        raise RuntimeError("Paper sources changed during build; rerun after edits finish")
    shutil.copy2(build / "primary/main.pdf", paper / "main.pdf")
    shutil.copy2(archive, paper / "sca_overleaf.zip")
    receipt = {
        "paper_sha256": sha256(paper / "main.pdf"),
        "source_zip_sha256": sha256(paper / "sca_overleaf.zip"),
        "source_files": len(inputs), "standalone_build_passed": True,
        "primary_overfull_warnings": warnings,
        "standalone_overfull_warnings": standalone_warnings,
        "build_directory": str(build),
    }
    (paper / "delivery_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
