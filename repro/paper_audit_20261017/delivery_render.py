"""Verify source hashes, independent PDF rendering and changes from a prior delivery."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import pypdfium2 as pdf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(page, scale=1):
    bitmap = page.render(scale=scale)
    image = bitmap.to_pil().copy()
    bitmap.close()
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    paper = root / "paper"
    receipt = json.loads((paper / "delivery_receipt.json").read_text())
    build = Path(receipt["build_directory"])
    standalone = build / "standalone/main.pdf"
    with zipfile.ZipFile(paper / "sca_overleaf.zip") as archive:
        sources = json.loads(archive.read("SOURCE_SHA256.json"))
        for name, expected in sources.items():
            assert sha(paper / name) == expected, f"Stale source: {name}"
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
            assert sha(build / "extracted" / name) == expected, name
    assert sha(paper / "main.pdf") == receipt["paper_sha256"]
    assert sha(paper / "sca_overleaf.zip") == receipt["source_zip_sha256"]
    args.out.mkdir(parents=True, exist_ok=True)
    current, independent, previous = (pdf.PdfDocument(path) for path in
                                     (paper / "main.pdf", standalone, args.previous))
    assert len(current) == len(independent) == len(previous)
    pages = []
    for index in range(len(current)):
        page, other, before = current[index], independent[index], previous[index]
        assert page.get_size() == other.get_size() == before.get_size()
        images = [render(item) for item in (page, other, before)]
        hashes = [hashlib.sha256(image.tobytes()).hexdigest() for image in images]
        assert hashes[0] == hashes[1], f"Independent rendering differs: {index + 1}"
        changed = hashes[0] != hashes[2]
        pages.append({"page": index + 1, "primary_render_sha256": hashes[0],
                      "independent_render_sha256": hashes[1],
                      "previous_render_sha256": hashes[2], "changed": changed})
        if index == 0 or changed:
            render(page, 2).save(args.out / f"current_page{index + 1}_144dpi.png")
            if changed:
                render(before, 2).save(args.out / f"previous_page{index + 1}_144dpi.png")
        for image in images:
            image.close()
        for item in (page, other, before):
            item.close()
    warnings = {}
    for kind in ("primary", "standalone"):
        lines = (build / kind / "main.log").read_text(errors="replace").splitlines()
        warnings[kind] = [line for line in lines if "Warning" in line or "Overfull" in line]
    report = {"status": "passed", "source_files_verified": len(sources),
              "pages": len(current), "independent_rendering_exact": True,
              "render_dpi": 72, "review_dpi": 144,
              "changed_pages": [row["page"] for row in pages if row["changed"]],
              "render_receipts": pages, "warnings": warnings,
              "artifacts": {str(path.relative_to(root)): sha(path) for path in
                            (paper / "main.pdf", paper / "sca_overleaf.zip",
                             paper / "delivery_receipt.json", paper / "main.tex")},
              "previous_pdf": str(args.previous), "previous_pdf_sha256": sha(args.previous),
              "visual_review": "pending human-agent inspection of the saved 144dpi pages"}
    (args.out / "render_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                     ("status", "source_files_verified", "pages", "changed_pages", "warnings")}, indent=2))


if __name__ == "__main__":
    main()
