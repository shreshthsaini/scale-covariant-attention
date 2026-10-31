"""Inventory the generated-image tree for the HF migration.

Walks <out_root> and reports one row per "set" (<campaign>/<method>/<size> leaf directory that holds regular
files), with file counts and bytes of regular files only (symlink trees are reported separately).
Usage: python inventory_out.py /scratch/09032/saini_2/sca/out > inventory.csv
"""
import csv
import os
import sys


def walk(root):
    rows = {}
    links = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel = os.path.relpath(dirpath, root)
        for name in filenames + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
            path = os.path.join(dirpath, name)
            st = os.lstat(path)
            top = rel.split(os.sep)[0] if rel != "." else name
            if os.path.islink(path):
                links[top] = links.get(top, 0) + 1
                continue
            parts = rel.split(os.sep)
            set_id = "/".join(parts[:3]) if len(parts) >= 3 else rel
            r = rows.setdefault(set_id, {"png": 0, "json": 0, "other": 0, "bytes": 0})
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            r["png" if ext == "png" else "json" if ext == "json" else "other"] += 1
            r["bytes"] += st.st_size
    return rows, links


def main():
    root = sys.argv[1]
    rows, links = walk(root)
    w = csv.writer(sys.stdout)
    w.writerow(["set_id", "n_png", "n_json", "n_other", "bytes"])
    for k in sorted(rows):
        r = rows[k]
        w.writerow([k, r["png"], r["json"], r["other"], r["bytes"]])
    for k in sorted(links):
        print(f"#links,{k},{links[k]}", file=sys.stderr)


if __name__ == "__main__":
    main()
