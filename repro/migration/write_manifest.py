"""Merge the archive plan, the build sums and the upload status into manifests.

Usage: python write_manifest.py <staging> <out_csv> [--uploaded repo1,repo2,...]
Writes <out_csv> (all rows, every phase) and <staging>/<repo>/MANIFEST.csv for each staged repo. Rows of repos named
in --uploaded whose archive exists get status uploaded_verified (run verify_hf.py first); staged rows of other repos get
staged_not_uploaded; planned rows without an archive get planned_not_uploaded. The calibration repo is listed file by
file from <staging>/sca-calibration.
"""
import argparse
import csv
import hashlib
import os

REPO_TYPE = {"sca-calibration": "model", "sca-eval-outputs": "dataset", "sca-generated-images": "dataset",
             "sca-frozen-code": "dataset"}
COLS = ["hf_repo", "repo_type", "path_in_repo", "phase", "status", "source", "n_png", "n_json", "n_files",
        "src_bytes", "archive_bytes", "sha256"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("staging")
    ap.add_argument("out_csv")
    ap.add_argument("--uploaded", default="")
    a = ap.parse_args()
    uploaded = set(filter(None, a.uploaded.split(",")))
    sums = {}
    for name in os.listdir(f"{a.staging}/sums"):
        aid, path, sha, size = open(f"{a.staging}/sums/{name}").read().rstrip("\n").split("\t")
        sums[aid] = (sha, int(size))
    rows = []
    calib = f"{a.staging}/sca-calibration"
    for dirpath, dirnames, files in os.walk(calib):
        dirnames[:] = sorted(d for d in dirnames if d != ".cache")
        for name in sorted(files):
            p = os.path.join(dirpath, name)
            rel = os.path.relpath(p, calib)
            sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
            src = "/scratch/09032/saini_2/sca/" + rel if rel.startswith("calib/") else "(model card)"
            rows.append(["shreshthsaini/sca-calibration", "model", rel, "1a-headline",
                         "uploaded_verified" if "sca-calibration" in uploaded else "staged_not_uploaded",
                         src, 0, 0, 1, os.path.getsize(p), os.path.getsize(p), sha])
    with open(f"{a.staging}/plan.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            first = open(r["list_file"]).readline().strip()
            source = os.path.join(r["root"], os.path.dirname(first)) if r["n_files"] != "0" else r["root"]
            if r["archive_id"].startswith(("eval_", "code_")):
                source = os.path.join(r["root"], first.split("/")[0])
            elif r["archive_id"] in ("img_linktrees", "misc", "eval__toplevel"):
                source = r["root"] + " (" + r["archive_id"] + ")"
            sha, size = sums.get(r["archive_id"], ("", ""))
            if sha:
                status = "uploaded_verified" if r["repo"] in uploaded else "staged_not_uploaded"
            else:
                status = "planned_not_uploaded"
            rows.append(["shreshthsaini/" + r["repo"], REPO_TYPE[r["repo"]], r["path_in_repo"], r["phase"], status,
                         source, r["n_png"], r["n_json"], r["n_files"], r["src_bytes"], size, sha])
    with open(a.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows)
    for repo in REPO_TYPE:
        d = f"{a.staging}/{repo}"
        if repo != "sca-calibration" and os.path.isdir(d):
            with open(f"{d}/MANIFEST.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(COLS)
                w.writerows(x for x in rows if x[0].endswith("/" + repo))
    done = [x for x in rows if x[4] == "uploaded_verified"]
    print(f"rows={len(rows)} uploaded_verified={len(done)} "
          f"uploaded_GB={sum(int(x[10] or 0) for x in done) / 1e9:.2f} "
          f"not_uploaded_src_GB={sum(int(x[9]) for x in rows if x[4] != 'uploaded_verified') / 1e9:.2f}")


if __name__ == "__main__":
    main()
