"""Plan the Hugging Face archives of the SCA migration (2026-10-31).

Writes <staging>/plan.tsv with one row per archive and a file list per archive under <staging>/lists/.
Columns: archive_id, repo, path_in_repo, root, compression, phase, n_png, n_json, n_files, src_bytes, list_file.
Images: one plain tar per <campaign>/<method>/<size> set (members relative to sca/out), sets over 10 GB split into
parts of whole images. Phase "1*" rows are uploaded; phase "2" rows are planned but not uploaded (HF free-tier
private storage cap of 100 GB, see MIGRATION.md). Rebuild the plan with:
  python plan_archives.py /scratch/09032/saini_2/sca/hf_staging
"""
import os
import sys

SCA = "/scratch/09032/saini_2/sca"
OUT = f"{SCA}/out"
SHARD_BYTES = 10e9

HEADLINE = ["dupfix/sca_v2/4096x4096"] + [f"main/{m}/4096x4096" for m in
            ("sca_nb", "area", "full", "dype", "sega_official", "ultraimage")] + [
    f"baseline_expansion_20261017_{c}/{m}/4096x4096" for c, m in
    (("v1", "hrdit_dev_author"), ("rest145", "hrdit_dev_author"), ("dev4", "hrdit_dev_author"),
     ("v1", "sigma_dev_matched"), ("a4k191", "sigma_dev_matched"), ("dev4", "sigma_dev_matched"))]
RERUN = ["dupfix/sca_v2/2048x4096", "dupfix/sca_v2/2048x6144", "dupfix_seed1/sca_v2/4096x4096",
         "dupfix_seed2/sca_v2/4096x4096", "dupfix_prog/sca_v2_prog/4096x4096", "dupfix_krea/sca_v2_krea/4096x4096"]
LATENCY_CAMPAIGNS = ["latency", "latency2", "latency3", "latency_curve", "latency_curve_v2", "latency_hrdit"]
PENDING_EVAL = ["dupfix_hrgen_krea/sca_v2_krea/4096x4096"]


def files_under(root, rel):
    out = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, rel)):
        dirnames.sort()
        for name in sorted(filenames):
            p = os.path.join(dirpath, name)
            out.append((os.path.relpath(p, root), os.lstat(p).st_size, os.path.islink(p)))
    return out


def leaf_sets(root):
    """<campaign>/<method>/<size> directories holding regular files, plus campaign-level extras."""
    sets = {}
    for rel, size, link in files_under(root, "."):
        if link:
            continue
        parts = rel.split(os.sep)
        key = "/".join(parts[:3]) if len(parts) > 3 else "/".join(parts[:-1])
        if parts[1:2] in (["intermediates"], ["wandb"]):
            key = "/".join(parts[:2])
        sets.setdefault(key, []).append((rel, size))
    return sets


def main():
    staging = sys.argv[1]
    os.makedirs(f"{staging}/lists", exist_ok=True)
    rows = []

    def add(aid, repo, path, root, comp, phase, members):
        lst = f"{staging}/lists/{aid}.txt"
        with open(lst, "w") as f:
            f.write("\n".join(m for m, _ in members) + "\n")
        n_png = sum(m.endswith(".png") for m, _ in members)
        n_json = sum(m.endswith(".json") for m, _ in members)
        rows.append([aid, repo, path, root, comp, phase, n_png, n_json, len(members),
                     sum(s for _, s in members), lst])

    sets = leaf_sets(OUT)
    linked = sorted({rel.split(os.sep)[0] for rel, _, link in files_under(OUT, ".") if link})
    lat = {k for k in sets if k.split("/")[0] in LATENCY_CAMPAIGNS}
    for key in sorted(sets):
        members = sets[key]
        phase = ("1a-headline" if key in HEADLINE else "1b-rerun" if key in RERUN else
                 "1c-latency" if key in lat else "1d-pending-eval" if key in PENDING_EVAL else "2")
        base = key.replace("/", "__")
        total = sum(s for _, s in members)
        if total <= SHARD_BYTES:
            add(f"img_{base}", "sca-generated-images", f"images/{base}.tar", OUT, "none", phase, members)
            continue
        stems = {}
        for m, s in members:
            stems.setdefault(m.rsplit(".", 1)[0], []).append((m, s))
        groups = [stems[k] for k in sorted(stems)]
        k = int(-(-total // SHARD_BYTES))
        per = -(-len(groups) // k)
        for i in range(k):
            chunk = [x for g in groups[i * per:(i + 1) * per] for x in g]
            if chunk:
                add(f"img_{base}_part{i + 1}of{k}", "sca-generated-images", f"images/{base}.part{i + 1}of{k}.tar",
                    OUT, "none", phase, chunk)
    link_members = [(rel, 0) for rel, _, l in files_under(OUT, ".") if l and rel.split(os.sep)[0] in linked]
    add("img_linktrees", "sca-generated-images", "images/linktrees.tar", OUT, "none", "1a-headline", link_members)

    ev = f"{SCA}/eval"
    for name in sorted(os.listdir(ev)):
        p = os.path.join(ev, name)
        members = [(rel, s) for rel, s, _ in files_under(ev, name)] if os.path.isdir(p) else [(name, os.lstat(p).st_size)]
        aid = f"eval_{name}" if os.path.isdir(p) else None
        if aid:
            add(aid, "sca-eval-outputs", f"eval/{name}.tar.zst", ev, "zstd", "1a-headline", members)
    top = [(n, os.lstat(os.path.join(ev, n)).st_size) for n in sorted(os.listdir(ev)) if not os.path.isdir(os.path.join(ev, n))]
    add("eval__toplevel", "sca-eval-outputs", "eval/_toplevel_files.tar.zst", ev, "zstd", "1a-headline", top)
    for name in ("logs", "wandb"):
        add(f"run_{name}", "sca-eval-outputs", f"{name}.tar.zst", SCA, "zstd", "1a-headline",
            [(rel, s) for rel, s, _ in files_under(SCA, name)])

    code = f"{SCA}/code"
    for tag in sorted(os.listdir(code)):
        add(f"code_{tag}", "sca-frozen-code", f"code/{tag}.tar.zst", code, "zstd", "1a-headline",
            [(rel, s) for rel, s, _ in files_under(code, tag)])
    add("web_assets", "sca-frozen-code", "web_assets.tar.zst", SCA, "zstd", "1a-headline",
        [(rel, s) for rel, s, _ in files_under(SCA, "web_assets")])
    misc_dirs = ["dispatch", "prepared", "telemetry", "task_patch_history", "continuation", "runtime_repairs_20261017",
                 "integrate_scripts", "finish_agent", "method_redesign_20261017", "figcache", "visual_check",
                 "baseline_candidates12_1077042", "baseline_candidates46_1077042", "paper_numeric_audit_20261017",
                 "metric_base_parity_20261017", "full6k_grid_fix_validation_1076947"]
    misc = [(n, os.lstat(os.path.join(SCA, n)).st_size) for n in sorted(os.listdir(SCA))
            if os.path.isfile(os.path.join(SCA, n)) and not os.path.islink(os.path.join(SCA, n))]
    for d in misc_dirs:
        if os.path.isdir(os.path.join(SCA, d)):
            misc += [(rel, s) for rel, s, _ in files_under(SCA, d)]
    add("misc", "sca-frozen-code", "misc_scratch_files.tar.zst", SCA, "zstd", "1a-headline", misc)

    with open(f"{staging}/plan.tsv", "w") as f:
        f.write("archive_id\trepo\tpath_in_repo\troot\tcompression\tphase\tn_png\tn_json\tn_files\tsrc_bytes\tlist_file\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    by = {}
    for r in rows:
        by.setdefault((r[1], r[5]), [0, 0])
        by[(r[1], r[5])][0] += 1
        by[(r[1], r[5])][1] += r[9]
    for (repo, phase), (n, b) in sorted(by.items()):
        print(f"{repo:22s} {phase:16s} archives={n:4d} {b / 1e9:8.2f} GB")


if __name__ == "__main__":
    main()
