"""Incremental Hugging Face sync of the SCA mirrors, run by sync_incremental.sh (MIGRATION.md Section 7).

Uploads to the four private repos only what is new or changed since the last sync and records every upload in
results/migration_manifest.csv (columns of write_manifest.py; a row is updated in place or appended):
  sca-eval-outputs      eval/<group>.tar.zst for each group under sca/eval with a DONE marker that is not on the Hub or
                        whose file count or byte total differs from its manifest row (members relative to eval/, as in
                        the migration); eval/_toplevel_files.tar.zst when the loose top-level files change.
  sca-calibration       every regular file under sca/calib that is missing remotely or differs by SHA256, and
                        calib/PART_LINKS.txt when the symlinks change.
  sca-frozen-code       code/<tag>.tar.zst for new frozen code copies; web_assets/trace/<method>/<prompt>.tar.zst for
                        trace renders that are new or changed since web_assets.tar.zst (members relative to sca/).
  sca-generated-images  image sets that plan_archives.py puts in phase 1 and that have no archive on the Hub yet
                        (headline, then reruns, latency, pending-eval); images/linktrees.tar when the symlink trees
                        change.
An upload happens only if the account's private storage stays at or under --cap-gb. A set that does not fit is skipped,
logged, and listed in the manifest as planned_not_uploaded. Image sets and frozen code copies that change after their
upload are reported, never re-uploaded. Nothing is deleted locally or on the Hub: new archives fill empty slots of the
migration staging tree, revised ones go to <staging>/sync/rev/<run>/. Inputs modified in the last 15 minutes, eval
groups without DONE and image sets named by a pending or running task wait for a later run.

Usage: python sync_hf.py <git checkout> [--cap-gb 97] [--dry-run]
Logs go to stderr and <staging>/sync/runs/<run>.log; stdout gets one summary line for .agent/notes/sync_log.md.
Exit code 0 on success (including a no-op), 1 if an upload failed, 2 if the Hub could not be read.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
SCA = "/scratch/09032/saini_2/sca"
OWNER = "shreshthsaini"
REPO_TYPE = {"sca-calibration": "model", "sca-eval-outputs": "dataset", "sca-frozen-code": "dataset",
             "sca-generated-images": "dataset"}
STAGING = f"{SCA}/hf_staging"
SYNC = f"{STAGING}/sync"
COLS = ["hf_repo", "repo_type", "path_in_repo", "phase", "status", "source", "n_png", "n_json", "n_files",
        "src_bytes", "archive_bytes", "sha256"]
OK, PLANNED = "uploaded_verified", "planned_not_uploaded"
QUIET_S = 15 * 60
WEB_BASE_LIST = f"{STAGING}/lists/web_assets.txt"
WEB_BASE_TIME = 1793474839  # 2026-10-31 14:27:19 CDT: the migration listed web_assets/ for web_assets.tar.zst
PHASE_RANK = {"1a-headline": 0, "1b-rerun": 1, "1c-latency": 2, "1d-pending-eval": 3}
REUPLOAD_KINDS = {"calib", "eval", "trace", "linktrees"}  # a changed source of these kinds is uploaded again


def set_base(path):
    """images/<base>[.partKofN].tar -> <base>."""
    return re.sub(r"\.part\d+of\d+$", "", path[len("images/"):-len(".tar")])


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def git_sha1(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def read_list(path):
    with open(path) as f:
        return [x for x in f.read().split("\n") if x]


def newest_mtime(root, members):
    newest = 0.0
    for rel in members:
        try:
            newest = max(newest, os.lstat(os.path.join(root, rel)).st_mtime)
        except FileNotFoundError:
            return float("inf")
    return newest


def plan_source(r, members):
    """The source column exactly as write_manifest.py derives it."""
    first = members[0] if members else ""
    source = os.path.join(r["root"], os.path.dirname(first)) if r["n_files"] != "0" else r["root"]
    if r["archive_id"].startswith(("eval_", "code_")):
        source = os.path.join(r["root"], first.split("/")[0])
    elif r["archive_id"] in ("img_linktrees", "misc", "eval__toplevel"):
        source = r["root"] + " (" + r["archive_id"] + ")"
    return source


@dataclass
class Item:
    kind: str     # calib, eval, code, trace, linktrees, image
    repo: str
    path: str     # path in the Hub repo
    phase: str
    source: str
    n_png: int
    n_json: int
    n_files: int
    src_bytes: int
    action: str   # new, changed, adopt (on the Hub, no manifest row), record (on the Hub and identical)
    aid: str = ""
    root: str = ""
    members: list = field(default_factory=list)
    comp: str = "none"
    local: str = ""
    sha: str = ""
    size: int = 0
    group: str = ""

    @property
    def est(self):
        return self.size if self.local else self.src_bytes + 1536 * self.n_files + 10240

    def row(self, status=OK):
        done = status == OK
        return [f"{OWNER}/{self.repo}", REPO_TYPE[self.repo], self.path, self.phase, status, self.source,
                str(self.n_png), str(self.n_json), str(self.n_files), str(self.src_bytes),
                str(self.size) if done else "", self.sha if done else ""]


def item_from_plan(r, kind, action):
    members = read_list(r["list_file"])
    return Item(kind=kind, repo=r["repo"], path=r["path_in_repo"], phase=r["phase"], source=plan_source(r, members),
                n_png=int(r["n_png"]), n_json=int(r["n_json"]), n_files=int(r["n_files"]),
                src_bytes=int(r["src_bytes"]), action=action, aid=r["archive_id"], root=r["root"], members=members,
                comp=r["compression"])


class Sync:
    def __init__(self, a):
        self.dry = a.dry_run
        self.cap = int(a.cap_gb * 1e9)
        self.ts = time.strftime("%Y%m%d_%H%M%S")
        self.now = time.time()
        os.makedirs(f"{SYNC}/runs", exist_ok=True)
        self.logf = open(f"{SYNC}/runs/{self.ts}{'_dry' if self.dry else ''}.log", "a")
        self.manifest = os.path.join(a.repo, "results", "migration_manifest.csv")
        with open(self.manifest, newline="") as f:
            data = list(csv.reader(f))
        if data[0] != COLS:
            raise SystemExit(f"unexpected manifest header in {self.manifest}")
        self.rows = data[1:]
        self.reindex()
        self.dirty = False
        self.uploaded, self.recorded, self.failed, self.would = [], [], [], []
        self.notes = {"skipped for the cap": [], "changed but not re-uploaded": [], "deferred": [],
                      "on the Hub without a manifest row": [], "eval groups changed without DONE": []}

    def log(self, msg):
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, file=sys.stderr, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    def note(self, kind, what):
        self.notes[kind].append(what)
        self.log(f"{kind}: {what}")

    def reindex(self):
        self.idx = {(r[0], r[2]): i for i, r in enumerate(self.rows)}

    def row_of(self, repo, path):
        i = self.idx.get((f"{OWNER}/{repo}", path))
        return None if i is None else self.rows[i]

    def unchanged(self, repo, path, n_files, src_bytes):
        r = self.row_of(repo, path)
        return r is not None and r[4] == OK and int(r[8]) == n_files and int(r[9]) == src_bytes

    def put_rows(self, new_rows):
        """Upsert rows; planned rows of the same image sets under other paths (an older split) are replaced."""
        img = f"{OWNER}/sca-generated-images"
        keys = {(r[0], r[2]) for r in new_rows}
        bases = {set_base(r[2]) for r in new_rows if r[0] == img and r[2].startswith("images/")}
        if bases:
            keep = [r for r in self.rows if not (r[0] == img and r[4] == PLANNED and (r[0], r[2]) not in keys
                                                 and r[2].startswith("images/") and set_base(r[2]) in bases)]
            if len(keep) != len(self.rows):
                self.rows, self.dirty = keep, True
                self.reindex()
        for row in new_rows:
            i = self.idx.get((row[0], row[2]))
            if i is None:
                self.idx[(row[0], row[2])] = len(self.rows)
                self.rows.append(row)
                self.dirty = True
            elif self.rows[i] != row:
                self.rows[i] = row
                self.dirty = True

    # ---------------------------------------------------------------- Hub state
    def read_hub(self):
        from huggingface_hub import HfApi
        self.api = HfApi()
        self.remote = {}
        for repo, rtype in REPO_TYPE.items():
            info = self.api.repo_info(f"{OWNER}/{repo}", repo_type=rtype)
            if info.private is not True:
                raise RuntimeError(f"{OWNER}/{repo} is not private; refusing to upload")
            files = {}
            for e in self.api.list_repo_tree(f"{OWNER}/{repo}", repo_type=rtype, recursive=True):
                if getattr(e, "size", None) is not None and hasattr(e, "blob_id"):
                    files[e.path] = (e.size, e.lfs.sha256 if getattr(e, "lfs", None) else None, e.blob_id)
            self.remote[repo] = files
        total = 0
        for kind, lister in (("model", self.api.list_models), ("dataset", self.api.list_datasets),
                             ("space", self.api.list_spaces)):
            for r in lister(author=OWNER, expand=["private"]):
                if not r.private:
                    continue
                try:
                    used = self.api.repo_info(r.id, repo_type=kind, expand=["usedStorage"]).used_storage
                except Exception:
                    used = None
                if used is None:
                    used = sum(getattr(e, "size", 0) or 0 for e in
                               self.api.list_repo_tree(r.id, repo_type=kind, recursive=True))
                total += used
        self.usage = self.usage0 = total
        self.log(f"Hub read: private storage {total / 1e9:.2f} GB, cap {self.cap / 1e9:.0f} GB; remote files "
                 + ", ".join(f"{k}={len(v)}" for k, v in self.remote.items()))

    def same_as_remote(self, repo, path, local, sha):
        e = self.remote[repo].get(path)
        if e is None or e[0] != os.path.getsize(local):
            return False
        return e[1] == sha if e[1] else e[2] == git_sha1(local)

    # ---------------------------------------------------------------- candidates
    def fresh_plan(self):
        d = f"{SYNC}/plan"
        os.makedirs(d, exist_ok=True)
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "plan_archives.py"), d],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"plan_archives.py failed: {r.stderr.strip()[-400:]}")
        with open(f"{d}/plan.tsv") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    def calib_items(self):
        root, repo, items, links = f"{SCA}/calib", "sca-calibration", [], []
        for dirpath, dirnames, files in os.walk(root):
            dirnames.sort()
            for name in sorted(files) + dirnames:
                p = os.path.join(dirpath, name)
                if os.path.islink(p):
                    links.append(f"./{os.path.relpath(p, root)} -> {os.readlink(p)}")
                    continue
                if name in dirnames:
                    continue
                rel = "calib/" + os.path.relpath(p, root)
                st = os.lstat(p)
                if self.now - st.st_mtime < QUIET_S:
                    self.note("deferred", rel)
                    continue
                sha = sha256_file(p)
                it = Item(kind="calib", repo=repo, path=rel, phase="1a-headline", source=f"{SCA}/{rel}", n_png=0,
                          n_json=0, n_files=1, src_bytes=st.st_size, action="new", local=p, sha=sha,
                          size=st.st_size)
                if self.same_as_remote(repo, rel, p, sha):
                    r = self.row_of(repo, rel)
                    if r is None or r[4] != OK or r[11] != sha:
                        it.action = "record"
                        items.append(it)
                    continue
                it.action = "new" if rel not in self.remote[repo] else "changed"
                items.append(it)
        os.makedirs(f"{SYNC}/calib", exist_ok=True)
        part_links = f"{SYNC}/calib/PART_LINKS.txt"
        content = "\n".join(sorted(links)) + "\n"
        if not os.path.exists(part_links) or open(part_links).read() != content:
            with open(part_links, "w") as f:
                f.write(content)
        rel = "calib/PART_LINKS.txt"
        sha = sha256_file(part_links)
        if not self.same_as_remote(repo, rel, part_links, sha):
            items.append(Item(kind="calib", repo=repo, path=rel, phase="1a-headline", source=f"{SCA}/{rel}", n_png=0,
                              n_json=0, n_files=1, src_bytes=len(content.encode()),
                              action="new" if rel not in self.remote[repo] else "changed", local=part_links, sha=sha,
                              size=os.path.getsize(part_links)))
        return items

    def eval_items(self, plan):
        repo, items = "sca-eval-outputs", []
        for r in plan:
            if r["repo"] != repo or not r["archive_id"].startswith("eval_"):
                continue
            path, n_files, src_bytes = r["path_in_repo"], int(r["n_files"]), int(r["src_bytes"])
            on_hub = path in self.remote[repo]
            if on_hub and self.unchanged(repo, path, n_files, src_bytes):
                continue
            if r["archive_id"] != "eval__toplevel":
                name = r["archive_id"][len("eval_"):]
                if not os.path.exists(f"{SCA}/eval/{name}/DONE"):
                    if on_hub:
                        self.note("eval groups changed without DONE", name)
                    continue
            it = item_from_plan(r, "eval", "new" if not on_hub else "changed" if self.row_of(repo, path) else "adopt")
            if newest_mtime(it.root, it.members) > self.now - QUIET_S:
                self.note("deferred", path)
                continue
            items.append(it)
        return items

    def code_items(self, plan):
        repo, items = "sca-frozen-code", []
        for r in plan:
            if r["repo"] != repo or not r["archive_id"].startswith("code_"):
                continue
            path = r["path_in_repo"]
            if path in self.remote[repo]:
                if self.unchanged(repo, path, int(r["n_files"]), int(r["src_bytes"])):
                    continue
                if self.row_of(repo, path):
                    self.note("changed but not re-uploaded", path)
                    continue
                it = item_from_plan(r, "code", "adopt")
            else:
                it = item_from_plan(r, "code", "new")
            if newest_mtime(it.root, it.members) > self.now - QUIET_S:
                self.note("deferred", path)
                continue
            items.append(it)
        return items

    def trace_items(self):
        repo, root, items = "sca-frozen-code", f"{SCA}/web_assets/trace", []
        base_members = set(read_list(WEB_BASE_LIST)) if os.path.exists(WEB_BASE_LIST) else None
        base_time = os.path.getmtime(WEB_BASE_LIST) if base_members is not None else WEB_BASE_TIME
        units = {}
        for dirpath, dirnames, files in os.walk(root):
            dirnames.sort()
            for name in sorted(files):
                p = os.path.join(dirpath, name)
                parts = os.path.relpath(p, root).split(os.sep)
                unit = "/".join(parts[:2]) if len(parts) >= 3 else (parts[0] + "/_files" if len(parts) == 2
                                                                     else "_files")
                st = os.lstat(p)
                units.setdefault(unit, []).append((os.path.relpath(p, SCA), st.st_size, st.st_mtime))
        for unit, mem in sorted(units.items()):
            path = f"web_assets/trace/{unit}.tar.zst"
            n_files, src_bytes = len(mem), sum(s for _, s, _ in mem)
            on_hub, row = path in self.remote[repo], self.row_of(repo, path)
            if on_hub and self.unchanged(repo, path, n_files, src_bytes):
                continue
            if not on_hub and row is None and all((base_members is None or rel in base_members) and mt <= base_time
                                                  for rel, _, mt in mem):
                continue  # already inside web_assets.tar.zst
            if max(mt for _, _, mt in mem) > self.now - QUIET_S:
                self.note("deferred", path)
                continue
            src = os.path.join(root, unit[:-len("/_files")] if unit.endswith("/_files") else
                               "" if unit == "_files" else unit)
            items.append(Item(kind="trace", repo=repo, path=path, phase="1a-headline", source=src.rstrip("/"),
                              n_png=sum(m.endswith(".png") for m, _, _ in mem),
                              n_json=sum(m.endswith(".json") for m, _, _ in mem), n_files=n_files,
                              src_bytes=src_bytes, action="new" if not on_hub else "changed" if row else "adopt",
                              aid="trace_" + unit.replace("/", "__"), root=SCA, members=[m for m, _, _ in mem],
                              comp="zstd"))
        return items

    def image_items(self, plan):
        repo = "sca-generated-images"
        remote_bases = {set_base(p) for p in self.remote[repo] if p.startswith("images/") and p.endswith(".tar")}
        busy = ""
        for d in ("pending", "running"):
            q = f"{SCA}/taskq/{d}"
            for name in sorted(os.listdir(q)) if os.path.isdir(q) else []:
                try:
                    with open(os.path.join(q, name), errors="replace") as f:
                        busy += f.read() + "\n"
                except (IsADirectoryError, FileNotFoundError):
                    pass
        sets, link_items = {}, []
        for r in plan:
            if r["repo"] != repo:
                continue
            if r["archive_id"] == "img_linktrees":
                path = r["path_in_repo"]
                if path in self.remote[repo] and self.unchanged(repo, path, int(r["n_files"]), int(r["src_bytes"])):
                    continue
                link_items.append(item_from_plan(r, "linktrees", "changed" if self.row_of(repo, path) else
                                                 "new" if path not in self.remote[repo] else "adopt"))
                continue
            sets.setdefault(set_base(r["path_in_repo"]), []).append(r)
        new_sets = []
        for base, rs in sorted(sets.items()):
            phase = rs[0]["phase"]
            if not phase.startswith("1"):
                continue
            n_files, src_bytes = sum(int(r["n_files"]) for r in rs), sum(int(r["src_bytes"]) for r in rs)
            if base in remote_bases:
                up = [r for r in self.rows if r[0] == f"{OWNER}/{repo}" and r[4] == OK and r[2].startswith("images/")
                      and set_base(r[2]) == base]
                if not up:
                    parts = self.adopt_from_build_records(rs)
                    if parts:
                        new_sets.append((-1, base, parts))
                    else:
                        self.note("on the Hub without a manifest row", base)
                elif (sum(int(r[8]) for r in up), sum(int(r[9]) for r in up)) != (n_files, src_bytes):
                    self.note("changed but not re-uploaded", f"{base} ({sum(int(r[8]) for r in up)} -> {n_files} files)")
                continue
            parts = [item_from_plan(r, "image", "new") for r in sorted(rs, key=lambda r: r["path_in_repo"])]
            for it in parts:
                it.group = base
            first = parts[0].members[0].split("/") if parts[0].members else []
            if len(first) >= 2 and f"/sca/out/{first[0]}/{first[1]}" in busy:
                self.note("deferred", f"{base} (named by a pending or running task)")
                continue
            if max(newest_mtime(it.root, it.members) for it in parts) > self.now - QUIET_S:
                self.note("deferred", f"{base} (modified in the last 15 minutes)")
                continue
            new_sets.append((PHASE_RANK.get(phase, 9), base, parts))
        quiet_links = []
        for it in link_items:
            if newest_mtime(it.root, it.members) > self.now - QUIET_S:
                self.note("deferred", it.path)
            else:
                quiet_links.append(it)
        return quiet_links, [parts for _, _, parts in sorted(new_sets, key=lambda x: (x[0], x[1]))]

    def adopt_from_build_records(self, rs):
        """Record items for an image set that an earlier sync built and uploaded but did not record (no rebuild)."""
        parts = []
        for r in sorted(rs, key=lambda r: r["path_in_repo"]):
            it = item_from_plan(r, "image", "record")
            rec = f"{SYNC}/built/{it.repo}/{it.path}.json"
            b = json.load(open(rec)) if os.path.exists(rec) else None
            if not (b and b["fp"] == [it.n_files, it.src_bytes] and os.path.exists(b["archive"])
                    and self.same_as_remote(it.repo, it.path, b["archive"], b["sha256"])):
                return []
            it.local, it.sha, it.size, it.group = b["archive"], b["sha256"], b["size"], set_base(it.path)
            parts.append(it)
        return parts

    # ---------------------------------------------------------------- build, upload, record
    def build(self, it):
        """Archive it (or reuse an earlier sync build of the same source); sets it.local, it.sha, it.size."""
        fp = [it.n_files, it.src_bytes]
        rec = f"{SYNC}/built/{it.repo}/{it.path}.json"
        if os.path.exists(rec):
            b = json.load(open(rec))
            if b["fp"] == fp and os.path.exists(b["archive"]) and os.path.getsize(b["archive"]) == b["size"]:
                it.local, it.sha, it.size = b["archive"], b["sha256"], b["size"]
                self.log(f"reuse {it.local}")
                return
        target = f"{STAGING}/{it.repo}/{it.path}"
        fresh_slot = not os.path.lexists(target)
        if not fresh_slot:
            target = f"{SYNC}/rev/{self.ts}/{it.repo}/{it.path}"
        lst = f"{SYNC}/lists/{self.ts}/{it.repo}/{it.path}.txt"
        for d in (os.path.dirname(target), os.path.dirname(lst), os.path.dirname(rec)):
            os.makedirs(d, exist_ok=True)
        with open(lst, "w") as f:
            f.write("\n".join(it.members) + "\n")
        tmp = target + ".tmp"
        zstd = "| zstd -T2 -3 -q " if it.comp == "zstd" else ""
        cmd = (f"set -o pipefail; tar --sort=name --format=gnu --no-recursion -cf - -C {shlex.quote(it.root)} "
               f"-T {shlex.quote(lst)} {zstd}| tee {shlex.quote(tmp)} | sha256sum | cut -d' ' -f1")
        self.log(f"build {target} ({it.n_files} files, {it.src_bytes / 1e9:.3f} GB source)")
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
        sha = r.stdout.strip()
        if r.returncode or len(sha) != 64:
            raise RuntimeError(f"archive build failed for {it.path}: {r.stderr.strip()[-300:]}")
        os.rename(tmp, target)
        it.local, it.sha, it.size = target, sha, os.path.getsize(target)
        with open(rec + ".tmp", "w") as f:
            json.dump({"fp": fp, "archive": target, "sha256": sha, "size": it.size, "run": self.ts}, f)
        os.replace(rec + ".tmp", rec)
        sums = f"{STAGING}/sums/{it.aid}.tsv"
        if fresh_slot and it.aid.startswith(("eval_", "code_", "img_")) and not os.path.exists(sums):
            os.makedirs(os.path.dirname(sums), exist_ok=True)
            with open(sums, "w") as f:
                f.write(f"{it.aid}\t{it.path}\t{sha}\t{it.size}\n")

    def upload(self, repo, items, message):
        """One Hub commit for items, then a size and SHA256 check of each; returns the paths that failed it."""
        from huggingface_hub import CommitOperationAdd
        rid, rtype = f"{OWNER}/{repo}", REPO_TYPE[repo]
        for attempt in (1, 2, 3):
            try:
                ops = [CommitOperationAdd(path_in_repo=it.path, path_or_fileobj=it.local) for it in items]
                self.api.create_commit(repo_id=rid, repo_type=rtype, operations=ops, commit_message=message)
                break
            except Exception as e:
                self.log(f"upload attempt {attempt} to {rid} failed: {e!r}")
                if attempt == 3:
                    return [it.path for it in items]
                time.sleep(60 * attempt)
        info = {e.path: e for e in self.api.get_paths_info(rid, [it.path for it in items], repo_type=rtype)}
        bad = []
        for it in items:
            e = info.get(it.path)
            lfs = getattr(e, "lfs", None)
            if e is None or getattr(e, "size", None) != it.size or (
                    lfs.sha256 != it.sha if lfs else e.blob_id != git_sha1(it.local)):
                bad.append(it.path)
        return bad

    def skip_cap(self, group, est):
        what = group[0].group or group[0].path
        self.note("skipped for the cap", f"{what} ({est / 1e9:.2f} GB)")
        if group[0].kind == "image" and not self.dry:
            self.put_rows([it.row(PLANNED) for it in group])

    def process(self, repo, groups, message):
        """groups: lists of items that go up together (the parts of one image set) or not at all."""
        send = []
        reserved = 0
        for g in groups:
            for it in [x for x in g if x.action == "record"]:
                self.recorded.append(it.path)
                if not self.dry:
                    self.put_rows([it.row(OK)])
            g = [x for x in g if x.action != "record"]
            if not g:
                continue
            est = sum(it.est for it in g)
            if self.usage + reserved + est > self.cap:
                self.skip_cap(g, est)
                continue
            if self.dry:
                for it in g:
                    self.would.append(f"{repo}:{it.path} ({it.action}, ~{it.est / 1e9:.3f} GB)")
                    self.log(f"dry run: would {it.action} {repo}:{it.path} (~{it.est / 1e9:.3f} GB)")
                reserved += est
                continue
            try:
                for it in g:
                    if not it.local:
                        self.build(it)
            except Exception as e:
                self.failed.append(f"{g[0].group or g[0].path}: {e}")
                self.log(f"FAILED build: {e}")
                continue
            keep = []
            for it in g:
                if it.action == "adopt" and self.same_as_remote(repo, it.path, it.local, it.sha):
                    self.recorded.append(it.path)
                    self.put_rows([it.row(OK)])
                elif it.action == "adopt" and it.kind not in REUPLOAD_KINDS:
                    self.note("changed but not re-uploaded", f"{it.path} (differs from the Hub copy)")
                else:
                    keep.append(it)
            size = sum(it.size for it in keep)
            if keep and self.usage + reserved + size > self.cap:
                self.skip_cap(keep, size)
                continue
            reserved += size
            send += keep
        if not send:
            return
        bad = self.upload(repo, send, message + ": " + ", ".join(it.path for it in send[:3])
                          + (f" and {len(send) - 3} more" if len(send) > 3 else ""))
        verified = []
        for it in send:
            if it.path in bad:
                self.failed.append(f"{repo}:{it.path} not verified")
                self.log(f"FAILED verify {repo}:{it.path}")
                continue
            verified.append(it.row(OK))
            self.usage += it.size
            self.uploaded.append(it)
            self.log(f"uploaded and verified {repo}:{it.path} ({it.size / 1e9:.3f} GB, sha256 {it.sha[:12]})")
        if verified:
            self.put_rows(verified)
            self.write_manifest()

    def write_manifest(self):
        if not self.dirty or self.dry:
            return
        tmp = os.path.join(os.path.dirname(self.manifest), ".migration_manifest.csv.synctmp")
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLS)
            w.writerows(self.rows)
        os.replace(tmp, self.manifest)
        self.dirty = False
        self.log(f"wrote {self.manifest} ({len(self.rows)} rows)")

    def sync_repo_manifests(self, message):
        """Keep MANIFEST.csv in each archive repo equal to its rows of results/migration_manifest.csv."""
        for repo in ("sca-eval-outputs", "sca-frozen-code", "sca-generated-images"):
            buf = io.StringIO(newline="")
            w = csv.writer(buf)
            w.writerow(COLS)
            w.writerows(r for r in self.rows if r[0] == f"{OWNER}/{repo}")
            data = buf.getvalue().encode()
            e = self.remote[repo].get("MANIFEST.csv")
            if e is not None and e[0] == len(data) and (
                    e[1] == hashlib.sha256(data).hexdigest() if e[1] else
                    e[2] == hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()):
                continue
            if self.dry:
                self.would.append(f"{repo}:MANIFEST.csv")
                self.log(f"dry run: would refresh {repo}:MANIFEST.csv")
                continue
            d = f"{SYNC}/manifests/{self.ts}/{repo}"
            os.makedirs(d, exist_ok=True)
            with open(f"{d}/MANIFEST.csv", "wb") as f:
                f.write(data)
            it = Item(kind="manifest", repo=repo, path="MANIFEST.csv", phase="", source="", n_png=0, n_json=0,
                      n_files=1, src_bytes=len(data), action="changed", local=f"{d}/MANIFEST.csv",
                      sha=hashlib.sha256(data).hexdigest(), size=len(data))
            if self.upload(repo, [it], message):
                self.failed.append(f"{repo}:MANIFEST.csv not verified")
            else:
                self.log(f"refreshed {repo}:MANIFEST.csv ({len(data)} bytes)")

    def summary(self):
        def names(xs, k=3):
            return ", ".join(xs[:k]) + (f" and {len(xs) - k} more" if len(xs) > k else "")
        parts = []
        if self.dry:
            parts.append(f"dry run, would upload {len(self.would)}" + (f": {names(self.would)}" if self.would else ""))
        elif self.uploaded:
            gb = sum(it.size for it in self.uploaded) / 1e9
            parts.append(f"uploaded {len(self.uploaded)} ({gb:.2f} GB): {names([it.path for it in self.uploaded])}")
        else:
            parts.append("nothing new to upload")
        if self.recorded:
            parts.append(f"recorded without upload {len(self.recorded)}: {names(self.recorded)}")
        for k, v in self.notes.items():
            if v:
                parts.append(f"{k} {len(v)}: {names(v)}")
        if self.failed:
            parts.append(f"FAILED {len(self.failed)}: {names(self.failed, 2)}")
        parts.append(f"private storage {self.usage / 1e9:.2f} GB of {self.cap / 1e9:.0f} GB cap")
        return "HF: " + "; ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--cap-gb", type=float, default=97.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    s = Sync(a)
    s.log(f"sync_hf start (dry_run={s.dry})")
    try:
        s.read_hub()
    except Exception as e:
        s.log(f"FAILED to read the Hub: {e!r}")
        print(f"HF: FAILED to read the Hub ({type(e).__name__}); nothing uploaded")
        return 2
    msg = f"SCA sync {time.strftime('%Y-%m-%d %H:%M')}"
    try:
        plan = s.fresh_plan()
        s.process("sca-calibration", [[it] for it in s.calib_items()], msg)
        s.process("sca-eval-outputs", [[it] for it in s.eval_items(plan)], msg)
        s.process("sca-frozen-code", [[it] for it in s.code_items(plan) + s.trace_items()], msg)
        link_items, image_sets = s.image_items(plan)
        s.process("sca-generated-images", [[it] for it in link_items], msg)
        for parts in image_sets:
            s.process("sca-generated-images", [parts], msg)
    except Exception as e:
        s.failed.append(f"{type(e).__name__}: {e}")
        s.log(f"FAILED: {e!r}")
    s.write_manifest()
    try:
        s.sync_repo_manifests(msg + ": MANIFEST.csv")
    except Exception as e:
        s.failed.append(f"MANIFEST.csv refresh: {type(e).__name__}")
        s.log(f"FAILED MANIFEST.csv refresh: {e!r}")
    line = s.summary()
    s.log(line)
    print(line)
    return 1 if s.failed else 0


if __name__ == "__main__":
    sys.exit(main())
