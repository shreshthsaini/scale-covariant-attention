"""Compare a local staging folder with a Hugging Face repo: every local file must exist remotely with the same size,
and with the same SHA256 for LFS/Xet-stored files (git blob files are compared by size and git SHA1).

Usage: python verify_hf.py <repo_id> <repo_type> <local_folder> [--sha-cache sums.txt]
Exit code 0 when everything matches. Prints one summary line per repo.
"""
import argparse
import hashlib
import os
import sys

from huggingface_hub import HfApi


def sha256(path, cache):
    if path in cache:
        return cache[path]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    cache[path] = h.hexdigest()
    return cache[path]


def git_sha1(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id")
    ap.add_argument("repo_type")
    ap.add_argument("local")
    ap.add_argument("--sha-cache", help="file of '<sha256>  <relative path>' lines to skip rehashing")
    a = ap.parse_args()
    cache = {}
    if a.sha_cache and os.path.exists(a.sha_cache):
        for line in open(a.sha_cache):
            digest, rel = line.rstrip("\n").split("  ", 1)
            cache[os.path.join(a.local, rel)] = digest
    api = HfApi()
    info = api.repo_info(a.repo_id, repo_type=a.repo_type)
    remote = {}
    for e in api.list_repo_tree(a.repo_id, repo_type=a.repo_type, recursive=True, expand=True):
        if getattr(e, "size", None) is not None and hasattr(e, "blob_id"):
            remote[e.path] = e
    bad, n, total = [], 0, 0
    for dirpath, _, files in os.walk(a.local):
        for name in files:
            p = os.path.join(dirpath, name)
            rel = os.path.relpath(p, a.local)
            if rel.startswith(".cache" + os.sep):
                continue
            n += 1
            size = os.path.getsize(p)
            total += size
            e = remote.get(rel)
            if e is None:
                bad.append(f"missing {rel}")
            elif e.size != size:
                bad.append(f"size {rel} local={size} remote={e.size}")
            elif e.lfs is not None:
                if e.lfs.sha256 != sha256(p, cache):
                    bad.append(f"sha256 {rel}")
            elif e.blob_id != git_sha1(p):
                bad.append(f"blob {rel}")
    print(f"{a.repo_id} private={info.private} local_files={n} remote_files={len(remote)} "
          f"bytes={total} mismatches={len(bad)}")
    for b in bad[:20]:
        print("  ", b)
    sys.exit(1 if bad or info.private is not True else 0)


if __name__ == "__main__":
    main()
