"""Recreate zero-byte files listed in each installed distribution's RECORD (size 0) that a scratch purge removed."""
import csv, glob, os, sys
site = sys.argv[1]; dry = "--dry" in sys.argv
made = 0
for rec in glob.glob(os.path.join(site, "*.dist-info", "RECORD")):
    for row in csv.reader(open(rec, newline="")):
        if len(row) < 3 or row[2] != "0":
            continue
        p = os.path.normpath(os.path.join(site, row[0]))
        if not p.startswith(site) or os.path.exists(p):
            continue
        made += 1
        if not dry:
            os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w").close()
print(site, "restored" if not dry else "missing", made)
