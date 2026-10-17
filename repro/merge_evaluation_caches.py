#!/usr/bin/env python3
"""Merge isolated scorer caches atomically, rejecting unequal key collisions."""

import argparse
import json
from pathlib import Path


def merge(destination, sources):
    destination = Path(destination)
    merged = json.loads(destination.read_text()) if destination.exists() else {}
    initial = len(merged)
    for source in sources:
        entries = json.loads(Path(source).read_text())
        for key, value in entries.items():
            if key in merged and merged[key] != value:
                raise ValueError(f"Unequal cache collision for {key} from {source}; destination unchanged")
            merged[key] = value
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".merge.tmp")
    temporary.write_text(json.dumps(merged, indent=2) + "\n")
    temporary.replace(destination)
    return {"initial_entries": initial, "merged_entries": len(merged), "added_entries": len(merged) - initial}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    parser.add_argument("sources", nargs="+")
    args = parser.parse_args()
    print(json.dumps(merge(args.destination, args.sources)))
