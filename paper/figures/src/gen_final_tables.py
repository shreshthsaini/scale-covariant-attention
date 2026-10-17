"""Generate complete E2/E9 tables from the archived evaluation evidence.

Run with the project's uv interpreter, for example:
  python paper/figures/src/gen_final_tables.py e2_8192 --out /tmp/e2_8192.tex
E2 defaults to per-image metrics; E9 also requires corrected distribution results.
--part per-image or distributions selects an independently complete table group.
No partial table or placeholder is written when validation fails. This module has
no import side effects, so its validation/rendering functions can be tested on
completed analogous data without relabeling those data as E2 or E9.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
# Same labels, units, and precision as gen_tables.py, which is not import-safe.
METRICS = (
    ("ir", "ImageReward", 1, 3),
    ("aesthetic", "Aesthetic", 1, 2),
    ("clip", "CLIP score", 1, 2),
    ("pickscore", "PickScore", 1, 2),
    ("hpsv2", r"HPSv2 ($\times100$)", 100, 2),
    ("hpsv21", r"HPSv2.1 ($\times100$)", 100, 2),
    ("musiq", "MUSIQ", 1, 1),
    ("musiq_crop1024", r"MUSIQ$_\text{c}$", 1, 1),
    ("clipiqa", r"CLIP-IQA ($\times100$)", 100, 1),
    ("arniqa", r"ARNIQA ($\times100$)", 100, 1),
)
DISTRIBUTIONS = (
    ("fid", r"FID$\downarrow$", 1, 1),
    ("kid", r"KID$\downarrow$ ($\times10^3$)", 1000, 2),
    ("patch_fid", r"FID$_\text{p}\downarrow$", 1, 1),
    ("patch_kid", r"KID$_\text{p}\downarrow$ ($\times10^3$)", 1000, 2),
)
LABELS = {"area": "AReA", "sca_nb": "SCA", "sca_krea": "SCA",
          "dype": "DyPE", "sega_official": "SEGA"}
PROFILES = {
    "e2_8192": (12, "8192x8192", ("area", "sca_nb"), "hi12.jsonl", r"FLUX.1-dev at $8192^2$"),
    "e2_6144": (100, "6144x6144", ("area", "sca_nb"), "eval100.jsonl", r"FLUX.1-dev at $6144^2$"),
    "e9_krea": (100, "4096x4096", ("area", "dype", "sega_official", "sca_krea"), "eval100.jsonl", r"FLUX.1-Krea-dev at $4096^2$"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def complete_meta(path, extra=False):
    meta = json.loads(path.read_text())
    # eval_extra.py has no all_metrics_available field; complete means no errors.
    require(meta.get("complete") is True and (meta.get("all_metrics_available") is True
            or (extra and "all_metrics_available" not in meta)),
            f"Incomplete evaluation: {path}")
    require(not meta.get("errors"), f"Evaluation errors: {path}")
    return meta


def number(value):
    result = float(value)
    require(math.isfinite(result), f"Nonfinite value: {value}")
    return result


def per_image(base, extra, methods, prompt_ids, resolution, seed="0"):
    """Validate exact manifest coverage and independently rederive all ten means."""
    means, values, sources = {}, {}, []
    for folder, prefix in ((base, ""), (extra, "extra_")):
        meta_path = folder / (prefix + "metrics.json")
        complete_meta(meta_path, extra=bool(prefix))
        source = folder / (prefix + "per_image.csv")
        summary_path = folder / (prefix + "summary.csv")
        sources.extend((meta_path, source, summary_path))
        data = rows(source)
        expected = {(m, p) for m in methods for p in prompt_ids}
        keys = [(r["method"], r["prompt_id"]) for r in data]
        require(len(keys) == len(set(keys)) and set(keys) == expected,
                f"Duplicate, missing, or unexpected image rows: {source}")
        require(all(r["res"] == resolution and r["seed"] == seed for r in data),
                f"Mixed resolution or seed: {source}")
        summary = {}
        for row in rows(summary_path):
            key = (row["method"], row["metric"])
            require(key not in summary, f"Duplicate summary entry: {summary_path}: {key}")
            summary[key] = row
        for key, _, _, _ in METRICS:
            if key not in data[0]:
                continue
            for method in methods:
                dest = (method, key)
                require(dest not in values, f"Metric present in both inputs: {dest}")
                values[dest] = {r["prompt_id"]: number(r[key]) for r in data if r["method"] == method}
                means[dest] = math.fsum(values[dest].values()) / len(prompt_ids)
                row = summary.get(dest)
                require(row is not None and row["res"] == resolution
                        and int(row["n_prompts"]) == len(prompt_ids), f"Bad summary coverage: {dest}")
                require(math.isclose(means[dest], number(row["value"]), rel_tol=1e-7, abs_tol=1e-7),
                        f"Summary disagrees with image values: {dest}")
    require(set(values) == {(m, k) for m in methods for k, *_ in METRICS}, "Missing per-image metric")
    return means, sources


def paired(base, extra, methods, means, count, resolution, seed="0"):
    """Use the existing implementation, including its sorted-metric RNG order."""
    method = methods[-1]
    result = {}
    with tempfile.TemporaryDirectory(prefix="sca-final-tables-") as tmp:
        for reference in methods[:-1]:
            output = Path(tmp) / "paired.csv"
            subprocess.run([sys.executable, str(ROOT / "scripts/paired_bootstrap.py"),
                            "--method", method, "--reference", reference, "--draws", "2000",
                            "--out", str(output), str(base / "per_image.csv"),
                            str(extra / "extra_per_image.csv")], check=True)
            for row in rows(output):
                key = row["metric"]
                if key not in {m[0] for m in METRICS}:
                    continue
                require(row["res"] == resolution and row["seed"] == seed
                        and int(row["n_prompts"]) == count, "Paired bootstrap coverage mismatch")
                delta, low, high = map(number, (row["delta"], row["ci_low"], row["ci_high"]))
                require(low <= high and math.isclose(delta, means[method, key] - means[reference, key],
                        rel_tol=6e-6, abs_tol=1e-6), "Paired bootstrap mean mismatch")
                require((reference, key) not in result, "Duplicate paired metric")
                result[reference, key] = (delta, low, high)
    require(len(result) == (len(methods) - 1) * len(METRICS), "Missing paired metric")
    return result


def distributions(base, methods, count, resolution):
    """Accept only corrected row assembly and reference-image cluster intervals."""
    meta_path = base / "metrics.json"
    meta = complete_meta(meta_path)
    config = meta["config"]
    draws = int(config["bootstrap"])
    require(draws >= 1000 and int(config["bootstrap_seed"]) == 0, "Invalid distribution bootstrap protocol")
    require(config["reference_method"] == "area", "Distribution reference must be AReA")
    for key, *_ in DISTRIBUTIONS:
        status = meta.get("status", {}).get(key, {})
        require(status.get("status") == "complete"
                and status.get("feature_assembly") == "reference_prompt_row_v2"
                and status.get("reference_clusters") == "images"
                and status.get("reference_rows") == 195
                and status.get("reference_cluster_count") == 168,
                f"Missing corrected 195-row/168-image-cluster provenance: {key}")
    means, intervals = {}, {}
    sources = [meta_path, base / "summary.csv", base / "paired_deltas.csv"]
    for path, paired_rows in ((sources[1], False), (sources[2], True)):
        for row in rows(path):
            key, method = row["metric"], row["method"]
            if key not in {m[0] for m in DISTRIBUTIONS}:
                continue
            require(method in methods and row["res"] == resolution
                    and int(row["n_prompts"]) == count
                    and int(row["bootstrap_valid"]) == draws, f"Incomplete distribution row: {path}")
            value, low, high = map(number, (row["value"], row["ci_low"], row["ci_high"]))
            require(low <= high, "Invalid distribution interval")
            destination = intervals if paired_rows else means
            require((method, key) not in destination, "Duplicate distribution metric")
            if paired_rows:
                require(method != "area" and row["reference_method"] == "area"
                        and row["delta_definition"] == "method_minus_reference", "Invalid distribution contrast")
                intervals[method, key] = (value, low, high)
            else:
                means[method, key] = value
    require(set(means) == {(m, k) for m in methods for k, *_ in DISTRIBUTIONS}, "Missing distribution mean")
    require(set(intervals) == {(m, k) for m in methods if m != "area" for k, *_ in DISTRIBUTIONS},
            "Missing distribution contrast")
    for (method, key), (delta, _, _) in intervals.items():
        require(math.isclose(delta, means[method, key] - means["area", key], rel_tol=1e-7, abs_tol=1e-7),
                "Distribution contrast disagrees with means")
    return means, intervals, draws, sources


def fmt(value, metric, signed=False):
    scaled = value * metric[2]
    # Avoid displaying a negative zero after rounding.
    if abs(scaled) < 0.5 * 10 ** -metric[3]:
        scaled = 0.0
    text = format(scaled, ("+" if signed else "") + f".{metric[3]}f")
    return "$" + text + "$"


def interval(values, metric, stacked=False):
    delta, low, high = values
    point = fmt(delta, metric, True)
    bounds = "[" + fmt(low, metric, True) + ", " + fmt(high, metric, True) + "]"
    return (r"\shortstack[r]{" + point + r"\\" + bounds + "}") if stacked else point + " " + bounds


def table(headers, data, caption, label):
    return "\n".join([
        r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}",
        r"\caption{" + caption + "}", r"\label{tab:" + label + "}",
        r"\begin{tabular}{l" + "r" * (len(headers) - 1) + "}", r"\toprule",
        " & ".join(headers) + r" \\", r"\midrule",
        *(" & ".join(row) + r" \\" for row in data),
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ])


def render_per_image(name, title, methods, means, deltas, count, seed="0"):
    description = f"{title}, {count} matched prompts, seed {seed}. "
    note = ("All metrics are higher-is-better. Intervals are paired prompt-bootstrap 95\\% "
            "percentile intervals (2,000 draws, bootstrap seed 0).")
    if len(methods) == 2:
        baseline, method = methods
        data = [[m[1], fmt(means[baseline, m[0]], m), fmt(means[method, m[0]], m),
                 interval(deltas[baseline, m[0]], m)] for m in METRICS]
        return table(["Metric", LABELS[baseline], LABELS[method], r"SCA $-$ AReA [95\% CI]"], data,
                     description + note, name + "_per_image")
    mean_table = table(["Metric"] + [LABELS[m] for m in methods],
                       [[metric[1]] + [fmt(means[m, metric[0]], metric) for m in methods] for metric in METRICS],
                       description + "Per-image metric means; all metrics are higher-is-better.", name + "_means")
    delta_table = table(["Metric"] + [r"SCA $-$ " + LABELS[m] for m in methods[:-1]],
                        [[metric[1]] + [interval(deltas[m, metric[0]], metric, stacked=True) for m in methods[:-1]]
                         for metric in METRICS], description + "SCA differences [95\\% CI]. " + note, name + "_paired")
    return mean_table + "\n" + delta_table


def render_distributions(name, title, methods, means, deltas, count, draws):
    description = (f"{title}, {count} generated prompts per method. Corrected reference crop assembly "
                   "preserves all 195 reference-row weights. All four metrics are lower-is-better. ")
    output = table(["Metric"] + [LABELS[m] for m in methods],
                   [[metric[1]] + [fmt(means[m, metric[0]], metric) for m in methods] for metric in DISTRIBUTIONS],
                   description, name + "_distribution_means")
    contrasts = [m for m in methods if m != "area"]
    output += "\n" + table(["Metric"] + [LABELS[m] + r" $-$ AReA" for m in contrasts],
        [[metric[1]] + [interval(deltas[m, metric[0]], metric, stacked=True) for m in contrasts] for metric in DISTRIBUTIONS],
        description + f"Differences [95\\% CI] use {draws:,} percentile-bootstrap draws, seed 0, "
        "resampling generated prompts and 168 reference-image identity clusters independently. "
        "The reference draw is shared between methods within each paired replicate.", name + "_distribution_paired")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=PROFILES)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--part", choices=("all", "per-image", "distributions"))
    args = parser.parse_args()
    count, resolution, methods, manifest, title = PROFILES[args.experiment]
    part = args.part or ("all" if args.experiment == "e9_krea" else "per-image")
    archive = args.results / "evidence"
    manifest_path = archive / "prompts" / manifest
    prompt_ids = [json.loads(line)["id"] for line in manifest_path.read_text().splitlines() if line.strip()]
    require(len(prompt_ids) == len(set(prompt_ids)) == count, "Manifest count or uniqueness mismatch")
    base = archive / "eval" / (args.experiment + "_base")
    extra = archive / "eval" / (args.experiment + "_extra")
    # Validate complete per-image coverage even for a distribution-only output.
    means, sources = per_image(base, extra, methods, prompt_ids, resolution)
    output = ""
    if part in ("all", "per-image"):
        deltas = paired(base, extra, methods, means, count, resolution)
        output += render_per_image(args.experiment, title, methods, means, deltas, count)
        sources.append(ROOT / "scripts/paired_bootstrap.py")
    if part in ("all", "distributions"):
        dist_means, dist_deltas, draws, dist_sources = distributions(base, methods, count, resolution)
        output += "\n" + render_distributions(args.experiment, title, methods, dist_means, dist_deltas, count, draws)
        sources.extend(dist_sources)
    sources.extend((manifest_path, Path(__file__).resolve()))
    provenance = ["% Generated from validated complete evidence. Source SHA-256:"]
    for source in sorted(set(sources)):
        display = source.relative_to(ROOT) if source.is_relative_to(ROOT) else source
        provenance.append(f"% {hashlib.sha256(source.read_bytes()).hexdigest()}  {display}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=args.out.parent, prefix=".table-", delete=False) as stream:
        stream.write("\n".join(provenance) + "\n" + output)
        temporary = stream.name
    os.replace(temporary, args.out)
    print(f"Validated {args.experiment}: {count} prompts x {len(methods)} methods; wrote {args.out}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"No table generated: {error}") from error
