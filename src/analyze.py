#!/usr/bin/env python3
"""
Turn Inspect .eval logs into a per-section adherence table, a summary CSV/JSON, and a
bar chart. The chart needs matplotlib; the table and CSV don't.

    python src/analyze.py --logs-dir vendor/model_spec_evals/logs
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

from inspect_ai.log import read_eval_log

SECTIONS = ["style", "chain_of_command", "best_work", "seek_truth", "stay_in_bounds"]
ROOT = Path(__file__).resolve().parent.parent


def summarize(path: str) -> dict:
    log = read_eval_log(path)
    model = log.eval.model
    epochs = getattr(log.eval.config, "epochs", None)
    # Per-section + overall means live in the scorer's grouped metrics.
    per_section, overall = {}, None
    for score in (log.results.scores if log.results else []):
        for name, metric in score.metrics.items():
            val = getattr(metric, "value", None)
            key = name.split("/")[-1] if "/" in name else name
            if key in SECTIONS:
                per_section[key] = val
            elif key in ("all", "mean", "accuracy"):
                overall = val
    if overall is None and per_section:
        overall = sum(per_section.values()) / len(per_section)

    usage = {}
    for mu in (log.stats.model_usage or {}).items() if log.stats else []:
        mname, u = mu
        usage[mname] = {
            "input": getattr(u, "input_tokens", None),
            "output": getattr(u, "output_tokens", None),
            "cache_read": getattr(u, "input_tokens_cache_read", None),
            "reasoning": getattr(u, "reasoning_tokens", None),
            "total": getattr(u, "total_tokens", None),
        }
    return {"model": model, "epochs": epochs, "overall": overall,
            "sections": per_section, "usage": usage, "log": Path(path).name}


def print_table(rows: list[dict]) -> None:
    cols = ["model", "overall"] + SECTIONS
    w = {c: max(len(c), *(len(_cell(r, c)) for r in rows)) for c in cols}
    line = "  ".join(c.ljust(w[c]) for c in cols)
    print(line); print("-" * len(line))
    for r in rows:
        print("  ".join(_cell(r, c).ljust(w[c]) for c in cols))


def _cell(r: dict, c: str) -> str:
    if c == "model":
        return str(r["model"])
    v = r["overall"] if c == "overall" else r["sections"].get(c)
    return "—" if v is None else f"{v*100:.1f}%"


def write_summary(rows: list[dict], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(rows, indent=2))
    with (outdir / "summary.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "overall"] + SECTIONS)
        for r in rows:
            wr.writerow([r["model"], r["overall"]] + [r["sections"].get(s) for s in SECTIONS])
    print(f"\nwrote {outdir/'summary.json'} and summary.csv")


def make_chart(rows: list[dict], out: Path, title: str = "Model Spec adherence") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping chart (pip install matplotlib).")
        return
    rows = sorted(rows, key=lambda r: (r["overall"] or 0))
    labels = [str(r["model"]).replace("openai/", "") for r in rows]
    vals = [(r["overall"] or 0) * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(rows) + 1.5))
    bars = ax.barh(labels, vals, color="#10a37f")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Model Spec adherence (%)")
    ax.set_title(title)
    ax.bar_label(bars, fmt="%.0f%%", padding=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", help=".eval log files")
    ap.add_argument("--logs-dir", help="directory of .eval logs")
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--title", default="Model Spec adherence")
    args = ap.parse_args()

    paths = list(args.logs)
    if args.logs_dir:
        paths += sorted(glob.glob(str(Path(args.logs_dir) / "*.eval")))
    if not paths:
        raise SystemExit("No .eval logs given. Pass files or --logs-dir.")

    rows = [summarize(p) for p in paths]
    print_table(rows)
    write_summary(rows, Path(args.outdir))
    make_chart(rows, Path(args.outdir) / "figures" / "adherence.png", title=args.title)


if __name__ == "__main__":
    main()
