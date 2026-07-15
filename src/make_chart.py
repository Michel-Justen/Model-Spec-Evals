#!/usr/bin/env python3
"""
Build the "Model Spec compliance by model" chart. Two modes:

  (default)     OpenAI's published lineup (reference) + our independently measured
                GPT-5.6 Sol, which OpenAI did not report.
  --ours-only   Only the models WE measured, all on one identical methodology
                (same harness, dataset, grader, sampling) — the cleanest apples-to-apples.

Colors validated colorblind-safe via the dataviz palette validator (teal / orange).

    python src/make_chart.py
    python src/make_chart.py --ours-only
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parent.parent
TEAL = "#0E8B77"
ORANGE = "#C85A22"
INK = "#1a1a1a"
MUTED = "#6b6b6b"

# Models we have measured on our own methodology: key -> (label, date, log dirs)
MEASURED = {
    "gpt5-thinking": ("GPT-5 Thinking", "Aug 2025", ["runs/anchor-gpt5-thinking", "runs/anchor-topup"]),
    "sol": ("GPT-5.6 Sol", "Jul 2026", ["runs/sol", "runs/sol-topup"]),
    "gpt5.5": ("GPT-5.5 Thinking", "Apr 2026", ["runs/gpt55"]),
    # add as measured, e.g.:
    "gpt5.4": ("GPT-5.4 Thinking", "Mar 2026", ["runs/gpt54"]),
}
# chronological order for the ours-only chart
OURS_ORDER = ["gpt5-thinking", "gpt5.4", "gpt5.5", "sol"]


CACHE = ROOT / "results" / ".estimate_cache.json"


def estimate(dirs: list[str], refresh: bool = False) -> tuple[float, float, int]:
    """Overall compliance + fixed-benchmark 95% half-width (sampling noise), n prompts.

    Reading the (large) .eval logs is slow, so results are cached by log-dir set in
    results/.estimate_cache.json. Pass refresh=True or delete the cache to recompute.
    """
    key = "|".join(sorted(dirs))
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    if not refresh and key in cache:
        o, c, n = cache[key]
        return o, c, n

    paths = [p for d in dirs for p in glob.glob(str(ROOT / d / "*.eval"))]
    by_prompt: dict[str, list[float]] = defaultdict(list)
    for path in paths:
        log = read_eval_log(path)
        for s in (log.samples or []):
            sc = s.scores.get("model_graded_spec_section_compliance") if s.scores else None
            if sc is None or not isinstance(sc.value, (int, float)):
                continue
            by_prompt[s.id].append(float(sc.value))
    n = len(by_prompt)
    overall = sum(sum(v) / len(v) for v in by_prompt.values()) / n
    var = sum((sum(v) / len(v)) * (1 - sum(v) / len(v)) / len(v) for v in by_prompt.values())
    ci = 1.96 * math.sqrt(var) / n

    cache[key] = [overall, ci, n]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=2))
    return overall, ci, n


def render(bars, xs, *, title, subtitle, caption, out, width, legend_handles=None,
           groups=(), single_color=None):
    """bars: list of (label, date, value, is_ours, ci)."""
    fig, ax = plt.subplots(figsize=(width, 6))
    for x, (label, date, val, ours, ci) in zip(xs, bars):
        color = single_color or (ORANGE if ours else TEAL)
        ax.bar(x, val, width=0.72, color=color, zorder=3,
               yerr=(ci if ci else None), capsize=4 if ci else 0,
               error_kw=dict(ecolor=INK, elinewidth=1.4, capthick=1.4))
        ax.text(x, val + (ci or 0) + 0.018, f"{val*100:.0f}%", ha="center",
                fontsize=12, fontweight="bold" if ours else "normal", color=INK)

    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "20%", "40%", "60%", "80%", "100%"], fontsize=10, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{l}\n{d}" for l, d, *_ in bars], fontsize=9.5, color=INK)
    ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.yaxis.grid(True, color="#ededed", zorder=0)
    ax.set_axisbelow(True)

    for gx, gtext in groups:
        ax.text(gx, -0.155, gtext, ha="center", color=MUTED, fontsize=9, style="italic")

    fig.text(0.065, 0.955, title, fontsize=17, fontweight="bold", color=INK, va="top")
    fig.text(0.065, 0.905, subtitle, fontsize=10.5, color=MUTED, va="top")
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", bbox_to_anchor=(1.0, 1.005),
                  ncol=2, frameon=False, fontsize=10, handlelength=1.2, columnspacing=1.4)
    fig.text(0.065, 0.035, caption, fontsize=8, color=MUTED)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.18, top=0.80, left=0.08, right=0.96)
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours-only", action="store_true",
                    help="chart only the models we measured, on one methodology")
    args = ap.parse_args()

    meas = {}
    for key, (label, date, dirs) in MEASURED.items():
        val, ci, n = estimate(dirs)
        meas[key] = (label, date, val, ci)
        print(f"  {label}: {val*100:.1f}% ±{ci*100:.1f} (n={n})")
    gpt5t = meas["gpt5-thinking"]

    if args.ours_only:
        bars = [(meas[k][0], meas[k][1], meas[k][2], True, meas[k][3])
                for k in OURS_ORDER if k in meas]
        xs = list(range(len(bars)))
        render(
            bars, xs, width=max(11, 2.4 * len(bars) + 5), single_color=TEAL,
            title="Independent Model Spec compliance",
            subtitle="Every model measured on the same harness, dataset, grader (GPT-5), and "
                     "sampling.\nReasoning models at high effort. Error bars are 95% CIs.",
            caption=f"Independent replication on OpenAI's open eval harness + dataset (n=587 prompts). "
                    f"Validation: our GPT-5 Thinking = {gpt5t[2]*100:.1f}% vs OpenAI's published 89%.",
            out=ROOT / "results" / "figures" / "compliance_ours_only.png",
        )
    else:
        sol = meas["sol"]
        bars = [
            ("GPT-4o", "May 2024", 0.72, False, None),
            ("GPT-5 Instant", "Aug 2025", 0.82, False, None),
            ("GPT-5.3 Instant", "Mar 2026", 0.84, False, None),
            ("OpenAI o3", "Apr 2025", 0.80, False, None),
            ("GPT-5 Thinking", "Aug 2025", 0.89, False, None),
            ("GPT-5.4 Thinking", "Mar 2026", 0.87, False, None),
            ("GPT-5.6 Sol", "Jul 2026", sol[2], True, sol[3]),
        ]
        xs = [0, 1, 2, 3.7, 4.7, 5.7, 6.7]
        render(
            bars, xs, width=11,
            title="Overall Model Spec compliance by model",
            subtitle="Higher = better adherence to OpenAI's Model Spec. Teal bars are OpenAI's "
                     "published figures;\nGPT-5.6 Sol — which OpenAI did not report — is measured "
                     "independently here.",
            caption=f"Independent replication using OpenAI's open eval harness + dataset. Method "
                    f"validated: our GPT-5 Thinking = {gpt5t[2]*100:.1f}% vs OpenAI's 89%. "
                    f"GPT-5.6 Sol error bar = 95% CI (n=587 prompts). Grader: GPT-5.",
            out=ROOT / "results" / "figures" / "compliance_by_model.png",
            legend_handles=[Patch(facecolor=TEAL, label="OpenAI (published)"),
                            Patch(facecolor=ORANGE, label="This work (independent)")],
            groups=[(1.0, "non-reasoning"), (5.2, "reasoning")],
        )


if __name__ == "__main__":
    main()
