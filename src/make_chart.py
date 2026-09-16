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
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
TEAL = "#0E8B77"
ORANGE = "#C85A22"
INK = "#1a1a1a"
MUTED = "#6b6b6b"

# Categorical palette (dataviz skill, validated colorblind-safe on a light surface;
# worst adjacent CVD ΔE 9.1, target ≥8). Low fill-vs-surface contrast on aqua/yellow
# is covered by the relief rule: every bar carries a direct value label.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # blue, orange, aqua, yellow, magenta

# Models we have measured on our own methodology: key -> (label, release date, key in SCORES)
MEASURED = {
    "gpt5-thinking": ("GPT-5 Thinking", "Aug 2025", "GPT-5 Thinking"),
    "sol": ("GPT-5.6 Sol", "Jul 2026", "GPT-5.6 Sol"),
    "gpt5.5": ("GPT-5.5 Thinking", "Apr 2026", "GPT-5.5"),
    "gpt5.4": ("GPT-5.4 Thinking", "Mar 2026", "GPT-5.4"),
    "astra": ("GPT-6 Astra", "Sep 2026", "GPT-6 Astra"),
}
# chronological order for the ours-only chart
OURS_ORDER = ["gpt5-thinking", "gpt5.4", "gpt5.5", "sol", "astra"]


# Per-prompt scores written by src/analyze_scores.py: {model: {prompt_id: [[score, filter_blocked], ...]}}.
# Single source of truth, so the chart can never drift from the reported numbers.
SCORES = ROOT / "results" / "per_prompt_scores.json"


def estimate(model_key: str, excl_filter: bool = False) -> tuple[float, float, int]:
    """Overall compliance + fixed-benchmark 95% half-width (sampling noise), n prompts.

    Prompt-weighted: each prompt contributes its own pass rate across epochs, so models
    with more epochs are not weighted differently. The CI is the fixed-benchmark one —
    sampling noise over repeated answers to THESE prompts, using the Bessel-corrected
    per-prompt variance (identical to analyze_scores.py); it is not a CI over prompts.
    With excl_filter, answers the OpenAI API blocked (stop_reason content_filter, whose
    boilerplate the grader scored) are dropped.
    """
    data = json.loads(SCORES.read_text())[model_key]
    rates, var = {}, 0.0
    for pid, answers in data.items():
        vals = [float(s) for s, blocked in answers if not (excl_filter and blocked)]
        if not vals:
            continue
        rates[pid] = sum(vals) / len(vals)
        if len(vals) > 1:
            var += statistics.variance(vals) / len(vals)
    n = len(rates)
    return sum(rates.values()) / n, 1.96 * math.sqrt(var) / n, n


def render(bars, xs, *, title, subtitle, caption, out, width, legend_handles=None,
           groups=(), single_color=None, bar_width=0.72, label_fmt="{:.0f}%", title_size=17,
           hide_yaxis=False, ylabel=None, colors=None, source=None):
    """bars: list of (label, date, value, is_ours, ci)."""
    fig, ax = plt.subplots(figsize=(width, 6))
    for i, (x, (label, date, val, ours, ci)) in enumerate(zip(xs, bars)):
        color = (colors[i] if colors else (single_color or (ORANGE if ours else TEAL)))
        ax.bar(x, val, width=bar_width, color=color, zorder=3,
               yerr=(ci if ci else None), capsize=4 if ci else 0,
               error_kw=dict(ecolor=INK, elinewidth=1.4, capthick=1.4))
        ax.text(x, val + (ci or 0) + 0.018, label_fmt.format(val * 100), ha="center",
                fontsize=12, fontweight="bold" if ours else "normal", color=INK)

    ax.set_ylim(0, 1)  # bars always anchored at 0 (honest heights), even when axis is hidden
    if hide_yaxis:
        ax.set_yticks([])
    else:
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0", "20%", "40%", "60%", "80%", "100%"], fontsize=10, color=MUTED)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=11, color=INK, labelpad=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{l}\n{d}" for l, d, *_ in bars], fontsize=9.5, color=INK)
    ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    if not hide_yaxis:
        ax.yaxis.grid(True, color="#ededed", zorder=0)
    ax.set_axisbelow(True)

    for gx, gtext in groups:
        ax.text(gx, -0.155, gtext, ha="center", color=MUTED, fontsize=9, style="italic")

    fig.text(0.065, 0.955, title, fontsize=title_size, fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.065, 0.905, subtitle, fontsize=10.5, color=MUTED, va="top")
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", bbox_to_anchor=(1.0, 1.005),
                  ncol=2, frameon=False, fontsize=10, handlelength=1.2, columnspacing=1.4)
    if source:
        fig.text(0.065, 0.052, caption, fontsize=8, color=MUTED)
        fig.text(0.065, 0.026, source, fontsize=7.5, color=MUTED, style="italic")
    else:
        fig.text(0.065, 0.035, caption, fontsize=8, color=MUTED)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.18, top=0.80, left=0.11, right=0.96)
    fig.savefig(out, dpi=160, facecolor="white")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours-only", action="store_true",
                    help="chart only the models we measured, on one methodology")
    ap.add_argument("--ramp", action="store_true",
                    help="(with --ours-only) chronological single-hue blue ramp instead of categorical")
    ap.add_argument("--excl-filter", action="store_true",
                    help="exclude API filter-blocked answers instead of scoring them as published")
    args = ap.parse_args()

    meas = {}
    for key, (label, date, score_key) in MEASURED.items():
        val, ci, n = estimate(score_key, args.excl_filter)
        meas[key] = (label, date, val, ci)
        print(f"  {label}: {val*100:.1f}% ±{ci*100:.1f} (n={n})")
    gpt5t = meas["gpt5-thinking"]

    if args.ours_only:
        bars = [(meas[k][0], meas[k][1], meas[k][2], True, meas[k][3])
                for k in OURS_ORDER if k in meas]
        xs = list(range(len(bars)))
        # blue ordinal ramp, oldest→newest = light→dark (validated with --ordinal)
        RAMP = ["#9dc3f0", "#5497e8", "#2a78d6", "#1c5cab", "#0d366b"]
        palette = RAMP[:len(bars)] if args.ramp else CAT[:len(bars)]
        out = "compliance_ours_only_ramp.png" if args.ramp else "compliance_ours_only.png"
        if args.excl_filter:
            out = out.replace(".png", "_excl_filter.png")
        render(
            bars, xs, width=9, colors=palette, bar_width=0.5, label_fmt="{:.1f}%", title_size=16,
            ylabel="Compliance rate",
            title="Independent Evaluation of OpenAI's Model Spec Evals Scores",
            subtitle="",
            caption='All models run at "high" reasoning effort. Error bars are 95% confidence intervals.',
            source="Source & methodology: github.com/Michel-Justen/Model-Spec-Evals",
            out=ROOT / "results" / "figures" / out,
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
