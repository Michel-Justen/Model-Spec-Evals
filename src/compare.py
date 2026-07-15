#!/usr/bin/env python3
"""
Paired comparison of two models on the shared Model Spec benchmark.

Both models are scored on the same 587 prompts, so a paired test on the per-prompt
compliance rate controls for prompt difficulty and is much more powerful than comparing
two independent confidence intervals. Reports each model's rate and the paired gap with
a 95% CI and p-value.

    python src/compare.py --a runs/anchor-gpt5-thinking runs/anchor-topup \
                          --b runs/sol runs/sol-topup \
                          --labels "GPT-5 Thinking" "GPT-5.6 Sol"

Each --a / --b entry is a .eval file or a directory of them.
"""
from __future__ import annotations

import argparse
import glob
import math
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log


def expand(entries: list[str]) -> list[str]:
    out: list[str] = []
    for x in entries:
        p = Path(x)
        out += sorted(glob.glob(str(p / "*.eval"))) if p.is_dir() else [x]
    return out


def prompt_rates(paths: list[str]) -> dict[str, float]:
    """Per-prompt compliance rate = mean over all that model's samples for the prompt."""
    by_prompt: dict[str, list[float]] = defaultdict(list)
    for path in paths:
        log = read_eval_log(path)
        for s in (log.samples or []):
            sc = s.scores.get("model_graded_spec_section_compliance") if s.scores else None
            if sc is None or not isinstance(sc.value, (int, float)):
                continue
            by_prompt[s.id].append(float(sc.value))
    return {pid: sum(v) / len(v) for pid, v in by_prompt.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True, help="model A logs/dirs")
    ap.add_argument("--b", nargs="+", required=True, help="model B logs/dirs")
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    args = ap.parse_args()

    ra, rb = prompt_rates(expand(args.a)), prompt_rates(expand(args.b))
    common = sorted(set(ra) & set(rb))
    la, lb = args.labels
    if not common:
        raise SystemExit("No shared prompts between the two log sets.")

    mean_a = sum(ra[p] for p in common) / len(common)
    mean_b = sum(rb[p] for p in common) / len(common)
    diffs = [ra[p] - rb[p] for p in common]
    n = len(diffs)
    md = sum(diffs) / n
    var = sum((d - md) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    z = md / se if se > 0 else float("inf")
    p = math.erfc(abs(z) / math.sqrt(2))  # two-tailed normal approx

    print(f"prompts compared: {n}")
    print(f"  {la}: {mean_a*100:.1f}%")
    print(f"  {lb}: {mean_b*100:.1f}%")
    print(f"paired gap ({la} - {lb}): {md*100:.2f} pts "
          f"(95% CI ±{1.96*se*100:.2f} -> [{(md-1.96*se)*100:.2f}, {(md+1.96*se)*100:.2f}])")
    print(f"z = {z:.2f}, two-tailed p = {p:.4f}  "
          f"{'SIGNIFICANT at 0.05' if p < 0.05 else 'not significant at 0.05'}")


if __name__ == "__main__":
    main()
