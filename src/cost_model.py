#!/usr/bin/env python3
"""
Estimate the dollar cost of a Model Spec eval run.

Calibrated to the token counts OpenAI published in the model_spec_evals README for one
full pass (587 prompts, 1 epoch, 1 grader sample): the grader used ~33.3M tokens, the
candidate ~259K. The grader dominates because it re-reads the full ~50K-token Model Spec
on every grading call, so cost scales with prompts * epochs * grader_samples.

    python src/cost_model.py
    python src/cost_model.py --epochs 5 --grader-samples 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

N_PROMPTS = 587  # runnable prompts (596 - 9 skipped)

# --- Calibration constants (per grading call / per candidate call), from the README anchor ---
GRADER_INPUT_PER_CALL = 32_778_942 / N_PROMPTS   # ~55,842  (full spec + rubric + convo)
GRADER_OUTPUT_PER_CALL = 559_609 / N_PROMPTS     # ~953
GRADER_REASON_PER_CALL = 450_304 / N_PROMPTS     # ~767  (billed at output rate)

# Candidate token profiles per call. Non-reasoning calibrated from gpt-4o-mini anchor;
# reasoning profile is a conservative estimate (reasoning models emit far more output).
CANDIDATE_PROFILES = {
    False: {"input": 49_788 / N_PROMPTS, "output": 208_949 / N_PROMPTS, "reasoning": 0.0},
    True:  {"input": 49_788 / N_PROMPTS, "output": 700.0, "reasoning": 1_500.0},
}

# Fraction of the grader's input that is the shared, cacheable prefix (spec + template).
# The spec alone is ~50K of the ~55.8K input tokens.
CACHEABLE_PREFIX_FRAC = 50_000 / GRADER_INPUT_PER_CALL

HERE = Path(__file__).resolve().parent
PRICES_PATH = HERE.parent / "config" / "prices.yaml"


def load_prices() -> dict:
    if yaml is None or not PRICES_PATH.exists():
        # fall back to the documented placeholder for the grader only
        return {"grader": {"openai/gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.0}},
                "candidates": {}}
    return yaml.safe_load(PRICES_PATH.read_text())


def grader_cost(calls: int, price: dict, cache: bool) -> float:
    """USD cost of `calls` grader calls at the given per-1M price dict."""
    in_tok = GRADER_INPUT_PER_CALL * calls
    out_tok = (GRADER_OUTPUT_PER_CALL + GRADER_REASON_PER_CALL) * calls
    if cache:
        cached = in_tok * CACHEABLE_PREFIX_FRAC
        fresh = in_tok - cached
        in_cost = (fresh * price["input"] + cached * price["cached_input"]) / 1e6
    else:
        in_cost = in_tok * price["input"] / 1e6
    return in_cost + out_tok * price["output"] / 1e6


def candidate_cost(calls: int, price: dict) -> float:
    prof = CANDIDATE_PROFILES[bool(price.get("reasoning", False))]
    in_cost = prof["input"] * calls * price["input"] / 1e6
    out_cost = (prof["output"] + prof["reasoning"]) * calls * price["output"] / 1e6
    return in_cost + out_cost


def estimate(epochs: int, grader_samples: int, prices: dict, n_prompts: int = N_PROMPTS) -> dict:
    grader_calls = n_prompts * epochs * grader_samples
    candidate_calls = n_prompts * epochs
    gp = next(iter(prices["grader"].values()))
    g_nocache = grader_cost(grader_calls, gp, cache=False)
    g_cache = grader_cost(grader_calls, gp, cache=True)

    per_model = {}
    for cid, cp in (prices.get("candidates") or {}).items():
        c_cost = candidate_cost(candidate_calls, cp)
        per_model[cid] = {
            "candidate_usd": c_cost,
            "total_nocache_usd": c_cost + g_nocache,
            "total_cache_usd": c_cost + g_cache,
        }
    return {
        "epochs": epochs, "grader_samples": grader_samples,
        "grader_calls": grader_calls, "candidate_calls": candidate_calls,
        "grader_input_tokens": GRADER_INPUT_PER_CALL * grader_calls,
        "grader_usd_nocache": g_nocache, "grader_usd_cache": g_cache,
        "per_model": per_model,
    }


def fmt_usd(x: float) -> str:
    return f"${x:,.0f}" if x >= 100 else f"${x:,.2f}"


def print_scenarios(prices: dict, scenarios: list[tuple[str, int, int]]) -> None:
    gname = next(iter(prices["grader"]))
    print(f"\nCost model — OpenAI Model Spec eval  |  {N_PROMPTS} prompts  |  grader = {gname}")
    print(f"grader price: {prices['grader'][gname]}  (per 1M tokens)")
    print("Grader input tokens = 55.8K/call × prompts × epochs × grader_samples (spec re-read each call).")
    print("cache column assumes the ~50K-token spec prefix hits the prompt cache (ideal upper bound).\n")
    hdr = f"{'scenario':<12}{'epochs':>7}{'g.samp':>7}{'grader calls':>14}{'grader Btok':>13}" \
          f"{'grader $ (no cache)':>21}{'grader $ (cached)':>19}"
    print(hdr); print("-" * len(hdr))
    for name, ep, gs in scenarios:
        e = estimate(ep, gs, prices)
        print(f"{name:<12}{ep:>7}{gs:>7}{e['grader_calls']:>14,}"
              f"{e['grader_input_tokens']/1e9:>12.2f}B"
              f"{fmt_usd(e['grader_usd_nocache']):>21}{fmt_usd(e['grader_usd_cache']):>19}")

    # Full 3-model program totals at the two "real run" presets
    print("\nPer-candidate + 3-model program totals (candidate share included):")
    for name, ep, gs in scenarios:
        if name in ("smoke", "pilot"):
            continue
        e = estimate(ep, gs, prices)
        if not e["per_model"]:
            continue
        print(f"\n  [{name}]  epochs={ep} grader_samples={gs}")
        prog_nc = prog_c = 0.0
        for cid, m in e["per_model"].items():
            if cid == "openai/gpt-4o-mini":
                continue  # not a target model, just a price reference
            prog_nc += m["total_nocache_usd"]; prog_c += m["total_cache_usd"]
            print(f"    {cid:<26} no-cache {fmt_usd(m['total_nocache_usd']):>10}   "
                  f"cached {fmt_usd(m['total_cache_usd']):>10}")
        print(f"    {'— 3 target models total —':<26} no-cache {fmt_usd(prog_nc):>10}   "
              f"cached {fmt_usd(prog_c):>10}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--grader-samples", type=int)
    args = ap.parse_args()
    prices = load_prices()

    if args.epochs and args.grader_samples:
        scenarios = [("custom", args.epochs, args.grader_samples)]
    else:
        scenarios = [
            ("smoke", 1, 1), ("pilot", 2, 1),
            ("moderate", 5, 3), ("blog(OpenAI)", 20, 5),
        ]
    print_scenarios(prices, scenarios)
    print("\nPrices are placeholders — edit config/prices.yaml. The per-call token constants "
          "come from OpenAI's published run; re-measure from your own logs to refine them.")


if __name__ == "__main__":
    main()
