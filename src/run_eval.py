#!/usr/bin/env python3
"""
Launcher around `inspect eval` for the Model Spec suite.

Reads config/models.yaml for the grader and presets so you don't have to remember the
-T flags, and prints the command before running (with --dry-run it prints without spending).

    python src/run_eval.py --model openai/gpt-5.4 --preset pilot
    python src/run_eval.py --model openai/gpt-5.6-sol --preset moderate
    python src/run_eval.py --model openai/gpt-5.4 --preset smoke --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # so the inspect subprocess inherits OPENAI_API_KEY
CONFIG = yaml.safe_load((ROOT / "config" / "models.yaml").read_text())
EVALS_DIR = ROOT / "vendor" / "model_spec_evals"
DATASET_DIR = ROOT / "vendor" / "model_spec_dataset" / "dataset"
TASK = "src/model_spec_evals/tasks.py"
# use the inspect binary from this interpreter's env, so PATH doesn't matter
INSPECT = str(Path(sys.executable).parent / "inspect")


def stratified_ids(k: int, seed: int = 0) -> list[str]:
    """Deterministically pick k prompt ids from EACH top_level_section (balanced subset)."""
    import random
    from collections import defaultdict

    sys.path.insert(0, str(EVALS_DIR / "src"))
    from model_spec_evals.datasets import model_spec_eval_dataset

    ds = model_spec_eval_dataset(str(DATASET_DIR))
    buckets: dict[str, list[str]] = defaultdict(list)
    for s in ds:
        buckets[s.metadata["top_level_section"]].append(s.id)
    rng = random.Random(seed)
    ids: list[str] = []
    for sec in sorted(buckets):
        b = sorted(buckets[sec])
        rng.shuffle(b)
        ids += b[:k]
    return ids


def build_cmd(model, preset, epochs=None, grader_samples=None, limit=None,
              stratify=None, reasoning_effort=None) -> list[str]:
    p = CONFIG["presets"][preset]
    epochs = epochs or p["epochs"]
    grader_samples = grader_samples if grader_samples is not None else p["grader_samples"]
    limit = limit if limit is not None else p.get("limit")
    grader = CONFIG["grader"]

    cmd = [
        INSPECT, "eval", TASK,
        "--model", model,
        "--display", "rich",
        "--epochs", str(epochs),
        "-T", f"dataset_dir={DATASET_DIR}",
        "-T", f"grader_model={grader}",
        "-T", f"num_grader_samples={grader_samples}",
    ]
    if reasoning_effort:
        cmd += ["--reasoning-effort", reasoning_effort]
    if stratify:
        ids = stratified_ids(stratify)
        print(f"[stratify] {stratify}/section → {len(ids)} prompts")
        cmd += ["--sample-id", ",".join(ids)]  # overrides limit
    elif limit:
        cmd += ["--limit", str(limit)]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Inspect model id, e.g. openai/gpt-5-6-sol")
    ap.add_argument("--preset", default="smoke", choices=list(CONFIG["presets"]))
    ap.add_argument("--epochs", type=int, help="override preset")
    ap.add_argument("--grader-samples", type=int, help="override preset")
    ap.add_argument("--limit", type=int, help="override preset prompt cap")
    ap.add_argument("--stratify", type=int, help="pick N prompts per spec section (balanced subset)")
    ap.add_argument("--reasoning-effort", help="candidate reasoning effort: minimal|low|medium|high")
    ap.add_argument("--dry-run", action="store_true", help="print the command, spend nothing")
    args = ap.parse_args()

    cmd = build_cmd(args.model, args.preset, args.epochs, args.grader_samples, args.limit,
                    args.stratify, args.reasoning_effort)
    print("cd", EVALS_DIR)
    print(" ".join(cmd), "\n")
    if args.dry_run:
        print("[dry-run] not executing.")
        return
    if not DATASET_DIR.exists():
        sys.exit(f"Dataset not found at {DATASET_DIR}. Run ./setup.sh first.")
    raise SystemExit(subprocess.call(cmd, cwd=EVALS_DIR))


if __name__ == "__main__":
    main()
