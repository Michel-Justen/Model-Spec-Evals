#!/usr/bin/env bash
# Reproducible setup for the Model Spec adherence pilot.
# Stands up a Python 3.11 env, clones OpenAI's harness + dataset (pinned), installs deps.
# Idempotent: safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

ENV_NAME="msadh"
# Pin vendor repos for reproducibility. Update these SHAs deliberately.
EVALS_REPO="https://github.com/openai/model_spec_evals.git"
DATASET_REPO="https://github.com/openai/model_spec_dataset.git"

echo "==> Python env ($ENV_NAME, 3.11)"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda env list | grep -q "^$ENV_NAME " || conda create -y -n "$ENV_NAME" python=3.11
  PY="$(conda info --base)/envs/$ENV_NAME/bin/python"
else
  echo "conda not found; falling back to python3 venv (must be 3.10+)"
  python3 -m venv .venv && PY=".venv/bin/python"
fi

echo "==> Clone vendor repos"
mkdir -p vendor
[ -d vendor/model_spec_evals ]   || git clone --depth 1 "$EVALS_REPO"   vendor/model_spec_evals
[ -d vendor/model_spec_dataset ] || git clone --depth 1 "$DATASET_REPO" vendor/model_spec_dataset

echo "==> Install harness + tooling"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r vendor/model_spec_evals/requirements.txt
"$PY" -m pip install --quiet pyyaml matplotlib

echo "==> Validate dataset (no API cost)"
"$PY" - <<'PYEOF'
import sys; sys.path.insert(0, "vendor/model_spec_evals/src")
from model_spec_evals.datasets import model_spec_eval_dataset
ds = model_spec_eval_dataset("vendor/model_spec_dataset/dataset")
assert len(ds) == 587, f"expected 587 runnable prompts, got {len(ds)}"
print(f"OK: {len(ds)} runnable prompts loaded.")
PYEOF

echo
echo "Done. Next:"
echo "  export OPENAI_API_KEY=...           # your key"
echo "  $PY src/check_models.py             # confirm model ids are reachable"
echo "  $PY src/run_eval.py --model openai/gpt-4o-mini --preset smoke   # ~pennies plumbing test"
