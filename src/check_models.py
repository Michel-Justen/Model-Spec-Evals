#!/usr/bin/env python3
"""
Pre-flight: confirm the candidate + grader model ids in config/models.yaml are actually
reachable on your OpenAI account BEFORE spending on a full run. Sends one 1-token ping
per model (costs ~nothing). Requires OPENAI_API_KEY.

    python src/check_models.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
CONFIG = yaml.safe_load((ROOT / "config" / "models.yaml").read_text())


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set. Export it and retry.")
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai package missing — run inside the msadh env.")

    client = OpenAI()
    ids = [CONFIG["grader"]] + list(CONFIG["candidates"])
    print(f"Checking {len(ids)} model id(s) via the models endpoint (free, no generation)...\n")
    ok, bad = [], []
    for full in ids:
        api_id = full.split("/", 1)[1] if "/" in full else full
        try:
            client.models.retrieve(api_id)  # 404/permission error if not accessible
            print(f"  ✅ {full}")
            ok.append(full)
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0][:120]
            print(f"  ❌ {full}  —  {msg}")
            bad.append((full, msg))
    print(f"\nreachable: {len(ok)}/{len(ids)}")
    if bad:
        print("Fix the ids in config/models.yaml (and config/prices.yaml) for the ❌ rows.")
        sys.exit(1)


if __name__ == "__main__":
    main()
