#!/usr/bin/env python3
"""
Where did GPT-5.6 Sol lose compliance vs GPT-5 Thinking? Ranks spec sub-sections by the
drop, then pulls concrete example prompts (the prompt, the rubric, Sol's response, and the
grader's reasoning for why Sol failed while GPT-5 Thinking passed).

Per-model extraction is cached to results/drill_<key>.json so the (slow) log read runs once.

    python src/drilldown.py
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parent.parent
SCORER = "model_graded_spec_section_compliance"
MODELS = {
    "sol": ["runs/sol"],
    "gpt5-thinking": ["runs/anchor-gpt5-thinking"],
}


def as_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        parts = []
        for m in x:
            role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else "")
            content = getattr(m, "content", None)
            if content is None and isinstance(m, dict):
                content = m.get("content")
            if isinstance(content, list):
                content = " ".join(
                    (getattr(c, "text", None) or (c.get("text", "") if isinstance(c, dict) else "") or "")
                    for c in content)
            parts.append(f"[{role}] {content}")
        return "\n".join(parts)
    return str(x)


def extract(key: str, dirs: list[str]) -> dict:
    cache = ROOT / "results" / f"drill_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    prompts: dict[str, dict] = {}
    for path in [p for d in dirs for p in glob.glob(str(ROOT / d / "*.eval"))]:
        log = read_eval_log(path)
        for s in (log.samples or []):
            sc = s.scores.get(SCORER) if s.scores else None
            if sc is None or not isinstance(sc.value, (int, float)):
                continue
            md = s.metadata or {}
            rec = prompts.setdefault(s.id, {
                "section": md.get("section_id"), "top": md.get("top_level_section"),
                "focus": md.get("focus_id"), "vals": [], "prompt": None, "target": None, "fail": None,
            })
            rec["vals"].append(float(sc.value))
            if rec["prompt"] is None:
                rec["prompt"] = as_text(s.input)[:800]
                rec["target"] = as_text(s.target)[:600]
            if float(sc.value) == 0.0 and rec["fail"] is None:
                resp = ""
                for m in reversed(s.messages or []):
                    if getattr(m, "role", None) == "assistant":
                        resp = as_text(getattr(m, "content", ""))
                        break
                rec["fail"] = {"critique": as_text(sc.explanation)[:700], "response": resp[:600]}
    cache.write_text(json.dumps(prompts))
    print(f"  cached {key}: {len(prompts)} prompts")
    return prompts


def rate(rec) -> float:
    return sum(rec["vals"]) / len(rec["vals"])


def main() -> None:
    sol = extract("sol", MODELS["sol"])
    gpt = extract("gpt5-thinking", MODELS["gpt5-thinking"])
    common = sorted(set(sol) & set(gpt))

    def agg(level: str, min_n: int):
        d = defaultdict(lambda: {"s": [], "g": []})
        for pid in common:
            d[sol[pid][level]]["s"].append(rate(sol[pid]))
            d[sol[pid][level]]["g"].append(rate(gpt[pid]))
        rows = [(k, sum(v["g"]) / len(v["g"]) - sum(v["s"]) / len(v["s"]),
                 sum(v["g"]) / len(v["g"]), sum(v["s"]) / len(v["s"]), len(v["s"]))
                for k, v in d.items() if len(v["s"]) >= min_n]
        return sorted(rows, key=lambda r: -r[1])

    print("\n=== Drop by top-level section (GPT-5 Thinking -> Sol) ===")
    print(f"{'section':<18}{'drop':>7}{'GPT5T':>8}{'Sol':>8}{'n':>5}")
    for k, drop, g, s, n in agg("top", 1):
        print(f"{k:<18}{drop*100:>6.1f}{g*100:>8.1f}{s*100:>8.1f}{n:>5}")

    print("\n=== Biggest drops by spec sub-section (section_id, n>=4) ===")
    print(f"{'section_id':<34}{'drop':>7}{'GPT5T':>8}{'Sol':>8}{'n':>5}")
    top_sections = agg("section", 4)[:10]
    for k, drop, g, s, n in top_sections:
        print(f"{str(k):<34}{drop*100:>6.1f}{g*100:>8.1f}{s*100:>8.1f}{n:>5}")

    print("\n\n########## EXAMPLE PROMPTS (Sol failed, GPT-5 Thinking passed) ##########")
    shown = 0
    for sec, drop, g, s, n in top_sections[:4]:
        # prompts in this sub-section where GPT-5 Thinking was compliant and Sol was not
        cands = [pid for pid in common if sol[pid]["section"] == sec
                 and rate(gpt[pid]) >= 0.75 and rate(sol[pid]) <= 0.25 and sol[pid]["fail"]]
        for pid in cands[:2]:
            r = sol[pid]
            print(f"\n----- section: {sec}  |  focus: {r['focus']}  |  "
                  f"GPT-5 Thinking {rate(gpt[pid])*100:.0f}% vs Sol {rate(sol[pid])*100:.0f}% -----")
            print(f"PROMPT:\n{r['prompt']}")
            print(f"\nRUBRIC (what compliance requires):\n{r['target']}")
            print(f"\nSOL'S RESPONSE:\n{r['fail']['response']}")
            print(f"\nGRADER — why Sol was non-compliant:\n{r['fail']['critique']}")
            shown += 1
    print(f"\n\n(shown {shown} examples)")


if __name__ == "__main__":
    main()
