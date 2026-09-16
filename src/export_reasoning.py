#!/usr/bin/env python3
"""
Export reasoning summaries + answers from .eval logs to JSONL, one row per answer, so a
monitor (e.g. for verbalized eval awareness) can run over them without touching inspect.

OpenAI never returns the raw chain of thought; what the logs hold is the API's reasoning
*summary* (written by a separate summarizer), and many answers get none. Rows without a
summary are still exported with has_summary=false so coverage can be reported honestly.

    python src/export_reasoning.py "run logs | July 2026 Model Spec Eval Independent Test/sol" \\
        --label gpt-5.6-sol --out results/reasoning/gpt-5.6-sol.jsonl

Output rows: label, model, prompt_id, epoch, metadata (section/focus fields from the dataset),
score (1 = compliant), grader_explanation, prompt, reasoning_summary, has_summary,
reasoning_redacted, reasoning_tokens, answer.

NOTE: rows contain full prompt/answer text, including sensitive prompts. Keep results/reasoning/
out of git (see .gitignore).
"""
from __future__ import annotations

import argparse
import glob
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from inspect_ai._util.content import ContentReasoning
from inspect_ai.log import read_eval_log_samples

SCORER = "model_graded_spec_section_compliance"


def resolver(sample):
    atts = getattr(sample, "attachments", None) or {}

    def res(text):
        if isinstance(text, str) and text.startswith("attachment://"):
            return atts.get(text.removeprefix("attachment://"), text)
        return text
    return res


def text_of(content, res) -> str:
    if isinstance(content, str):
        return res(content)
    parts = []
    for c in content or []:
        t = getattr(c, "text", None)
        if t:
            parts.append(res(t))
    return "\n".join(parts)


def row(sample, label: str, model: str) -> dict:
    res = resolver(sample)
    summary, redacted = "", False
    for m in sample.messages:
        if m.role == "assistant" and isinstance(m.content, list):
            for c in m.content:
                if isinstance(c, ContentReasoning):
                    redacted = redacted or bool(c.redacted)
                    s = res(c.summary) if c.summary else ("" if c.redacted else res(c.reasoning or ""))
                    if s:
                        summary = (summary + "\n\n" + s).strip()
    user_msgs = [m for m in sample.messages if m.role == "user"]
    prompt = text_of(user_msgs[-1].content, res) if user_msgs else ""
    answer = res(sample.output.completion) if sample.output and sample.output.completion else ""
    sc = (sample.scores or {}).get(SCORER)
    usage = (sample.model_usage or {}).get(model)
    return {
        "label": label,
        "model": model,
        "prompt_id": sample.id,
        "epoch": sample.epoch,
        "metadata": sample.metadata or {},
        "score": sc.value if sc is not None else None,
        "grader_explanation": res(sc.explanation) if sc is not None and sc.explanation else None,
        "prompt": prompt,
        "reasoning_summary": summary,
        "has_summary": bool(summary),
        "reasoning_redacted": redacted,
        "reasoning_tokens": getattr(usage, "reasoning_tokens", None) if usage else None,
        "answer": answer,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help=".eval files or directories of them")
    ap.add_argument("--label", required=True, help="short model label for the rows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="stop after N rows (for testing)")
    a = ap.parse_args()

    paths: list[str] = []
    for x in a.inputs:
        paths += sorted(glob.glob(str(Path(x) / "*.eval"))) if Path(x).is_dir() else [x]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = with_summary = 0
    with out.open("w") as fh:
        for p in paths:
            from inspect_ai.log import read_eval_log
            model = read_eval_log(p, header_only=True).eval.model
            for s in read_eval_log_samples(p, all_samples_required=False):
                r = row(s, a.label, model)
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
                with_summary += r["has_summary"]
                if a.limit and n >= a.limit:
                    break
            if a.limit and n >= a.limit:
                break
    pct = 100 * with_summary / n if n else 0
    print(f"wrote {out}: {n} rows, {with_summary} with a reasoning summary ({pct:.0f}%)")


if __name__ == "__main__":
    main()
