#!/usr/bin/env python3
"""
Verify what was actually SENT to the API in an eval log, not what the header says.

The log header records the requested config, and a provider can silently drop parameters:
inspect 0.3.246 routed gpt-6-astra through Chat Completions and dropped the reasoning params
while the header still said reasoning_effort=high. This reads the recorded request/response
payloads for the first few samples and checks:

  candidate  Responses API used; reasoning.effort / reasoning.summary as expected; effort
             echoed back by the API; reasoning summary stored in the log
  grader     input item shape, and that the candidate's readable summary text is NOT in the
             grader's request (the grader must not see the reasoning summary)

    python src/verify_requests.py LOG.eval --candidate openai/gpt-6-astra \\
        --expect-effort high --expect-summary detailed
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

warnings.filterwarnings("ignore")
from inspect_ai._util.content import ContentReasoning
from inspect_ai.log import read_eval_log_sample, read_eval_log_sample_summaries

GRADER = "openai/gpt-5"


def summary_text(sample) -> str:
    for m in sample.messages:
        if m.role == "assistant" and isinstance(m.content, list):
            for c in m.content:
                if isinstance(c, ContentReasoning) and c.summary:
                    return c.summary
    return ""


def model_events(sample, candidate: str):
    """First *successful* candidate call, then the first grader call after it.

    Failed candidate attempts (API error, later retried) are skipped and counted, so the
    check reads the call that actually produced the answer. Handles candidate == grader.
    """
    cand = grad = None
    retried = 0
    for ev in sample.events:
        if getattr(ev, "event", None) != "model" or not ev.call:
            continue
        resp = ev.call.response if isinstance(ev.call.response, dict) else {}
        if cand is None and str(ev.model) == candidate:
            if getattr(ev, "error", None) or "output" not in resp:
                retried += 1
                continue
            cand = ev
        elif cand is not None and grad is None and str(ev.model) == GRADER:
            grad = ev
    return cand, grad, retried


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--candidate", required=True, help="e.g. openai/gpt-6-astra")
    ap.add_argument("--expect-effort")
    ap.add_argument("--expect-summary")
    ap.add_argument("-n", type=int, default=5, help="samples to check")
    a = ap.parse_args()

    fails: list[str] = []
    with_summary = checked = 0
    for sm in read_eval_log_sample_summaries(a.log)[: a.n]:
        s = read_eval_log_sample(a.log, sm.id, sm.epoch, resolve_attachments=True)
        checked += 1
        summ = summary_text(s)
        with_summary += bool(summ)
        cand, grad, retried = model_events(s, a.candidate)
        tag = f"{sm.id[:8]}/e{sm.epoch}"
        if cand is None:
            fails.append(f"{tag}: no candidate call recorded")
            continue

        req = cand.call.request or {}
        resp = cand.call.response if isinstance(cand.call.response, dict) else {}
        api = "responses" if "input" in req else ("chat" if "messages" in req else "?")
        sent = req.get("reasoning") or {}
        echoed = (resp.get("reasoning") or {}).get("effort")
        usage = cand.output.usage if cand.output else None
        rtok = getattr(usage, "reasoning_tokens", None) if usage else None

        g_items, leak = [], None
        if grad is not None:
            greq = grad.call.request or {}
            inp = greq.get("input", [])
            g_items = [x.get("type") for x in inp] if isinstance(inp, list) else []
            probe = summ.strip()[30:90] if len(summ.strip()) > 90 else summ.strip()
            leak = bool(probe) and probe in json.dumps(greq)

        print(f"{tag}  retried={retried} api={api:9s} sent={sent}  echoed_effort={echoed}  reasoning_tokens={rtok}  "
              f"summary_chars={len(summ)}  grader_items={g_items}  summary_in_grader={leak}")

        if api != "responses":
            fails.append(f"{tag}: candidate used {api} API, not Responses")
        if a.expect_effort and sent.get("effort") != a.expect_effort:
            fails.append(f"{tag}: effort sent={sent.get('effort')!r}, expected {a.expect_effort!r}")
        if a.expect_effort and echoed and echoed != a.expect_effort:
            fails.append(f"{tag}: API echoed effort={echoed!r}")
        if a.expect_summary and sent.get("summary") != a.expect_summary:
            fails.append(f"{tag}: summary sent={sent.get('summary')!r}, expected {a.expect_summary!r}")
        if grad is None:
            fails.append(f"{tag}: no grader call recorded")
        elif leak:
            fails.append(f"{tag}: candidate's reasoning summary text appears in the grader request")

    print(f"\nreasoning summaries stored: {with_summary}/{checked} samples")
    if fails:
        print("FAIL")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
