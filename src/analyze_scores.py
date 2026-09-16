"""Cross-model scores, fixed-benchmark CIs and paired gaps from every .eval log per model folder.

Answers are pooled per prompt id across all .eval files in a model's folder (subfolders such as the
medium-effort Astra pilot are not read). Filter handling: an answer is "filter-blocked" when the API
returned stop_reason content_filter (cyber_policy) and the grader scored OpenAI's boilerplate;
bio_policy blocks come back as errors and are never scored.

Usage: python src/analyze_scores.py [--per-prompt-out results/per_prompt_scores.json] | tee results/<name>.txt
Run with the msadh-astra env (inspect 0.3.263) and never while an eval is running (memory/swap).
"""
import argparse, glob, json, math, statistics as st, warnings, collections
warnings.filterwarnings("ignore")
from inspect_ai.log import read_eval_log_samples

ap = argparse.ArgumentParser()
ap.add_argument("--per-prompt-out", default="results/per_prompt_scores.json")
ap.add_argument("--logs-root", default="runs",
                help="directory holding one folder of .eval logs per model (default: runs)")
args = ap.parse_args()

B = args.logs_root
MODELS = {"GPT-5 Thinking": "anchor-gpt5-thinking", "GPT-5.4": "gpt54", "GPT-5.5": "gpt55", "GPT-5.6 Sol": "sol", "GPT-6 Astra": "astra"}
def stop_of(s):
    try: return s.output.stop_reason if s.output else None
    except Exception: return "no-output"
data, integ = {}, {}
for name, folder in MODELS.items():
    d = collections.defaultdict(list); crit = collections.Counter(); noext = []; noout = []; per_file = {}
    for p in sorted(glob.glob(f"{B}/{folder}/*.eval")):
        n = 0
        for s in read_eval_log_samples(p, all_samples_required=False):
            sc = (s.scores or {}).get("model_graded_spec_section_compliance")
            if sc is None or not isinstance(sc.value, (int, float)): continue
            stop = stop_of(s)
            blocked = stop == "content_filter"          # the API filter's boilerplate was graded as the answer
            if stop == "no-output": noout.append(s.id[:8])
            crit[len((sc.metadata or {}).get("critiques", []))] += 1
            if (sc.explanation or "").startswith("Could not extract any critiques"): noext.append(s.id[:8])
            d[s.id].append((float(sc.value), blocked)); n += 1
        per_file[p.rsplit("_", 1)[-1][:8]] = n
    data[name] = d; integ[name] = (crit, noext, noout)
    apd = collections.Counter(len(v) for v in d.values())
    print(f"loaded {name}: {sum(len(v) for v in d.values())} scored answers, {len(d)} prompts | "
          f"per file {per_file} | answers-per-prompt {dict(sorted(apd.items()))}", flush=True)
json.dump({m: {p: v for p, v in d.items()} for m, d in data.items()}, open(args.per_prompt_out, "w"))
print(f"saved {args.per_prompt_out} (prompt ids + per-answer scores + filter flag only)")

def rates(d, excl=False):
    out = {}
    for pid, v in d.items():
        vals = [x for x, b in v if not (excl and b)]
        if vals: out[pid] = sum(vals) / len(vals)
    return out
def ci_fixed(d):
    var = sum(st.variance([x for x, _ in v]) / len(v) for v in d.values() if len(v) > 1)
    return 1.96 * math.sqrt(var) / len(d)
def blocked_prompts(d): return {p for p, v in d.items() if any(b for _, b in v)}
def paired(x, y, drop=frozenset()):
    ids = [p for p in x if p in y and p not in drop]
    diff = [x[p] - y[p] for p in ids]
    se = st.stdev(diff) / math.sqrt(len(diff)); mu = st.mean(diff)
    return mu * 100, 1.96 * se * 100, math.erfc(abs(mu / se) / math.sqrt(2)), len(ids)

print("\n=== INTEGRITY ===")
for m, (crit, noext, noout) in integ.items():
    print(f"{m:15s} grades-per-answer distribution={dict(sorted(crit.items()))} | silent 0s ('Could not extract'): {len(noext)} {noext[:6]} | scored with no output: {len(noout)} {noout[:6]}")

print("\n=== SCORES (prompt-weighted) ===")
print(f"{'model':15s} {'as published':>13s} {'excl. filter':>13s} {'fixed-bench CI':>15s} | filter-blocked answers / prompts")
for m, d in data.items():
    a, e = rates(d), rates(d, True)
    nb = sum(b for v in d.values() for _, b in v)
    print(f"{m:15s} {sum(a.values())/len(a)*100:12.2f}% {sum(e.values())/len(e)*100:12.2f}% {ci_fixed(d)*100:+14.2f} | {nb} / {len(blocked_prompts(d))}")
for ref in ["GPT-6 Astra", "GPT-5 Thinking"]:
    print(f"\n=== PAIRED GAPS: {ref} minus other (positive = {ref} higher) ===")
    for m in MODELS:
        if m == ref: continue
        o = paired(rates(data[ref]), rates(data[m]))
        c = paired(rates(data[ref]), rates(data[m]), blocked_prompts(data[ref]) | blocked_prompts(data[m]))
        print(f"  vs {m:15s} as published: {o[0]:+.2f} ±{o[1]:.2f} p={o[2]:.1e} n={o[3]} | filter-blocked prompts dropped: {c[0]:+.2f} ±{c[1]:.2f} p={c[2]:.1e} n={c[3]}")
