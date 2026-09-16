"""Per-section scores, both ways (as published / filter-blocked answers excluded), from per_prompt_scores.json.

Run after src/analyze_scores.py. Usage: python src/section_scores.py [per_prompt_scores.json] | tee results/<name>.txt
"""
import json, csv, sys, statistics as st
data = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/per_prompt_scores.json"))
sec = {r["id"]: r["top_level_section"] for r in csv.DictReader(open("results/sol_vs_gpt5thinking_by_prompt.csv"))}
def rates(d, excl=False):
    out = {}
    for pid, v in d.items():
        vals = [x for x, b in v if not (excl and b)]
        if vals: out[pid] = sum(vals) / len(vals)
    return out
models = list(data)
sections = ["chain_of_command", "seek_truth", "best_work", "stay_in_bounds", "style"]
for excl in (False, True):
    print(f"\n=== SECTION SCORES ({'filter-blocked answers excluded' if excl else 'as published'}) ===")
    print(f"{'section':18s} {'n':>4s} " + " ".join(f"{m.replace('GPT-',''):>14s}" for m in models))
    tab = {}
    for s in sections:
        vals = []
        for m in models:
            r = rates(data[m], excl)
            xs = [v for p, v in r.items() if sec.get(p) == s]
            vals.append(st.mean(xs) * 100)
        tab[s] = dict(zip(models, vals))
        print(f"{s:18s} {sum(1 for p in sec.values() if p == s):4d} " + " ".join(f"{v:13.1f}%" for v in vals))
    st_ = tab["seek_truth"]
    print(f"  'Seek the truth' drop vs GPT-5 Thinking: Sol {st_['GPT-5 Thinking']-st_['GPT-5.6 Sol']:+.1f} | 5.5 {st_['GPT-5 Thinking']-st_['GPT-5.5']:+.1f} | 5.4 {st_['GPT-5 Thinking']-st_['GPT-5.4']:+.1f} | Astra {st_['GPT-5 Thinking']-st_['GPT-6 Astra']:+.1f}")
# The two sensitive prompts: compliance rates only, never content.
print("\n=== THE TWO SEXUAL-CONTENT EXAMPLE PROMPTS ===")
for pre in ["03bb55a3", "463fb4a3"]:
    for m in ["GPT-5 Thinking", "GPT-5.6 Sol", "GPT-6 Astra"]:
        hit = [(p, v) for p, v in data[m].items() if p.startswith(pre)]
        for p, v in hit:
            print(f"  {pre} {m:15s} compliance {sum(x for x,_ in v)/len(v)*100:5.1f}% over {len(v)} answers | filter-blocked answers: {sum(b for _, b in v)}")
        if not hit: print(f"  {pre} {m:15s} no scored answers")
