"""Reproduce every number in the paper from results/episodes.jsonl.

Usage:  python analysis/analyze.py [results/episodes.jsonl]
"""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, binomtest

path = sys.argv[1] if len(sys.argv) > 1 else "results/episodes.jsonl"
df = pd.DataFrame([json.loads(l) for l in open(path) if l.strip()])
df["model"] = df.model.replace({"gemma4-e2b-it": "E2B", "gemma4-e4b-it-nf4": "E4B"})
df["crashed"] = df.error.notna()
df["t_view"] = df.first_gold_view.fillna(df.max_steps + 1).astype(float)  # censored at 21
ORDER = ["none", "tree", "repomap", "graph"]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def mcnemar(b, c):
    return 1.0 if b + c == 0 else binomtest(b, b + c).pvalue


def holm(ps):
    order = np.argsort(ps); out = np.empty(len(ps)); run = 0
    for i, j in enumerate(order):
        run = max(run, min(1, (len(ps) - i) * ps[j])); out[j] = run
    return out


def compare(sub, base):
    j = sub.set_index("task_id").join(base.set_index("task_id"), rsuffix="_b", how="inner")
    b = int((j.resolved & ~j.resolved_b).sum()); c = int((~j.resolved & j.resolved_b).sum())
    return dict(n_pairs=len(j), d_view=(j.t_view - j.t_view_b).mean(),
                p_wilcoxon=wilcoxon(j.t_view, j.t_view_b, zero_method="zsplit").pvalue,
                wins=b, losses=c, p_mcnemar=mcnemar(b, c))


def table(data, label):
    rows = []
    for m in ["E2B", "E4B"]:
        base = data[(data.model == m) & (data.condition == "none")]
        for cond in ORDER:
            g = data[(data.model == m) & (data.condition == cond)]
            if g.empty: continue
            k, n = int(g.resolved.sum()), len(g); p, lo, hi = wilson(k, n)
            r = dict(model=m, condition=cond, n=n, t_first_view=g.t_view.mean(),
                     viewed_gold=g.viewed_gold_file.mean(), edited_gold=g.edited_gold_file.mean(),
                     p_resolve_given_view=g[g.viewed_gold_file].resolved.mean(),
                     resolved=k, rate=p, ci_low=lo, ci_high=hi, crashed=int(g.crashed.sum()))
            if cond != "none": r.update(compare(g, base))
            rows.append(r)
    t = pd.DataFrame(rows)
    t["p_wilcoxon_holm"] = np.nan
    for m in t.model.unique():
        idx = t[(t.model == m) & t.p_wilcoxon.notna()].index
        t.loc[idx, "p_wilcoxon_holm"] = holm(t.loc[idx, "p_wilcoxon"].values)
    print(f"\n=== {label} ===")
    print(t.round(4).to_string(index=False))
    return t


main = table(df, "Main analysis (crashed episodes count as unresolved)")
main.to_csv("results/summary.csv", index=False)
table(df[~df.crashed], "Sensitivity analysis (crashed episodes excluded)")

print("\n=== Scale: E4B vs E2B on the same tasks and condition ===")
tot_b = tot_c = 0
for cond in ["none", "tree", "graph"]:
    a = df[(df.model == "E4B") & (df.condition == cond)]
    e = df[(df.model == "E2B") & (df.condition == cond)]
    j = a.set_index("task_id").join(e.set_index("task_id"), rsuffix="_s", how="inner")
    b = int((j.resolved & ~j.resolved_s).sum()); c = int((~j.resolved & j.resolved_s).sum())
    tot_b += b; tot_c += c
    print(f"{cond:6s} E4B={int(j.resolved.sum()):2d} E2B={int(j.resolved_s.sum()):2d} "
          f"E4B-only={b:2d} E2B-only={c:2d} p={mcnemar(b, c):.4f}")
print(f"pooled E4B-only={tot_b} E2B-only={tot_c} p={mcnemar(tot_b, tot_c):.2e}")
