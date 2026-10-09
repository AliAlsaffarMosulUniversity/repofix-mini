"""Figure 1: localization time vs. resolution rate, per model and condition."""
import pandas as pd, matplotlib.pyplot as plt
s = pd.read_csv("results/summary.csv")
COL = {"E2B": "#2a78d6", "E4B": "#eb6834"}
LAB = {"E2B": "Gemma 4 E2B (fp16)", "E4B": "Gemma 4 E4B (NF4)"}
order = ["none", "tree", "repomap", "graph"]; off = {"E2B": -0.12, "E4B": 0.12}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a85", "xtick.color": "#4a4a46", "ytick.color": "#4a4a46"})
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
for m, g in s.groupby("model"):
    g = g.set_index("condition").reindex([c for c in order if c in set(g.condition)])
    x = [order.index(c) + off[m] for c in g.index]
    a.plot(x, g.t_first_view, "o", ms=8, color=COL[m], mec="white", mew=1.5, label=LAB[m], zorder=3)
    b.errorbar(x, g.rate * 100, yerr=[(g.rate - g.ci_low) * 100, (g.ci_high - g.rate) * 100],
               fmt="o", ms=8, color=COL[m], mec="white", mew=1.5, elinewidth=2, capsize=0, label=LAB[m], zorder=3)
for ax, t, yl in [(a, "A. Localization: step of first view of the faulty file", "steps (lower = faster)"),
                  (b, "B. Repair: tasks resolved (95% Wilson CI)", "% resolved")]:
    ax.set_xticks(range(4)); ax.set_xticklabels(order); ax.set_title(t, loc="left", fontsize=10.5, color="#1a1a19")
    ax.set_ylabel(yl, color="#4a4a46"); ax.grid(axis="y", color="#e4e3dc", lw=0.8); ax.set_axisbelow(True)
a.set_ylim(0, 12); b.set_ylim(0, 32)
a.legend(frameon=False, loc="lower left")
fig.text(0.01, -0.03, "E4B was not run with repomap. 60 tasks per cell; crashed episodes counted as unresolved.",
         fontsize=8.5, color="#6b6b66")
fig.tight_layout(); fig.savefig("results/figure1.png", dpi=200, bbox_inches="tight")
