"""Regenerate the four charts embedded in README.md.

Every threshold, filter, attribute list and metric used here comes from the
shared `nba2k` package, which the notebooks import too -- this script used to
be a copy-paste fork of notebook logic, which meant a methodology fix had to be
applied twice and the README could silently disagree with the analysis.

Usage:
    python scripts/make_readme_charts.py
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nba2k import (  # noqa: E402
    MIN_MINUTES,
    cluster_profile,
    fit_clusters,
    load_merged,
    performance_sample,
    project_pca,
    rating_residuals,
    value_metrics,
    value_sample,
)
from nba2k.constants import CLUSTER_NAMES  # noqa: E402
from nba2k.metrics import minutes_sensitivity  # noqa: E402

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 150

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images"
)
N_LABEL = 8

df = load_merged()
print(f"Loaded {len(df):,} players")


# ---------------------------------------------------------------------------
# Chart 1: NB02 - Overall rating vs real 2025-26 PIE, with the fitted line and
# the 95% prediction interval the over/under-rated chart is judged against.
# ---------------------------------------------------------------------------
perf = performance_sample(df)
res = rating_residuals(perf, x="overall", y="nba_pie")

pearson_r, _ = stats.pearsonr(res["overall"], res["nba_pie"])
spearman_r, _ = stats.spearmanr(res["overall"], res["nba_pie"])

sens = minutes_sensitivity(performance_sample(df, min_minutes=0))
sens_note = "  ".join(
    f"{int(row.min_minutes)}min: r={row.pearson_r:.2f} (n={int(row.n)})"
    for row in sens.itertuples()
)

fig, ax = plt.subplots(figsize=(9, 6))
ordered = res.sort_values("overall")
ax.fill_between(
    ordered["overall"], ordered["pi_low"], ordered["pi_high"],
    color="#d62728", alpha=0.10, lw=0, label="95% prediction interval",
)
ax.scatter(
    res["overall"], res["nba_pie"],
    alpha=0.5, s=35, color="#1f77b4", edgecolor="white", linewidth=0.4,
)
ax.plot(ordered["overall"], ordered["fitted"], color="#d62728", lw=2, label="OLS fit")
ax.set_title(
    f"NBA 2K26 Overall vs Real PIE (minutes >= {MIN_MINUTES}, n={len(res)})\n"
    f"Pearson r={pearson_r:.2f}, Spearman rho={spearman_r:.2f}"
)
ax.set_xlabel("NBA 2K26 Overall Rating")
ax.set_ylabel("2025-26 PIE (Player Impact Estimate)")
ax.legend(loc="upper left", fontsize=9)
ax.text(
    0.5, -0.16, f"Not threshold-shopped - {sens_note}",
    transform=ax.transAxes, ha="center", fontsize=7.5, color="#555555",
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_overall_vs_pie.png", bbox_inches="tight")
plt.close()
print("Saved 01_overall_vs_pie.png")


# ---------------------------------------------------------------------------
# Chart 2: NB03 - PCA projection of attribute-based k-means clusters. Cluster
# names come from centroid profiles, not from label indices.
# ---------------------------------------------------------------------------
cl, info = fit_clusters(df)
coords, pca = project_pca(info)
cl["pca1"], cl["pca2"] = coords[:, 0], coords[:, 1]
print("Cluster naming (label -> archetype):")
for label, name in sorted(info["mapping"].items()):
    print(f"  {label} -> {name}")

fig, ax = plt.subplots(figsize=(10, 7.5))
sns.scatterplot(
    data=cl, x="pca1", y="pca2", hue="cluster_label", hue_order=CLUSTER_NAMES,
    palette="tab10", alpha=0.75, s=55, edgecolor="white", linewidth=0.4, ax=ax,
)
ax.set_title("PCA Projection of NBA 2K26 Attribute-Based Clusters (k=7)")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
ax.legend(title="Data-driven cluster", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_clusters_pca.png")
plt.close()
print("Saved 02_clusters_pca.png")
print(cluster_profile(cl, info)["count"].to_string())


# ---------------------------------------------------------------------------
# Chart 3: NB04 - Moneyball. Production is PIE x minutes (an impact volume
# comparable to a season salary), not PIE alone.
# ---------------------------------------------------------------------------
val = value_metrics(value_sample(df))

fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(
    val["salary_usd"] / 1e6, val["pie_minutes"], c=val["value_score"],
    cmap="RdYlGn", s=45, alpha=0.85, edgecolor="white", linewidth=0.3,
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("2025-26 Salary ($M, log scale)")
ax.set_ylabel("Season impact volume: PIE x minutes (log scale)")
ax.set_title(
    "Real Production Volume vs. Real Salary\n"
    "(green = more production per dollar than the league fit predicts)"
)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Value score (-studentized residual of log salary on log production)")

for _, row in val.nlargest(6, "value_score").iterrows():
    ax.annotate(row["name"], (row["salary_usd"] / 1e6, row["pie_minutes"]), fontsize=8,
                xytext=(5, 5), textcoords="offset points")
for _, row in val.nsmallest(6, "value_score").iterrows():
    ax.annotate(row["name"], (row["salary_usd"] / 1e6, row["pie_minutes"]), fontsize=8,
                xytext=(5, -10), textcoords="offset points")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_moneyball_value.png")
plt.close()
print("Saved 03_moneyball_value.png")
print("Best value:", ", ".join(val.nlargest(N_LABEL, "value_score")["name"]))
print("Worst value:", ", ".join(val.nsmallest(N_LABEL, "value_score")["name"]))


# ---------------------------------------------------------------------------
# Chart 4: NB02 - Most over/under-rated players, ranked by studentized residual
# from the PIE-on-overall fit rather than by a z-score difference.
# ---------------------------------------------------------------------------
top_over = res.nsmallest(N_LABEL, "studentized")
top_under = res.nlargest(N_LABEL, "studentized")
combined = pd.concat([top_over, top_under]).sort_values("studentized")

fig, ax = plt.subplots(figsize=(9.5, 7))
colors = ["#d62728" if s < 0 else "#2ca02c" for s in combined["studentized"]]
bars = ax.barh(combined["name"], combined["studentized"], color=colors)
for bar, outside in zip(bars, combined["outside_pi"]):
    if outside:
        bar.set_edgecolor("black")
        bar.set_linewidth(1.1)
ax.axvline(0, color="black", lw=0.8)
ax.axvline(-2, color="grey", lw=0.7, ls="--")
ax.axvline(2, color="grey", lw=0.7, ls="--")
ax.set_xlabel("Studentized residual of real PIE on 2K26 Overall")
ax.set_title(
    "Most Over- and Under-Rated Players by NBA 2K26\n"
    f"(vs. real 2025-26 PIE, {MIN_MINUTES}+ minutes; green = under-rated, "
    "red = over-rated;\nblack outline = outside the 95% prediction interval)"
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_over_under_rated.png")
plt.close()
print("Saved 04_over_under_rated.png")
print("Over-rated:", ", ".join(top_over["name"]))
print("Under-rated:", ", ".join(top_under["name"]))
print(f"{int(res['outside_pi'].sum())} of {len(res)} players fall outside the 95% PI")

print("Done.")
