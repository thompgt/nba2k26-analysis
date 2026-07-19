"""
Regenerates the 3-4 charts embedded in README.md directly from the real
processed dataset (data/processed/players_merged.csv), using the same
methodology as the corresponding analysis notebooks. Not run as part of the
data pipeline -- this is a one-off/occasional utility to keep README images
in sync with the analysis.

Usage:
    python scripts/make_readme_charts.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 150

REFERENCE_DATE = pd.Timestamp("2025-10-01")
OUT_DIR = "images"

df = pd.read_csv("data/processed/players_merged.csv")
print(f"Loaded {len(df):,} players")


# ---------------------------------------------------------------------------
# Chart 1: NB02 - Overall rating vs real 2025-26 PIE (does the game rating
# track real on-court impact?)
# ---------------------------------------------------------------------------
perf = df[
    (df["stats_match_score"] >= 90) & df["nba_pie"].notna() & df["nba_min"].notna()
].copy()
MIN_MINUTES = 500
perf_reliable = perf[perf["nba_min"] >= MIN_MINUTES].copy()

pearson_r, pearson_p = stats.pearsonr(perf_reliable["overall"], perf_reliable["nba_pie"])
spearman_r, _ = stats.spearmanr(perf_reliable["overall"], perf_reliable["nba_pie"])

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(
    perf_reliable["overall"], perf_reliable["nba_pie"],
    alpha=0.5, s=35, color="#1f77b4", edgecolor="white", linewidth=0.4,
)
slope, intercept = np.polyfit(perf_reliable["overall"], perf_reliable["nba_pie"], 1)
xs = np.linspace(perf_reliable["overall"].min(), perf_reliable["overall"].max(), 100)
ax.plot(xs, slope * xs + intercept, color="#d62728", lw=2)
ax.set_title(
    f"NBA 2K26 Overall vs Real PIE (minutes >= {MIN_MINUTES}, n={len(perf_reliable)})\n"
    f"Pearson r={pearson_r:.2f}, Spearman rho={spearman_r:.2f}"
)
ax.set_xlabel("NBA 2K26 Overall Rating")
ax.set_ylabel("2025-26 PIE (Player Impact Estimate)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_overall_vs_pie.png")
plt.close()
print("Saved 01_overall_vs_pie.png")


# ---------------------------------------------------------------------------
# Chart 2: NB03 - PCA projection of attribute-based k-means clusters (data-
# driven archetypes vs 2K's designer-authored archetype labels)
# ---------------------------------------------------------------------------
ATTRS = [
    "agility", "ball_handle", "block", "close_shot", "defensive_consistency",
    "defensive_rebound", "draw_foul", "driving_dunk", "free_throw", "hands",
    "help_defense_iq", "hustle", "interior_defense", "layup", "mid_range_shot",
    "offensive_consistency", "offensive_rebound", "overall_durability",
    "pass_accuracy", "pass_iq", "pass_perception", "pass_vision", "perimeter_defense",
    "post_control", "post_fade", "post_hook", "shot_iq", "speed", "speed_with_ball",
    "stamina", "standing_dunk", "steal", "strength", "three_point_shot", "vertical",
]
RANDOM_STATE = 42
cl = df.dropna(subset=ATTRS).copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(cl[ATTRS].values)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
cl["pca1"], cl["pca2"] = X_pca[:, 0], X_pca[:, 1]

K = 7
kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
cl["cluster"] = kmeans.labels_

LABELS = {
    0: "Elite Two-Way Superstars", 1: "3-and-D Connectors", 2: "Do-It-All Forwards",
    3: "Shot-Creating Lead Guards", 4: "Skilled Post Bigs",
    5: "Movement Shooters/Bench Guards", 6: "Rim-Running Bigs",
}
cl["cluster_label"] = cl["cluster"].map(LABELS)

fig, ax = plt.subplots(figsize=(10, 7.5))
sns.scatterplot(
    data=cl, x="pca1", y="pca2", hue="cluster_label",
    hue_order=[LABELS[c] for c in range(K)],
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


# ---------------------------------------------------------------------------
# Chart 3: NB04 - Moneyball: real production (PIE) vs real salary
# ---------------------------------------------------------------------------
val = df[
    (df["salary_match_score"] >= 90) & df["salary_usd"].notna() & (df["salary_usd"] > 0)
    & (df["stats_match_score"] >= 90) & df["nba_pie"].notna()
    & (df["nba_min"] >= 500)
].copy()
val["log_salary"] = np.log10(val["salary_usd"])

def zscore(s):
    return (s - s.mean()) / s.std()

val["pie_z"] = zscore(val["nba_pie"])
val["salary_z"] = zscore(val["log_salary"])
val["value_gap"] = val["pie_z"] - val["salary_z"]

fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(
    val["salary_usd"] / 1e6, val["nba_pie"], c=val["value_gap"],
    cmap="RdYlGn", s=45, alpha=0.85, edgecolor="white", linewidth=0.3,
)
ax.set_xscale("log")
ax.set_xlabel("2025-26 Salary ($M, log scale)")
ax.set_ylabel("2025-26 PIE")
ax.set_title(
    "Real Production (PIE) vs. Real Salary\n"
    "(green = underpaid for production, red = overpaid)"
)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Value gap (production z - salary z)")

for _, row in val.nlargest(6, "value_gap").iterrows():
    ax.annotate(row["name"], (row["salary_usd"] / 1e6, row["nba_pie"]), fontsize=8,
                xytext=(5, 5), textcoords="offset points")
for _, row in val.nsmallest(6, "value_gap").iterrows():
    ax.annotate(row["name"], (row["salary_usd"] / 1e6, row["nba_pie"]), fontsize=8,
                xytext=(5, -10), textcoords="offset points")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_moneyball_value.png")
plt.close()
print("Saved 03_moneyball_value.png")


# ---------------------------------------------------------------------------
# Chart 4: NB02 - Most over/under-rated players (2K26 Overall vs real PIE gap)
# ---------------------------------------------------------------------------
def zscore2(s):
    return (s - s.mean()) / s.std()

perf_flag = perf_reliable.copy()
perf_flag["overall_z"] = zscore2(perf_flag["overall"])
perf_flag["pie_z"] = zscore2(perf_flag["nba_pie"])
perf_flag["gap"] = perf_flag["overall_z"] - perf_flag["pie_z"]

top_over = perf_flag.sort_values("gap", ascending=False).head(8)
top_under = perf_flag.sort_values("gap").head(8)
combined = pd.concat([top_under, top_over]).sort_values("gap")

fig, ax = plt.subplots(figsize=(9, 7))
colors = ["#2ca02c" if g < 0 else "#d62728" for g in combined["gap"]]
ax.barh(combined["name"], combined["gap"], color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Gap: standardized Overall - standardized real PIE")
ax.set_title(
    "Most Over- and Under-Rated Players by NBA 2K26\n"
    "(vs. real 2025-26 PIE, rotation players only, green = under-rated, red = over-rated)"
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_over_under_rated.png")
plt.close()
print("Saved 04_over_under_rated.png")

print("Done.")
