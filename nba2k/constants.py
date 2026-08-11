"""Single source of truth for every threshold and column list in the analysis.

These were previously duplicated verbatim across the notebooks and
`scripts/make_readme_charts.py`. Each threshold is arbitrary in the sense that
no natural cut point exists; `nba2k.metrics.minutes_sensitivity` exists so the
headline correlation can be shown not to depend on the particular choice.
"""

import pandas as pd

#: The 35 granular 2K26 attributes used for clustering and nearest-neighbour work.
ATTRS = [
    "agility", "ball_handle", "block", "close_shot", "defensive_consistency",
    "defensive_rebound", "draw_foul", "driving_dunk", "free_throw", "hands",
    "help_defense_iq", "hustle", "interior_defense", "layup", "mid_range_shot",
    "offensive_consistency", "offensive_rebound", "overall_durability",
    "pass_accuracy", "pass_iq", "pass_perception", "pass_vision", "perimeter_defense",
    "post_control", "post_fade", "post_hook", "shot_iq", "speed", "speed_with_ball",
    "stamina", "standing_dunk", "steal", "strength", "three_point_shot", "vertical",
]

#: 2K's own six category rollups.
CATEGORY_COLS = [
    "cat_outside_scoring", "cat_athleticism", "cat_inside_scoring",
    "cat_playmaking", "cat_defense", "cat_rebounding",
]

#: Minimum minutes for a player to count as a rotation player. Roughly 6-7
#: minutes a night over a full season; below this, rate stats like PIE are
#: dominated by small-sample noise. See `minutes_sensitivity()`.
MIN_MINUTES = 500

#: Minimum rapidfuzz name-match score to trust a cross-source join.
MATCH_SCORE_MIN = 90

#: Pinned so clustering, PCA and any train/test split are reproducible.
RANDOM_STATE = 42

#: Number of k-means clusters (chosen for interpretability in notebook 03).
K_CLUSTERS = 7

#: Age is computed relative to the start of the 2025-26 season.
REFERENCE_DATE = pd.Timestamp("2025-10-01")

#: Prime pay-scale window: past the rookie wage scale, before veteran decline.
PRIME_AGE_MIN, PRIME_AGE_MAX = 25, 32

#: Canonical cluster names, in the order `nba2k.clustering.CLUSTER_SIGNATURES`
#: defines them. These are *names*, not label indices -- see
#: `nba2k.clustering.assign_cluster_names` for why that distinction matters.
CLUSTER_NAMES = [
    "Elite Two-Way Superstars",
    "3-and-D Connectors",
    "Do-It-All Forwards",
    "Shot-Creating Lead Guards",
    "Skilled Post Bigs",
    "Movement Shooters/Bench Guards",
    "Rim-Running Bigs",
]
