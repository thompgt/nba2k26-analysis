"""Shared analysis code for the NBA 2K26 ratings project.

`scripts/make_readme_charts.py` used to be a copy-paste fork of the notebooks:
the attribute list, the minutes threshold, the match-score filters, the z-scoring
and the cluster names all appeared verbatim in both places, so any methodology
fix had to be made twice and the README charts could silently drift away from
the analysis they claim to summarise. Everything shared now lives here and is
imported by both the notebooks and the chart script, and is unit-tested under
`tests/`.
"""

from .constants import (
    ATTRS,
    CATEGORY_COLS,
    CLUSTER_NAMES,
    K_CLUSTERS,
    MATCH_SCORE_MIN,
    MIN_MINUTES,
    RANDOM_STATE,
    REFERENCE_DATE,
)
from .clustering import cluster_profile, fit_clusters, project_pca
from .data import (
    add_age,
    load_merged,
    performance_sample,
    salary_sample,
    value_sample,
)
from .metrics import (
    minutes_sensitivity,
    rating_residuals,
    value_metrics,
    zscore,
)

__all__ = [
    "ATTRS",
    "CATEGORY_COLS",
    "CLUSTER_NAMES",
    "K_CLUSTERS",
    "MATCH_SCORE_MIN",
    "MIN_MINUTES",
    "RANDOM_STATE",
    "REFERENCE_DATE",
    "add_age",
    "cluster_profile",
    "fit_clusters",
    "load_merged",
    "minutes_sensitivity",
    "performance_sample",
    "project_pca",
    "rating_residuals",
    "salary_sample",
    "value_metrics",
    "value_sample",
    "zscore",
]
