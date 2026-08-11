"""Attribute-space clustering, with cluster *names* derived from centroids.

The bug this module exists to fix: both notebook 03 and the README chart script
mapped cluster names onto k-means label *indices* with a hardcoded dict
(``{0: "Elite Two-Way Superstars", 1: "3-and-D Connectors", ...}``). k-means
label indices are arbitrary -- they depend on centroid initialisation order,
which depends on the scikit-learn version, the BLAS implementation and the
input row order, none of which are pinned by ``random_state`` alone. Any of
those changing permutes the labels, and every chart and takeaway silently
becomes wrong while still looking completely plausible.

Names are therefore assigned by *matching centroid profiles*: each archetype is
defined by a signature over the standardized centroid (what makes a
rim-running big a rim-running big is a high offensive-rebound/block/standing-dunk
profile and a floor-level three-point shot, not the number 6), the
name-to-cluster assignment is solved as a bijective assignment problem, and the
result is checked against independent assertions before being returned.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .constants import ATTRS, CLUSTER_NAMES, K_CLUSTERS, RANDOM_STATE


def _mean_of(centroid, cols):
    return float(np.mean([centroid[c] for c in cols]))


def _level(centroid):
    """How elevated the whole profile is -- an "overall goodness" proxy."""
    return float(np.mean(list(centroid.values())))


#: Each archetype scores every standardized centroid; higher = better fit.
#: These read as basketball descriptions, not as indices, which is the point.
CLUSTER_SIGNATURES = {
    "Elite Two-Way Superstars": lambda c: 3.0 * _level(c),
    "3-and-D Connectors": lambda c: (
        _mean_of(c, ["perimeter_defense", "steal", "defensive_consistency",
                     "help_defense_iq", "pass_perception"])
        - _mean_of(c, ["post_control", "post_hook"])
    ),
    "Do-It-All Forwards": lambda c: -float(np.mean(np.abs(list(c.values())))),
    "Shot-Creating Lead Guards": lambda c: (
        _mean_of(c, ["ball_handle", "speed_with_ball", "pass_vision", "pass_iq",
                     "layup", "draw_foul"])
        - _mean_of(c, ["interior_defense", "offensive_rebound"])
    ),
    "Skilled Post Bigs": lambda c: (
        _mean_of(c, ["post_control", "post_hook", "post_fade", "strength",
                     "standing_dunk"])
        - _mean_of(c, ["speed_with_ball", "agility"])
    ),
    "Movement Shooters/Bench Guards": lambda c: (
        _mean_of(c, ["three_point_shot", "free_throw", "speed"])
        - _mean_of(c, ["interior_defense", "post_hook", "offensive_rebound",
                       "defensive_rebound", "block"])
        - _level(c)
    ),
    "Rim-Running Bigs": lambda c: (
        _mean_of(c, ["offensive_rebound", "block", "standing_dunk",
                     "defensive_rebound", "interior_defense"])
        - _mean_of(c, ["three_point_shot", "ball_handle", "pass_vision"])
    ),
}

assert set(CLUSTER_SIGNATURES) == set(CLUSTER_NAMES), (
    "CLUSTER_NAMES and CLUSTER_SIGNATURES must describe the same archetypes"
)


def standardized_centroids(kmeans, attrs=ATTRS):
    """Centroids as a DataFrame in standardized units, indexed by label."""
    return pd.DataFrame(kmeans.cluster_centers_, columns=list(attrs))


def assign_cluster_names(centroids, names=None):
    """Map k-means label -> archetype name by centroid profile, not by index.

    Every archetype signature scores every centroid; the scores are
    standardized per archetype so that signatures with different natural scales
    compete fairly, and the globally best one-to-one assignment is solved with
    the Hungarian algorithm. This guarantees a bijection (no two clusters get
    the same name, no name goes unused) regardless of how the labels came out.

    Returns ``(mapping, score_frame)`` where ``mapping`` is ``{label: name}``.
    """
    names = list(names or CLUSTER_NAMES)
    labels = list(centroids.index)
    if len(names) != len(labels):
        raise ValueError(
            f"{len(names)} archetype names for {len(labels)} clusters -- "
            "the naming scheme only makes sense for the k it was written for"
        )

    raw = pd.DataFrame(
        {
            name: [CLUSTER_SIGNATURES[name](centroids.loc[label].to_dict())
                   for label in labels]
            for name in names
        },
        index=labels,
    )
    # Standardize each archetype's column so no single signature's scale
    # dominates the assignment objective.
    scores = raw.apply(lambda col: (col - col.mean()) / (col.std(ddof=0) or 1.0))

    row_ind, col_ind = linear_sum_assignment(-scores.values)
    mapping = {labels[r]: names[c] for r, c in zip(row_ind, col_ind)}
    return mapping, scores


def check_cluster_mapping(centroids, mapping):
    """Assert the assignment agrees with independent, basketball-legible facts.

    The assignment problem always returns *some* bijection, so it cannot fail
    loudly on its own. These checks can, and are the actual safety net: if the
    attribute data or k changes enough that these no longer hold, the run stops
    instead of publishing a confidently mislabelled chart.
    """
    by_name = {name: centroids.loc[label] for label, name in mapping.items()}
    problems = []

    def rank(name, attr, want, among=None):
        pool = among or list(by_name)
        values = {n: by_name[n][attr] for n in pool}
        winner = (max if want == "max" else min)(values, key=values.get)
        if winner != name:
            problems.append(
                f"expected {name!r} to have the {want} {attr} among {pool}, "
                f"got {winner!r}"
            )

    rank("Rim-Running Bigs", "offensive_rebound", "max")
    rank("Rim-Running Bigs", "three_point_shot", "min")
    rank("Shot-Creating Lead Guards", "ball_handle", "max",
         among=[n for n in by_name if n != "Elite Two-Way Superstars"])
    rank("Skilled Post Bigs", "post_hook", "max",
         among=[n for n in by_name if n != "Elite Two-Way Superstars"])
    rank("3-and-D Connectors", "perimeter_defense", "max",
         among=[n for n in by_name if n != "Elite Two-Way Superstars"])

    elite = by_name["Elite Two-Way Superstars"]
    elite_level = float(elite.mean())
    for name, centroid in by_name.items():
        if name != "Elite Two-Way Superstars" and float(centroid.mean()) >= elite_level:
            problems.append(
                f"{name!r} has a higher mean attribute profile "
                f"({float(centroid.mean()):.2f}) than 'Elite Two-Way Superstars' "
                f"({elite_level:.2f})"
            )

    if problems:
        raise AssertionError(
            "Cluster naming failed its sanity checks -- the archetype signatures "
            "no longer describe this clustering, so the names would be wrong:\n  "
            + "\n  ".join(problems)
        )
    return True


def fit_clusters(df, attrs=ATTRS, k=K_CLUSTERS, random_state=RANDOM_STATE,
                 check=True, sort_key="slug", n_init=50):
    """Standardize `attrs`, fit k-means, and attach profile-derived names.

    Two determinism guards beyond `random_state`, both of which were needed in
    practice: rows are sorted by `sort_key` first, because k-means++ draws its
    seeds from the data in row order and simply shuffling the input CSV was
    enough to land in a different local optimum (verified -- with the sort in
    place, two shuffles of the same data give byte-identical labels); and
    `n_init` is raised well above scikit-learn's default so the reported
    solution is less dependent on any single initialisation.

    Returns ``(frame, info)``. ``frame`` is the complete-attribute subset of
    `df` with `cluster` (the arbitrary integer label) and `cluster_label` (the
    stable archetype name) columns. ``info`` carries the fitted `scaler`,
    `kmeans`, the standardized `centroids`, the `mapping` and the signature
    `scores`.
    """
    frame = df.dropna(subset=list(attrs)).copy()
    if sort_key and sort_key in frame.columns:
        frame = frame.sort_values(sort_key).copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(frame[list(attrs)].values)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)
    frame["cluster"] = kmeans.labels_

    centroids = standardized_centroids(kmeans, attrs)
    mapping, scores = assign_cluster_names(centroids)
    if check:
        check_cluster_mapping(centroids, mapping)
    frame["cluster_label"] = frame["cluster"].map(mapping)

    info = {
        "scaler": scaler,
        "kmeans": kmeans,
        "X": X,
        "centroids": centroids,
        "mapping": mapping,
        "signature_scores": scores,
    }
    return frame, info


def cluster_profile(frame, info, attrs=ATTRS):
    """Mean attribute profile per named cluster, in raw (unstandardized) units."""
    raw = pd.DataFrame(
        info["scaler"].inverse_transform(info["kmeans"].cluster_centers_),
        columns=list(attrs),
    )
    raw.index = [info["mapping"][label] for label in raw.index]
    raw["count"] = frame["cluster_label"].value_counts().reindex(raw.index).values
    return raw


def project_pca(info, n_components=2, random_state=RANDOM_STATE):
    """2D PCA projection of the standardized attribute space."""
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(info["X"])
    return coords, pca
