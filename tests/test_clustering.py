"""Tests for profile-based cluster naming.

The bug being guarded against: names were hardcoded to k-means label indices,
which are arbitrary. The decisive test here permutes the labels and checks that
each player keeps the same *name*.
"""

import numpy as np
import pandas as pd
import pytest

from nba2k import clustering
from nba2k.constants import ATTRS, CLUSTER_NAMES


def _synthetic_players(n_per=30, seed=0):
    """Seven synthetic archetypes with deliberately distinct attribute profiles."""
    rng = np.random.default_rng(seed)
    profiles = {
        "Elite Two-Way Superstars": 92,
        "3-and-D Connectors": 78,
        "Do-It-All Forwards": 76,
        "Shot-Creating Lead Guards": 82,
        "Skilled Post Bigs": 79,
        "Movement Shooters/Bench Guards": 72,
        "Rim-Running Bigs": 75,
    }
    tweaks = {
        "3-and-D Connectors": {"perimeter_defense": +12, "steal": +10,
                               "defensive_consistency": +10, "help_defense_iq": +10,
                               "pass_perception": +10, "post_control": -12,
                               "post_hook": -12},
        "Shot-Creating Lead Guards": {"ball_handle": +14, "speed_with_ball": +14,
                                      "pass_vision": +12, "pass_iq": +12, "layup": +12,
                                      "draw_foul": +12, "interior_defense": -14,
                                      "offensive_rebound": -14},
        "Skilled Post Bigs": {"post_control": +16, "post_hook": +16, "post_fade": +14,
                              "strength": +12, "standing_dunk": +12,
                              "speed_with_ball": -14, "agility": -14},
        "Movement Shooters/Bench Guards": {"three_point_shot": +12, "free_throw": +10,
                                           "speed": +8, "interior_defense": -12,
                                           "post_hook": -12, "offensive_rebound": -12,
                                           "defensive_rebound": -12, "block": -12},
        "Rim-Running Bigs": {"offensive_rebound": +18, "block": +16,
                             "standing_dunk": +16, "defensive_rebound": +14,
                             "interior_defense": +14, "three_point_shot": -25,
                             "ball_handle": -18, "pass_vision": -16},
    }

    rows = []
    for i, (name, base) in enumerate(profiles.items()):
        for j in range(n_per):
            row = {a: base + rng.normal(0, 2.0) for a in ATTRS}
            for attr, delta in tweaks.get(name, {}).items():
                row[attr] += delta
            row["slug"] = f"{i}-{j}"
            row["true_archetype"] = name
            row["overall"] = base
            rows.append(row)
    return pd.DataFrame(rows)


def test_recovers_the_planted_archetypes():
    frame, info = clustering.fit_clusters(_synthetic_players())
    agreement = pd.crosstab(frame["true_archetype"], frame["cluster_label"])
    # Every planted archetype should map to exactly one named cluster.
    assert (agreement.max(axis=1) / agreement.sum(axis=1) > 0.9).all()
    for name in agreement.index:
        assert agreement.loc[name].idxmax() == name


def test_naming_is_a_bijection():
    _, info = clustering.fit_clusters(_synthetic_players())
    assert sorted(info["mapping"].values()) == sorted(CLUSTER_NAMES)
    assert len(set(info["mapping"])) == len(CLUSTER_NAMES)


def test_names_survive_a_label_permutation():
    """The decisive test: relabel every cluster and the names must follow the
    centroids, not the integers. A hardcoded {0: ..., 1: ...} dict fails here.
    """
    players = _synthetic_players()
    frame, info = clustering.fit_clusters(players)

    permutation = {0: 3, 1: 5, 2: 0, 3: 6, 4: 1, 5: 2, 6: 4}
    permuted = info["centroids"].rename(index=permutation).sort_index()
    permuted_mapping, _ = clustering.assign_cluster_names(permuted)

    for old_label, name in info["mapping"].items():
        assert permuted_mapping[permutation[old_label]] == name


def test_row_order_does_not_change_anyones_cluster():
    """k-means++ seeds from the data in row order; fit_clusters sorts first so
    that a reshuffled input CSV cannot silently produce a different solution."""
    players = _synthetic_players()
    a, _ = clustering.fit_clusters(players)
    b, _ = clustering.fit_clusters(players.sample(frac=1.0, random_state=17))
    a = a.set_index("slug")["cluster_label"]
    b = b.set_index("slug")["cluster_label"]
    assert (a == b.reindex(a.index)).all()


def test_sanity_checks_reject_a_scrambled_mapping():
    """check_cluster_mapping must actually be able to fail."""
    _, info = clustering.fit_clusters(_synthetic_players())
    labels = list(info["centroids"].index)
    names = [info["mapping"][x] for x in labels]
    scrambled = dict(zip(labels, names[1:] + names[:1]))
    with pytest.raises(AssertionError):
        clustering.check_cluster_mapping(info["centroids"], scrambled)


def test_wrong_k_is_rejected_rather_than_mislabelled():
    centroids = pd.DataFrame(np.zeros((3, len(ATTRS))), columns=ATTRS)
    with pytest.raises(ValueError):
        clustering.assign_cluster_names(centroids)


def test_cluster_profile_is_in_raw_units_and_counts_players():
    frame, info = clustering.fit_clusters(_synthetic_players())
    profile = clustering.cluster_profile(frame, info)
    assert sorted(profile.index) == sorted(CLUSTER_NAMES)
    assert profile["count"].sum() == len(frame)
    assert profile.loc["Rim-Running Bigs", "three_point_shot"] < \
        profile.loc["Movement Shooters/Bench Guards", "three_point_shot"]
    assert 40 < profile.loc["Elite Two-Way Superstars", "overall_durability"] < 120
