"""Tests over the committed dataset itself.

These are the checks that would have caught the audit's data-pipeline findings,
and they double as drift detection: if a re-scrape or an upstream schema change
breaks an invariant the analysis depends on, CI fails instead of the notebooks
quietly producing different numbers.
"""

import numpy as np
import pandas as pd
import pytest

from nba2k import constants, data


@pytest.fixture(scope="module")
def merged():
    return data.load_merged()


def test_merged_file_loads_and_has_the_expected_shape(merged):
    assert len(merged) > 300
    for col in ["name", "overall", "nba_pie", "nba_min", "salary_usd",
                "stats_match_score", "salary_match_score"]:
        assert col in merged.columns


def test_every_attribute_column_is_present(merged):
    missing = [a for a in constants.ATTRS if a not in merged.columns]
    assert missing == []


def test_overall_is_in_a_plausible_range(merged):
    assert merged["overall"].between(40, 99).all()


def test_salary_matching_recovered_the_injured_stars(merged):
    """Regression test for the age-blocking bug.

    These four are the expensive, no-stats-row players that age-blocked
    matching silently dropped -- precisely the cases the "overpaid" analysis
    exists to surface. Each must now carry a salary.
    """
    recovered = ["Tyrese Haliburton", "Kyrie Irving", "Fred VanVleet", "Damian Lillard"]
    have_salary = merged.set_index("name")["salary_usd"]
    for player in recovered:
        assert player in have_salary.index, f"{player} missing from the merged table"
        assert pd.notna(have_salary.loc[player]), f"{player} still has no salary"


def test_split_season_salary_is_summed_not_maxed(merged):
    """Lillard was paid by two teams in 2025-26; taking the max understated it."""
    row = merged.loc[merged["name"] == "Damian Lillard"].iloc[0]
    assert row["salary_rows"] == 2
    assert row["salary_usd"] == pytest.approx(13_398_800 + 22_516_603)


def test_fallback_matches_are_flagged(merged):
    """Unblocked fallback matches are riskier, so they must be identifiable."""
    assert "salary_match_fallback" in merged.columns
    assert "stats_match_fallback" in merged.columns
    assert merged["salary_match_fallback"].sum() > 0
    # ... and every fallback match cleared the stricter threshold.
    fallback = merged[merged["salary_match_fallback"]]
    assert (fallback["salary_match_score"] >= 95).all()


def test_match_scores_never_fall_below_the_base_threshold(merged):
    for col in ["stats_match_score", "salary_match_score"]:
        assert (merged[col].dropna() >= 85).all()


def test_samples_are_nested_and_respect_their_filters(merged):
    perf = data.performance_sample(merged)
    val = data.value_sample(merged)
    assert (perf["nba_min"] >= constants.MIN_MINUTES).all()
    assert (perf["stats_match_score"] >= constants.MATCH_SCORE_MIN).all()
    assert (val["salary_usd"] > 0).all()
    assert set(val["slug"]) <= set(perf["slug"])
    assert 150 < len(perf) < 350


def test_add_age_is_plausible_for_nba_players(merged):
    aged = data.add_age(merged)
    ages = aged["age"].dropna()
    assert ages.between(17, 45).all()
    assert 22 < ages.median() < 32


def test_no_duplicate_players(merged):
    assert not merged["slug"].duplicated().any()
