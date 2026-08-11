"""Tests for the metrics that the audit found to be wrong.

These are mostly *property* tests: they encode the statistical reason each old
metric was replaced, so a regression back to the old behaviour fails loudly.
"""

import numpy as np
import pandas as pd
import pytest

from nba2k import metrics


@pytest.fixture
def correlated_frame():
    """x and y correlated at roughly r = 0.7, like overall vs PIE."""
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(0, 1, n)
    y = 0.7 * x + np.sqrt(1 - 0.7 ** 2) * rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "y": y, "nba_min": rng.uniform(500, 2500, n)})


def test_zscore_matches_pandas_convention(correlated_frame):
    z = metrics.zscore(correlated_frame["x"])
    assert z.mean() == pytest.approx(0, abs=1e-12)
    assert z.std() == pytest.approx(1, abs=1e-12)


def test_zscore_gap_is_contaminated_by_x(correlated_frame):
    """The old metric's fatal flaw, stated as a test.

    z(x) - z(y) correlates with x at sqrt((1-r)/2), which is 0.39 at r = 0.7 --
    so the old "most over-rated" ranking was substantially a ranking of who has
    the most extreme rating, not who diverges most from their rating.
    """
    gap = metrics.zscore_gap(correlated_frame, "x", "y")
    r = np.corrcoef(gap, correlated_frame["x"])[0, 1]
    assert r > 0.3


def test_studentized_residual_is_orthogonal_to_x(correlated_frame):
    """The replacement metric's defining property: independent of the predictor."""
    res = metrics.rating_residuals(correlated_frame, x="x", y="y")
    r = np.corrcoef(res["studentized"], res["x"])[0, 1]
    assert abs(r) < 0.05


def test_residuals_recover_a_planted_outlier():
    """A player deliberately placed far below the line must rank most over-rated."""
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(0, 1, n)
    y = 0.8 * x + rng.normal(0, 0.3, n)
    frame = pd.DataFrame({"x": x, "y": y})
    frame.loc[0, "y"] = frame.loc[0, "x"] * 0.8 - 3.0  # way below the fit

    res = metrics.rating_residuals(frame, x="x", y="y")
    assert res["studentized"].idxmin() == 0
    assert res.loc[0, "outside_pi"]


def test_prediction_interval_covers_about_95_percent(correlated_frame):
    res = metrics.rating_residuals(correlated_frame, x="x", y="y", alpha=0.05)
    coverage = 1 - res["outside_pi"].mean()
    assert 0.90 < coverage < 0.99
    assert (res["pi_low"] < res["pi_high"]).all()


def test_rating_residuals_needs_enough_rows():
    with pytest.raises(ValueError):
        metrics.rating_residuals(pd.DataFrame({"x": [1.0], "y": [2.0]}), x="x", y="y")


def _value_frame():
    """Two players with identical per-minute impact but very different volume,
    both paid the same. The higher-volume one is unambiguously better value.
    """
    rng = np.random.default_rng(2)
    n = 120
    minutes = rng.uniform(500, 2500, n)
    pie = rng.uniform(0.05, 0.20, n)
    salary = 1e6 + 40 * pie * minutes * 1000 + rng.normal(0, 1e5, n)
    frame = pd.DataFrame({
        "name": [f"p{i}" for i in range(n)],
        "nba_min": minutes, "nba_pie": pie, "salary_usd": salary,
    })
    frame.loc[0, ["nba_min", "nba_pie", "salary_usd"]] = [600, 0.12, 10_000_000]
    frame.loc[1, ["nba_min", "nba_pie", "salary_usd"]] = [2400, 0.12, 10_000_000]
    frame.loc[0, "name"] = "low_minutes"
    frame.loc[1, "name"] = "high_minutes"
    return frame


def test_old_rate_vs_total_metric_ties_players_of_very_different_volume():
    """Why the old moneyball metric was wrong: PIE is a rate, so two players
    with the same PIE and the same salary score identically even though one
    played four times as many minutes."""
    frame = _value_frame()
    gap = metrics.zscore(frame["nba_pie"]) - metrics.zscore(np.log10(frame["salary_usd"]))
    assert gap.iloc[0] == pytest.approx(gap.iloc[1])


def test_value_metrics_prefers_the_higher_volume_player():
    out = metrics.value_metrics(_value_frame()).set_index("name")
    assert out.loc["high_minutes", "value_score"] > out.loc["low_minutes", "value_score"]
    assert out.loc["high_minutes", "dollars_per_pie_minute"] < out.loc["low_minutes", "dollars_per_pie_minute"]
    assert out.loc["high_minutes", "pie_minutes"] == pytest.approx(288.0)


def test_value_metrics_drops_nonpositive_production():
    frame = _value_frame()
    frame.loc[2, "nba_pie"] = -0.01
    out = metrics.value_metrics(frame)
    assert out.attrs["dropped_nonpositive_production"] == 1
    assert (out["pie_minutes"] > 0).all()


def test_minutes_sensitivity_reports_every_threshold(correlated_frame):
    table = metrics.minutes_sensitivity(
        correlated_frame, x="x", y="y", thresholds=(0, 1000, 2000)
    )
    assert list(table["min_minutes"]) == [0, 1000, 2000]
    assert (table["n"].diff().dropna() < 0).all()  # stricter cut = fewer players


def test_benjamini_hochberg_is_more_conservative_than_raw_p():
    p = [0.001, 0.02, 0.03, 0.04, 0.2, 0.5, 0.7, 0.9]
    out = metrics.benjamini_hochberg(p, alpha=0.05)
    assert (out["q_value"] >= out["p_value"] - 1e-12).all()
    assert out["significant"].sum() < sum(x < 0.05 for x in p)
    assert out["q_value"].is_monotonic_increasing


def test_benjamini_hochberg_keeps_an_overwhelming_signal():
    out = metrics.benjamini_hochberg([1e-12] * 3 + [0.6, 0.7], alpha=0.05)
    assert out["significant"].tolist() == [True, True, True, False, False]


def test_bootstrap_ci_brackets_the_point_estimate(correlated_frame):
    lo, hi = metrics.bootstrap_corr_ci(
        correlated_frame["x"], correlated_frame["y"], n_boot=400, random_state=0
    )
    r = np.corrcoef(correlated_frame["x"], correlated_frame["y"])[0, 1]
    assert lo < r < hi


def test_bootstrap_ci_is_wide_for_tiny_samples():
    rng = np.random.default_rng(3)
    x = rng.normal(size=5)
    y = rng.normal(size=5)
    lo, hi = metrics.bootstrap_corr_ci(x, y, n_boot=400, random_state=0)
    assert hi - lo > 0.8  # an r from 5 players says essentially nothing
