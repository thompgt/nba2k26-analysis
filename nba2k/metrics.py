"""The project's headline metrics, with the statistics done properly.

Two of these replace metrics that were wrong in a way that changed the answer,
so the reasoning is spelled out at length.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .constants import MIN_MINUTES


def zscore(s):
    """Standardize a Series (sample standard deviation, matching pandas)."""
    return (s - s.mean()) / s.std()


def zscore_gap(df, a, b):
    """The *old* over/under-rated metric: z(a) - z(b). Kept only so notebooks
    can show side by side why it was replaced -- do not use it for rankings.

    If a and b correlate at r, then z(a) - z(b) has covariance structure such
    that E[z(a) - z(b) | z(a)] = (1 - r) * z(a). At r = 0.70 that means 30% of
    a player's standardized rating passes straight through into their "gap"
    score, so the ranking is dominated by *extremeness on a*, not by
    disagreement between a and b. A player sitting exactly on the regression
    line still gets a large gap purely for being far from the mean. That is
    regression to the mean, not a scouting error.
    """
    return zscore(df[a]) - zscore(df[b])


def rating_residuals(df, x="overall", y="nba_pie", alpha=0.05, dropna=True):
    """Regress `y` on `x` and return per-player residual diagnostics.

    This is the correct replacement for `zscore_gap`. Ranking by the residual
    from the fitted line asks the right question -- "given how this player is
    rated, is their real production higher or lower than the *typical* player
    with that rating?" -- and is by construction uncorrelated with `x`, so
    being a superstar or a scrub no longer buys you a place at the top of the
    list.

    Residuals are *externally studentized* (each observation's residual divided
    by an error estimate computed with that observation left out), which puts
    every player on a comparable t-distributed scale and stops one big outlier
    from inflating the scale that judges it.

    Added columns:
      ``fitted``       predicted y at this player's x
      ``residual``     raw y - fitted
      ``leverage``     hat-matrix diagonal
      ``studentized``  externally studentized residual (the ranking column)
      ``pi_low``/``pi_high``  ``1 - alpha`` prediction interval for y at this x
      ``outside_pi``   True if the player's actual y falls outside that interval

    Negative ``studentized`` = produced less than the fit predicts for their
    rating = **over-rated**. Positive = **under-rated**.
    """
    out = df.dropna(subset=[x, y]).copy() if dropna else df.copy()
    if len(out) < 3:
        raise ValueError(f"need at least 3 complete rows to regress {y} on {x}")

    X = sm.add_constant(out[[x]].astype(float))
    model = sm.OLS(out[y].astype(float), X).fit()
    influence = model.get_influence()

    out["fitted"] = model.fittedvalues
    out["residual"] = model.resid
    out["leverage"] = influence.hat_matrix_diag
    out["studentized"] = influence.resid_studentized_external

    pred = model.get_prediction(X).summary_frame(alpha=alpha)
    out["pi_low"] = pred["obs_ci_lower"].values
    out["pi_high"] = pred["obs_ci_upper"].values
    out["outside_pi"] = (out[y] < out["pi_low"]) | (out[y] > out["pi_high"])

    out.attrs["model"] = model
    out.attrs["alpha"] = alpha
    return out


def value_metrics(df, salary_col="salary_usd", pie_col="nba_pie", min_col="nba_min",
                  alpha=0.05):
    """Production-vs-pay metrics that compare like with like.

    The previous moneyball metric was ``z(PIE) - z(log salary)``. PIE is a
    *rate* -- share of the game's total statistical events a player accounted
    for while on the floor -- while salary is a season *total*. A 520-minute
    minimum-salary player with a decent per-possession rate therefore beat a
    2,500-minute starter mechanically, because the numerator ignored how much
    of the season they actually played while the denominator did not.

    Here production is measured as **PIE x minutes**, an impact *volume*
    comparable to a season salary, and value is the residual from regressing
    log10(salary) on log10(PIE-minutes) -- so, as in `rating_residuals`, the
    ranking is not just a proxy for being expensive.

    Added columns:
      ``pie_minutes``               PIE x minutes played (season impact volume)
      ``log_pie_minutes``           log10 of the above
      ``log_salary``                log10 salary
      ``dollars_per_pie_minute``    salary / PIE-minutes; the plain-language
                                    version of the same quantity (lower = better value)
      ``expected_log_salary``       fit of log salary on log PIE-minutes
      ``pay_residual``              studentized residual of that fit
      ``value_score``               ``-pay_residual``; positive = paid less than
                                    production predicts (bargain), negative = overpaid

    Rows with non-positive PIE-minutes (a player whose net box-score impact was
    negative over the season) cannot be logged and are dropped; the count is
    recorded in ``result.attrs["dropped_nonpositive_production"]``.
    """
    out = df.dropna(subset=[salary_col, pie_col, min_col]).copy()
    out = out[out[salary_col] > 0]
    out["pie_minutes"] = out[pie_col] * out[min_col]

    n_before = len(out)
    out = out[out["pie_minutes"] > 0].copy()
    dropped = n_before - len(out)

    out["log_pie_minutes"] = np.log10(out["pie_minutes"])
    out["log_salary"] = np.log10(out[salary_col])
    out["dollars_per_pie_minute"] = out[salary_col] / out["pie_minutes"]

    fitted = rating_residuals(
        out, x="log_pie_minutes", y="log_salary", alpha=alpha, dropna=False
    )
    out["expected_log_salary"] = fitted["fitted"]
    out["pay_residual"] = fitted["studentized"]
    out["value_score"] = -fitted["studentized"]

    out.attrs["model"] = fitted.attrs["model"]
    out.attrs["dropped_nonpositive_production"] = int(dropped)
    return out


def minutes_sensitivity(df, x="overall", y="nba_pie",
                        thresholds=(0, 250, MIN_MINUTES, 1000, 1500)):
    """Is the headline correlation an artefact of the 500-minute cut?

    Recomputes Pearson and Spearman correlation between `x` and `y` at several
    minutes floors, so a reader can see the choice wasn't threshold-shopped.
    """
    rows = []
    for t in thresholds:
        sub = df[(df["nba_min"] >= t)].dropna(subset=[x, y])
        if len(sub) < 3:
            continue
        pearson_r, pearson_p = stats.pearsonr(sub[x], sub[y])
        spearman_r, _ = stats.spearmanr(sub[x], sub[y])
        rows.append({
            "min_minutes": t,
            "n": len(sub),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_r,
        })
    return pd.DataFrame(rows)


def benjamini_hochberg(pvalues, alpha=0.05):
    """Benjamini-Hochberg FDR control over a family of p-values.

    Returns a DataFrame with the original p-values, their BH-adjusted q-values
    and a `significant` flag. Needed anywhere this project tests many
    correlations at once (notebook 05 runs ~30 position x attribute-pair tests,
    where roughly 1.5 spurious hits at p < 0.05 are expected by chance alone).
    """
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q_sorted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)
    q = np.empty(n)
    q[order] = q_sorted
    return pd.DataFrame({"p_value": p, "q_value": q, "significant": q < alpha})


def bootstrap_corr_ci(x, y, n_boot=2000, alpha=0.05, random_state=None):
    """Percentile bootstrap CI for a Pearson correlation.

    More honest than a bare p-value for the small per-position samples in
    notebook 05: an r from 12 players comes with an interval so wide that
    quoting the point estimate alone is misleading.
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return (np.nan, np.nan)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([
        stats.pearsonr(x[i], y[i])[0] if np.std(x[i]) > 0 and np.std(y[i]) > 0 else np.nan
        for i in idx
    ])
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0:
        return (np.nan, np.nan)
    return tuple(np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
