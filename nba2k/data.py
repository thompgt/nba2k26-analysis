"""Loading and filtering the merged player table."""

import os

import pandas as pd

from .constants import MATCH_SCORE_MIN, MIN_MINUTES, REFERENCE_DATE

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED_PATH = os.path.join(REPO_ROOT, "data", "processed", "players_merged.csv")


def load_merged(path=None):
    """Load `data/processed/players_merged.csv`.

    Resolved relative to the repository root rather than the caller's working
    directory, so notebooks (run from `notebooks/`) and scripts (run from the
    repo root) get the same file without either hard-coding `../`.
    """
    return pd.read_csv(path or MERGED_PATH)


def add_age(df, reference_date=REFERENCE_DATE):
    """Add an `age` column in years as of `reference_date`."""
    out = df.copy()
    birthdate = pd.to_datetime(out["birthdate"], errors="coerce")
    out["age"] = (reference_date - birthdate).dt.days / 365.25
    return out


def performance_sample(df, min_minutes=MIN_MINUTES, match_score_min=MATCH_SCORE_MIN):
    """Rotation players with a high-confidence stats match and a PIE value."""
    return df[
        (df["stats_match_score"] >= match_score_min)
        & df["nba_pie"].notna()
        & df["nba_min"].notna()
        & (df["nba_min"] >= min_minutes)
    ].copy()


def salary_sample(df, match_score_min=MATCH_SCORE_MIN):
    """Players with a high-confidence salary match and a positive salary."""
    return df[
        (df["salary_match_score"] >= match_score_min)
        & df["salary_usd"].notna()
        & (df["salary_usd"] > 0)
    ].copy()


def value_sample(df, min_minutes=MIN_MINUTES, match_score_min=MATCH_SCORE_MIN):
    """Rotation players with both a high-confidence salary and stats match."""
    perf = performance_sample(df, min_minutes=min_minutes, match_score_min=match_score_min)
    return perf[
        (perf["salary_match_score"] >= match_score_min)
        & perf["salary_usd"].notna()
        & (perf["salary_usd"] > 0)
    ].copy()
