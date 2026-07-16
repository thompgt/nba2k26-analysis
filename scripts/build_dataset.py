"""Fuzzy-match 2K26 ratings <-> real NBA stats <-> salary into one player table.

There's no shared player ID across these three sources (2kratings.com slugs,
stats.nba.com PLAYER_ID, HoopsHype's internal player IDs), so matching is done
by normalized full-name similarity (rapidfuzz), same approach as fifa-analysis's
`build_dataset.py`. Unlike the FIFA project we don't block by age (2K bio ages
aren't always populated/reliable for recent debuts), so we lean a bit harder on
the match-score threshold and take the single best match per player.

A player traded mid-season appears multiple times in the NBA stats table (one
row per team + a "TOT" combined row); we keep the TOT row when present so each
player has exactly one season-total stat line.

Outputs:
  data/processed/players_2k26_clean.csv   full scraped 2K26 ratings, lightly cleaned
  data/processed/players_merged.csv       2K26 players matched to real stats / salary
  data/processed/match_stats.json         match-rate diagnostics
"""

import json
import os
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

NAME_MATCH_THRESHOLD = 88


def normalize_name(name):
    if pd.isna(name):
        return ""
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    name = name.lower().strip()
    for suffix in [" jr.", " jr", " sr.", " sr", " ii", " iii", " iv"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    return name


def _fix_misparsed_attribute_columns(df):
    """A handful of archived pages (~3%) had an extra hidden badge digit
    between an attribute's value and its label, which an earlier version of
    the scraper's parser folded into the column name instead of the label,
    e.g. producing a `6_close_shot` column (value correct, name wrong)
    instead of populating `close_shot` for that row. Fold any `<digits>_<attr>`
    column back into its real `<attr>` column (fixed at the source in
    `scrape_2k_ratings.py` for future scrapes; this repairs already-scraped
    data without a full re-scrape).
    """
    import re

    misparsed = [c for c in df.columns if re.match(r"^\d+_[a-z_]+$", c)]
    for col in misparsed:
        real_col = re.sub(r"^\d+_", "", col)
        if real_col in df.columns:
            df[real_col] = df[real_col].combine_first(df[col])
        else:
            df[real_col] = df[col]
    return df.drop(columns=misparsed)


def load_2k():
    df = pd.read_csv(os.path.join(RAW_DIR, "2k26_ratings.csv"))
    # Drop the site's non-player utility pages (filter/comparison tools) that
    # slipped through the slug-based exclusion list into early scrapes.
    df = df[~df["name"].str.contains("Filter Tool|Comparison Tool", na=False)]
    df = _fix_misparsed_attribute_columns(df)
    # A small number of very-recent draftees had a 2kratings.com page archived
    # before their initial rating was published (no Overall yet) -- drop them,
    # there's no rating to validate.
    df = df.dropna(subset=["overall"]).copy()
    df["name_norm"] = df["name"].map(normalize_name)
    return df


def load_nba_stats():
    df = pd.read_csv(os.path.join(RAW_DIR, "nba_stats_2025_26.csv"))
    # Prefer the combined "TOT" team-total row for players traded mid-season.
    df["is_tot"] = df["TEAM_ABBREVIATION"].eq("TOT")
    df = df.sort_values("is_tot", ascending=False).drop_duplicates(subset="PLAYER_ID", keep="first")
    df["name_norm"] = df["PLAYER_NAME"].map(normalize_name)
    return df


def load_salaries():
    df = pd.read_csv(os.path.join(RAW_DIR, "nba_salaries_2025_26.csv"))
    df["name_norm"] = df["name"].map(normalize_name)
    # A few players are on two-way / partially-guaranteed rows duplicated across
    # trade-deadline team pages; keep the highest salary_usd figure per player.
    df = df.sort_values("salary_usd", ascending=False).drop_duplicates(subset="name_norm", keep="first")
    return df


def fuzzy_match(left, right, right_name_col="name_norm"):
    """For each row in `left`, find the best-matching row index in `right`
    by normalized-name similarity. Returns (matched_idx, score) Series aligned
    to left.index.
    """
    choices = right[right_name_col].tolist()
    choice_idx = right.index.tolist()

    matched_idx = pd.Series(index=left.index, dtype="float64")
    scores = pd.Series(index=left.index, dtype="float64")

    for i, name in left["name_norm"].items():
        if not name:
            continue
        best = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
        if best and best[1] >= NAME_MATCH_THRESHOLD:
            matched_idx.loc[i] = choice_idx[best[2]]
            scores.loc[i] = best[1]

    return matched_idx, scores


def main():
    os.makedirs(PROC_DIR, exist_ok=True)

    ratings = load_2k()
    stats = load_nba_stats()
    salaries = load_salaries()

    ratings.to_csv(os.path.join(PROC_DIR, "players_2k26_clean.csv"), index=False)

    stats_idx, stats_scores = fuzzy_match(ratings, stats)
    sal_idx, sal_scores = fuzzy_match(ratings, salaries)

    merged = ratings.copy()
    merged["stats_match_score"] = stats_scores
    merged["salary_match_score"] = sal_scores

    stats_cols = [c for c in stats.columns if c not in ("name_norm", "is_tot")]
    stats_lookup = stats[stats_cols].copy()
    stats_lookup.columns = ["nba_" + c.lower() if c not in ("PLAYER_ID",) else "nba_player_id" for c in stats_cols]
    stats_aligned = stats_lookup.reindex(stats_idx).reset_index(drop=True)
    stats_aligned.index = merged.index
    merged = pd.concat([merged, stats_aligned], axis=1)

    sal_lookup = salaries[["salary_usd", "player_option", "team_option", "two_way_contract"]].copy()
    sal_aligned = sal_lookup.reindex(sal_idx).reset_index(drop=True)
    sal_aligned.index = merged.index
    merged = pd.concat([merged, sal_aligned], axis=1)

    match_stats = {
        "players_2k26": int(len(ratings)),
        "nba_stats_players": int(len(stats)),
        "salary_players": int(len(salaries)),
        "matched_to_nba_stats": int(merged["nba_player_id"].notna().sum()),
        "matched_to_salary": int(merged["salary_usd"].notna().sum()),
        "matched_to_both": int((merged["nba_player_id"].notna() & merged["salary_usd"].notna()).sum()),
    }

    merged.to_csv(os.path.join(PROC_DIR, "players_merged.csv"), index=False)
    with open(os.path.join(PROC_DIR, "match_stats.json"), "w") as f:
        json.dump(match_stats, f, indent=2)
    print(json.dumps(match_stats, indent=2))


if __name__ == "__main__":
    main()
