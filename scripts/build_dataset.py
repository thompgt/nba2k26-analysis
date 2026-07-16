"""Fuzzy-match 2K26 ratings <-> real NBA stats <-> salary into one player table.

There's no shared player ID across these three sources (2kratings.com slugs,
stats.nba.com PLAYER_ID, HoopsHype's internal player IDs), so matching is done
by normalized full-name similarity (rapidfuzz), blocked by age (+/-1 year),
same approach as fifa-analysis's `build_dataset.py`. Age blocking matters a
lot here: common first/last-name combinations are common enough in a
500-player league that unblocked fuzzy matching produces real false
positives (e.g. "Jaylen Nowell" incorrectly matching to "Jaylen Wells",
"Keon Johnson" to "Keldon Johnson") that a plain similarity-score threshold
doesn't reliably catch.

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

NAME_MATCH_THRESHOLD = 85


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
    # Some single-position players' bio text has no "Archetype:" label between
    # Position and Height, so the position-parsing window swallowed the next
    # field's label (e.g. position2 ends up as literally "Height:"). Blank out
    # any bio text field that looks like it captured a stray "<Label>:" token.
    bio_text_cols = ["position", "position2", "archetype", "nationality", "team", "college", "hometown"]
    for col in bio_text_cols:
        df.loc[df[col].astype(str).str.match(r"^[A-Za-z() ]+:$", na=False), col] = np.nan
    # A small number of very-recent draftees had a 2kratings.com page archived
    # before their initial rating was published (no Overall yet) -- drop them,
    # there's no rating to validate.
    df = df.dropna(subset=["overall"]).copy()
    df["name_norm"] = df["name"].map(normalize_name)
    # Approximate age as of the 2025-26 season (players are captured at
    # different points in the season, so this is +/- a few months) for
    # age-blocked fuzzy matching below -- this is what catches "Jaylen
    # Nowell" vs "Jaylen Wells"-style near-miss false matches on common
    # first/last name combinations.
    dob = pd.to_datetime(df["birthdate"], errors="coerce")
    df["age_2k"] = ((pd.Timestamp("2025-12-01") - dob).dt.days // 365)
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


def fuzzy_match(left, right, right_name_col="name_norm", left_age_col=None, right_age_col=None, age_tol=1):
    """For each row in `left`, find the best-matching row index in `right`
    by normalized-name similarity. Returns (matched_idx, score) Series aligned
    to left.index.

    If `left_age_col`/`right_age_col` are given, candidates are blocked to
    within `age_tol` years -- this is what keeps common first/last-name
    combinations (e.g. "Jaylen Nowell" vs "Jaylen Wells", "Keon Johnson" vs
    "Keldon Johnson") from fuzzy-matching to the wrong real player. Rows with
    a missing age fall back to an unblocked match.
    """
    use_age_blocking = left_age_col is not None and right_age_col is not None

    matched_idx = pd.Series(index=left.index, dtype="float64")
    scores = pd.Series(index=left.index, dtype="float64")

    if use_age_blocking:
        right_by_age = {}
        for age, sub in right.groupby(right_age_col):
            if pd.isna(age):
                continue
            right_by_age[int(age)] = sub

    all_choices = right[right_name_col].tolist()
    all_choice_idx = right.index.tolist()

    for i, row in left.iterrows():
        name = row["name_norm"]
        if not name:
            continue

        candidates = right
        if use_age_blocking:
            age = row.get(left_age_col)
            if pd.notna(age):
                age = int(age)
                parts = [right_by_age.get(a) for a in range(age - age_tol, age + age_tol + 1)]
                parts = [p for p in parts if p is not None]
                candidates = pd.concat(parts) if parts else right.iloc[0:0]

        if candidates.empty:
            continue
        choices = candidates[right_name_col].tolist() if use_age_blocking else all_choices
        choice_idx = candidates.index.tolist() if use_age_blocking else all_choice_idx

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

    stats_idx, stats_scores = fuzzy_match(
        ratings, stats, left_age_col="age_2k", right_age_col="AGE",
    )

    # Salaries have no age column of their own; borrow one via an exact
    # normalized-name lookup against the (reliable, real) NBA stats table so
    # the ratings<->salary match can also be age-blocked.
    age_lookup = stats.drop_duplicates("name_norm").set_index("name_norm")["AGE"]
    salaries = salaries.copy()
    salaries["age_ref"] = salaries["name_norm"].map(age_lookup)
    sal_idx, sal_scores = fuzzy_match(
        ratings, salaries, left_age_col="age_2k", right_age_col="age_ref",
    )

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
