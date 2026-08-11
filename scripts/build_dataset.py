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

Age blocking has one failure mode we handle explicitly: the salary table has no
age column of its own, so it borrows one from the NBA stats table by exact
normalized name. Any salaried player with *no* stats row at all (season-ending
injury, waived-and-unsigned) therefore gets a NaN age and can never appear in a
blocked candidate set -- which silently dropped exactly the injured stars on
huge contracts an "overpaid" analysis exists to find (Haliburton, Irving,
Lillard, VanVleet). Rows that find no blocked match now fall back to an
*unblocked* match at a stricter threshold (95 vs 85) and the recovered pairs
are logged and flagged in the output (`*_match_fallback`).

The NBA stats endpoint (`LeagueDashPlayerStats`) already returns one
season-aggregated row per player -- no per-team rows and no "TOT" combined row
-- so no de-duplication is needed there; this is asserted rather than assumed.

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
# Threshold for the unblocked fallback used when age blocking yields no
# candidate at all. Stricter than the blocked threshold precisely because the
# age guard that catches "Jaylen Nowell" vs "Jaylen Wells" is not available.
FALLBACK_MATCH_THRESHOLD = 95


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
    """Load the season stat lines.

    `LeagueDashPlayerStats` aggregates the whole season server-side, so it
    returns exactly one row per player -- there are no per-team rows for
    mid-season trades and no "TOT" combined row to prefer. Earlier versions of
    this script carried TOT-preferring de-duplication code that never fired;
    the invariant it assumed is asserted here instead.
    """
    df = pd.read_csv(os.path.join(RAW_DIR, "nba_stats_2025_26.csv"))
    assert not df["PLAYER_ID"].duplicated().any(), (
        "nba_stats_2025_26.csv has duplicate PLAYER_IDs -- the endpoint's "
        "one-row-per-player contract changed; season-total aggregation "
        "(or TOT-row selection) is now required here."
    )
    assert not df["TEAM_ABBREVIATION"].eq("TOT").any(), (
        "nba_stats_2025_26.csv contains TOT rows -- see the assertion above."
    )
    df["name_norm"] = df["PLAYER_NAME"].map(normalize_name)
    return df


def load_salaries():
    """Load 2025-26 salaries, one row per player.

    A player who changed teams mid-season appears once per team page (e.g.
    Damian Lillard: $13.4M Portland + $22.5M Milwaukee). Those are two separate
    real payments in the same season, so we **sum** them -- an earlier version
    kept only the largest row, understating split-season pay by up to 37%.
    `salary_rows` records how many rows were combined so downstream analysis can
    flag or exclude split-season players.
    """
    df = pd.read_csv(os.path.join(RAW_DIR, "nba_salaries_2025_26.csv"))
    df["name_norm"] = df["name"].map(normalize_name)
    df = df[df["name_norm"] != ""].copy()

    agg = df.groupby("name_norm", as_index=False).agg(
        name=("name", "first"),
        salary_usd=("salary_usd", "sum"),
        salary_usd_max_row=("salary_usd", "max"),
        salary_rows=("salary_usd", "size"),
        player_option=("player_option", "any"),
        team_option=("team_option", "any"),
        two_way_contract=("two_way_contract", "any"),
    )
    split = agg[agg["salary_rows"] > 1]
    if len(split):
        print(f"Split-season salaries summed for {len(split)} player(s):")
        for _, r in split.iterrows():
            print(f"  {r['name']}: {r['salary_rows']} rows -> ${r['salary_usd']:,.0f} "
                  f"(largest single row ${r['salary_usd_max_row']:,.0f})")
    return agg


def fuzzy_match(left, right, right_name_col="name_norm", left_age_col=None,
                right_age_col=None, age_tol=1, fallback_threshold=FALLBACK_MATCH_THRESHOLD,
                label="right", verbose=True):
    """For each row in `left`, find the best-matching row index in `right`
    by normalized-name similarity. Returns (matched_idx, score) Series aligned
    to left.index.

    If `left_age_col`/`right_age_col` are given, candidates are blocked to
    within `age_tol` years -- this is what keeps common first/last-name
    combinations (e.g. "Jaylen Nowell" vs "Jaylen Wells", "Keon Johnson" vs
    "Keldon Johnson") from fuzzy-matching to the wrong real player.

    Blocking is a filter on the *right* table too: a right-hand row with no age
    is in no age bucket and so is invisible to every blocked search. To avoid
    silently dropping those (which, for the salary table, means precisely the
    injured stars with no stats row to borrow an age from), any left row that
    finds no blocked match is retried unblocked at `fallback_threshold`, which
    is stricter than `NAME_MATCH_THRESHOLD` because the age guard is absent.

    Returns (matched_idx, score, used_fallback) Series aligned to left.index.
    """
    use_age_blocking = left_age_col is not None and right_age_col is not None

    matched_idx = pd.Series(index=left.index, dtype="float64")
    scores = pd.Series(index=left.index, dtype="float64")
    used_fallback = pd.Series(False, index=left.index, dtype="bool")

    if use_age_blocking:
        right_by_age = {}
        for age, sub in right.groupby(right_age_col):
            if pd.isna(age):
                continue
            right_by_age[int(age)] = sub

    all_choices = right[right_name_col].tolist()
    all_choice_idx = right.index.tolist()

    recovered = []
    for i, row in left.iterrows():
        name = row["name_norm"]
        if not name:
            continue

        candidates = right
        blocked = False
        if use_age_blocking:
            age = row.get(left_age_col)
            if pd.notna(age):
                blocked = True
                age = int(age)
                parts = [right_by_age.get(a) for a in range(age - age_tol, age + age_tol + 1)]
                parts = [p for p in parts if p is not None]
                candidates = pd.concat(parts) if parts else right.iloc[0:0]

        if not candidates.empty:
            choices = candidates[right_name_col].tolist() if blocked else all_choices
            choice_idx = candidates.index.tolist() if blocked else all_choice_idx
            best = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
            if best and best[1] >= NAME_MATCH_THRESHOLD:
                matched_idx.loc[i] = choice_idx[best[2]]
                scores.loc[i] = best[1]
                continue

        # No blocked match. Retry unblocked at a stricter threshold so that
        # right-hand rows with a missing/unknown age are still reachable.
        if not blocked:
            continue
        best = process.extractOne(name, all_choices, scorer=fuzz.token_sort_ratio)
        if best and best[1] >= fallback_threshold:
            matched_idx.loc[i] = all_choice_idx[best[2]]
            scores.loc[i] = best[1]
            used_fallback.loc[i] = True
            recovered.append((row.get("name", name), best[0], best[1]))

    if verbose and recovered:
        print(f"Recovered {len(recovered)} {label} match(es) via unblocked fallback "
              f"(score >= {fallback_threshold}):")
        for left_name, right_name, score in recovered:
            print(f"  {left_name!r} -> {right_name!r} (score {score:.0f})")

    return matched_idx, scores, used_fallback


def main():
    os.makedirs(PROC_DIR, exist_ok=True)

    ratings = load_2k()
    stats = load_nba_stats()
    salaries = load_salaries()

    ratings.to_csv(os.path.join(PROC_DIR, "players_2k26_clean.csv"), index=False)

    stats_idx, stats_scores, stats_fallback = fuzzy_match(
        ratings, stats, left_age_col="age_2k", right_age_col="AGE", label="stats",
    )

    # Salaries have no age column of their own; borrow one via an exact
    # normalized-name lookup against the (reliable, real) NBA stats table so
    # the ratings<->salary match can also be age-blocked.
    # Salaried players with no stats row at all (season-ending injury, waived)
    # get a NaN age here and are therefore invisible to the blocked search --
    # the unblocked fallback inside fuzzy_match() is what recovers them.
    age_lookup = stats.drop_duplicates("name_norm").set_index("name_norm")["AGE"]
    salaries = salaries.copy()
    salaries["age_ref"] = salaries["name_norm"].map(age_lookup)
    print(f"{salaries['age_ref'].isna().sum()} of {len(salaries)} salary rows have no "
          f"age reference (no exact-name stats row) and are unreachable by age-blocked matching")
    sal_idx, sal_scores, sal_fallback = fuzzy_match(
        ratings, salaries, left_age_col="age_2k", right_age_col="age_ref", label="salary",
    )

    merged = ratings.copy()
    merged["stats_match_score"] = stats_scores
    merged["stats_match_fallback"] = stats_fallback
    merged["salary_match_score"] = sal_scores
    merged["salary_match_fallback"] = sal_fallback

    stats_cols = [c for c in stats.columns if c not in ("name_norm", "is_tot")]
    stats_lookup = stats[stats_cols].copy()
    stats_lookup.columns = ["nba_" + c.lower() if c not in ("PLAYER_ID",) else "nba_player_id" for c in stats_cols]
    stats_aligned = stats_lookup.reindex(stats_idx).reset_index(drop=True)
    stats_aligned.index = merged.index
    merged = pd.concat([merged, stats_aligned], axis=1)

    sal_lookup = salaries[
        ["salary_usd", "salary_rows", "player_option", "team_option", "two_way_contract"]
    ].copy()
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
        "stats_matches_via_fallback": int(merged["stats_match_fallback"].sum()),
        "salary_matches_via_fallback": int(merged["salary_match_fallback"].sum()),
        "split_season_salaries": int((merged["salary_rows"] > 1).sum()),
    }

    merged.to_csv(os.path.join(PROC_DIR, "players_merged.csv"), index=False)
    with open(os.path.join(PROC_DIR, "match_stats.json"), "w") as f:
        json.dump(match_stats, f, indent=2)
    print(json.dumps(match_stats, indent=2))


if __name__ == "__main__":
    main()
