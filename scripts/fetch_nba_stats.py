"""Pull real 2025-26 NBA regular-season per-player stats via `nba_api`.

`nba_api` is a thin, no-auth-required Python client around the same JSON
endpoints stats.nba.com's own site uses, so no scraping/bot-evasion is needed.

We pull two "Base" + "Advanced" league-wide per-player tables and join them:
  - Base: games played/minutes, points, rebounds, assists, shooting splits
    (FG%, 3P%, FT%), steals, blocks, turnovers, etc.
  - Advanced: offensive/defensive/net rating, usage%, true shooting%, PIE,
    assist%, rebound%, etc. -- the "how good, adjusted for role/pace" numbers.

Output: data/raw/nba_stats_2025_26.csv (one row per player-team stint; a
player traded mid-season gets one row per team plus a TOT row, matching
stats.nba.com's own convention).
"""

import os
import time

from nba_api.stats.endpoints import leaguedashplayerstats

SEASON = "2025-26"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_PATH = os.path.join(RAW_DIR, "nba_stats_2025_26.csv")

BASE_KEEP = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE",
    "GP", "W", "L", "MIN",
    "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "BLKA", "PF", "PFD", "PTS",
    "PLUS_MINUS",
]

ADV_KEEP = [
    "PLAYER_ID", "TEAM_ID",
    "OFF_RATING", "DEF_RATING", "NET_RATING",
    "AST_PCT", "AST_TO", "AST_RATIO",
    "OREB_PCT", "DREB_PCT", "REB_PCT",
    "TM_TOV_PCT", "EFG_PCT", "TS_PCT", "USG_PCT", "PACE", "PIE",
]


def fetch(measure_type=None, retries=3):
    kwargs = dict(season=SEASON, season_type_all_star="Regular Season", timeout=60)
    if measure_type:
        kwargs["measure_type_detailed_defense"] = measure_type
    last_err = None
    for _ in range(retries):
        try:
            resp = leaguedashplayerstats.LeagueDashPlayerStats(**kwargs)
            return resp.get_data_frames()[0]
        except Exception as e:
            last_err = e
            time.sleep(3)
    raise last_err


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print(f"Fetching Base per-player stats for {SEASON}...")
    base = fetch()[BASE_KEEP]

    print(f"Fetching Advanced per-player stats for {SEASON}...")
    adv = fetch(measure_type="Advanced")[ADV_KEEP]

    merged = base.merge(adv, on=["PLAYER_ID", "TEAM_ID"], how="left")
    merged.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
