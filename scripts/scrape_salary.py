"""Scrape 2025-26 NBA player salaries from HoopsHype's per-team pages.

hoopshype.com/salaries/ is a Next.js app; the league-wide `/salaries/players/`
page only server-renders the first ~20-25 highest-paid contracts (the rest
loads client-side via an internal GraphQL API we don't have direct access
to). Each *team* page (`/salaries/<team_slug>/`), however, server-renders
its full (~15-15 player) roster with multi-year salary data embedded in a
`__NEXT_DATA__` JSON blob, so we hit all 30 team pages instead of the
league-wide one.

`season: 2026` in that JSON denotes the salary for the season that *ends* in
2026, i.e. the 2025-26 season we want (spot-checked against known real
salaries, e.g. Jayson Tatum's $54,126,450 figure for `season=2026`, which
matches his actual reported 2025-26 salary).

Politeness: fetching goes through `polite_http.get()`, which checks robots.txt,
sends an identifying User-Agent with a contact address, honours any declared
Crawl-delay, and retries 429/5xx with backoff rather than parsing an error page
into empty rows.

Output: data/raw/nba_salaries_2025_26.csv
"""

import csv
import json
import os
import time

from bs4 import BeautifulSoup

from polite_http import FetchError, RobotsDisallowed, crawl_delay, get

TEAM_SLUGS = [
    "atlanta_hawks", "boston_celtics", "brooklyn_nets", "charlotte_hornets",
    "chicago_bulls", "cleveland_cavaliers", "dallas_mavericks", "denver_nuggets",
    "detroit_pistons", "golden_state_warriors", "houston_rockets", "indiana_pacers",
    "los_angeles_clippers", "los_angeles_lakers", "memphis_grizzlies", "miami_heat",
    "milwaukee_bucks", "minnesota_timberwolves", "new_orleans_pelicans",
    "new_york_knicks", "oklahoma_city_thunder", "orlando_magic",
    "philadelphia_76ers", "phoenix_suns", "portland_trail_blazers",
    "sacramento_kings", "san_antonio_spurs", "toronto_raptors", "utah_jazz",
    "washington_wizards",
]

SALARY_SEASON = 2026  # season ending in 2026 = the 2025-26 season
SLEEP = 1.0

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_PATH = os.path.join(RAW_DIR, "nba_salaries_2025_26.csv")


def team_salaries(team_slug):
    r = get(f"https://hoopshype.com/salaries/{team_slug}/", timeout=25)
    if r.status_code != 200:
        print(f"  {team_slug}: HTTP {r.status_code}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    tag = soup.find("script", id="__NEXT_DATA__")
    if tag is None:
        return []
    data = json.loads(tag.string)

    rows = []
    queries = data["props"]["pageProps"]["dehydratedState"]["queries"]
    for q in queries:
        d = q.get("state", {}).get("data")
        if not isinstance(d, dict) or "contracts" not in d:
            continue
        contracts = d["contracts"].get("contracts", [])
        for c in contracts:
            name = c.get("playerName")
            for season in c.get("seasons", []):
                if season.get("season") == SALARY_SEASON:
                    rows.append(
                        {
                            "hoopshype_player_id": c.get("playerID"),
                            "name": name,
                            "team_slug": team_slug,
                            "salary_usd": season.get("salary"),
                            "player_option": season.get("playerOption"),
                            "team_option": season.get("teamOption"),
                            "two_way_contract": season.get("twoWayContract"),
                        }
                    )
        break  # first query with contracts data is the team-roster one
    return rows


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    fieldnames = [
        "hoopshype_player_id", "name", "team_slug", "salary_usd",
        "player_option", "team_option", "two_way_contract",
    ]

    delay = max(SLEEP, crawl_delay("https://hoopshype.com/", default=SLEEP))

    all_rows = []
    for i, slug in enumerate(TEAM_SLUGS):
        try:
            rows = team_salaries(slug)
        except (RobotsDisallowed, FetchError) as e:
            print(f"  {slug}: FAILED ({e})")
            rows = []
        except Exception as e:
            print(f"  {slug}: FAILED ({e})")
            rows = []
        print(f"[{i+1}/{len(TEAM_SLUGS)}] {slug}: {len(rows)} players")
        all_rows.extend(rows)
        time.sleep(delay)

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"wrote {OUT_PATH} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
