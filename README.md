# NBA 2K26 Player Ratings Analysis

A data science project on **NBA 2K26** (the video game's) in-game player ratings:
demographics, how well the game's Overall rating and attribute categories proxy
real-world NBA performance (validated against real 2025-26 season stats and
salaries via `nba_api` and HoopsHype), in the same spirit as
[`fifa-analysis`](../fifa-analysis)'s EA FC 26 study.

**This project is about the NBA 2K26 video game's player ratings, not a
real-stats-only project.** (There are separate, unrelated projects at
`nba-player-clustering` and `nba_scout` that analyze real NBA statistics only —
this one is specifically about how 2K Sports rates players in-game, and how
that compares to reality.)

## Data sources

- **NBA 2K26 ratings**: [`2kratings.com`](https://www.2kratings.com/) individual
  player pages, scraped via `scripts/scrape_2k_ratings.py`. **Important
  caveat**: by the time this project was built (mid-2026), the live
  2kratings.com site had already rolled forward to previewing *NBA 2K27*
  ratings (2K27 releases ~Sept 2026) — its live per-attribute list pages are
  labeled "on NBA 2K27", and individual player pages only keep a single
  historical Overall number per past edition, not the full attribute
  breakdown. To get real NBA 2K26 attribute-level data, the script instead
  pulls **Internet Archive (Wayback Machine) snapshots** of individual player
  pages captured while NBA 2K26 was current (mostly Aug 2025 - Feb 2026),
  verified via each snapshot's `<title>` tag reading "... NBA 2K26 Rating".
  This works well but is **not exhaustive**: the Wayback Machine didn't crawl
  every current-roster player during that window, so the dataset covers
  roughly 350-450 of the ~550 players who appeared on NBA rosters in the
  2025-26 season (stars, rotation players, and anyone whose page happened to
  get crawled) rather than the full league. Bench/two-way/mid-season-signee
  players are under-represented. Official `nba.2k.com/2k26/ratings` was also
  checked as a fallback and rejected: it only lists a JS-rendered Top 100.
- **Real 2025-26 NBA season stats**: [`nba_api`](https://github.com/swar/nba_api)
  (`LeagueDashPlayerStats`, Base + Advanced), which hits stats.nba.com's own
  JSON endpoints directly — no scraping or bot-evasion needed. Pulled by
  `scripts/fetch_nba_stats.py`.
- **Salaries**: [`hoopshype.com`](https://hoopshype.com/salaries/) per-team
  salary pages (`scripts/scrape_salary.py`). The league-wide `/salaries/players/`
  page is a Next.js app that only server-renders its top ~20 contracts
  client-side-paginated beyond that; the 30 per-team pages, however,
  server-render each team's full roster with multi-year contract data in an
  embedded `__NEXT_DATA__` JSON blob, which we parse directly instead of the
  HTML table.

## Project structure

```
scripts/                     data acquisition + processing pipeline
  scrape_2k_ratings.py         scrapes NBA 2K26 player attributes via Wayback Machine
  fetch_nba_stats.py           pulls real 2025-26 season stats via nba_api
  scrape_salary.py             scrapes 2025-26 salaries from HoopsHype (per-team pages)
  build_dataset.py             fuzzy-matches the three sources into one player table
data/
  raw/                        untracked, gitignored (regenerate via scripts/)
  processed/                  small merged/cleaned CSVs, tracked in git
notebooks/
  01_demographics.ipynb        who's rated: position, height/weight/wingspan, age,
                                nationality/college, team, badges/archetypes
  02_rating_validation.ipynb   2K26 Overall + attributes vs real stats/salary ground truth
```

## Reproducing

```
pip install -r requirements.txt
python scripts/scrape_2k_ratings.py
python scripts/fetch_nba_stats.py
python scripts/scrape_salary.py
python scripts/build_dataset.py
jupyter notebook
```

No API keys are required for any of these — `nba_api` hits stats.nba.com's
public JSON endpoints directly, and the two scrapers use `curl_cffi`'s Chrome
TLS impersonation (plain `requests` gets a 403 from 2kratings.com/the Wayback
Machine's edge fairly often; `curl_cffi` reliably gets 200s).

## Limitations to keep in mind

- **2K26 ratings coverage is a real but incomplete slice of the league**
  (~350-450 of ~550 players), biased toward players whose 2kratings.com page
  got crawled by the Internet Archive during the 2K26 window — likely skewed
  toward more notable players, similar in spirit to fifa-analysis's
  Transfermarkt/Sofascore match-rate caveat.
- 2K26 attribute pages reflect a point-in-time snapshot (mostly the initial
  "launch" rating from ~Aug 2025, before most in-season roster updates), while
  the real stats are full 2025-26 season totals. A player's 2K rating may not
  reflect a late-season hot/cold streak the stats do capture.
- Name matching across sources is fuzzy (rapidfuzz, `players_merged.csv`
  records a match score per source) — spot-check any single-player finding
  against the raw data before treating it as ground truth.
