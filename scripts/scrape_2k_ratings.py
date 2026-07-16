"""Scrape full NBA 2K26 player attribute ratings from 2kratings.com.

Why the Wayback Machine, not the live site
-------------------------------------------
2kratings.com is a fan-run ratings database that continuously rolls forward:
by mid-2026 the "current" live site has already moved on to previewing NBA 2K27
ratings (2K27 releases ~Sept 2026), and the live per-attribute list pages
(`/lists/*-attribute`) are now labeled "on NBA 2K27". The live individual player
pages only keep a single historical *Overall* number per past edition (via a
"Ratings Over the Years" chart), not the full ~40-attribute breakdown we need
for validation.

However, individual 2kratings.com player pages (e.g. `/jayson-tatum`) *were*
crawled by the Internet Archive while NBA 2K26 was the current edition (mostly
Aug-Dec 2025), and at that time the page's full attribute breakdown reflected
NBA 2K26 (verified: the archived page `<title>` reads "... NBA 2K26 Rating").
So this script:

  1. Queries the Wayback Machine CDX API for all `2kratings.com/<player-slug>`
     captures between 2025-08-01 and 2026-03-01 (the NBA 2K26 window), picking
     one capture per player.
  2. Fetches each archived snapshot (`id_` flag = raw, unrewritten HTML).
  3. Verifies the page title says "NBA 2K26" (skips anything that doesn't,
     e.g. a stray early/late capture that slipped in already labeled 2K27).
  4. Parses bio info (nationality, team, position, height/weight/wingspan,
     draft/college, badges) and the full attribute breakdown (six categories:
     Outside Scoring, Athleticism, Inside Scoring, Playmaking, Defense,
     Rebounding, each with several sub-attributes) plus Overall/Potential.

Coverage limitation: the Wayback Machine did not crawl every current-roster
player during the 2K26 window (backup bench players, two-way contracts, and
players who joined/were traded mid-season are less likely to have been
crawled). This gives a real but incomplete slice of the league -- expect
roughly 350-450 players out of ~550 on active NBA rosters. This is documented
again in the README and notebooks.

Output: data/raw/2k26_ratings.csv
"""

import csv
import json
import os
import re
import time

from curl_cffi import requests as creq

CDX_URL = (
    "https://web.archive.org/cdx/search/cdx?url=2kratings.com&matchType=domain"
    "&output=json&from=20250801&to=20260301&filter=statuscode:200&collapse=urlkey&limit=200000"
)
IMPERSONATE = "chrome124"
SLEEP = 0.5

# Non-player paths / list pages / category pages we don't want to treat as player slugs.
EXCLUDE_SLUGS = {
    "teams", "lists", "current-teams", "all-time-teams", "all-decade-teams", "wnba-teams",
    "mynba-eras", "countries", "updates", "wp-content", "wp-json", "category", "tag",
    "author", "page", "confirm-email", "about", "contact", "privacy", "terms",
    "wnba-current-teams", "all-star", "draft", "rosters",
    "attributes-filter", "badges-filter", "compare-players",
}

# Attribute categories and the sub-attributes under them, in the order they
# appear on a player's "Attributes" tab.
CATEGORIES = [
    "Outside Scoring", "Athleticism", "Inside Scoring",
    "Playmaking", "Defense", "Rebounding",
]

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_PATH = os.path.join(RAW_DIR, "2k26_ratings.csv")


def get(url, timeout=30, retries=3):
    last_err = None
    for _ in range(retries):
        try:
            r = creq.get(url, impersonate=IMPERSONATE, timeout=timeout)
            return r
        except Exception as e:
            last_err = e
            time.sleep(2)
    raise last_err


def discover_player_slugs():
    """Query the CDX API for candidate NBA 2K26-era player-page slugs."""
    r = get(CDX_URL, timeout=60)
    data = json.loads(r.text)
    rows = data[1:] if data else []

    slug_to_ts = {}
    for urlkey, ts, orig, *_ in rows:
        path = urlkey.split(")/", 1)[1] if ")/" in urlkey else ""
        path = path.split("?")[0]
        if not re.match(r"^[a-z0-9-]+$", path):
            continue
        slug = path
        if slug in EXCLUDE_SLUGS:
            continue
        if any(ch.isdigit() for ch in slug):
            # Historical/classic-team player variants embed a season, e.g.
            # "al-harrington-2006-07-golden-state-warriors" -- skip those.
            continue
        if slug.endswith("-teams") or "all-time" in slug or "all-decade" in slug or "classic" in slug:
            continue
        slug_to_ts[slug] = ts  # collapse=urlkey already gives us one row per key
    return slug_to_ts


def clean_int(text):
    text = (text or "").strip()
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def parse_bio(soup, text):
    bio = {}

    def field(label, stop_at=None):
        idx = text.find(label)
        if idx == -1:
            return None
        chunk = text[idx + len(label):idx + len(label) + 120]
        return chunk.split("|")[0].strip()

    bio["nationality"] = field("Nationality:|")
    bio["team"] = field("Team:|")
    jersey = field("Jersey: #")
    bio["jersey"] = jersey

    # Position: "PF|/|SF" style around the "Position:|" marker
    idx = text.find("Position:|")
    if idx != -1:
        chunk = text[idx + len("Position:|"): idx + len("Position:|") + 40]
        parts = chunk.split("|Archetype:")[0].split("|")
        parts = [p for p in parts if p and p != "/"]
        bio["position"] = parts[0] if parts else None
        bio["position2"] = parts[1] if len(parts) > 1 else None

    bio["archetype"] = field("Archetype:|")

    height_raw = field("Height:|")
    if height_raw:
        m = re.match(r"([\d'\"]+)\s*\((\d+)cm\)", height_raw)
        bio["height_ftin"] = m.group(1) if m else height_raw
        bio["height_cm"] = int(m.group(2)) if m else None

    weight_raw = field("Weight:|")
    if weight_raw:
        m = re.match(r"(\d+)lbs\s*\((\d+)kg\)", weight_raw)
        bio["weight_lb"] = int(m.group(1)) if m else None
        bio["weight_kg"] = int(m.group(2)) if m else None

    wingspan_raw = field("Wingspan:|")
    if wingspan_raw:
        m = re.match(r"([\d'\"]+)\s*\((\d+)cm\)", wingspan_raw)
        bio["wingspan_ftin"] = m.group(1) if m else wingspan_raw
        bio["wingspan_cm"] = int(m.group(2)) if m else None

    yrs = field("Year(s) in the NBA: ")
    bio["years_in_nba"] = clean_int(yrs) if yrs else None

    bio["birthdate"] = field("Birthdate: ")
    bio["hometown"] = field("Hometown: ")
    bio["college"] = field("Prior to  NBA:\n Duke".split(":")[0] + ": ") if False else field("Prior to  NBA:\n")

    return bio


def _split_value_label(box, full_text):
    """A card header / li reads as '<value> <Label...>' once flattened to text
    (the attribute-box value comes first in the archived page markup). Strip
    the leading value token (as it literally appears, e.g. "90" or "3,052")
    off the front of the text to recover the label.
    """
    value_text = box.get_text(strip=True)
    label = full_text
    if label.startswith(value_text):
        label = label[len(value_text):].strip()
    # A few rows have an extra hidden badge/exponent digit between the value
    # and the label (e.g. "93 6 Close Shot"); strip any further leading
    # digits/punctuation so the label starts at the first letter.
    label = re.sub(r"^[\d,\s]+", "", label)
    return label


def parse_attributes(soup):
    """Parse the Attributes tab: category composites + sub-attribute values.

    Archived (2025-era) 2kratings.com pages render each header/row as
    `<attribute-box value> <label text>`, e.g. "90 Outside Scoring" or
    "90 Close Shot", rather than label-then-value like the modern live site.
    """
    attrs = {}
    nav = soup.find("div", id="nav-attributes")
    if nav is None:
        return attrs

    for card in nav.find_all("div", class_="card"):
        header = card.find("h4")
        if header is None:
            continue
        header_box = header.find("span", class_="attribute-box")
        if header_box is None:
            continue
        header_text = header.get_text(" ", strip=True)
        cat_name = _split_value_label(header_box, header_text)

        if cat_name == "Total Attributes":
            attrs["total_attributes"] = clean_int(header_box.get_text(strip=True).replace(",", ""))
            continue
        if cat_name == "Potential":
            val = header_box.get_text(strip=True)
            attrs["potential_grade"] = val
            attrs["potential"] = clean_int(val)
            continue
        if cat_name == "Intangibles":
            attrs["intangibles"] = clean_int(header_box.get_text(strip=True))
            continue

        if cat_name in CATEGORIES:
            key = "cat_" + cat_name.lower().replace(" ", "_")
            attrs[key] = clean_int(header_box.get_text(strip=True))

            for li in card.find_all("li"):
                value_box = li.find("span", class_="attribute-box")
                if value_box is None:
                    continue
                li_text = li.get_text(" ", strip=True)
                label = _split_value_label(value_box, li_text)
                if not label:
                    continue
                attr_key = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
                attrs[attr_key] = clean_int(value_box.get_text(strip=True))

    return attrs


def parse_player_page(slug, html):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text() if title_tag else ""
    if "NBA 2K26" not in title:
        return None, title

    name = title.split(" NBA 2K26")[0].strip()
    text = soup.get_text("|", strip=True)

    row = {"slug": slug, "name": name}
    row.update(parse_bio(soup, text))

    # Overall rating: the bio card's badge row reads "...|<n>|OVERALL|..."
    m = re.search(r"\|(\d+)\|OVERALL\|", text)
    if not m:
        m = re.search(r"NBA 2K26 Rating is (\d+)", text)
    row["overall"] = int(m.group(1)) if m else None

    row.update(parse_attributes(soup))
    return row, title


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("Discovering NBA 2K26-era player pages via Wayback CDX API...")
    slug_to_ts = discover_player_slugs()
    print(f"Found {len(slug_to_ts)} candidate player slugs")

    rows = []
    skipped = []
    for i, (slug, ts) in enumerate(sorted(slug_to_ts.items())):
        url = f"https://web.archive.org/web/{ts}id_/https://www.2kratings.com/{slug}"
        try:
            r = get(url, timeout=30)
        except Exception as e:
            print(f"  [{i+1}/{len(slug_to_ts)}] {slug}: FAILED ({e})")
            skipped.append(slug)
            continue

        if r.status_code != 200:
            skipped.append(slug)
            continue

        try:
            row, title = parse_player_page(slug, r.text)
        except Exception as e:
            print(f"  [{i+1}/{len(slug_to_ts)}] {slug}: parse error ({e})")
            skipped.append(slug)
            continue

        if row is None:
            print(f"  [{i+1}/{len(slug_to_ts)}] {slug}: skipped, title={title!r}")
            skipped.append(slug)
            continue

        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(slug_to_ts)}] ...{len(rows)} parsed so far")
        time.sleep(SLEEP)

    print(f"Parsed {len(rows)} players, skipped {len(skipped)}")

    fieldnames = set()
    for row in rows:
        fieldnames.update(row.keys())
    # Keep a stable, readable column order: identity/bio first, then attributes.
    preferred = [
        "slug", "name", "team", "nationality", "position", "position2", "archetype",
        "height_ftin", "height_cm", "weight_lb", "weight_kg", "wingspan_ftin", "wingspan_cm",
        "years_in_nba", "birthdate", "hometown", "college", "jersey",
        "overall", "potential_grade", "total_attributes", "intangibles",
        "cat_outside_scoring", "cat_athleticism", "cat_inside_scoring",
        "cat_playmaking", "cat_defense", "cat_rebounding",
    ]
    remaining = sorted(fieldnames - set(preferred))
    fieldnames = preferred + remaining

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
