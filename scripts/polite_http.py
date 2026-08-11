"""Shared, deliberately well-behaved HTTP helpers for the two scrapers.

Both scrapers here talk to sites that block a bare `requests` call, and both
previously handled that by impersonating a Chrome TLS fingerprint via
`curl_cffi` while sending nothing that identified the project or gave an
operator a way to contact us -- and by retrying only on transport exceptions,
so an HTTP 429 or 503 body was parsed as if it were a successful page and
turned into an empty/garbage row.

This module fixes all three of those:

* `robots_allows()` fetches and parses the target host's robots.txt (cached per
  host) and is checked before any fetch; `RESPECT_ROBOTS=0` in the environment
  can override it for a one-off manual run, but the default is to obey.
* Every request sends a descriptive `User-Agent` carrying the project URL and a
  contact address, plus a `From` header, so an operator seeing this traffic can
  identify and reach whoever is running it.
* `get()` treats 429 and 5xx as retryable failures with exponential backoff and
  honours a `Retry-After` header, rather than handing the error body to the
  parser.

`curl_cffi`'s Chrome impersonation is still used, because both sites' edge
(Cloudflare in front of 2kratings.com/the Wayback Machine, and HoopsHype)
rejects the default TLS fingerprint outright. That is a real caveat, documented
in the README: it is bot *detection* evasion, not paywall or authentication
evasion, and it is paired here with rate limiting, robots.txt compliance and
identifying contact info so the traffic is polite and attributable rather than
covert.
"""

import os
import time
import urllib.robotparser
from urllib.parse import urlparse

from curl_cffi import requests as creq

PROJECT_URL = "https://github.com/thompgt/nba2k26-analysis"
CONTACT_EMAIL = "thomas.pequegnot04@gmail.com"

USER_AGENT = (
    "nba2k26-analysis/1.0 (research/academic; "
    f"+{PROJECT_URL}; contact: {CONTACT_EMAIL}) "
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "From": CONTACT_EMAIL}

IMPERSONATE = "chrome124"
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_RETRY_AFTER = 120  # never sleep longer than this on a Retry-After header

_robots_cache = {}


class FetchError(RuntimeError):
    """A URL could not be fetched successfully after all retries."""


class RobotsDisallowed(RuntimeError):
    """robots.txt for the target host disallows this URL for our User-Agent."""


def _robots_for(url):
    """Fetch + parse robots.txt for `url`'s host, cached. None if unavailable."""
    parts = urlparse(url)
    root = f"{parts.scheme}://{parts.netloc}"
    if root in _robots_cache:
        return _robots_cache[root]

    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(root + "/robots.txt")
    try:
        r = creq.get(root + "/robots.txt", impersonate=IMPERSONATE,
                     headers=HEADERS, timeout=20)
        if r.status_code == 200:
            parser.parse(r.text.splitlines())
        elif 400 <= r.status_code < 500:
            # RFC 9309: 4xx means "no restrictions".
            parser.parse([])
        else:
            parser = None
    except Exception:
        parser = None

    _robots_cache[root] = parser
    return parser


def robots_allows(url, user_agent=USER_AGENT):
    """True if robots.txt permits fetching `url`.

    A missing or unreachable robots.txt is treated as permissive (RFC 9309),
    which is also the behaviour when `RESPECT_ROBOTS=0` is set.
    """
    if os.environ.get("RESPECT_ROBOTS", "1") == "0":
        return True
    parser = _robots_for(url)
    if parser is None:
        return True
    return parser.can_fetch(user_agent, url)


def crawl_delay(url, default=0.5, user_agent=USER_AGENT):
    """robots.txt Crawl-delay for this host, or `default` if none is declared."""
    parser = _robots_for(url)
    if parser is None:
        return default
    try:
        declared = parser.crawl_delay(user_agent) or parser.crawl_delay("*")
    except Exception:
        declared = None
    return max(float(declared), default) if declared else default


def get(url, timeout=30, retries=3, backoff=2.0, check_robots=True):
    """Fetch `url` politely.

    Raises `RobotsDisallowed` if robots.txt forbids it, and `FetchError` if all
    attempts fail -- crucially including the case where the server answered 429
    or 5xx, which the previous implementation returned to the caller as if it
    were a usable page.
    """
    if check_robots and not robots_allows(url):
        raise RobotsDisallowed(f"robots.txt disallows {url} for our User-Agent")

    last_err = None
    for attempt in range(retries):
        try:
            r = creq.get(url, impersonate=IMPERSONATE, headers=HEADERS, timeout=timeout)
        except Exception as e:
            last_err = f"transport error: {e}"
        else:
            if r.status_code not in RETRY_STATUSES:
                return r
            last_err = f"HTTP {r.status_code}"
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = min(float(retry_after), MAX_RETRY_AFTER)
                except ValueError:
                    wait = backoff * (2 ** attempt)
                if attempt < retries - 1:
                    time.sleep(wait)
                    continue

        if attempt < retries - 1:
            time.sleep(backoff * (2 ** attempt))

    raise FetchError(f"{url}: giving up after {retries} attempts ({last_err})")
