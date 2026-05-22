# ============================================================
# WHAT THIS FILE IS
# ============================================================
# A script that talks to Google's Places API to collect
# real restaurant data from Pune.
#
# HOW TO RUN:
#   Step 1: Get a Google Places API key (free, 5 mins)
#           console.cloud.google.com → Enable "Places API" → Credentials
#   Step 2: In your terminal, set the key as an environment variable:
#           export PLACES_API_KEY="AIza...your_key_here"
#   Step 3: python collect_places.py
# ============================================================

import os           # lets you read environment variables (like your API key)
import json         # converts between Python dicts and JSON text
import time         # lets you pause execution (time.sleep)
import logging      # prints status messages with timestamps
import requests     # makes HTTP calls — this is how you "talk" to an API
import pandas as pd # turns lists of dicts into a table (DataFrame)
from pathlib import Path  # cleaner way to work with file paths
from dotenv import load_dotenv  # loads environment variables from a .env file

load_dotenv()  # read environment variables from .env file (if it exists)

# ── CONCEPT: Logging ──────────────────────────────────────────
# Instead of print(), we use logging. It automatically adds
# timestamps and severity levels (INFO, WARNING, ERROR).
# You will see lines like: "10:32:01  INFO  Collecting: Baner"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
# __name__ is a built-in Python variable. When this file runs
# directly, __name__ equals "__main__". In logging it labels
# which module each message came from.


# ── CONCEPT: Environment Variables ───────────────────────────
# NEVER put API keys directly in code. If you push to GitHub,
# they become public. Instead, read them from the environment.
# os.getenv("KEY", "fallback") returns the value or "fallback"
# if the variable doesn't exist.
API_KEY  = os.getenv("PLACES_API_KEY", "")
BASE_URL = "https://maps.googleapis.com/maps/api/place"

# ── CONCEPT: Constants at the top ────────────────────────────
# Put magic numbers here, not buried in functions.
# Makes them easy to find and change.
RADIUS_M = 1500    # search radius in metres per locality
DELAY_S  = 0.3     # seconds to sleep between API calls
OUT_DIR  = Path("data")   # output folder


# ── CONCEPT: Data as a list of dicts ─────────────────────────
# Each locality is a dict with three keys.
# A list of dicts is the natural Python structure for tabular
# data before you load it into pandas.
LOCALITIES = [
    {"name": "Koregaon Park", "lat": 18.5362, "lon": 73.8938},
    {"name": "Baner",         "lat": 18.5590, "lon": 73.7868},
    {"name": "Viman Nagar",   "lat": 18.5679, "lon": 73.9143},
    {"name": "FC Road",       "lat": 18.5195, "lon": 73.8398},
    {"name": "Kothrud",       "lat": 18.5074, "lon": 73.8077},
    {"name": "Hinjewadi",     "lat": 18.5912, "lon": 73.7389},
    {"name": "Aundh",         "lat": 18.5590, "lon": 73.8079},
    {"name": "Camp",          "lat": 18.5167, "lon": 73.8794},
]


# ============================================================
# FUNCTION 1: nearby_search
# ============================================================
# WHAT: Calls Google's Nearby Search endpoint.
#       Returns a dict (Python's version of JSON).
#
# HOW AN HTTP REQUEST WORKS:
#   Your script → internet → Google's server
#   Google's server looks up restaurants → sends back JSON text
#   requests.get() receives that text → converts it to a dict
#
# PARAMETERS:
#   lat, lon     — centre point to search around
#   page_token   — if Google gave you a token, pass it here
#                  to get the next page of 20 results
def nearby_search(lat: float, lon: float, page_token: str = None) -> dict:

    # Build the query parameters — these go into the URL
    # like: ...nearbysearch/json?location=18.53,73.89&radius=1500&...
    params = {
        "location": f"{lat},{lon}",   # f-string: inserts variables into string
        "radius":   RADIUS_M,
        "type":     "restaurant",
        "key":      API_KEY,
    }

    # If we're fetching page 2 or 3, Google only needs the token
    if page_token:
        params = {"pagetoken": page_token, "key": API_KEY}

    # CONCEPT: requests.get()
    # This is the actual HTTP GET request. Like typing a URL in a browser,
    # but from code. timeout=10 means "give up after 10 seconds".
    response = requests.get(
        f"{BASE_URL}/nearbysearch/json",
        params=params,
        timeout=10
    )

    # CONCEPT: raise_for_status()
    # If Google returned an error (like 403 Forbidden or 500 Server Error),
    # this line raises an exception so your code doesn't silently get
    # wrong data and continue. Always do this after an API call.
    response.raise_for_status()

    # .json() parses the response text into a Python dict.
    # Try: print(response.text) before this line to see raw JSON.
    return response.json()


# ============================================================
# FUNCTION 2: place_details
# ============================================================
# WHAT: The Nearby Search above returns basic info.
#       This call fetches richer data for ONE specific restaurant
#       using its unique place_id.
#
# WHY TWO CALLS?
#   Google's API is designed this way to save bandwidth.
#   You search broadly first, then fetch details only for what you need.
def place_details(place_id: str) -> dict:

    params = {
        "place_id": place_id,
        # "fields" controls exactly what you get back.
        # Only request what you need — extra fields cost API credits.
        "fields": (
            "name,formatted_address,geometry,rating,"
            "user_ratings_total,price_level,"
            "opening_hours,types,business_status"
        ),
        "key": API_KEY,
    }

    response = requests.get(
        f"{BASE_URL}/details/json",
        params=params,
        timeout=10
    )
    response.raise_for_status()

    # The details response wraps everything in a "result" key.
    # .get("result", {}) safely returns {} if "result" is missing,
    # instead of crashing with a KeyError.
    return response.json().get("result", {})


# ============================================================
# FUNCTION 3: parse_hours
# ============================================================
# WHAT: Google returns opening hours as a list of strings:
#   ["Monday: 11:00 AM – 11:00 PM", "Tuesday: ...", ...]
#   We join them into one string separated by " | "
#
# CONCEPT: Defensive programming with .get()
#   Always use dict.get(key, fallback) when the key might be
#   missing. Direct access dict[key] raises KeyError if absent.
def parse_hours(opening_hours: dict) -> str:
    if not opening_hours:   # None or empty dict both evaluate to False
        return ""
    texts = opening_hours.get("weekday_text", [])
    return " | ".join(texts)   # join() takes a list and glues it with separator


# ============================================================
# FUNCTION 4: parse_cuisine
# ============================================================
# WHAT: Google gives vague "types" like ["restaurant", "food",
#       "point_of_interest"]. We filter out the useless ones
#       and take the first specific tag if any.
#
# NOTE: Google doesn't give specific cuisine (no "biryani").
#       We'll enrich this from Kaggle data in the cleaning step.
def parse_cuisine(types: list) -> str:
    skip = {"restaurant", "food", "point_of_interest",
            "establishment", "store"}
    # List comprehension: build a new list of items that pass the filter.
    # [expression  for item in list  if condition]
    specific = [t for t in (types or []) if t not in skip]
    return specific[0] if specific else "restaurant"


# ============================================================
# FUNCTION 5: flatten
# ============================================================
# WHAT: Takes a raw API result dict (messy, nested) and turns
#       it into a flat dict with only the columns we want.
#       Each call to flatten() produces ONE row in our CSV.
#
# CONCEPT: Nested vs flat data
#   Raw API:  {"geometry": {"location": {"lat": 18.5, "lng": 73.8}}}
#   Flattened: {"lat": 18.5, "lon": 73.8}
#   Flat is what pandas and ML models expect.
def flatten(place: dict, locality: str) -> dict:

    # Drill into the nested geometry dict safely
    geo = place.get("geometry", {}).get("location", {})

    return {
        "place_id":        place.get("place_id", ""),
        "restaurant_name": place.get("name", ""),
        "locality":        locality,
        "address":         place.get("formatted_address", ""),
        "lat":             geo.get("lat"),
        "lon":             geo.get("lng"),
        "rating":          place.get("rating"),          # float, e.g. 4.2
        "review_count":    place.get("user_ratings_total"),
        "price_range":     place.get("price_level"),     # int 1-4 or None
        "cuisine_tag":     parse_cuisine(place.get("types", [])),
        "opening_hours":   parse_hours(place.get("opening_hours", {})),
        "business_status": place.get("business_status", ""),
    }


# ============================================================
# FUNCTION 6: collect_locality
# ============================================================
# WHAT: The core collection loop for a single locality.
#       Handles pagination (up to 3 pages = 60 restaurants),
#       deduplication, and rate limiting.
#
# CONCEPT: Sets for deduplication
#   seen_ids is a Python set — like a list but:
#   - No duplicates allowed
#   - Checking "is X in here?" is O(1) — instant, regardless of size
#   We pass it in from outside so it persists across all localities.
def collect_locality(locality: dict, seen_ids: set) -> list:

    rows = []
    page_token = None

    # We allow up to 3 pages. Each page = 20 results max.
    # So one locality can give us up to 60 restaurants.
    for page_num in range(3):

        # Google requires a 2-second pause before using a page token.
        # If you don't wait, the API returns INVALID_REQUEST.
        # This is an example of a real-world API quirk you must know.
        if page_token:
            time.sleep(2.2)

        # Make the API call
        data = nearby_search(locality["lat"], locality["lon"], page_token)

        # CONCEPT: Status checking
        # Google doesn't always raise HTTP errors for API-level problems.
        # A 200 OK response can still contain "status": "REQUEST_DENIED".
        # Always check the API's own status field.
        status = data.get("status")

        if status == "REQUEST_DENIED":
            log.error("API key rejected. Check your key and that Places API is enabled.")
            raise SystemExit(1)   # stop the whole program immediately

        if status not in ("OK", "ZERO_RESULTS"):
            log.warning(f"Unexpected status '{status}' for {locality['name']}")
            break

        results = data.get("results", [])
        log.info(f"  Page {page_num + 1}: {len(results)} places found")

        # Loop through each restaurant in this page
        for place in results:
            pid = place.get("place_id", "")

            # Skip if we've seen this restaurant before
            # (can happen at locality borders)
            if pid in seen_ids:
                continue
            seen_ids.add(pid)   # mark as seen

            # Be polite to the API — don't hammer it
            time.sleep(DELAY_S)

            # Fetch richer details for this specific restaurant
            detail = place_details(pid)

            # Merge the search result and detail result together.
            # {**a, **b} creates a new dict combining both.
            # Keys in b overwrite keys in a if there's overlap.
            merged = {**place, **detail}

            row = flatten(merged, locality["name"])
            rows.append(row)

        # Get the token for the next page (or None if this is the last)
        page_token = data.get("next_page_token")

        if not page_token:
            break   # no more pages, stop

    return rows


# ============================================================
# FUNCTION 7: run  (the main function)
# ============================================================
# CONCEPT: Why have a main function?
#   Putting everything in a function (instead of loose at the
#   top level) means:
#   1. You can import this file elsewhere without it executing
#   2. Variables are local, not global — cleaner
#   3. Easier to test
def run():

    # Guard: fail early with a clear message, not halfway through
    if not API_KEY:
        log.error("PLACES_API_KEY not set. Run: export PLACES_API_KEY='your_key' ")
        raise SystemExit(1)

    # Create the output folder if it doesn't exist
    # exist_ok=True means don't crash if it already exists
    OUT_DIR.mkdir(exist_ok=True)

    all_rows = []    # will collect every flat restaurant dict
    seen_ids = set() # shared across all localities for dedup

    for locality in LOCALITIES:
        log.info(f"Collecting: {locality['name']}")

        # CONCEPT: try/except
        # requests.RequestException covers all network errors:
        # no internet, timeout, DNS failure, etc.
        # We catch it so ONE bad locality doesn't stop all others.
        try:
            rows = collect_locality(locality, seen_ids)
            all_rows.extend(rows)   # extend adds multiple items; append adds one
            log.info(f"  → {len(rows)} new restaurants (total: {len(all_rows)})")
        except requests.RequestException as e:
            log.warning(f"  Network error for {locality['name']}: {e}. Skipping.")

        time.sleep(0.5)  # brief pause between localities

    # ── Save raw JSON ──────────────────────────────────────────
    # Keep the raw data always. If your CSV cleaning breaks,
    # you can re-run the cleaning without re-calling the API.
    raw_path = OUT_DIR / "raw_places.json"
    with open(raw_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    log.info(f"Raw JSON saved → {raw_path}")

    # ── Save CSV via pandas ────────────────────────────────────
    # pd.DataFrame() takes a list of dicts and makes a table.
    # Each dict becomes a row. Keys become column names.
    df = pd.DataFrame(all_rows)
    csv_path = OUT_DIR / "data/raw/restaurants_raw.csv"
    df.to_csv(csv_path, index=False)
    # index=False: don't write the row numbers (0,1,2...) as a column

    # ── Print a quick summary ──────────────────────────────────
    log.info(f"Total restaurants : {len(df)}")
    log.info(f"Localities covered: {df['locality'].nunique()}")
    log.info(f"Missing rating    : {df['rating'].isna().sum()}")
    log.info(f"Missing price     : {df['price_range'].isna().sum()}")


# ── CONCEPT: if __name__ == "__main__" ───────────────────────
# Python sets __name__ = "__main__" only when you RUN this file.
# If another file imports it, __name__ = "collect_places" instead.
# This pattern means: only execute run() when running directly.
if __name__ == "__main__":
    run()