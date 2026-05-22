import os
import json
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

API_KEY  = os.getenv("PLACES_API_KEY", "")
BASE_URL = "https://maps.googleapis.com/maps/api/place"
RADIUS_M = 1500
OUT_DIR  = Path("data")

# ONLY the failed localities
FAILED_LOCALITIES = [
    {"name": "FC Road",  "lat": 18.5195, "lon": 73.8398},
    {"name": "Kothrud",  "lat": 18.5074, "lon": 73.8077},
    {"name": "Aundh",    "lat": 18.5590, "lon": 73.8079},
    {"name": "Camp",     "lat": 18.5167, "lon": 73.8794},
]

# CONCEPT: Retry with longer timeout
# The original script used timeout=10 seconds.
# We increase it to 30 here — slower connection gets more time.
# We also add retries — if one attempt fails, try again up to 3 times.
def get_with_retry(url: str, params: dict, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            log.warning(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                wait = (attempt + 1) * 5   # wait 5s, then 10s, then 15s
                log.info(f"  Waiting {wait}s before retry...")
                time.sleep(wait)
    raise requests.RequestException(f"All {retries} attempts failed")


def nearby_search(lat, lon, page_token=None):
    params = {
        "location": f"{lat},{lon}",
        "radius":   RADIUS_M,
        "type":     "restaurant",
        "key":      API_KEY,
    }
    if page_token:
        params = {"pagetoken": page_token, "key": API_KEY}
    return get_with_retry(f"{BASE_URL}/nearbysearch/json", params)


def place_details(place_id):
    params = {
        "place_id": place_id,
        "fields": (
            "name,formatted_address,geometry,rating,"
            "user_ratings_total,price_level,"
            "opening_hours,types,business_status"
        ),
        "key": API_KEY,
    }
    return get_with_retry(f"{BASE_URL}/details/json", params)


def parse_hours(opening_hours):
    if not opening_hours:
        return ""
    return " | ".join(opening_hours.get("weekday_text", []))


def parse_cuisine(types):
    skip = {"restaurant", "food", "point_of_interest", "establishment", "store"}
    specific = [t for t in (types or []) if t not in skip]
    return specific[0] if specific else "restaurant"


def flatten(place, locality):
    geo = place.get("geometry", {}).get("location", {})
    return {
        "place_id":        place.get("place_id", ""),
        "restaurant_name": place.get("name", ""),
        "locality":        locality,
        "address":         place.get("formatted_address", ""),
        "lat":             geo.get("lat"),
        "lon":             geo.get("lng"),
        "rating":          place.get("rating"),
        "review_count":    place.get("user_ratings_total"),
        "price_range":     place.get("price_level"),
        "cuisine_tag":     parse_cuisine(place.get("types", [])),
        "opening_hours":   parse_hours(place.get("opening_hours", {})),
        "business_status": place.get("business_status", ""),
    }


def run():
    if not API_KEY:
        log.error("PLACES_API_KEY not set.")
        raise SystemExit(1)

    # Load existing data so we don't duplicate restaurants
    existing_csv = OUT_DIR / "restaurants_raw.csv"
    existing_df  = pd.read_csv(existing_csv)
    seen_ids     = set(existing_df["place_id"].tolist())
    log.info(f"Loaded {len(existing_df)} existing restaurants. Will skip duplicates.")

    new_rows = []

    for locality in FAILED_LOCALITIES:
        log.info(f"Collecting: {locality['name']}")
        page_token = None

        for page_num in range(3):
            if page_token:
                time.sleep(2.2)

            try:
                data   = nearby_search(locality["lat"], locality["lon"], page_token)
                status = data.get("status")

                if status == "REQUEST_DENIED":
                    log.error("API key rejected.")
                    raise SystemExit(1)

                if status not in ("OK", "ZERO_RESULTS"):
                    log.warning(f"Status '{status}' for {locality['name']}")
                    break

                results = data.get("results", [])
                log.info(f"  Page {page_num + 1}: {len(results)} places found")

                for place in results:
                    pid = place.get("place_id", "")
                    if pid in seen_ids:
                        continue
                    seen_ids.add(pid)

                    time.sleep(0.3)
                    detail = place_details(pid)
                    merged = {**place, **detail}
                    new_rows.append(flatten(merged, locality["name"]))

                page_token = data.get("next_page_token")
                if not page_token:
                    break

            except requests.RequestException as e:
                log.warning(f"  Failed {locality['name']}: {e}. Moving on.")
                break

        log.info(f"  → collected so far: {len(new_rows)} new restaurants")
        time.sleep(1)   # longer pause between localities

    if new_rows:
        # Merge new rows into existing CSV
        new_df    = pd.DataFrame(new_rows)
        combined  = pd.concat([existing_df, new_df], ignore_index=True)
        combined.to_csv(existing_csv, index=False)
        log.info(f"\nDone. Added {len(new_rows)} restaurants.")
        log.info(f"Total now: {len(combined)} across {combined['locality'].nunique()} localities.")
    else:
        log.warning("No new restaurants collected.")


if __name__ == "__main__":
    run()