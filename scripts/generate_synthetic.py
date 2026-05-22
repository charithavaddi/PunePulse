# ============================================================
# WHAT THIS FILE IS
# ============================================================
# A script that generates fake-but-realistic restaurant data
# for Pune. No API needed — pure Python + math.
#
# WHY SYNTHETIC DATA?
#   You can't get real hourly footfall data for Pune restaurants.
#   So you encode real knowledge (Pune eats dinner at 9pm,
#   monsoon kills dine-in, Ganesh Chaturthi is huge) into math,
#   then add randomness to make it look like real data.
#
# HOW TO RUN:
#   python generate_synthetic.py
# ============================================================

import numpy as np           # numerical computing, arrays, random numbers
import pandas as pd          # DataFrames — tabular data
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# CONCEPT: Random seed
# rng = random number generator. seed=42 means: every time you
# run this script, you get the SAME random numbers.
# This is called "reproducibility" — critical in data science.
# Without a seed, your CSV changes every run, making debugging hell.
rng = np.random.default_rng(seed=42)

OUT_DIR = Path("data")


# ============================================================
# RESTAURANT PROFILES
# ============================================================
# CONCEPT: Tuples inside a list
# Each restaurant is a tuple of fixed values in a known order.
# We'll convert these to dicts later (easier to work with by name).
#
# Order: name, locality, cuisine, price_range(1-4),
#        capacity(max/hour), base_orders(avg weekday noon), lat, lon
RESTAURANT_PROFILES = [
    ("Vaishali",        "FC Road",       "South Indian",  2, 120, 90,  18.5199, 73.8405),
    ("Cafe Goodluck",   "FC Road",       "Irani Cafe",    1, 80,  60,  18.5185, 73.8415),
    ("Shabree",         "Kothrud",       "Maharashtrian", 2, 100, 70,  18.5070, 73.8080),
    ("Malaka Spice",    "Koregaon Park", "Pan Asian",     3, 80,  55,  18.5370, 73.8950),
    ("Panchali",        "Aundh",         "North Indian",  2, 90,  65,  18.5595, 73.8075),
    ("Sujata Mastani",  "Shivajinagar",  "Desserts",      1, 150, 120, 18.5310, 73.8480),
    ("The Urban Cafe",  "Baner",         "Cafe",          3, 70,  45,  18.5592, 73.7870),
    ("Burger Factory",  "Hinjewadi",     "Fast Food",     2, 110, 85,  18.5915, 73.7385),
    ("Spice Garden",    "Hadapsar",      "Biryani",       2, 130, 100, 18.5020, 73.9255),
    ("Irani Chai Corner","Camp",         "Irani Cafe",    1, 60,  50,  18.5170, 73.8800),
]

START_DATE = datetime(2024, 1, 1)
END_DATE   = datetime(2024, 12, 31)

# CONCEPT: Dict with tuple values
# Key = date string. Value = (festival name, demand multiplier).
# A multiplier of 1.6 means 60% more customers than a normal day.
FESTIVALS = {
    "2024-03-25": ("Gudi Padwa",       1.40),  # Pune's new year
    "2024-09-07": ("Ganesh Chaturthi", 1.60),  # biggest Pune festival
    "2024-09-08": ("Ganesh Chaturthi", 1.55),
    "2024-09-09": ("Ganesh Chaturthi", 1.45),
    "2024-11-02": ("Diwali",           1.70),
    "2024-11-03": ("Diwali",           1.65),
    "2024-12-31": ("New Year Eve",     1.80),
}

# Months when monsoon hits Pune (June=6 through September=9)
MONSOON_MONTHS = {6, 7, 8, 9}


# ============================================================
# FUNCTION 1: hourly_weights
# ============================================================
# WHAT: Returns a 24-element array (one value per hour 0-23).
#       Each value = fraction of daily orders in that hour.
#       All values sum to exactly 1.0.
#
# CONCEPT: Probability distribution
#   This is the shape of demand across a day.
#   Hour 13 (1pm) has weight 0.14 → 14% of daily orders happen then.
#   This is called a "demand curve" or "load profile".
#   In ML later, this becomes a FEATURE — a signal the model learns from.
def hourly_weights(is_weekend: bool) -> np.ndarray:

    # np.zeros(24) creates an array of 24 zeros: [0,0,0,...,0]
    weights = np.zeros(24)

    # Fill in non-zero hours
    weights[8]  = 0.03   # light breakfast
    weights[9]  = 0.05
    weights[10] = 0.04

    # Lunch: stronger on weekdays (office workers), moderate on weekends
    weights[12] = 0.10 if not is_weekend else 0.08
    weights[13] = 0.14 if not is_weekend else 0.10
    weights[14] = 0.10 if not is_weekend else 0.08

    # Evening snacks — the chai/vada pav window, very Pune
    weights[17] = 0.06
    weights[18] = 0.07
    weights[19] = 0.06

    # Dinner peak — later on weekends
    weights[20] = 0.10
    weights[21] = 0.12
    weights[22] = 0.08 if is_weekend else 0.06
    weights[23] = 0.04 if is_weekend else 0.02

    # Sunday brunch bump (only on weekends)
    if is_weekend:
        weights[11] = 0.06

    # CONCEPT: Normalisation
    # After setting values manually, they might not sum to exactly 1.0.
    # Dividing by the sum forces them to sum to 1.0.
    # This is called normalising, and you'll do it constantly in ML.
    weights = weights / weights.sum()

    return weights


# ============================================================
# FUNCTION 2: generate_footfall
# ============================================================
# WHAT: The main simulation loop.
#       For each restaurant × each day × each hour,
#       compute expected footfall using multipliers,
#       then add realistic noise via Poisson distribution.
#
# CONCEPT: Multiplicative demand model
#   final_demand = base × weekend_mult × monsoon_mult × festival_mult × seasonal_mult × noise
#   Each factor either boosts or reduces the base.
#   This is the simplest demand model — and also the most interpretable.
def generate_footfall(restaurants: list, dates: pd.DatetimeIndex) -> pd.DataFrame:

    rows = []

    for rest in restaurants:
        log.info(f"  Generating footfall for: {rest['restaurant_name']}")
        base = rest["base_orders"]   # average orders on a normal weekday at noon

        for date in dates:
            date_str   = date.strftime("%Y-%m-%d")   # "2024-09-07"
            is_weekend = date.weekday() >= 5          # Mon=0...Sat=5, Sun=6
            month      = date.month

            # ── Build the daily demand multiplier ──────────────
            # Start at 1.0 — meaning "no change from base"
            day_mult = 1.0

            # Weekend boost: people eat out more
            if is_weekend:
                # rng.uniform(low, high) gives a random float between low and high
                # This adds realistic variation — not every weekend is the same
                day_mult *= rng.uniform(1.20, 1.40)

            # Monsoon dip: heavy rain keeps people home
            if month in MONSOON_MONTHS:
                day_mult *= rng.uniform(0.70, 0.85)

            # Festival spike: check today + 2 days after (effect tapers)
            for days_ago in range(3):
                check_date = (date - timedelta(days=days_ago)).strftime("%Y-%m-%d")
                if check_date in FESTIVALS:
                    _, festival_mult = FESTIVALS[check_date]
                    # taper: full effect on day 0, 85% on day 1, 70% on day 2
                    taper = 1.0 - days_ago * 0.15
                    day_mult *= (1 + (festival_mult - 1) * taper)
                    break

            # Seasonal trend (October-December is peak restaurant season in Pune)
            seasonal = {
                1: 0.85, 2: 0.90, 3: 1.05, 4: 1.00, 5: 1.00,
                6: 0.80, 7: 0.75, 8: 0.78, 9: 1.10,
                10: 1.15, 11: 1.20, 12: 1.25
            }
            day_mult *= seasonal[month]

            # Daily random noise (±12%) — real life is never perfectly predictable
            day_mult *= rng.uniform(0.88, 1.12)

            # Total expected customers today
            daily_total = base * day_mult
            # Cap at physical capacity: 14 peak hours × max covers/hour
            daily_total = min(daily_total, rest["capacity"] * 14)

            # Spread the daily total across 24 hours using our demand curve
            hw = hourly_weights(is_weekend)
            hourly_expected = daily_total * hw
            # hourly_expected is now an array of 24 floats,
            # e.g. [0, 0, 0, ..., 3.2, 8.7, 12.1, ...]

            # ── Generate each hourly row ────────────────────────
            for hour in range(24):
                expected = hourly_expected[hour]

                if expected < 0.5:
                    footfall = 0   # don't bother with near-zero hours
                else:
                    # CONCEPT: Poisson distribution
                    # Real customer arrivals follow a Poisson distribution.
                    # It takes a mean (lambda) and returns a random integer.
                    # Key property: variance = mean.
                    # So if expected=10, you might get 7, 8, 10, 12, 13...
                    # This is more realistic than adding simple random noise.
                    footfall = int(rng.poisson(expected))
                    # Physically can't exceed the restaurant's capacity
                    footfall = min(footfall, rest["capacity"])

                rows.append({
                    "restaurant_name": rest["restaurant_name"],
                    "locality":        rest["locality"],
                    "cuisine":         rest["cuisine"],
                    "date":            date_str,
                    "hour":            hour,
                    "footfall":        footfall,
                    "is_weekend":      int(is_weekend),  # 0 or 1, not True/False
                    "month":           month,
                    "is_monsoon":      int(month in MONSOON_MONTHS),
                    "festival":        FESTIVALS.get(date_str, ("none", 1.0))[0],
                    "day_of_week":     date.strftime("%A"),   # "Monday", "Tuesday"...
                })

    # CONCEPT: pd.DataFrame from list of dicts
    # pandas sees each dict as a row. All keys become columns.
    # Missing keys in any dict become NaN (Not a Number).
    return pd.DataFrame(rows)


# ============================================================
# FUNCTION 3: generate_orders
# ============================================================
# WHAT: Creates individual order records from the footfall data.
#       Each visitor places 1-3 dish orders.
#       We only simulate 90 days (not full year) to keep file size sane.
#
# CONCEPT: Why not generate an order per footfall row?
#   Full year × 10 restaurants × 24 hours × ~8 avg footfall = ~700k rows.
#   With 1-3 dishes each, that's 1-2 million rows.
#   For a portfolio project, 90-day sample (~100k orders) is plenty.
def generate_orders(footfall_df: pd.DataFrame, restaurants: list) -> pd.DataFrame:

    # Build a lookup: restaurant name → restaurant dict
    # This lets us find a restaurant's details in O(1) time
    rest_lookup = {r["restaurant_name"]: r for r in restaurants}

    # Cuisine → list of (dish_name, price_inr) tuples
    CUISINE_MENU = {
        "South Indian":  [("Masala Dosa", 80),   ("Idli Sambar", 60),  ("Filter Coffee", 30)],
        "Irani Cafe":    [("Bun Maska", 30),      ("Irani Chai", 20),   ("Kheema Pav", 80)],
        "Maharashtrian": [("Vada Pav", 20),       ("Misal Pav", 60),    ("Puran Poli", 50)],
        "Pan Asian":     [("Dimsum", 180),         ("Ramen", 250),       ("Thai Curry", 280)],
        "North Indian":  [("Butter Chicken", 260), ("Dal Makhani", 180), ("Naan", 40)],
        "Desserts":      [("Mastani", 80),         ("Ice Cream", 60),    ("Falooda", 90)],
        "Cafe":          [("Cold Coffee", 120),    ("Brownie", 90),      ("Sandwich", 110)],
        "Fast Food":     [("Burger", 120),         ("Fries", 60),        ("Shake", 90)],
        "Biryani":       [("Chicken Biryani", 180),("Veg Biryani", 140), ("Raita", 40)],
    }

    # Sample 90 random days from the full year
    all_dates = footfall_df["date"].unique()
    sample_size = min(90, len(all_dates))
    # np.random.choice picks random elements from an array
    sampled_dates = set(np.random.choice(all_dates, size=sample_size, replace=False))

    # Filter footfall to only those 90 days
    subset = footfall_df[footfall_df["date"].isin(sampled_dates)]

    rows = []
    order_id = 1

    # iterrows() gives you (index, row_as_Series) for each row
    # Note: iterrows() is slow on huge DataFrames. Fine here.
    for _, record in subset.iterrows():
        if record["footfall"] == 0:
            continue

        rest    = rest_lookup[record["restaurant_name"]]
        cuisine = rest["cuisine"]
        menu    = CUISINE_MENU.get(cuisine, CUISINE_MENU["Cafe"])  # fallback

        # One order per customer (footfall = number of customers that hour)
        for _ in range(record["footfall"]):

            # Each customer orders 1, 2, or 3 items
            # p=[0.3, 0.5, 0.2] = probabilities for each option
            n_items = int(rng.choice([1, 2, 3], p=[0.3, 0.5, 0.2]))

            # Pick n_items random dishes from the menu (with replacement)
            dish_indices = rng.choice(len(menu), size=n_items, replace=True)
            total_price  = sum(menu[i][1] for i in dish_indices)
            # Small price noise (±5%) — real prices vary slightly
            total_price  = round(total_price * rng.uniform(0.95, 1.05), 2)

            # Group size: solo / pair / family / large group
            group_size = int(rng.choice([1, 2, 4, 6], p=[0.25, 0.40, 0.25, 0.10]))

            rows.append({
                "order_id":        order_id,
                "restaurant_name": record["restaurant_name"],
                "locality":        record["locality"],
                "cuisine":         cuisine,
                "date":            record["date"],
                "hour":            record["hour"],
                "day_of_week":     record["day_of_week"],
                "is_weekend":      record["is_weekend"],
                "is_monsoon":      record["is_monsoon"],
                "festival":        record["festival"],
                "primary_dish":    menu[dish_indices[0]][0],
                "n_items":         n_items,
                "order_value_inr": total_price,
                "group_size":      group_size,
                "price_range":     rest["price_range"],
            })
            order_id += 1

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================
def run():
    OUT_DIR.mkdir(exist_ok=True)

    # Convert list-of-tuples to list-of-dicts for easier access by name
    restaurants = [
        {
            "restaurant_name": r[0], "locality":    r[1],
            "cuisine":         r[2],  "price_range": r[3],
            "capacity":        r[4],  "base_orders": r[5],
            "lat":             r[6],  "lon":         r[7],
        }
        for r in RESTAURANT_PROFILES
    ]

    # pd.date_range: generates every date between two dates
    # freq="D" means daily frequency
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    # ── Generate footfall ──────────────────────────────────────
    log.info("Generating footfall...")
    footfall_df = generate_footfall(restaurants, dates)
    footfall_path = OUT_DIR / "data/cleaned/synthetic_footfall.csv"
    footfall_df.to_csv(footfall_path, index=False)
    log.info(f"Footfall: {len(footfall_df):,} rows → {footfall_path}")

    # ── Generate orders ────────────────────────────────────────
    log.info("Generating orders (90-day sample)...")
    orders_df = generate_orders(footfall_df, restaurants)
    orders_path = OUT_DIR / "data/cleaned/synthetic_orders.csv"
    orders_df.to_csv(orders_path, index=False)
    log.info(f"Orders: {len(orders_df):,} rows → {orders_path}")

    # ── Sanity checks — always verify your generated data ──────
    # These help you catch bugs: if 2am is your peak hour, something's wrong.
    log.info("\n── Sanity check: avg footfall by hour ──")
    hourly_avg = footfall_df.groupby("hour")["footfall"].mean().round(1)
    for h in [8, 12, 13, 17, 20, 21]:
        log.info(f"  {h:02d}:00 → avg {hourly_avg[h]} customers")

    log.info("\n── Sanity check: monsoon impact ──")
    monsoon_comp = (
        footfall_df
        .groupby("is_monsoon")["footfall"]
        .mean()
        .round(2)
    )
    log.info(f"  Non-monsoon avg/hour : {monsoon_comp.get(0, 'N/A')}")
    log.info(f"  Monsoon avg/hour     : {monsoon_comp.get(1, 'N/A')}")


if __name__ == "__main__":
    run()