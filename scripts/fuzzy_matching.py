import pandas as pd
from rapidfuzz import fuzz
import os

# =====================================================
# CREATE OUTPUT DIRECTORY
# =====================================================

os.makedirs("data/merged", exist_ok=True)

# =====================================================
# LOAD DATASETS
# =====================================================

df1 = pd.read_csv("data/cleaned/restaurants_cleaned.csv")
df2 = pd.read_csv("data/cleaned/resta_data.csv")

print("\nDatasets Loaded Successfully")

print("\nrestaurants_cleaned shape:", df1.shape)
print("resta_data shape:", df2.shape)

# =====================================================
# COLUMN DEFINITIONS
# =====================================================

# restaurants_cleaned.csv
NAME1 = "restaurant_name"
ADDRESS1 = "address"

# resta_data.csv
NAME2 = "name"
ADDRESS2 = "address"

# =====================================================
# TEXT CLEANING FUNCTIONS
# =====================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    return (
        str(text)
        .lower()
        .strip()
        .replace("&", "and")
        .replace("-", " ")
        .replace(".", "")
        .replace(",", "")
    )


def clean_address(text):

    if pd.isna(text):
        return ""

    text = (
        str(text)
        .lower()
        .strip()
    )

    # Address standardization
    replacements = {
        "road": "rd",
        "street": "st",
        "lane": "ln",
        "avenue": "ave",
        "nagar": "ngr",
        "society": "soc",
        "phase": "ph",
        "building": "bldg",
        ",": "",
        ".": ""
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


# =====================================================
# NORMALIZE DATA
# =====================================================

print("\nCleaning text fields...")

# Dataset 1
df1["name_clean"] = df1[NAME1].apply(clean_text)
df1["address_clean"] = df1[ADDRESS1].apply(clean_address)

# Dataset 2
df2["name_clean"] = df2[NAME2].apply(clean_text)
df2["address_clean"] = df2[ADDRESS2].apply(clean_address)

# =====================================================
# FUZZY MATCHING
# =====================================================

print("\nStarting fuzzy matching...")

matches = []

# Matching threshold
THRESHOLD = 88

for idx1, row1 in df1.iterrows():

    best_score = 0
    best_match = None

    name1 = row1["name_clean"]
    address1 = row1["address_clean"]

    # -----------------------------------------
    # OPTIONAL BLOCKING
    # Compare only rows with similar address start
    # -----------------------------------------

    address_token = address1.split()[0] if address1 else ""

    candidate_rows = df2[
        df2["address_clean"].str.contains(
            address_token,
            na=False
        )
    ]

    # Fallback if no candidates found
    if len(candidate_rows) == 0:
        candidate_rows = df2

    # -----------------------------------------
    # MATCHING LOOP
    # -----------------------------------------

    for idx2, row2 in candidate_rows.iterrows():

        name2 = row2["name_clean"]
        address2 = row2["address_clean"]

        # -------------------------------------
        # NAME SIMILARITY
        # -------------------------------------

        name_score = fuzz.token_sort_ratio(
            name1,
            name2
        )

        # -------------------------------------
        # ADDRESS SIMILARITY
        # -------------------------------------

        address_score = fuzz.token_sort_ratio(
            address1,
            address2
        )

        # -------------------------------------
        # COMBINED WEIGHTED SCORE
        # -------------------------------------

        combined_score = (
            0.7 * name_score
            + 0.3 * address_score
        )

        # Track best match
        if combined_score > best_score:
            best_score = combined_score
            best_match = row2

    # -----------------------------------------
    # SAVE HIGH-CONFIDENCE MATCHES
    # -----------------------------------------

    if best_score >= THRESHOLD:

        merged_row = {

            # Dataset 1
            "place_id": row1["place_id"],
            "restaurant_name": row1["restaurant_name"],
            "locality": row1["locality"],
            "address_google": row1["address"],

            # Dataset 2
            "restId": best_match["restId"],
            "zomato_name": best_match["name"],
            "address_zomato": best_match["address"],

            # Match quality
            "match_score": round(best_score, 2),

            # Useful merged fields
            "rating_google": row1["rating"],
            "rating_zomato": best_match["aggregate_rating"],

            "review_count_google": row1["review_count"],
            "rating_votes_zomato": best_match["rating_votes"],

            "price_range": row1["price_range"],
            "business_type": row1["business_type"],
            "cuisine_type": row1["cuisine_type"],

            "cost_for_two": best_match["cost_for_two"],

            "lat": row1["lat"],
            "lon": row1["lon"]
        }

        matches.append(merged_row)

# =====================================================
# CREATE MATCHED DATAFRAME
# =====================================================

matched_df = pd.DataFrame(matches)

# =====================================================
# REMOVE DUPLICATE MATCHES
# =====================================================

matched_df.drop_duplicates(
    subset=["place_id", "restId"],
    inplace=True
)

# =====================================================
# SAVE OUTPUT
# =====================================================

output_path = "data/merged/restaurant_matches.csv"

matched_df.to_csv(
    output_path,
    index=False
)

# =====================================================
# RESULTS
# =====================================================

print("\n===================================")
print("FUZZY MATCHING COMPLETE")
print("===================================")

print("\nTotal Matches Found:")
print(len(matched_df))

print("\nSample Matches:")
print(matched_df.head())

print("\nSaved merged dataset to:")
print(output_path)

print("\nAverage Match Score:")
print(round(matched_df["match_score"].mean(), 2))