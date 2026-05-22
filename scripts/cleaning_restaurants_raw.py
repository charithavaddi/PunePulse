import pandas as pd
import numpy as np

# =========================
# LOAD DATASET
# =========================

# Replace with your actual file path
df = pd.read_csv(r"C:\Users\chari\OneDrive\Desktop\Projects\Pune Restaurant Demand Intelligence Platform\data\raw\restaurants_raw.csv")

print("Original Shape:", df.shape)

# Preview
print(df.head())


# =========================================================
# STEP 1 — FILTER VALID FOOD BUSINESS TYPES
# =========================================================

valid_tags = [
    "restaurant",
    "bar",
    "night_club",
    "cafe",
    "meal_takeaway",
    "meal_delivery",
    "bakery"
]


def has_valid_tag(tag_string):

    if pd.isna(tag_string):
        return False

    tags = [
        tag.strip().lower()
        for tag in str(tag_string).split(",")
    ]

    return any(tag in valid_tags for tag in tags)


# Keep only relevant businesses
df = df[df["cuisine_tag"].apply(has_valid_tag)]

print("\nAfter Filtering Valid Business Types:")
print(df.shape)


# =========================================================
# STEP 2 — RENAME cuisine_tag
# =========================================================

# Google's cuisine_tag is actually business metadata
df.rename(
    columns={"cuisine_tag": "business_tags"},
    inplace=True
)

print("\nColumn Renamed:")
print(df.columns)


# =========================================================
# STEP 3 — HANDLE MISSING PRICE RANGE
# =========================================================

# Create missingness indicator BEFORE filling
df["price_range_missing"] = df["price_range"].isna()

# Fill missing values deliberately
df["price_range"] = df["price_range"].fillna("Unknown")

print("\nPrice Range Missing Values:")
print(df["price_range"].value_counts())


# =========================================================
# STEP 4 — STANDARDIZE BUSINESS TAGS
# =========================================================

def standardize_business_type(tag_string):

    tags = [
        tag.strip().lower()
        for tag in str(tag_string).split(",")
    ]

    # Priority ordering
    if "night_club" in tags:
        return "Night Club"

    elif "bar" in tags:
        return "Bar"

    elif "cafe" in tags:
        return "Cafe"

    elif "bakery" in tags:
        return "Bakery"

    elif "meal_delivery" in tags:
        return "Delivery"

    elif "meal_takeaway" in tags:
        return "Takeaway"

    elif "restaurant" in tags:
        return "Restaurant"

    else:
        return "Other"


df["business_type"] = df["business_tags"].apply(
    standardize_business_type
)

print("\nBusiness Type Distribution:")
print(df["business_type"].value_counts())


# =========================================================
# STEP 5 — CREATE CUISINE TYPE COLUMN
# =========================================================

# Rule-based cuisine inference
# You can improve this later with Zomato enrichment

def infer_cuisine(name):

    name = str(name).lower()

    # Indian cuisines
    if "biryani" in name:
        return "Biryani"

    elif "dhaba" in name:
        return "North Indian"

    elif "punjabi" in name:
        return "North Indian"

    elif "south indian" in name:
        return "South Indian"

    elif "dosa" in name:
        return "South Indian"

    elif "idli" in name:
        return "South Indian"

    elif "misal" in name:
        return "Maharashtrian"

    # Cafe/Bakery
    elif "cafe" in name:
        return "Cafe"

    elif "coffee" in name:
        return "Cafe"

    elif "bakery" in name:
        return "Bakery"

    # International
    elif "pizza" in name:
        return "Italian"

    elif "pasta" in name:
        return "Italian"

    elif "burger" in name:
        return "Fast Food"

    elif "fried chicken" in name:
        return "Fast Food"

    elif "chinese" in name:
        return "Chinese"

    elif "momos" in name:
        return "Chinese"

    elif "shawarma" in name:
        return "Middle Eastern"

    else:
        return "Unknown"


df["cuisine_type"] = df["restaurant_name"].apply(infer_cuisine)

print("\nCuisine Type Distribution:")
print(df["cuisine_type"].value_counts())


# =========================================================
# STEP 6 — CLEAN LOCALITY COLUMN
# =========================================================

# Optional: standardize locality formatting

if "locality" in df.columns:

    df["locality"] = (
        df["locality"]
        .astype(str)
        .str.strip()
        .str.title()
    )


# =========================================================
# STEP 7 — REMOVE DUPLICATES
# =========================================================

# Remove duplicates based on restaurant name + locality

if "locality" in df.columns:

    df.drop_duplicates(
        subset=["restaurant_name", "locality"],
        inplace=True
    )

else:

    df.drop_duplicates(
        subset=["restaurant_name"],
        inplace=True
    )

print("\nAfter Removing Duplicates:")
print(df.shape)


# =========================================================
# STEP 8 — BASIC DATA QUALITY CHECKS
# =========================================================

print("\n==========================")
print("DATA QUALITY REPORT")
print("==========================")

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Info:")
print(df.info())

print("\nBusiness Type Counts:")
print(df["business_type"].value_counts())

print("\nCuisine Counts:")
print(df["cuisine_type"].value_counts())

print("\nPrice Range Counts:")
print(df["price_range"].value_counts())


# =========================================================
# STEP 9 — SAVE CLEANED DATASET
# =========================================================


# Output path
output_file = "data/cleaned/restaurants_cleaned.csv"

# Save cleaned dataset
df.to_csv(output_file, index=False)

print("\n==========================")
print("CLEANING COMPLETE")
print("==========================")

print(f"\nSaved cleaned dataset as: {output_file}")
print("\nFinal Shape:", df.shape)
