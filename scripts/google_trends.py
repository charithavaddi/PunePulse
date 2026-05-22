from pytrends.request import TrendReq
import pandas as pd

# Connect to Google Trends
pytrends = TrendReq(hl='en-US', tz=330)

# Keywords to track
keywords = [
    "biryani Pune",
    "cafe Pune",
    "misal pav"
]

# Build request
pytrends.build_payload(
    kw_list=keywords,
    timeframe='2024-01-01 2024-12-31',
    geo='IN-MH'   # Maharashtra
)

# Get interest over time
trends_data = pytrends.interest_over_time()

# Remove 'isPartial' column if present
if 'isPartial' in trends_data.columns:
    trends_data = trends_data.drop(columns=['isPartial'])

# Save to CSV
trends_data.to_csv("pune_food_trends_2024.csv")

print(trends_data.head())
print("\nCSV saved successfully!")
