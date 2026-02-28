import pandas as pd
import numpy as np
from database import FlightDatabase

# --- CONFIGURATION ---
OUTPUT_FILE = "daily_training_data.csv"

def prepare_data():
    print("--- Preparing Data for Forecasting (Source: SQLite DB) ---")
    
    # 1. Load Reviews from the Database
    # This is the "Senior" way: Reading from a robust data store
    db = FlightDatabase()
    df_reviews = db.get_reviews()
    db.close()

    if df_reviews.empty:
        print("❌ Error: No reviews found in the database.")
        print("  - Did you run 'scrape_reviews.py'?")
        print("  - Did you run 'sentiment_analysis.py'?")
        return

    # Filter: We only want reviews that have been analyzed by the AI
    # (i.e., they have a sentiment score that is not 0.0 if it was a placeholder)
    # Or more simply, just ensure the column exists and isn't empty.
    print(f"Loaded {len(df_reviews)} reviews from database.")

    # 2. Convert Dates
    # The DB stores them as strings "YYYY-MM-DD HH:MM:SS". We need actual Datetime objects.
    df_reviews['date'] = pd.to_datetime(df_reviews['date'])
    
    # Remove the time part (we only care about the Day)
    df_reviews['date'] = df_reviews['date'].dt.normalize()

    # 3. Aggregate Sentiment by Day
    # Group by Date -> Calculate Mean Sentiment
    daily_sentiment = df_reviews.groupby('date')['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['ds', 'sentiment'] # Prophet needs 'ds'
    
    print(f"Aggregated into {len(daily_sentiment)} daily data points.")

    # 4. Generate Simulated Price History
    # (Since we just started scraping today, we simulate the price history 
    #  to demonstrate the forecasting capability)
    
    if daily_sentiment.empty:
        print("Error: No valid dates after aggregation.")
        return

    min_date = daily_sentiment['ds'].min()
    max_date = daily_sentiment['ds'].max()
    
    print(f"Date Range: {min_date.date()} to {max_date.date()}")
    
    # Create a continuous date range (filling any missing days)
    all_dates = pd.date_range(start=min_date, end=max_date)
    df_daily = pd.DataFrame({'ds': all_dates})
    
    # Merge with sentiment (fill missing days with 0/Neutral)
    df_daily = pd.merge(df_daily, daily_sentiment, on='ds', how='left').fillna(0)

    # --- CREATE SYNTHETIC PRICE LOGIC ---
    np.random.seed(42) # Fixed seed so results are consistent
    
    # Base Price
    prices = 300 
    
    # Weekend Spikes (Fri/Sun are expensive)
    is_weekend = df_daily['ds'].dt.dayofweek.isin([4, 6])
    prices += np.where(is_weekend, 50, 0)
    
    # Sentiment Impact (Higher Sentiment -> Higher Price Demand)
    prices += (df_daily['sentiment'] * 20) 
    
    # Random Market Noise
    prices += np.random.normal(0, 15, len(df_daily))
    
    df_daily['y'] = prices.round(2)

    # 5. Save Final Training Data
    # We save this as a CSV because Prophet (in the next step) likes CSVs/DataFrames.
    df_daily.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ Success! Training data saved to '{OUTPUT_FILE}'")
    print(df_daily.head())

if __name__ == "__main__":
    prepare_data()