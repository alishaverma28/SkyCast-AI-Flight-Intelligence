import pandas as pd
from transformers import pipeline
from database import FlightDatabase
from datetime import datetime

# --- CONFIGURATION ---
# We use DistilBART because it's faster than GPT and runs locally
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"

def generate_daily_briefing():
    print("--- Starting GenAI Summarization Engine ---")
    
    db = FlightDatabase()
    df = db.get_reviews()
    
    if df.empty:
        print("No data to summarize.")
        db.close()
        return

    # 1. Convert dates and find the latest date with data
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()
    print(f"Analyzing data for: {latest_date.date()}")

    # 2. Filter for ONLY the latest day's reviews
    daily_reviews = df[df['date'] == latest_date]
    
    if daily_reviews.empty:
        print("No reviews found for the latest date.")
        db.close()
        return

    # 3. Combine texts
    # We take the top 10 longest reviews to feed the LLM (Model has a token limit)
    # In a real system, we'd use a 'Map-Reduce' chain for unlimited text.
    texts = daily_reviews['content'].tolist()
    combined_text = " ".join(texts[:10]) 
    
    # Truncate to fit model context window roughly (3000 chars)
    combined_text = combined_text[:3000]

    print("Loading Generative Model...")
    summarizer = pipeline("summarization", model=SUMMARIZER_MODEL)

    print("Generating Executive Summary...")
    # We ask the model to create a concise summary
    summary_output = summarizer(combined_text, max_length=60, min_length=20, do_sample=False)
    final_summary = summary_output[0]['summary_text']
    
    print(f"📝 AI Summary: {final_summary}")

    # 4. Save to DB
    airline_name = daily_reviews['airline'].iloc[0] # Assuming one airline for now
    
    # Format date string for SQLite
    date_str = latest_date.strftime('%Y-%m-%d %H:%M:%S')
    
    db.save_summary(date_str, airline_name, final_summary)
    db.close()

if __name__ == "__main__":
    generate_daily_briefing()