import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from database import FlightDatabase # Import our new robust layer

# --- CONFIGURATION ---
AIRLINES = {
    "American Airlines": "https://www.airlinequality.com/airline-reviews/american-airlines",
    "Delta Air Lines": "https://www.airlinequality.com/airline-reviews/delta-air-lines"
}
PAGES_TO_SCRAPE = 3 # Keep it small for testing, increase later

def scrape_airline(airline_name, base_url, db):
    print(f"--- Scraping {airline_name} ---")
    new_reviews = []

    for i in range(1, PAGES_TO_SCRAPE + 1):
        url = f"{base_url}/page/{i}/"
        try:
            response = requests.get(url)
            if response.status_code != 200: break
            
            soup = BeautifulSoup(response.content, 'html.parser')
            review_articles = soup.find_all("article", itemprop="review")
            
            for article in review_articles:
                text_div = article.find("div", class_="text_content")
                if not text_div: continue
                
                review_text = text_div.get_text(strip=True)
                if "|" in review_text:
                    review_text = review_text.split("|", 1)[1].strip()

                date_tag = article.find("time", itemprop="datePublished")
                raw_date = date_tag["datetime"] if date_tag else None
                
                # Data Integrity: Skip bad data immediately
                if not raw_date or not review_text: continue

                new_reviews.append({
                    "date": raw_date,
                    "airline": airline_name,
                    "rating": 5, # Placeholder if rating scraping fails
                    "content": review_text,
                    "sentiment_score": 0.0, # Will be filled by AI later
                    "sentiment_label": "Pending"
                })
                
        except Exception as e:
            print(f"Error on page {i}: {e}")
            
        time.sleep(0.5) # Polite delay

    if new_reviews:
        df = pd.DataFrame(new_reviews)
        
        # --- THE FIX IS HERE ---
        # We convert the dates to standard strings (YYYY-MM-DD HH:MM:SS)
        # This makes SQLite happy because it loves strings.
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save to DB (The DB class handles deduplication!)
        db.save_reviews(df)

def main():
    # Initialize the connection once
    db = FlightDatabase()
    
    # Loop through multiple airlines (Scalability!)
    for name, url in AIRLINES.items():
        scrape_airline(name, url, db)
        
    print("--- Scraping Session Complete ---")
    db.close()

if __name__ == "__main__":
    main()