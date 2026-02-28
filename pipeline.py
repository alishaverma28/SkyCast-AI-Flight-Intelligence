import logging
import time
from scrape_reviews import main as run_scraper
from sentiment_analysis import process_sentiment
from generate_summaries import generate_daily_briefing
from prepare_forecast_data import prepare_data
from train_forecast_model import train_and_forecast

# --- LOGGING SETUP ---
# Professional logs save to a file AND print to screen
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # FIX: Added encoding='utf-8' to handle emojis on Windows
        logging.FileHandler("system.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def run_full_pipeline():
    logging.info("🚀 STARTING FLIGHT INTELLIGENCE PIPELINE")
    
    try:
        # Step 1: ETL (Extract)
        logging.info("STEP 1: Scraping new data...")
        run_scraper()
        
        # Step 2: NLP (Transform)
        logging.info("STEP 2: Running Sentiment Analysis...")
        process_sentiment()
        
        # Step 3: GenAI (Transform)
        logging.info("STEP 3: Generating Executive Summaries...")
        generate_daily_briefing()
        
        # Step 4: Data Prep (Transform)
        logging.info("STEP 4: Preparing Forecasting Datasets...")
        prepare_data()
        
        # Step 5: ML Training (Load)
        logging.info("STEP 5: Retraining Forecast Models...")
        train_and_forecast()
        
        logging.info("✅ PIPELINE COMPLETED SUCCESSFULLY.")
        
    except Exception as e:
        logging.error(f"❌ PIPELINE FAILED: {e}", exc_info=True)

if __name__ == "__main__":
    run_full_pipeline()