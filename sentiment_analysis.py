import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
from database import FlightDatabase
import sqlite3

MODEL_NAME = f"cardiffnlp/twitter-roberta-base-sentiment"

def process_sentiment():
    db = FlightDatabase()
    
    # 1. Fetch only reviews that haven't been analyzed yet (Optimization)
    # In this simplified version, we'll just fetch all and overwrite for simplicity,
    # but a Senior Eng would query: "SELECT * FROM reviews WHERE sentiment_label = 'Pending'"
    df = db.get_reviews()
    
    if df.empty:
        print("No reviews to process.")
        return

    print(f"--- Loading AI Model to process {len(df)} reviews ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    # 2. Batch Inference (Process locally)
    updated_rows = []
    
    print("Running inference...")
    for index, row in df.iterrows():
        # Skip if already processed (Simple caching logic)
        if row['sentiment_label'] != 'Pending' and row['sentiment_label'] is not None:
            continue

        try:
            encoded_input = tokenizer(str(row['content']), return_tensors='pt', truncation=True, max_length=512)
            output = model(**encoded_input)
            scores = softmax(output.logits[0].detach().numpy())
            
            ranking = np.argsort(scores)[::-1]
            top_label = ranking[0]
            
            if top_label == 0: 
                label = "Negative"
                score = -1 * scores[0]
            elif top_label == 1: 
                label = "Neutral"
                score = 0
            else: 
                label = "Positive"
                score = scores[2]
            
            # Update the DB row directly
            cursor = db.conn.cursor()
            cursor.execute("""
                UPDATE reviews 
                SET sentiment_score = ?, sentiment_label = ? 
                WHERE id = ?
            """, (float(score), label, row['id']))
            
        except Exception as e:
            print(f"Error on row {index}: {e}")

    db.conn.commit()
    print("✅ Sentiment Analysis Complete. Database updated.")
    db.close()

if __name__ == "__main__":
    process_sentiment()