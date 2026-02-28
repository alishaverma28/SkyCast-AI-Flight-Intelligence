import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
INPUT_FILE = "daily_training_data.csv"
OUTPUT_FORECAST_FILE = "forecast_results.csv"
DAYS_TO_PREDICT = 30

def train_and_forecast():
    print("--- Training Forecast Model ---")
    
    # 1. Load Training Data
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(df)} days of history.")
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 2. Initialize the Model
    # We turn on 'daily_seasonality' because flight prices change day-to-day
    model = Prophet(daily_seasonality=True)
    
    # 3. Add our "Secret Weapon" Feature
    # We tell Prophet: "Sentiment is an important factor. Pay attention to it."
    model.add_regressor('sentiment')

    # 4. Train the Model
    print("Fitting model to data...")
    model.fit(df)

    # 5. Create the "Future" Dataframe
    # This creates a list of dates for the next 30 days
    future = model.make_future_dataframe(periods=DAYS_TO_PREDICT)
    
    # --- CRITICAL STEP: Future Sentiment ---
    # The model needs to know: "What is the sentiment for these future days?"
    # Since we can't predict future feelings, we assume the sentiment
    # stays the same as the LAST day in our history (a "status quo" forecast).
    
    last_sentiment = df['sentiment'].iloc[-1]
    print(f"Forecasting with assumed future sentiment: {last_sentiment:.4f}")
    
    # Fill the 'sentiment' column in the future dataframe
    # We use forward fill (ffill) to propagate the last known sentiment into the future
    future['sentiment'] = df['sentiment'].iloc[-1] 
    
    # If there are any gaps in the history, fill them too
    future['sentiment'] = future['sentiment'].fillna(0)

    # 6. Make the Prediction
    print(f"Predicting prices for the next {DAYS_TO_PREDICT} days...")
    forecast = model.predict(future)

    # 7. Save the Results
    # We only keep the columns we need for the dashboard
    # ds = Date, yhat = Predicted Price, yhat_lower/upper = Confidence Interval
    cols_to_keep = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    final_df = forecast[cols_to_keep]
    
    final_df.to_csv(OUTPUT_FORECAST_FILE, index=False)
    print(f"Success! Forecast saved to '{OUTPUT_FORECAST_FILE}'")
    
    # Optional: Plot it right now to see
    fig1 = model.plot(forecast)
    plt.title("30-Day Price Forecast (influenced by Sentiment)")
    plt.show()

if __name__ == "__main__":
    train_and_forecast()