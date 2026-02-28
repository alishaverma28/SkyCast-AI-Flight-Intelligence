# ✈️ SkyCast AI – Flight Market Intelligence Platform

[🔗 Live Demo](https://skycast-ai-flight-intelligence-2twpsfrisp47j9yvaadeae.streamlit.app/)


## 🚀 Overview

SkyCast AI is an end-to-end Flight Market Intelligence Platform that combines:

- 📊 Machine Learning for airfare price prediction  
- 🧠 NLP for airline sentiment analysis  
- 🗄️ SQLite database integration  
- 📈 Interactive executive dashboard using Streamlit  

The platform transforms raw flight and review data into actionable business insights for pricing strategy and competitive analysis.

---

## 🎯 Business Problem

Airlines operate in a highly dynamic pricing environment. Ticket prices fluctuate due to:

- Demand changes  
- Route competition  
- Seasonality  
- Customer sentiment  

The objective of this project was to build a data-driven intelligence system that:

- Forecasts airfare prices  
- Analyzes competitor sentiment  
- Provides strategic decision-support insights  

---

## 🏗️ System Architecture

### 1️⃣ Data Processing Layer
- Data preprocessing using Pandas
- Feature engineering (date-time features, duration, route encoding)
- Data storage using SQLite
- Structured data pipeline design

### 2️⃣ Machine Learning Layer
- Model Used: XGBoost Regressor
- Task: Regression (Airfare Price Prediction)
- Performance:
  - R² Score ≈ 0.83
  - Low RMSE for accurate forecasting
- Hyperparameter tuning applied

### 3️⃣ NLP Sentiment Layer
- Airline review sentiment analysis
- Classification into:
  - Positive
  - Neutral
  - Negative
- Aggregated competitor sentiment metrics

### 4️⃣ Application Layer
- Interactive dashboard built with Streamlit
- Visualizations using Plotly
- KPI metrics and trend analysis
- Forecast confidence intervals
- Executive insight panel

---

## 📊 Key Features

✔️ Airfare Price Forecasting  
✔️ Confidence Interval Visualization  
✔️ Airline Sentiment Analysis  
✔️ Market KPI Dashboard  
✔️ SQLite Database Integration  
✔️ Interactive Filters (Route / Airline / Date)  
✔️ Business Strategy Recommendation Section  

---

## 🛠 Tech Stack

- Python  
- Pandas & NumPy  
- Scikit-learn  
- XGBoost  
- NLP (TextBlob / VADER)  
- SQLite  
- Streamlit  
- Plotly  

---

## 📈 Model Performance

| Metric | Value |
|--------|--------|
| R² Score | ~0.83 |
| Model | XGBoost Regressor |
| Task | Airfare Price Prediction |

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 What This Project Demonstrates

- End-to-end ML pipeline development  
- Feature engineering for structured data  
- Regression modeling with XGBoost  
- NLP sentiment analysis  
- Database design and integration  
- Business-focused data visualization  
- Converting ML outputs into executive insights  

---

## 💡 Future Enhancements

- Deploy on Streamlit Cloud  
- Add real-time API integration  
- Implement LSTM for advanced time-series forecasting  
- Automate data pipeline with scheduled jobs  

---

## 👩‍💻 Author

Alisha Verma  
Data Analyst | Machine Learning Enthusiast | AI Application Developer
