import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from database import FlightDatabase # connecting to your robust backend

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SkyCast AI | Enterprise Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PROFESSIONAL STYLING ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
            color: #171717;
        }
        
        /* Header Styling */
        h1, h2, h3 { font-weight: 700; letter-spacing: -0.5px; }
        
        /* Metric Cards */
        [data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* AI Summary Box */
        .ai-box {
            background-color: #eff6ff;
            border: 1px solid #bfdbfe;
            border-left: 5px solid #3b82f6;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 25px;
        }
        .ai-title {
            color: #1e40af;
            font-weight: 600;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }
        .ai-text {
            color: #1e3a8a;
            font-size: 1.1rem;
            line-height: 1.5;
            font-style: italic;
        }
        
        /* Clean up Streamlit UI */
        .block-container { padding-top: 2rem; }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 3. ROBUST DATA LOADING ---
@st.cache_data
def load_data():
    # 1. Load Forecast (Still CSV for this demo)
    try:
        df_forecast = pd.read_csv("forecast_results.csv")
        df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])
    except FileNotFoundError:
        st.error("❌ Forecast data missing. Run `train_forecast_model.py`.")
        st.stop()

    # 2. Load Reviews & Summary from DATABASE (Senior Architecture)
    try:
        db = FlightDatabase()
        df_reviews = db.get_reviews()
        df_summary = db.get_latest_summary()
        db.close()
        
        # Convert DB string dates to Datetime objects
        if not df_reviews.empty:
            df_reviews['date'] = pd.to_datetime(df_reviews['date'])
            
    except Exception as e:
        st.error(f"❌ Database Error: {e}")
        st.stop()
        
    return df_forecast, df_reviews, df_summary

df_forecast_raw, df_reviews_raw, df_summary = load_data()

# --- 4. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("✈️ SkyCast AI")
    st.caption("Enterprise Intelligence v2.0")
    st.divider()
    
    # REAL Filtering Logic
    # Get unique airlines from DB
    available_airlines = df_reviews_raw['airline'].unique().tolist() if not df_reviews_raw.empty else ["American Airlines"]
    
    selected_route = st.selectbox("Market Route", ["JFK ➝ LAX", "LHR ➝ DXB", "SIN ➝ SYD"])
    
    selected_airlines = st.multiselect(
        "Competitor Analysis", 
        options=available_airlines,
        default=available_airlines[:1] # Default to first airline
    )
    
    st.divider()
    forecast_days = st.slider("Prediction Horizon", 7, 30, 30)
    st.caption(f"Database contains {len(df_reviews_raw)} verified reviews.")

# --- 5. DATA PROCESSING (REACTIVE LAYER) ---

# Filter Reviews (REAL DATA)
if not df_reviews_raw.empty and selected_airlines:
    df_reviews_viz = df_reviews_raw[df_reviews_raw['airline'].isin(selected_airlines)].copy()
else:
    df_reviews_viz = df_reviews_raw.copy()

# Simulate Price Logic (Since we only scrape one price route)
df_forecast_viz = df_forecast_raw.copy()
price_multiplier = 1.0
if selected_route == "LHR ➝ DXB": price_multiplier = 1.45
elif selected_route == "SIN ➝ SYD": price_multiplier = 0.85

df_forecast_viz['yhat'] *= price_multiplier
df_forecast_viz['yhat_upper'] *= price_multiplier
df_forecast_viz['yhat_lower'] *= price_multiplier

# Slice Forecast
df_forecast_viz = df_forecast_viz.iloc[-(forecast_days+7):]

# --- 6. DASHBOARD UI ---

st.title(f"Market Intelligence: {selected_route}")

# --- A. GEN-AI EXECUTIVE BRIEFING ---
if not df_summary.empty:
    summary_text = df_summary.iloc[0]['summary_text']
    summary_date = pd.to_datetime(df_summary.iloc[0]['date']).strftime('%b %d, %Y')
    
    st.markdown(f"""
    <div class="ai-box">
        <div class="ai-title">🤖 Daily AI Executive Briefing • {summary_date}</div>
        <div class="ai-text">"{summary_text}"</div>
    </div>
    """, unsafe_allow_html=True)

# --- B. METRICS ROW ---
col1, col2, col3, col4 = st.columns(4)

# Price Metric
latest_price = df_forecast_viz.iloc[-1]['yhat']
start_price = df_forecast_viz.iloc[0]['yhat']
pct_change = ((latest_price - start_price) / start_price) * 100
col1.metric("Forecast Price", f"${latest_price:.0f}", f"{pct_change:.1f}%", delta_color="inverse")

# Sentiment Metric (Real Data)
if not df_reviews_viz.empty:
    avg_sent = df_reviews_viz['sentiment_score'].mean()
    sent_delta = "Improving" if avg_sent > 0 else "Declining"
    col2.metric("Net Sentiment", f"{avg_sent:.2f}", sent_delta)
else:
    col2.metric("Net Sentiment", "N/A", "No Data")

# Volatility Metric
uncertainty = df_forecast_viz.iloc[-1]['yhat_upper'] - df_forecast_viz.iloc[-1]['yhat_lower']
col3.metric("Market Volatility", f"±${uncertainty/2:.0f}", "Confidence Interval")

# Volume Metric
col4.metric("Analyzed Reviews", len(df_reviews_viz), "Real-time Source")

st.divider()

# --- C. MAIN CHARTS ---
c_left, c_right = st.columns([2, 1])

with c_left:
    st.subheader("📈 Price Forecast Model")
    
    fig = go.Figure()
    
    # Confidence Cloud
    fig.add_trace(go.Scatter(
        x=df_forecast_viz['ds'], y=df_forecast_viz['yhat_upper'],
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=df_forecast_viz['ds'], y=df_forecast_viz['yhat_lower'],
        fill='tonexty', mode='lines', line=dict(width=0),
        fillcolor='rgba(59, 130, 246, 0.1)', name='Confidence Range'
    ))
    
    # Main Trend Line
    fig.add_trace(go.Scatter(
        x=df_forecast_viz['ds'], y=df_forecast_viz['yhat'],
        mode='lines+markers', name='Projected Price',
        line=dict(color='#2563eb', width=3),
        marker=dict(size=6)
    ))

    fig.update_layout(
        yaxis=dict(title="Price (USD)", gridcolor='#f3f4f6'),
        xaxis=dict(showgrid=False),
        plot_bgcolor='white',
        height=400,
        margin=dict(l=10,r=10,t=10,b=10),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

with c_right:
    st.subheader("🧠 Sentiment Distribution")
    
    if not df_reviews_viz.empty:
        # Pie Chart for Sentiment
        sent_counts = df_reviews_viz['sentiment_label'].value_counts()
        
        fig_pie = px.pie(
            names=sent_counts.index, 
            values=sent_counts.values,
            hole=0.6,
            color=sent_counts.index,
            color_discrete_map={
                'Positive': '#22c55e', 
                'Neutral': '#94a3b8', 
                'Negative': '#ef4444'
            }
        )
        fig_pie.update_layout(
            showlegend=True, 
            height=250, 
            margin=dict(l=0,r=0,t=0,b=0),
            legend=dict(orientation="h")
        )
        
        # Add a center text for average score
        fig_pie.add_annotation(text=f"{avg_sent:.2f}", font_size=24, showarrow=False)
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # AI Recommendation
        if pct_change < -5:
            st.success("**Strategy:** BUY. Prices dropping.")
        elif pct_change > 5:
            st.error("**Strategy:** SELL/AVOID. Prices peaking.")
        else:
            st.info("**Strategy:** HOLD. Market stable.")

# --- D. DETAILED DATA TABLE ---
st.subheader("🗣️ Voice of the Customer (Live Feed)")

st.dataframe(
    df_reviews_viz[['date', 'airline', 'sentiment_label', 'sentiment_score', 'content']]
    .sort_values(by='date', ascending=False)
    .head(10),
    column_config={
        "date": "Date",
        "airline": "Carrier",
        "sentiment_label": "Verdict",
        "sentiment_score": st.column_config.ProgressColumn("AI Score", min_value=-1, max_value=1, format="%.2f"),
        "content": st.column_config.TextColumn("Review Snippet", width="large")
    },
    hide_index=True,
    use_container_width=True
)