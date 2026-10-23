import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import yfinance as yf

# Page config - must be first
st.set_page_config(
    page_title="HFT Reliance Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern attractive UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }

    .main-header p {
        margin: 0.4rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }

    .metric-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 14px;
        padding: 1.2rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 25px rgba(0,0,0,0.25);
        transition: transform 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
    }

    .rel-card { border-left: 4px solid #1e88e5; }
    .nifty-card { border-left: 4px solid #00c853; }
    .sensex-card { border-left: 4px solid #ab47bc; }
    .fx-card { border-left: 4px solid #ff9800; }

    .prediction-box {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
    }

    .prediction-value {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.05rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    div[data-testid="stSidebar"] .stNumberInput label,
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0 !important;
    }

    .feature-label {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-bottom: 0.2rem;
    }

    .footer {
        text-align: center;
        color: #718096;
        font-size: 0.85rem;
        margin-top: 2rem;
        padding: 1rem;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 1.6rem; }
        .prediction-value { font-size: 2rem; }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Constants taken from the EDA report (HFT Reliance Order Book Profiling Report)
# 5,985,000 ticks | 21 columns | RELIANCE.NS | no missing values
# -----------------------------------------------------------------------------
TARGET = "Last_Trade_Price"

# (min, max, mean) of each raw column as seen in the EDA report
EDA_STATS = {
    "Bid_Price_L1": (1158.22, 1248.59, 1200.56),
    "Ask_Price_L1": (1158.34, 1248.72, 1200.69),
    "Bid_Price_L2": (1157.87, 1248.49, 1200.34),
    "Ask_Price_L2": (1158.44, 1249.07, 1200.92),
    "Bid_Price_L3": (1157.40, 1248.32, 1200.01),
    "Ask_Price_L3": (1158.50, 1249.21, 1201.02),
    "Bid_Size_L1": (800, 11999, 6402),
    "Ask_Size_L1": (800, 11999, 6397),
    "Bid_Size_L2": (1500, 24999, 13246),
    "Ask_Size_L2": (1500, 24999, 13253),
    "Bid_Size_L3": (3000, 44999, 24008),
    "Ask_Size_L3": (3000, 44999, 23998),
    "Last_Trade_Price": (1158.22, 1248.72, 1200.63),
    "Last_Trade_Volume": (5, 1999, 1002),
    "Calculated_Mid_Price": (1158.32, 1248.62, 1200.63),
    "Calculated_Micro_Price": (1158.24, 1248.70, 1200.63),
    "Bid_Ask_Spread_INR": (0.05, 0.20, 0.127),
    "Order_Volume_Imbalance": (-0.8747, 0.8748, 0.0004),
}
PRICE_MIN, PRICE_MAX, PRICE_MEAN = EDA_STATS["Last_Trade_Price"]

# Typical distance (INR) of each depth level from the L1 mid price (from the means above)
LEVEL_OFFSETS = {
    "Bid_Price_L1": -0.06, "Ask_Price_L1": +0.06,
    "Bid_Price_L2": -0.29, "Ask_Price_L2": +0.29,
    "Bid_Price_L3": -0.61, "Ask_Price_L3": +0.39,
}

# Last_Trade_Side encoding (BUY / SELL). Change if you encoded it differently when training.
SIDE_ENCODING = {"BUY": 1, "SELL": 0}


# Load model artifacts (looked up next to app.py, regardless of where you launch it from)
APP_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    """Load scaler, selected features, and the best available model.
    Prefers best_model.pkl (from train_all_models.py), then random_forest_model.pkl,
    then models/Random_Forest.pkl. Models are saved with joblib; fall back to pickle.
    """
    import joblib

    def _load_any(path):
        try:
            return joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f)

    scaler = _load_any(os.path.join(APP_DIR, "scaler.pkl"))
    selected = list(_load_any(os.path.join(APP_DIR, "selected_features.pkl")))

    candidates = [
        os.path.join(APP_DIR, "best_model.pkl"),
        os.path.join(APP_DIR, "random_forest_model.pkl"),
        os.path.join(APP_DIR, "models", "Random_Forest.pkl"),
        os.path.join(APP_DIR, "models", "Hist_Gradient_Boosting.pkl"),
        os.path.join(APP_DIR, "models", "Extra_Trees.pkl"),
    ]
    model_path = next((p for p in candidates if os.path.isfile(p)), None)
    if model_path is None:
        raise FileNotFoundError(os.path.join(APP_DIR, "random_forest_model.pkl"))

    model = _load_any(model_path)
    model_name = os.path.basename(model_path).replace(".pkl", "").replace("_", " ")
    return model, scaler, selected, model_name

try:
    model, scaler, selected_features, MODEL_NAME = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Model file not found: **{os.path.basename(getattr(e, 'filename', str(e)))}**")
    st.info(
        "Train the model first to create `scaler.pkl`, `selected_features.pkl` and "
        "`random_forest_model.pkl` (or `best_model.pkl`), then refresh this page:\n\n"
        "```\npython train_all_models.py path\\to\\your_dataset.csv\n```"
    )
    st.stop()
except Exception as e:
    st.error(f"Failed to load model artifacts: {type(e).__name__}: {e}")
    st.info(
        "Re-run training:\n\n```\npython train_all_models.py path\\to\\your_dataset.csv\n```"
    )
    st.stop()


# Fetch live market data (Reliance has no public order-book feed on Yahoo, so we show price context only)
@st.cache_data(ttl=60)
def get_live_prices():
    tickers = {
        'RELIANCE': 'RELIANCE.NS',
        'NIFTY': '^NSEI',
        'SENSEX': '^BSESN',
        'USDINR': 'INR=X'
    }
    data = {}
    for name, ticker in tickers.items():
        try:
            hist = yf.Ticker(ticker).history(period='5d')
            if not hist.empty:
                last = hist.iloc[-1]
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else last['Open']
                data[name] = {
                    'price': float(last['Close']),
                    'volume': float(last['Volume']),
                    'change': float(last['Close'] - prev_close),
                    'change_pct': float((last['Close'] - prev_close) / prev_close * 100),
                }
                continue
        except Exception:
            pass
        data[name] = {'price': 0.0, 'volume': 0.0, 'change': 0.0, 'change_pct': 0.0}
    return data


@st.cache_data(ttl=300)
def get_historical_data(period='6mo', interval='1d'):
    """Fetch Yahoo Finance candles for the selected period and interval."""
    try:
        df = yf.Ticker('RELIANCE.NS').history(
            period=period, interval=interval, auto_adjust=False
        )[['Open', 'High', 'Low', 'Close', 'Volume']]
        if df.empty:
            return pd.DataFrame()

        # Add Nifty context using the same candle interval when Yahoo supports it.
        try:
            nifty = yf.Ticker('^NSEI').history(
                period=period, interval=interval, auto_adjust=False
            )[['Close']].rename(columns={'Close': 'NIFTY'})
            df = df.join(nifty, how='left')
        except Exception:
            df['NIFTY'] = np.nan
        return df
    except Exception:
        return pd.DataFrame()


def format_price(price):
    return f"₹{price:,.2f}"


def format_volume(vol):
    if vol >= 1e7:
        return f"{vol/1e7:.2f} Cr"
    elif vol >= 1e5:
        return f"{vol/1e5:.2f} L"
    else:
        return f"{vol:,.0f}"


def build_feature_row(v):
    """Turn raw sidebar inputs into every feature the model could have been trained on."""
    bid1, ask1 = v['Bid_Price_L1'], v['Ask_Price_L1']
    bs1, as1 = v['Bid_Size_L1'], v['Ask_Size_L1']
    tot = bs1 + as1
    feats = dict(v)
    feats['Last_Trade_Side'] = SIDE_ENCODING[v['Last_Trade_Side']]
    feats['Calculated_Mid_Price'] = (bid1 + ask1) / 2
    feats['Calculated_Micro_Price'] = (bid1 * as1 + ask1 * bs1) / tot if tot else (bid1 + ask1) / 2
    feats['Bid_Ask_Spread_INR'] = ask1 - bid1
    feats['Order_Volume_Imbalance'] = (bs1 - as1) / tot if tot else 0.0
    return feats


# Header
st.markdown("""
<div class="main-header">
    <h1>📈 HFT Reliance Price Predictor</h1>
    <p>AI-Powered Limit Order Book Price Prediction using Random Forest • RELIANCE.NS • Live Market Context</p>
</div>
""", unsafe_allow_html=True)

# Live prices section
live = get_live_prices()

st.markdown("### 📊 Live Market Overview")
cards = [
    ("rel-card", "🏢", "Reliance (RELIANCE.NS)", "#1e88e5", 'RELIANCE', True),
    ("nifty-card", "📊", "Nifty 50", "#00c853", 'NIFTY', False),
    ("sensex-card", "📈", "Sensex", "#ab47bc", 'SENSEX', False),
    ("fx-card", "💱", "USD / INR", "#ff9800", 'USDINR', False),
]
for col, (css, icon, label, color, key, show_vol) in zip(st.columns(4), cards):
    d = live[key]
    change_color = "#00c853" if d['change'] >= 0 else "#ff1744"
    price_txt = f"₹{d['price']:,.2f}" if key != 'NIFTY' and key != 'SENSEX' else f"{d['price']:,.2f}"
    vol_txt = f"Vol: {format_volume(d['volume'])}" if show_vol else "&nbsp;"
    with col:
        st.markdown(f"""
        <div class="metric-card {css}">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                <span style="font-size:1.4rem;">{icon}</span>
                <span style="font-weight:600;color:{color};">{label}</span>
            </div>
            <div style="font-size:1.5rem;font-weight:700;color:white;">{price_txt if d['price'] else 'N/A'}</div>
            <div style="color:{change_color};font-size:0.9rem;margin-top:4px;">
                {'▲' if d['change'] >= 0 else '▼'} {d['change_pct']:+.2f}%
            </div>
            <div style="color:#a0aec0;font-size:0.8rem;margin-top:6px;">{vol_txt}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs for Prediction, Order Book, Charts, EDA and About
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔮 Price Prediction", "📚 Order Book", "📈 Price Charts",
    "🧪 EDA Insights", "ℹ️ About Model", "🏆 Model Performance"
])

# ----------------------------- Sidebar inputs --------------------------------
# Base price: the training data only covers ~₹1158-1249, so if the live price is
# outside that range we default to the dataset mean (tree models can't extrapolate).
live_rel = live['RELIANCE']['price']
base_price = live_rel if PRICE_MIN <= live_rel <= PRICE_MAX else PRICE_MEAN

with st.sidebar:
    st.markdown("## 🎛️ Order Book Inputs")
    st.markdown("---")

    st.markdown("**Level 1 (Best Bid / Ask)**")
    bid_p1 = st.number_input("Bid Price L1 (₹)", 0.0, value=round(base_price + LEVEL_OFFSETS['Bid_Price_L1'], 2), format="%.2f")
    ask_p1 = st.number_input("Ask Price L1 (₹)", 0.0, value=round(base_price + LEVEL_OFFSETS['Ask_Price_L1'], 2), format="%.2f")
    bid_s1 = st.number_input("Bid Size L1", 0.0, value=float(EDA_STATS['Bid_Size_L1'][2]), format="%.0f")
    ask_s1 = st.number_input("Ask Size L1", 0.0, value=float(EDA_STATS['Ask_Size_L1'][2]), format="%.0f")

    st.markdown("**Level 2**")
    bid_p2 = st.number_input("Bid Price L2 (₹)", 0.0, value=round(base_price + LEVEL_OFFSETS['Bid_Price_L2'], 2), format="%.2f")
    ask_p2 = st.number_input("Ask Price L2 (₹)", 0.0, value=round(base_price + LEVEL_OFFSETS['Ask_Price_L2'], 2), format="%.2f")
    bid_s2 = st.number_input("Bid Size L2", 0.0, value=float(EDA_STATS['Bid_Size_L2'][2]), format="%.0f")
    ask_s2 = st.number_input("Ask Size L2", 0.0, value=float(EDA_STATS['Ask_Size_L2'][2]), format="%.0f")

    st.markdown("**Level 3**")
    bid_p3 = st.number_input("Bid Price L3 (₹)", 0.0, value=round(base_price + LEVEL_OFFSETS['Bid_Price_L3'], 2), format="%.2f")
    ask_p3 = st.number_input("Ask Price L3 (₹)", 0.0, value=round(base_price + LEVEL_OFFSETS['Ask_Price_L3'], 2), format="%.2f")
    bid_s3 = st.number_input("Bid Size L3", 0.0, value=float(EDA_STATS['Bid_Size_L3'][2]), format="%.0f")
    ask_s3 = st.number_input("Ask Size L3", 0.0, value=float(EDA_STATS['Ask_Size_L3'][2]), format="%.0f")

    st.markdown("**Last Trade**")
    trade_side = st.selectbox("Last Trade Side", ["BUY", "SELL"])
    trade_vol = st.number_input("Last Trade Volume", 0.0, value=float(EDA_STATS['Last_Trade_Volume'][2]), format="%.0f")

    st.markdown("---")
    st.markdown("💡 *Mid price, micro price, spread and imbalance are calculated automatically from Level 1.*")

raw_inputs = {
    'Bid_Price_L1': bid_p1, 'Bid_Size_L1': bid_s1, 'Ask_Price_L1': ask_p1, 'Ask_Size_L1': ask_s1,
    'Bid_Price_L2': bid_p2, 'Bid_Size_L2': bid_s2, 'Ask_Price_L2': ask_p2, 'Ask_Size_L2': ask_s2,
    'Bid_Price_L3': bid_p3, 'Bid_Size_L3': bid_s3, 'Ask_Price_L3': ask_p3, 'Ask_Size_L3': ask_s3,
    'Last_Trade_Side': trade_side, 'Last_Trade_Volume': trade_vol,
}
all_features = build_feature_row(raw_inputs)

# ------------------------------ Tab 1: Prediction ----------------------------
with tab1:
    st.markdown("#### Predict Reliance Last Trade Price")
    st.caption(
        f"Model: **{MODEL_NAME}** · top {len(selected_features)} features "
        f"(SelectKBest): **{', '.join(selected_features)}**"
    )

    # Input sanity checks
    problems = []
    if ask_p1 <= bid_p1:
        problems.append("Ask Price L1 must be higher than Bid Price L1 (crossed book).")
    if not (bid_p3 <= bid_p2 <= bid_p1 and ask_p1 <= ask_p2 <= ask_p3):
        problems.append("Depth prices should widen outward: Bid L3 ≤ L2 ≤ L1 < Ask L1 ≤ L2 ≤ L3.")
    for p in problems:
        st.warning(p)

    pred_col1, pred_col2 = st.columns([1.2, 1])

    with pred_col1:
        st.markdown("##### Feature Summary")
        missing = [f for f in selected_features if f not in all_features]
        if missing:
            st.error(f"These model features are not available in the app: {missing}")
            st.stop()

        def ref_for(f):
            lo, hi, mean = EDA_STATS.get(f, (None, None, None))
            return "—" if mean is None else f"{lo:,.2f} – {hi:,.2f}"

        summary_df = pd.DataFrame({
            'Feature': selected_features,
            'Value': [round(float(all_features[f]), 4) for f in selected_features],
            'Training Range': [ref_for(f) for f in selected_features],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        out_of_range = [
            f for f in selected_features
            if f in EDA_STATS and not (EDA_STATS[f][0] <= all_features[f] <= EDA_STATS[f][1])
        ]
        if out_of_range:
            st.warning(f"Outside the training range (prediction may be unreliable): {', '.join(out_of_range)}")

        if st.button("🚀 Predict Last Trade Price", use_container_width=True):
            input_df = pd.DataFrame([[all_features[f] for f in selected_features]], columns=selected_features)
            input_scaled = scaler.transform(input_df)
            prediction = float(model.predict(input_scaled)[0])

            st.session_state['prediction'] = prediction
            st.session_state['ref_mid'] = all_features['Calculated_Mid_Price']
            st.session_state['predicted_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with pred_col2:
        if 'prediction' in st.session_state:
            pred = st.session_state['prediction']
            ref_mid = st.session_state['ref_mid']
            diff = pred - ref_mid
            diff_pct = (diff / ref_mid) * 100 if ref_mid else 0

            st.markdown(f"""
            <div class="prediction-box">
                <div style="color:#a0aec0;font-size:0.95rem;">Predicted Last Trade Price</div>
                <div class="prediction-value">{format_price(pred)}</div>
                <div style="color:#cbd5e0;font-size:0.9rem;margin-top:0.5rem;">
                    vs Order-Book Mid: {format_price(ref_mid)}
                </div>
                <div style="color:{'#00c853' if diff >= 0 else '#ff1744'};font-size:1.1rem;font-weight:600;margin-top:0.4rem;">
                    {'▲' if diff >= 0 else '▼'} {diff:+,.2f} ({diff_pct:+.3f}%)
                </div>
                <div style="color:#718096;font-size:0.75rem;margin-top:1rem;">
                    Predicted at {st.session_state.get('predicted_at', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="prediction-box">
                <div style="color:#a0aec0;font-size:1rem;padding:2rem 0;">
                    👈 Adjust the order book in the sidebar<br>and click <b>Predict</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Derived microstructure metrics
    st.markdown("##### Microstructure Snapshot")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mid Price", format_price(all_features['Calculated_Mid_Price']))
    m2.metric("Micro Price", format_price(all_features['Calculated_Micro_Price']))
    m3.metric("Bid-Ask Spread", f"₹{all_features['Bid_Ask_Spread_INR']:.2f}")
    imb = all_features['Order_Volume_Imbalance']
    m4.metric("Order Imbalance", f"{imb:+.3f}", "Buy pressure" if imb > 0 else "Sell pressure")

# ------------------------------ Tab 2: Order Book ----------------------------
with tab2:
    st.markdown("#### Limit Order Book Depth (3 Levels)")

    bids = pd.DataFrame({
        'Level': ['L1', 'L2', 'L3'],
        'Price': [bid_p1, bid_p2, bid_p3],
        'Size': [bid_s1, bid_s2, bid_s3],
    })
    asks = pd.DataFrame({
        'Level': ['L1', 'L2', 'L3'],
        'Price': [ask_p1, ask_p2, ask_p3],
        'Size': [ask_s1, ask_s2, ask_s3],
    })

    fig_ob = go.Figure()
    fig_ob.add_trace(go.Bar(
        x=bids['Price'], y=bids['Size'], name='Bids',
        marker_color='rgba(0, 200, 83, 0.7)', text=bids['Level'], textposition='outside',
        width=0.12
    ))
    fig_ob.add_trace(go.Bar(
        x=asks['Price'], y=asks['Size'], name='Asks',
        marker_color='rgba(255, 23, 68, 0.7)', text=asks['Level'], textposition='outside',
        width=0.12
    ))
    fig_ob.add_vline(
        x=all_features['Calculated_Mid_Price'], line_dash='dash', line_color='#ffd700',
        annotation_text='Mid', annotation_position='top'
    )
    fig_ob.update_layout(
        height=420, template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title='Price (₹)', yaxis_title='Order Size (shares)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=20, r=20, t=40, b=20), bargap=0.2
    )
    fig_ob.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    fig_ob.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    st.plotly_chart(fig_ob, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🟢 Bid Side")
        st.dataframe(bids, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("##### 🔴 Ask Side")
        st.dataframe(asks, use_container_width=True, hide_index=True)

    total_bid = bid_s1 + bid_s2 + bid_s3
    total_ask = ask_s1 + ask_s2 + ask_s3
    depth_imb = (total_bid - total_ask) / (total_bid + total_ask) if (total_bid + total_ask) else 0
    st.caption(f"Total bid depth: **{total_bid:,.0f}** • Total ask depth: **{total_ask:,.0f}** • 3-level depth imbalance: **{depth_imb:+.3f}**")

# ------------------------------ Tab 3: Charts --------------------------------
with tab3:
    st.markdown("#### Historical Price Trends (RELIANCE.NS)")
    st.caption(
        "Choose a candle interval and lookback period. Yahoo Finance does not provide "
        "30-second candles; 1 minute is the finest standard interval."
    )

    timeframe_options = {
        "1 minute": "1m",
        "5 minutes": "5m",
        "15 minutes": "15m",
        "30 minutes": "30m",
        "1 hour": "1h",
        "1 day": "1d",
        "1 week": "1wk",
        "1 month": "1mo",
    }
    timeframe_col, period_col = st.columns([1, 1])
    with timeframe_col:
        timeframe_label = st.selectbox(
            "Candle Timeframe", list(timeframe_options.keys()), index=5
        )
    interval = timeframe_options[timeframe_label]

    # Yahoo's intraday history windows are more restricted than daily/weekly/monthly.
    if interval == "1m":
        period_options = ["1d", "5d"]
        default_period = "1d"
    elif interval in ("5m", "15m", "30m"):
        period_options = ["1d", "5d", "1mo", "2mo"]
        default_period = "5d"
    elif interval == "1h":
        period_options = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"]
        default_period = "1mo"
    elif interval == "1d":
        period_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
        default_period = "6mo"
    elif interval == "1wk":
        period_options = ["6mo", "1y", "2y", "5y", "10y", "max"]
        default_period = "2y"
    else:
        period_options = ["1y", "2y", "5y", "10y", "max"]
        default_period = "5y"

    with period_col:
        period = st.selectbox(
            "Historical Range",
            period_options,
            index=period_options.index(default_period),
            key=f"chart_period_{interval}",
        )

    hist = get_historical_data(period, interval)

    if hist.empty:
        st.info(
            "Yahoo Finance returned no candles for this timeframe/range. "
            "Try a shorter range or another interval."
        )
    else:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"Reliance Candlestick — {timeframe_label}", "Trading Volume")
        )

        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist['Open'], high=hist['High'],
            low=hist['Low'], close=hist['Close'], name='RELIANCE'
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=hist.index, y=hist['Volume'], name='Volume',
            marker_color='rgba(30, 136, 229, 0.5)'
        ), row=2, col=1)

        fig.update_layout(
            height=550,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=20, r=20, t=45, b=20),
            hovermode='x unified',
            xaxis_rangeslider_visible=False
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap remains available for the selected candle data.
        st.markdown("##### Price Correlation")
        corr_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume', 'NIFTY'] if c in hist.columns]
        corr_data = hist[corr_cols].dropna()
        if len(corr_data) >= 2 and len(corr_cols) >= 2:
            corr = corr_data.corr()
            fig_corr = px.imshow(
                corr,
                text_auto='.2f',
                color_continuous_scale='RdYlGn',
                aspect='auto',
                title='Correlation between Reliance OHLCV & Nifty 50'
            )
            fig_corr.update_layout(
                height=350,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.caption("Not enough overlapping Reliance and Nifty candles to calculate correlation.")

# ------------------------------ Tab 4: EDA -----------------------------------
with tab4:
    st.markdown("#### Insights from the Profiling Report")

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Observations", "5,985,000")
    e2.metric("Variables", "21")
    e3.metric("Missing Cells", "0 (0%)")
    e4.metric("Memory", "411 MiB")

    st.markdown("##### Descriptive statistics (from the report)")
    eda_df = pd.DataFrame(
        [(k, v[0], v[1], v[2]) for k, v in EDA_STATS.items()],
        columns=['Column', 'Minimum', 'Maximum', 'Mean']
    )
    st.dataframe(eda_df, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("##### Last Trade Side balance")
        fig_side = px.pie(
            names=['SELL', 'BUY'], values=[2994482, 2990518], hole=0.55,
            color_discrete_sequence=['#ff1744', '#00c853']
        )
        fig_side.update_layout(
            height=300, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_side, use_container_width=True)
    with right:
        st.markdown("##### Average price by depth level")
        levels = ['L3', 'L2', 'L1', 'L1 ', 'L2 ', 'L3 ']
        avg_prices = [
            EDA_STATS['Bid_Price_L3'][2], EDA_STATS['Bid_Price_L2'][2], EDA_STATS['Bid_Price_L1'][2],
            EDA_STATS['Ask_Price_L1'][2], EDA_STATS['Ask_Price_L2'][2], EDA_STATS['Ask_Price_L3'][2],
        ]
        fig_lvl = go.Figure(go.Bar(
            x=[f"Bid {l}" if i < 3 else f"Ask {l.strip()}" for i, l in enumerate(levels)],
            y=avg_prices,
            marker_color=['#00c853'] * 3 + ['#ff1744'] * 3
        ))
        fig_lvl.update_layout(
            height=300, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(range=[1199.5, 1201.5]),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_lvl, use_container_width=True)

    st.markdown("""
    **Key observations**
    - `Ticker` is constant (`RELIANCE.NS`) → dropped before modelling.
    - No missing values, no infinite values, no zeros in any numeric column.
    - All price columns share almost the same distribution (mean ≈ ₹1200, std ≈ ₹26.3) → they are **highly correlated** with the target.
    - Order sizes are roughly uniform: L1 (800–11,999), L2 (1,500–24,999), L3 (3,000–44,999).
    - Bid-ask spread is tiny (₹0.05 – ₹0.20); order imbalance is centred at 0 (range ±0.875).
    - Buy and sell trades are balanced (50.0% / 50.0%).
    """)

# ------------------------------ Tab 5: About ---------------------------------
with tab5:
    st.markdown(f"""
    ### About This Model

    This application predicts the **Reliance Industries (RELIANCE.NS) last trade price** from a snapshot of the **3-level limit order book**, using **{MODEL_NAME}** trained on ~6 million high-frequency ticks.

    #### Feature Selection
    - Original features: bid/ask price & size for Levels 1–3, last trade side & volume, mid price, micro price, spread and order volume imbalance
    - Target: `{TARGET}`
    - **SelectKBest** (f_regression) selected the top {len(selected_features)} most predictive features:
    """)

    for i, f in enumerate(selected_features, 1):
        st.markdown(f"  {i}. `{f}`")

    st.markdown("""
    #### Pipeline
    1. High-frequency order book data (5.98M rows, 21 columns)
    2. Drop constant `Ticker`; encode `Last_Trade_Side`; parse `Timestamp_IST`
    3. Feature selection with SelectKBest (f_regression)
    4. MinMaxScaler normalization
    5. Model training (Random Forest and other algorithms via `train_all_models.py`; this app loads the best / RF artifact)

    #### Derived Features (auto-calculated in this app)
    - **Mid Price** = (Bid L1 + Ask L1) / 2
    - **Micro Price** = (Bid L1 × Ask Size L1 + Ask L1 × Bid Size L1) / (Bid Size L1 + Ask Size L1)
    - **Spread** = Ask L1 − Bid L1
    - **Order Volume Imbalance** = (Bid Size L1 − Ask Size L1) / (Bid Size L1 + Ask Size L1)

    #### Notes
    - Training prices span only **₹1,158 – ₹1,249**. Tree-based models cannot extrapolate, so inputs outside this range are flagged.
    - Yahoo Finance provides no order book, so live data is shown for **market context only**.
    - Markets are volatile — this is **not financial advice**.

    #### How the project works — end to end
    1. **Input:** the prediction screen collects the order-book values and trade details available to the model.
    2. **Feature engineering:** the app calculates mid-price, micro-price, bid-ask spread, and order-volume imbalance from the best bid/ask levels and sizes.
    3. **Feature consistency:** the saved feature list determines which inputs are passed to the model, and the saved scaler transforms them in the same way as training.
    4. **Prediction:** the trained regression model returns an estimated `Last_Trade_Price`. This is a model estimate, not an executable quote or a guaranteed future price.
    5. **Market context:** the Price Charts tab fetches Yahoo Finance candles for Reliance and Nifty 50. These candles provide market context; they do not replace the historical order-book inputs used for model training.
    6. **Evaluation:** the Model Performance tab displays metrics saved by the training script, compares algorithms, and can plot actual versus predicted test samples.

    #### Model evaluation and limitations
    - `train_all_models.py` sorts the model comparison by **test RMSE** and saves `model_comparison.csv` plus `predictions_sample.csv`.
    - The training script uses a chronological 80/20 train/test split and scores models on the same held-out test rows (some slower algorithms may train on a capped subset of training rows).
    - A high score on this dataset does not guarantee performance on future market conditions. Monitor out-of-sample results and retrain when the data distribution changes.
    - Yahoo Finance intraday history has interval-specific retention limits, and it does not provide the order-book depth used by this model.

    #### Tech Stack
    - **Frontend / visualization**: Streamlit, custom CSS, Plotly
    - **ML**: scikit-learn regression algorithms, SelectKBest, MinMaxScaler
    - **Data**: high-frequency order-book dataset, pandas / NumPy, yfinance for market-context charts
    """)

# -------------------------- Tab 6: Model Performance -------------------------
with tab6:
    st.markdown("### 🏆 Compare the performance of every trained model")
    st.write(
        "This dashboard reads `model_comparison.csv`, which is generated by "
        "`train_all_models.py`. Every model is evaluated on the same chronological "
        "hold-out test rows, so the reported metrics can be compared fairly."
    )

    comparison_path = os.path.join(APP_DIR, "model_comparison.csv")
    predictions_path = os.path.join(APP_DIR, "predictions_sample.csv")

    if not os.path.isfile(comparison_path):
        st.warning("No model_comparison.csv file was found beside this app.")
        st.markdown(
            "To create the comparison report, run training from the project folder:\n\n"
            "```bash\npython train_all_models.py \"path/to/your_dataset.csv\"\n```"
        )
        st.info("After training finishes, refresh this page. The report will include all models that trained successfully.")
    else:
        try:
            comp = pd.read_csv(comparison_path)
            required_metrics = {"Model", "R2", "RMSE", "MAE"}
            missing_metrics = required_metrics - set(comp.columns)
            if missing_metrics:
                st.error(f"The comparison file is missing required columns: {', '.join(sorted(missing_metrics))}")
            elif comp.empty:
                st.info("The model comparison file is empty. Run model training again.")
            else:
                for col in ["R2", "RMSE", "MAE", "MAPE_%", "Train_R2",
                            "Train_Rows", "Fit_Seconds", "Predict_Seconds", "Rank"]:
                    if col in comp.columns:
                        comp[col] = pd.to_numeric(comp[col], errors="coerce")

                comp = comp.dropna(subset=["Model", "R2", "RMSE", "MAE"]).copy()
                comp = comp.sort_values("RMSE", ascending=True).reset_index(drop=True)
                best_row = comp.iloc[0]

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Models evaluated", f"{len(comp)}")
                m2.metric("Lowest RMSE", f"₹{best_row['RMSE']:.4f}", str(best_row["Model"]))
                m3.metric("Lowest MAE", f"₹{comp['MAE'].min():.4f}")
                m4.metric("Highest R²", f"{comp['R2'].max():.4f}")

                st.markdown("#### Full model scorecard")
                display_cols = [c for c in [
                    "Rank", "Model", "R2", "RMSE", "MAE", "MAPE_%",
                    "Train_R2", "Train_Rows", "Fit_Seconds", "Predict_Seconds"
                ] if c in comp.columns]
                st.dataframe(
                    comp[display_cols].style.format({
                        c: "{:.5f}" for c in ["R2", "RMSE", "MAE", "MAPE_%", "Train_R2"]
                        if c in display_cols
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                left, right = st.columns(2)
                with left:
                    st.markdown("#### R² score — higher is better")
                    r2_plot = comp.sort_values("R2", ascending=True)
                    fig_r2 = px.bar(
                        r2_plot, x="R2", y="Model", orientation="h",
                        color="R2", color_continuous_scale="Viridis",
                        text=r2_plot["R2"].map(lambda x: f"{x:.4f}")
                    )
                    fig_r2.update_traces(textposition="outside", cliponaxis=False)
                    fig_r2.update_layout(
                        height=max(350, 35 * len(comp)), template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10, r=45, t=25, b=20), coloraxis_showscale=False
                    )
                    st.plotly_chart(fig_r2, use_container_width=True)

                with right:
                    st.markdown("#### Prediction error (₹) — lower is better")
                    err = comp.melt(
                        id_vars=["Model"], value_vars=[c for c in ["RMSE", "MAE"] if c in comp.columns],
                        var_name="Metric", value_name="Error (₹)"
                    )
                    fig_err = px.bar(
                        err, x="Error (₹)", y="Model", color="Metric", barmode="group",
                        orientation="h", text=err["Error (₹)"].map(lambda x: f"₹{x:.3f}")
                    )
                    fig_err.update_traces(textposition="outside", cliponaxis=False)
                    fig_err.update_layout(
                        height=max(350, 35 * len(comp)), template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10, r=65, t=25, b=20)
                    )
                    st.plotly_chart(fig_err, use_container_width=True)

                if "MAPE_%" in comp.columns:
                    st.markdown("#### MAPE — average percentage error (lower is better)")
                    mape = comp.dropna(subset=["MAPE_%"]).sort_values("MAPE_%", ascending=True)
                    fig_mape = px.bar(
                        mape, x="Model", y="MAPE_%", color="MAPE_%",
                        color_continuous_scale="Plasma",
                        text=mape["MAPE_%"].map(lambda x: f"{x:.3f}%")
                    )
                    fig_mape.update_traces(textposition="outside", cliponaxis=False)
                    fig_mape.update_layout(
                        height=380, template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10, r=20, t=25, b=90), coloraxis_showscale=False,
                        xaxis_tickangle=-35
                    )
                    st.plotly_chart(fig_mape, use_container_width=True)

                if "Fit_Seconds" in comp.columns:
                    st.markdown("#### Training time — lower is faster")
                    fit = comp.dropna(subset=["Fit_Seconds"]).sort_values("Fit_Seconds", ascending=True)
                    fig_fit = px.bar(
                        fit, x="Model", y="Fit_Seconds", color="Fit_Seconds",
                        text=fit["Fit_Seconds"].map(lambda x: f"{x:.1f}s")
                    )
                    fig_fit.update_traces(textposition="outside", cliponaxis=False)
                    fig_fit.update_layout(
                        height=350, template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=10, r=15, t=25, b=90), coloraxis_showscale=False,
                        xaxis_tickangle=-35
                    )
                    st.plotly_chart(fig_fit, use_container_width=True)

                st.markdown("#### How to interpret these metrics")
                st.markdown("""
                - **R² (coefficient of determination):** closer to 1 is better. A score of 1 is a perfect fit; 0 means the model is no better than predicting the test-set mean; a negative score means it performs worse than that baseline.
                - **RMSE (root mean squared error):** typical error scale in the target's units (₹ here), with larger errors penalized more strongly. Lower is better.
                - **MAE (mean absolute error):** average absolute difference between actual and predicted prices, in rupees. Lower is better.
                - **MAPE (%):** average absolute error relative to the actual price. Lower is better; it can be unstable when actual values are near zero.
                - **Train R²:** fit score on a sample of training rows. If it is much higher than test R², the model may be overfitting.
                - **Fit seconds / Predict seconds:** measured training and inference duration for the run. Faster is useful, but accuracy and latency should both be considered.
                """)

                st.markdown("#### Actual vs predicted values (sample)")
                if os.path.isfile(predictions_path):
                    try:
                        pred_df = pd.read_csv(predictions_path)
                        if "Actual" in pred_df.columns and len(pred_df):
                            model_choices = [c for c in pred_df.columns if c != "Actual" and c in comp["Model"].astype(str).tolist()]
                            if model_choices:
                                chosen = st.selectbox("Choose a model to inspect", model_choices, key="performance_prediction_model")
                                n_show = min(300, len(pred_df))
                                chart_df = pred_df.iloc[:n_show][["Actual", chosen]].copy()
                                chart_df.insert(0, "Test sample", np.arange(1, len(chart_df) + 1))
                                chart_long = chart_df.melt(
                                    id_vars="Test sample", var_name="Series", value_name="Price (₹)"
                                )
                                fig_actual = px.line(
                                    chart_long, x="Test sample", y="Price (₹)", color="Series",
                                    title=f"Actual vs {chosen} prediction (first {n_show} saved test samples)"
                                )
                                fig_actual.update_layout(
                                    height=400, template="plotly_dark",
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
                                )
                                st.plotly_chart(fig_actual, use_container_width=True)
                            else:
                                st.caption("No matching model prediction columns were found in predictions_sample.csv.")
                        else:
                            st.caption("The prediction sample file has no Actual column or contains no rows.")
                    except Exception as e:
                        st.caption(f"Could not read prediction samples: {type(e).__name__}: {e}")
                else:
                    st.caption("For the actual-versus-predicted chart, run train_all_models.py to generate predictions_sample.csv.")

                st.caption(
                    "Important: these scores describe the saved training run and its held-out dataset—not guaranteed live-market accuracy. "
                    "The lowest RMSE is the ranking criterion used by the training script; verify the test design and avoid treating predictions as financial advice."
                )
        except Exception as e:
            st.error(f"Could not load model comparison results: {type(e).__name__}: {e}")

# Footer
st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit • Order book data profiled with data-profiling • Not financial advice
</div>
""", unsafe_allow_html=True)