import streamlit as st
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# ============================================================
# 1. SAFE MODEL PATH FIX & LOADING
# ============================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: Adjust PROJECT_DIR if your Model folder is elsewhere
PROJECT_DIR = os.path.dirname(APP_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "Model")


@st.cache_resource
def load_models():
    lr = pickle.load(open(os.path.join(MODEL_DIR, "linear_model.pkl"), "rb"))
    rf = pickle.load(open(os.path.join(MODEL_DIR, "rf_model.pkl"), "rb"))
    encoder = pickle.load(open(os.path.join(MODEL_DIR, "location_encoder.pkl"), "rb"))
    return lr, rf, encoder


lr_model, rf_model, encoder = load_models()


# ============================================================
# 2. UTILITY & LOGIC FUNCTIONS
# ============================================================
def format_inr(amount):
    amount = int(amount)
    s = str(amount)
    if len(s) > 3:
        last3 = s[-3:]
        rest = s[:-3][::-1]
        groups = [rest[i:i + 2] for i in range(0, len(rest), 2)]
        s = ",".join(groups)[::-1] + "," + last3
    return f"₹ {s}"


def quick_reco(price):
    if price < 3000000:
        return "💡 Best for first-time buyers & rental income."
    elif price < 7000000:
        return "📈 Good buy – stable locality with steady growth."
    elif price < 15000000:
        return "🚀 Strong investment – high appreciation zone."
    else:
        return "🏆 Luxury segment – long-term wealth asset."


def get_recommendation(price, area, bhk):
    if price < 3000000:
        badge, advice, confidence = "🟢 Affordable Zone", "Good option for budget buyers.", 70
    elif price < 7000000:
        badge, advice, confidence = "🔵 High Demand Area", "Balanced pricing with good resale value.", 82
    elif price < 15000000:
        badge, advice, confidence = "🟣 Premium Growth Zone", "Strong long-term appreciation potential.", 88
    else:
        badge, advice, confidence = "🔴 Elite Locality", "Luxury segment with premium lifestyle.", 92

    space_tip = "⚠ Compact layout." if area / bhk < 350 else "✅ Spacious & well-planned."
    low, high = int(price * 0.9), int(price * 1.1)
    return badge, advice, space_tip, confidence, low, high


# ============================================================
# 3. PAGE CONFIG & CUSTOM CSS
# ============================================================
st.set_page_config("House Price Prediction", "🏠", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
* { font-family:Inter,sans-serif; }
.stApp { background: radial-gradient(circle at top,#020617,#000); color:#e5e7eb; }
header,footer { display:none; }
.header { border:2px solid #22d3ee; border-radius:22px; padding:35px; margin-bottom:25px;
box-shadow:0 0 35px rgba(34,211,238,.4); text-align: center; }
label { color:white !important; font-weight:600; }
input, select { background:#020617 !important; color:white !important;
border:2px solid #22d3ee !important; border-radius:12px !important; }
button { border-radius:14px !important; font-weight:700 !important;
background:transparent !important; color:#22d3ee !important;
border:2px solid #22d3ee !important; width: 100%; }
button:hover { background:rgba(34,211,238,.1) !important; transform:scale(1.02); }
.result { margin-top:25px; padding:30px; border-radius:20px;
background:linear-gradient(135deg,#022c22,#064e3b);
border:2px solid #10b981; box-shadow:0 0 35px rgba(16,185,129,.6); }
.conf { margin-top:20px; padding:26px; border-radius:20px;
border:2px solid #22d3ee; background:#020617;
box-shadow:0 0 30px rgba(34,211,238,.35); }
.progress { height:12px; background:#020617; border-radius:999px;
border:1px solid #22d3ee; overflow:hidden; margin: 10px 0; }
.progress span { height:100%; display:block;
background:linear-gradient(90deg,#22d3ee,#10b981); }
.badge { display:inline-block; margin-top:10px; padding:6px 14px;
border-radius:999px; border:1px solid #38bdf8;
color:#7dd3fc; font-size:12px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 4. USER INTERFACE
# ============================================================
st.markdown("""
<div class="header">
<h1>🏠 House Price Prediction</h1>
<p style="color:#22d3ee;">AI-Powered Real Estate Valuation</p>
</div>
""", unsafe_allow_html=True)

locations = list(encoder.classes_)

c1, c2 = st.columns(2)
with c1:
    location = st.selectbox("📍 Location", locations)
    area = st.number_input("📐 Area (sqft)", min_value=300, value=1500)
with c2:
    bhk = st.selectbox("🛏 BHK", [1, 2, 3, 4, 5], index=2)
    bath = st.selectbox("🚿 Bathrooms", [1, 2, 3, 4, 5], index=1)

# ============================================================
# 5. PREDICTION ENGINE & VISUALIZATION
# ============================================================
if st.button("🚀 Calculate Property Price"):
    # Data Processing
    loc_code = encoder.transform([location])[0]
    X = np.array([[area, bhk, bath, loc_code]])

    # Ensemble Prediction (Average of Models)
    price = int(np.expm1(((lr_model.predict(X) + rf_model.predict(X)) / 2)[0]))

    # Get Insights
    badge, advice, space_tip, conf, low, high = get_recommendation(price, area, bhk)
    quick_tip = quick_reco(price)

    # 5a. UI Display - Result Card
    st.markdown(f"""
    <div class="result">
        <h2 style="margin:0;">{format_inr(price)}</h2>
        <div class="badge">{badge}</div>
        <p style="margin-top:15px;">{advice}</p>
        <p><b>Recommendation:</b> {quick_tip}</p>
    </div>
    """, unsafe_allow_html=True)

    # 5b. UI Display - Confidence Card
    st.markdown(f"""
    <div class="conf">
        <h4 style="margin:0;">📊 Price Confidence</h4>
        <div class="progress"><span style="width:{conf}%"></span></div>
        <p>Confidence Score: <b>{conf}%</b></p>
        <ul style="margin-bottom:0;">
            <li>{space_tip}</li>
            <li>Expected Price Range: {format_inr(low)} – {format_inr(high)}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 5c. Data Visualization - Chart
    st.markdown("### 📊 Market Price Distribution")

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#020617")
    ax.set_facecolor("#020617")

    bars = ax.bar(
        ["Min", "Predicted", "Max"],
        [low, price, high],
        width=0.55,
        color=['#1e293b', '#22d3ee', '#1e293b'],
        edgecolor='#22d3ee'
    )

    # Add Text Labels on bars
    for bar, val in zip(bars, [low, price, high]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                format_inr(val), ha="center", va="bottom", color="white", fontsize=9)

    ax.set_title("Real-Estate Price Analysis", color="white", pad=20)
    ax.tick_params(colors="white")
    # Hide spines for a cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)

    st.pyplot(fig)