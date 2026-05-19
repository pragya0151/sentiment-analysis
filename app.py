# ============================================================
# Sentiment Analysis App
# DistilBERT fine-tuned on Amazon Fine Food Reviews
# Built by Pragya Khullar
# ============================================================

import streamlit as st
import torch
import torch.nn as nn
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main { background-color: #ffffff; }
    
    .hero-container {
        padding: 1.5rem 0 1rem 0;
        border-bottom: 1px solid #e9ecef;
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: 3rem !important;
        font-weight: 800;
        color: #1a1a2e;
        margin: 0;
        line-height: 1.2;
        text-align: center;
        letter-spacing: -0.02rem;
    }
    .hero-subtitle {
        font-size: 4rem;
        color: #6c757d;
        margin: 0.5rem 0 0 0;
        text-align: center;
        letter-spacing: 0.02rem;
    }
    .result-positive {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left: 4px solid #28a745;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    .result-negative {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-left: 4px solid #dc3545;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    .result-label {
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0;
        color: #1a1a2e;
    }
    .result-confidence {
        font-size: 0.9rem;
        color: #495057;
        margin: 0.2rem 0 0 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
    }
    .impact-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid #e9ecef;
        height: 100%;
    }
    .impact-value-green {
        font-size: 1.8rem;
        font-weight: 700;
        color: #28a745;
        margin: 0;
    }
    .impact-value-large {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 0;
    }
    .impact-label {
        font-size: 0.75rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
        margin: 0;
    }
    .section-divider {
        border: none;
        border-top: 1px solid #e9ecef;
        margin: 1.5rem 0;
    }
    .example-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        height: 100%;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL LOADING
# ============================================================

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    def forward(self, logits):
        return logits / self.temperature

@st.cache_resource
def load_model():
    device = torch.device('cpu')
    tokenizer = DistilBertTokenizer.from_pretrained('models/sentiment_model')
    model = DistilBertForSequenceClassification.from_pretrained('models/sentiment_model')
    model = model.to(device)
    model.eval()
    scaler = TemperatureScaler()
    scaler.load_state_dict(torch.load(
        'models/temperature_scaler.pt', map_location=device))
    scaler.eval()
    return tokenizer, model, scaler, device

tokenizer, model, scaler, device = load_model()

# ============================================================
# INFERENCE
# ============================================================

def predict(texts):
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(
        texts, max_length=128, truncation=True,
        padding=True, return_tensors='pt'
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        scaled_logits = scaler(outputs.logits)
        probs = torch.nn.functional.softmax(scaled_logits, dim=1)
    probs = probs.cpu().numpy()
    predictions = ['Positive' if p[1] > 0.5 else 'Negative' for p in probs]
    confidences = [round(float(max(p)), 3) for p in probs]
    return predictions, confidences, probs

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### 🎯 Sentiment Analyzer")
    st.markdown("*DistilBERT fine-tuned on Amazon reviews*")
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("**Model Performance**")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class='metric-card'>
            <div class='metric-value'>0.95</div>
            <div class='metric-label'>Clean F1</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='metric-card'>
            <div class='metric-value'>0.86</div>
            <div class='metric-label'>Noisy F1</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("""<div class='metric-card'>
            <div class='metric-value'>80K</div>
            <div class='metric-label'>Training</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class='metric-card'>
            <div class='metric-value'>1.528</div>
            <div class='metric-label'>Temp Scale</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("**About**")
    st.markdown("""
    Built to identify negative reviews before they impact e-commerce revenue.
    - TF-IDF Baseline: 0.92 F1
    - DistilBERT: **0.95 F1**
    - Domain tested on Twitter data
    - Temperature scaled (T=1.528)
    """)
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.caption("Built by Pragya Khullar")
    st.caption("Dataset: Amazon Fine Food Reviews")

# ============================================================
# HERO HEADER
# ============================================================

st.markdown("""
<div class='hero-container'>
    <p class='hero-title'>🎯 Sentiment Analyzer</p>
    <p class='hero-subtitle'>Fine-tuned DistilBERT · SHAP Explainability · Business Impact · Amazon Fine Food Reviews</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "🔍 Single Review",
    "📦 Batch Prediction",
    "📈 Business Impact"
])

# ============================================================
# TAB 1 — SINGLE REVIEW
# ============================================================

with tab1:
    col_input, col_examples = st.columns([3, 1])

    with col_input:
        review_input = st.text_area(
            "Review",
            placeholder="Type or paste a product review here...",
            height=120,
            label_visibility="collapsed"
        )
        analyze_btn = st.button("Analyze Sentiment", type="primary")

    with col_examples:
        st.markdown("""
        <div class='example-box'>
            <p style='font-size:0.78rem; color:#6c757d; margin:0 0 0.6rem 0; text-transform:uppercase; letter-spacing:0.05rem;'>Try These</p>
            <p style='font-size:0.85rem; margin:0.4rem 0;'>😊 "Amazing product, best purchase ever!"</p>
            <p style='font-size:0.85rem; margin:0.4rem 0;'>😞 "Terrible quality, waste of money."</p>
            <p style='font-size:0.85rem; margin:0.4rem 0;'>😐 "It was okay, not what I expected."</p>
        </div>
        """, unsafe_allow_html=True)

    if analyze_btn and review_input.strip():
        with st.spinner("Analyzing..."):
            predictions, confidences, probs = predict(review_input)
            sentiment = predictions[0]
            confidence = confidences[0]

        if sentiment == "Positive":
            st.markdown(f"""
            <div class='result-positive'>
                <p class='result-label'>✅ Positive</p>
                <p class='result-confidence'>Confidence: {confidence*100:.1f}%</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-negative'>
                <p class='result-label'>❌ Negative</p>
                <p class='result-confidence'>Confidence: {confidence*100:.1f}%</p>
            </div>""", unsafe_allow_html=True)

        # Equal size side by side charts
        col_prob, col_shap = st.columns([1, 1])

        with col_prob:
            st.markdown("**Probability Breakdown**")
            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.barh(
                ['Negative', 'Positive'],
                [probs[0][0], probs[0][1]],
                color=['#dc3545', '#28a745'],
                height=0.4
            )
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            for bar, val in zip(bars, [probs[0][0], probs[0][1]]):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', va='center', fontsize=11)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_shap:
            st.markdown("**Why this prediction?**")
            st.caption("⏳ Generating SHAP explanation (30-60 seconds)...")
            with st.spinner(""):
                explainer = shap.Explainer(
                    lambda x: predict(list(x))[2],
                    masker=shap.maskers.Text(tokenizer),
                    output_names=["Negative", "Positive"]
                )
                shap_values = explainer([review_input])
                fig, ax = plt.subplots(figsize=(5, 4))
                shap.plots.waterfall(
                    shap_values[0, :, 1],
                    max_display=8,
                    show=False
                )
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    elif analyze_btn and not review_input.strip():
        st.warning("Please enter a review to analyze.")

# ============================================================
# TAB 2 — BATCH PREDICTION
# ============================================================

with tab2:
    st.markdown("Upload a CSV file to analyze multiple reviews at once.")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        label_visibility="collapsed"
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        col_preview, col_settings = st.columns([2, 1])
        with col_preview:
            st.markdown("**Preview**")
            st.dataframe(df.head(3), use_container_width=True)
        with col_settings:
            st.markdown("**Settings**")
            text_column = st.selectbox(
                "Select text column",
                options=df.columns.tolist()
            )
            st.caption(f"{len(df):,} rows detected")

        if st.button("Run Analysis", type="primary", use_container_width=True):
            texts = df[text_column].fillna('').astype(str).tolist()
            all_preds = []
            all_confs = []

            progress_bar = st.progress(0)
            status = st.empty()

            for i in range(0, len(texts), 32):
                batch = texts[i:i+32]
                preds, confs, _ = predict(batch)
                all_preds.extend(preds)
                all_confs.extend(confs)
                progress = min(i+32, len(texts)) / len(texts)
                progress_bar.progress(progress)
                status.caption(f"Processing {min(i+32, len(texts)):,} / {len(texts):,} reviews...")

            status.empty()
            progress_bar.empty()

            results_df = pd.DataFrame({
                'Review': texts,
                'Sentiment': all_preds,
                'Confidence': [f"{c*100:.1f}%" for c in all_confs]
            })

            positive_count = all_preds.count('Positive')
            negative_count = all_preds.count('Negative')
            avg_conf = sum(all_confs) / len(all_confs)

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Reviews", f"{len(texts):,}")
            c2.metric("Positive", f"{positive_count:,}")
            c3.metric("Negative", f"{negative_count:,}")
            c4.metric("Avg Confidence", f"{avg_conf*100:.1f}%")

            st.markdown("<br>", unsafe_allow_html=True)

            col_chart, col_table = st.columns([1, 2])
            with col_chart:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.pie(
                    [positive_count, negative_count],
                    labels=['Positive', 'Negative'],
                    colors=['#28a745', '#dc3545'],
                    autopct='%1.1f%%',
                    startangle=90
                )
                ax.set_title('Sentiment Distribution', fontsize=11)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col_table:
                st.markdown("**Results**")
                st.dataframe(results_df, use_container_width=True, height=250)

            csv = results_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Results CSV",
                data=csv,
                file_name="sentiment_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

# ============================================================
# TAB 3 — BUSINESS IMPACT CALCULATOR
# ============================================================

with tab3:
    st.markdown("Estimate the revenue impact of deploying this model for your business.")
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col_inputs, col_results = st.columns([1, 1])

    with col_inputs:
        st.markdown("**Your Business Parameters**")
        avg_order = st.slider("Average Order Value ($)", 10, 500, 35, 5)
        churn_rate = st.slider("Review-driven Churn Rate (%)", 1, 50, 15, 1)
        readers = st.slider("Avg Readers per Review", 10, 1000, 100, 10)
        daily_reviews = st.slider("Daily Reviews per Product", 10, 500, 50, 10)
        neg_rate = st.slider("Negative Review Rate (%)", 5, 50, 20, 5)

    with col_results:
        st.markdown("**Estimated Impact**")

        daily_neg = daily_reviews * (neg_rate / 100)
        daily_caught = daily_neg * 0.95
        daily_missed = daily_neg * 0.05
        rev_per_review = readers * (churn_rate / 100) * avg_order
        daily_rev = daily_caught * rev_per_review
        monthly_rev = daily_rev * 30
        annual_rev = daily_rev * 365

        st.markdown(f"""
        <div class='impact-card'>
            <div style='margin-bottom:1rem;'>
                <p class='impact-label'>Daily Revenue Protected</p>
                <p class='impact-value-green'>${daily_rev:,.0f}</p>
            </div>
            <div style='margin-bottom:1rem;'>
                <p class='impact-label'>Monthly Revenue Protected</p>
                <p class='impact-value-green'>${monthly_rev:,.0f}</p>
            </div>
            <div style='margin-bottom:1.5rem;'>
                <p class='impact-label'>Annual Revenue Protected</p>
                <p class='impact-value-large'>${annual_rev:,.0f}</p>
            </div>
            <hr style='border:none; border-top:1px solid #e9ecef; margin:1rem 0;'>
            <div style='display:flex; justify-content:space-between; text-align:center;'>
                <div>
                    <p class='impact-label'>Flagged Daily</p>
                    <p style='font-size:1.3rem; font-weight:700; margin:0; color:#28a745;'>{daily_caught:.1f}</p>
                </div>
                <div>
                    <p class='impact-label'>Missed Daily</p>
                    <p style='font-size:1.3rem; font-weight:700; margin:0; color:#dc3545;'>{daily_missed:.1f}</p>
                </div>
                <div>
                    <p class='impact-label'>Model Recall</p>
                    <p style='font-size:1.3rem; font-weight:700; margin:0; color:#1a1a2e;'>95%</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("⚠️ Estimates based on your inputs. Calibrate with actual sales data in production.")

    # Revenue chart
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("**Revenue Protection Over Time**")

    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        # Bar chart — daily/monthly/annual
        fig, ax = plt.subplots(figsize=(5, 3.5))
        periods = ['Daily', 'Monthly', 'Annual']
        values = [daily_rev, monthly_rev, annual_rev]
        colors = ['#3498db', '#9b59b6', '#28a745']
        bars = ax.bar(periods, values, color=colors, width=0.5)
        ax.set_ylabel('Revenue Protected ($)')
        ax.set_yscale('log')
        ax.set_title('Revenue Protected by Period', fontweight='bold')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() * 1.02,
                   f'${val:,.0f}',
                   ha='center', fontsize=9, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_chart2:
        # Pie chart — caught vs missed
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.pie(
            [daily_caught, daily_missed,
             daily_reviews * (1 - neg_rate/100)],
            labels=['Flagged\nNegative', 'Missed\nNegative', 'Positive\nCleared'],
            colors=['#28a745', '#dc3545', '#3498db'],
            autopct='%1.1f%%',
            startangle=90
        )
        ax.set_title('Daily Review Classification', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with st.expander("📊 How is this calculated?"):
        st.markdown(f"""
        **Formula:**
        Daily negative reviews  = {daily_reviews} × {neg_rate}% = {daily_neg:.1f}
        Caught by model         = {daily_neg:.1f} × 95% recall = {daily_caught:.1f}
        Revenue per review      = {readers} readers × {churn_rate}% churn × ${avg_order} = ${rev_per_review:,.0f}
        Daily revenue protected = {daily_caught:.1f} × ${rev_per_review:,.0f} = ${daily_rev:,.0f}
        Annual revenue protected= ${daily_rev:,.0f} × 365 = ${annual_rev:,.0f}

        **Parameters explained:**
        - **Churn rate:** % of readers who don't purchase after seeing a negative review
        - **Model recall:** 95% — measured on our test set
        - **Readers per review:** average product page visitors who read reviews
        """)