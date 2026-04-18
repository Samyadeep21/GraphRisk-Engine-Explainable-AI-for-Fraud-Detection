import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# ───────────────────────── CONFIG
st.set_page_config(page_title="GraphRisk Engine", layout="wide")

# ───────────────────────── STYLE
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #f8fafc, #eef2ff);
    color: #0F172A;
}

.header {
    font-size: 28px;
    font-weight: 700;
    color: white;
    padding: 20px;
    border-radius: 14px;
    background: linear-gradient(90deg, #2563EB, #22C55E);
    margin-bottom: 25px;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
}

.section {
    background: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
}

.section-title {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 12px;
}

.low { color: #22C55E; font-weight:600;}
.medium { color: #3B82F6; font-weight:600;}
.high { color: #F59E0B; font-weight:600;}
.critical { color: #EF4444; font-weight:600;}

</style>
""", unsafe_allow_html=True)

# ───────────────────────── SIDEBAR
st.sidebar.title("⚙️ Configuration")
api_url = st.sidebar.text_input("API Endpoint", "http://127.0.0.1:8000")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# ───────────────────────── HEADER
st.markdown('<div class="header">⚡ GraphRisk Engine — Explainable AI</div>', unsafe_allow_html=True)

# ───────────────────────── MAIN
if uploaded_file:

    with st.spinner("Running GNN + RL analysis..."):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{api_url}/analyze", files=files)
        data = response.json()

    summary = data["summary"]
    results = pd.DataFrame(data["all_results"])
    flagged = pd.DataFrame(data["flagged_nodes"])

    # 🔥 ADD RISK LEVEL (IMPORTANT FIX)
    def risk_bucket(score):
        if score < 0.497:
            return "Low"
        elif score < 0.498:
            return "Medium"
        elif score < 0.499:
            return "High"
        else:
            return "Critical"

    results["risk_level"] = results["risk_score"].apply(risk_bucket)

    # ───────── KPI
    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(f'<div class="card"><h4>Total</h4><h2>{summary["total_analyzed"]}</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="card"><h4>Flagged</h4><h2>{summary["flagged"]}</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="card"><h4>Approved</h4><h2>{summary["approved"]}</h2></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="card"><h4>Flag Rate (%)</h4><h2>{summary["flag_rate"]}</h2></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ───────── LAYOUT
    col1, col2 = st.columns([2.2, 1])

    # ───────── GRAPH (FIXED)
    with col1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Risk Distribution</div>', unsafe_allow_html=True)

        fig = px.histogram(
            results,
            x="risk_score",
            color="risk_level",
            nbins=28,
            color_discrete_map={
                "Low": "#22C55E",
                "Medium": "#3B82F6",
                "High": "#F59E0B",
                "Critical": "#EF4444"
            }
        )

        fig.update_traces(
            marker=dict(line=dict(color="#ffffff", width=1.2)),
            opacity=0.9
        )

        fig.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            bargap=0.12,
            margin=dict(l=40, r=20, t=30, b=50),

            font=dict(color="#0F172A"),

            xaxis=dict(
                title="Risk Score",
                tickfont=dict(color="#0F172A"),
                title_font=dict(color="#0F172A"),
                showgrid=False
            ),

            yaxis=dict(
                title="Transaction Count",
                tickfont=dict(color="#0F172A"),
                title_font=dict(color="#0F172A"),
                gridcolor="rgba(0,0,0,0.06)"
            )
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ───────── EXPLAINABILITY PANEL (FIXED)
    with col2:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧠 Explainability Panel</div>', unsafe_allow_html=True)

        top = results.sort_values("risk_score", ascending=False).iloc[0]
        risk = top["risk_score"]

        if risk < 0.497:
            level, cls = "Low", "low"
        elif risk < 0.498:
            level, cls = "Medium", "medium"
        elif risk < 0.499:
            level, cls = "High", "high"
        else:
            level, cls = "Critical", "critical"

        confidence = round(abs(risk - 0.5) * 2, 4)

        st.markdown(f"""
        <b>Node ID:</b> {int(top["node_id"])}<br><br>
        <b>Risk Score:</b> {risk:.4f}<br><br>
        <b>Risk Level:</b> <span class="{cls}">{level}</span><br><br>
        <b>Model Confidence:</b> {confidence:.4f}<br><br>

        <div style="background:#0F172A;color:white;padding:12px;border-radius:10px;">
        <b>Why Flagged?</b><br>
        • High transaction connectivity<br>
        • Suspicious graph behavior<br>
        • RL decision boundary exceeded
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ───────── TABLE
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📄 Transactions</div>', unsafe_allow_html=True)

    st.dataframe(results, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ───────── FLAGGED
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🚨 Flagged Transactions</div>', unsafe_allow_html=True)

    if flagged.empty:
        st.success("No suspicious transactions detected 🎉")
    else:
        st.dataframe(flagged, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Upload a CSV file to begin analysis")