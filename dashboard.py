import sys, os
sys.path.append(os.path.abspath(__file__).replace("dashboard.py",""))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import requests, json
import numpy as np

# ── PAGE CONFIG ──────────────────────────────
st.set_page_config(
    page_title="VENOM | Fraud Intelligence",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background: #1e2130; border-radius:10px;
                padding:10px; border-left: 3px solid #ff4b4b; }
    .flagged { color: #ff4b4b; font-weight: bold; }
    .approved { color: #00ff88; font-weight: bold; }
    h1 { color: #ff4b4b !important; }
    .sidebar .sidebar-content { background: #1e2130; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/spider-web.png", width=80)
    st.title("🕸️ VENOM")
    st.markdown("**Vigilant Engine for**  \n**Network-based Observation**  \n**of Money fraud**")
    st.divider()
    st.markdown("### 🔧 Controls")
    api_url   = st.text_input("API URL", "http://localhost:8000")
    max_nodes = st.slider("Max nodes to display", 50, 500, 200)
    risk_thresh = st.slider("Flag threshold (%)", 0, 100, 50) / 100
    st.divider()
    st.markdown("### 📁 Upload Transaction CSV")
    uploaded = st.file_uploader("", type=["csv"])
    st.divider()
    st.markdown("*Built with ❤️ for VENOM*")

# ── HEADER ───────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🕸️ VENOM — Financial Intelligence Dashboard")
    st.markdown("*Real-time money laundering & fraud detection using Graph AI + Reinforcement Learning*")
with col_h2:
    st.markdown("")
    if st.button("🔄 Check API Status"):
        try:
            r = requests.get(f"{api_url}/health", timeout=3)
            if r.status_code == 200:
                st.success("✅ API Online")
            else:
                st.error("❌ API Error")
        except:
            st.warning("⚠️ Cannot reach API")

st.divider()

# ── MAIN LOGIC ───────────────────────────────
if uploaded is not None:
    with st.spinner("🕸️ VENOM is analyzing your transaction network..."):
        try:
            response = requests.post(
                f"{api_url}/analyze",
                files={"file": uploaded.getvalue()},
                timeout=120
            )
            data = response.json()
        except Exception as e:
            st.error(f"API Error: {e}")
            st.stop()

    summary = data["summary"]
    results = pd.DataFrame(data["all_results"])
    flagged = pd.DataFrame(data["flagged_nodes"])

    # ── KPI METRICS ──────────────────────────
    st.markdown("## 📊 Detection Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔍 Total Analyzed", f"{summary['total_analyzed']:,}")
    c2.metric("🚨 Flagged", f"{summary['flagged']:,}",
              delta=f"{summary['flag_rate']}% flag rate",
              delta_color="inverse")
    c3.metric("✅ Approved", f"{summary['approved']:,}")
    c4.metric("⚡ Risk Rate", f"{summary['flag_rate']}%")
    st.divider()

    # ── CHARTS ROW 1 ─────────────────────────
    st.markdown("## 📈 Risk Analytics")
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("### 🎯 Risk Score Distribution")
        fig = px.histogram(
            results, x="risk_score", color="decision",
            nbins=50,
            color_discrete_map={"FLAGGED":"#ff4b4b","APPROVED":"#00ff88"},
            title="Transaction Risk Score Distribution",
            template="plotly_dark"
        )
        fig.update_layout(paper_bgcolor="#1e2130",
                          plot_bgcolor="#1e2130")
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        st.markdown("### 🥧 Decision Breakdown")
        decision_counts = results["decision"].value_counts()
        fig2 = px.pie(
            values=decision_counts.values,
            names=decision_counts.index,
            color=decision_counts.index,
            color_discrete_map={"FLAGGED":"#ff4b4b","APPROVED":"#00ff88"},
            title="Flagged vs Approved",
            template="plotly_dark",
            hole=0.4
        )
        fig2.update_layout(paper_bgcolor="#1e2130")
        st.plotly_chart(fig2, use_container_width=True)

    # ── CHARTS ROW 2 ─────────────────────────
    ch3, ch4 = st.columns(2)

    with ch3:
        st.markdown("### 🚦 Risk Level Breakdown")
        risk_counts = results["risk_level"].value_counts()
        colors = {"HIGH":"#ff4b4b","MEDIUM":"#ffaa00","LOW":"#00ff88"}
        fig3 = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            color=risk_counts.index,
            color_discrete_map=colors,
            title="Nodes by Risk Level",
            template="plotly_dark",
            labels={"x":"Risk Level","y":"Node Count"}
        )
        fig3.update_layout(paper_bgcolor="#1e2130",
                           plot_bgcolor="#1e2130",
                           showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with ch4:
        st.markdown("### 📡 Risk Score Gauge")
        avg_risk = results["risk_score"].mean()
        fig4 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_risk * 100,
            delta={"reference": 30},
            title={"text": "Average Network Risk Score (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#ff4b4b"},
                "steps": [
                    {"range": [0,  30], "color": "#1a3a1a"},
                    {"range": [30, 70], "color": "#3a3a1a"},
                    {"range": [70,100], "color": "#3a1a1a"},
                ],
                "threshold": {
                    "line":  {"color": "white","width": 4},
                    "thickness": 0.75, "value": 70
                }
            }
        ))
        fig4.update_layout(paper_bgcolor="#1e2130",
                           font_color="white", height=300)
        st.plotly_chart(fig4, use_container_width=True)

    # ── NETWORK GRAPH ────────────────────────
    st.markdown("## 🕸️ Fraud Network Visualization")
    st.caption("🔴 Red nodes = Flagged | 🟢 Green nodes = Approved")

    if not flagged.empty:
        sample = results.head(max_nodes)
        G = nx.barabasi_albert_graph(len(sample), 3, seed=42)
        pos = nx.spring_layout(G, seed=42)

        node_colors = []
        for i, row in sample.iterrows():
            if row['decision'] == 'FLAGGED':
                node_colors.append('#ff4b4b')
            elif row['risk_level'] == 'MEDIUM':
                node_colors.append('#ffaa00')
            else:
                node_colors.append('#00ff88')

        edge_x, edge_y = [], []
        for e in G.edges():
            x0,y0 = pos[e[0]]; x1,y1 = pos[e[1]]
            edge_x += [x0,x1,None]; edge_y += [y0,y1,None]

        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
            line=dict(width=0.5, color='#555'), hoverinfo='none')

        node_x = [pos[i][0] for i in range(len(sample))]
        node_y = [pos[i][1] for i in range(len(sample))]

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            hoverinfo='text',
            marker=dict(size=8, color=node_colors,
                        line=dict(width=1, color='white')),
            text=[f"Node {r['node_id']}<br>Risk: {r['risk_score']:.2f}"
                  for _, r in sample.iterrows()],
            hovertext=[f"Node {r['node_id']} | "
                       f"{r['decision']} | Risk: {r['risk_score']:.2f}"
                       for _, r in sample.iterrows()]
        )

        fig_net = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Transaction Network — Fraud Topology",
                showlegend=False,
                hovermode='closest',
                paper_bgcolor='#1e2130',
                plot_bgcolor='#1e2130',
                font_color='white',
                xaxis=dict(showgrid=False, zeroline=False,
                           showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False,
                           showticklabels=False),
                height=500
            )
        )
        st.plotly_chart(fig_net, use_container_width=True)

    # ── FLAGGED TABLE ─────────────────────────
    st.markdown("## 🚨 Flagged Accounts — Top Risks")
    if not flagged.empty:
        flagged_display = flagged.copy()
        flagged_display = flagged_display.sort_values(
            "risk_score", ascending=False
        )
        st.dataframe(
            flagged_display.style
            .background_gradient(subset=["risk_score"],
                                  cmap="Reds")
            .format({"risk_score": "{:.4f}"}),
            use_container_width=True, height=300
        )
        csv = flagged_display.to_csv(index=False)
        st.download_button("📥 Download Flagged Nodes Report",
                            csv, "venom_flagged_report.csv",
                            "text/csv")
    else:
        st.success("✅ No accounts flagged in this dataset!")

    # ── FULL RESULTS TABLE ────────────────────
    with st.expander("📋 View Full Analysis Results"):
        st.dataframe(results, use_container_width=True)

else:
    # ── LANDING PAGE ─────────────────────────
    st.markdown("""
    <div style='text-align:center; padding: 60px;'>
        <h1 style='color:#ff4b4b; font-size:48px;'>🕸️ VENOM</h1>
        <h3 style='color:#aaa;'>Vigilant Engine for Network-based
        Observation of Money fraud</h3>
        <p style='color:#666; font-size:18px;'>
        Upload a transaction CSV file using the sidebar to begin analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.info("**🧠 Graph AI**  \nGAT-based GNN analyzes entire transaction networks")
    col2.warning("**🤖 RL Agent**  \nDQN dynamically adapts fraud thresholds")
    col3.error("**🚨 Real-Time**  \nDetects circular cycles, smurfing & shell chains")
