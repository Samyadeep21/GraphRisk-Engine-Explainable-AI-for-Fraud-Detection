# ⚡ GraphRisk Engine — Explainable AI for Fraud Detection

> 🚀 A production-grade AI dashboard combining **Graph Neural Networks (GNN)** and **Reinforcement Learning (RL)** for intelligent fraud detection with explainability.

---

## 📌 Overview

GraphRisk Engine is an end-to-end system designed to detect fraudulent transactions using **graph-based learning** and **decision optimization via RL**.

Unlike traditional black-box models, this system focuses on:

✔ Transparent decision-making  
✔ Explainable risk scoring  
✔ Real-time fraud insights  
✔ Clean, production-level UI  

---

## 🧠 Core Idea

Financial fraud is inherently **relational**.

Instead of treating transactions independently, this system models:

- Users → Nodes  
- Transactions → Edges  
- Behavior → Graph Patterns  

Using:
- **GNN → Structural understanding**
- **RL → Decision optimization**

---

## 🏗️ Architecture

CSV Input → Backend API (FastAPI)
↓
GNN + RL Model
↓
Risk Scores + Decisions
↓
Streamlit Dashboard (UI Layer)


---

## ✨ Features

### 🔹 1. Risk Scoring Engine
- Assigns risk scores using GNN + RL
- Categorizes into:
  - Low 🟢
  - Medium 🔵
  - High 🟠
  - Critical 🔴

---

### 🔹 2. Interactive Dashboard

#### 📊 Risk Distribution
- Histogram with segmented risk levels
- Clean spacing + color-coded bins
- High readability (optimized UI)

#### 📈 KPI Metrics
- Total Transactions
- Flagged Transactions
- Approved Transactions
- Fraud Rate %

---

### 🔹 3. Explainability Panel (🔥 Key Highlight)

Provides **WHY a node is flagged**, not just prediction:

- Node ID
- Risk Score
- Risk Level
- Model Confidence
- Reasons:
  - High connectivity
  - Suspicious graph patterns
  - RL decision threshold breach

---

### 🔹 4. Transaction Explorer
- Full dataset visualization
- Risk-level tagging
- Clean tabular UI

---

### 🔹 5. Flagged Transaction Monitoring
- Separate section for suspicious nodes
- Clear success/failure indicators

---

## 🎨 UI/UX Highlights

This project went through **multiple UI redesign iterations** to reach production quality.

### Improvements Made:

❌ Initial Issues:
- Dark UI → low readability  
- Graph labels not visible  
- Poor spacing in charts  
- Explainability panel unreadable  

✅ Final Improvements:
- Light gradient background (blue → white)
- Glass-like card layout
- Clean typography + spacing
- High-contrast charts
- Proper axis visibility (fixed risk_score readability)
- Structured layout (2-column insights + graphs)

👉 Result:
> A **fintech-grade dashboard**, not a generic Streamlit app.

---

## ⚙️ Tech Stack

| Layer        | Technology |
|-------------|-----------|
| Frontend UI | Streamlit |
| Visualization | Plotly |
| Backend API | FastAPI |
| ML Models | GNN + RL |
| Data Handling | Pandas |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/graph-risk-engine.git
cd graph-risk-engine

2. Install dependencies
pip install -r requirements.txt
3. Start backend (FastAPI)
uvicorn main:app --reload
4. Run dashboard
streamlit run dashboard.py
📂 Input Format

Upload a CSV file containing:

node_id
transaction features
graph-related attributes

📊 Example Output
Risk Score: 0.4994
Risk Level: Critical
Decision: FLAGGED

🧩 Challenges Faced & Solutions
❌ Problem 1: UI Visibility Issues
Dark theme made text unreadable

✅ Solution:

Switched to light gradient UI
Improved contrast and font clarity
❌ Problem 2: Graph Readability
Risk score labels not visible
Bars overlapping

✅ Solution:

Added spacing (bargap)
Fixed axis color and fonts
Applied color segmentation
❌ Problem 3: Explainability Panel Not Clear

✅ Solution:

Structured panel with clear hierarchy
Added contrast block for "Why Flagged"
❌ Problem 4: Generic Dashboard Look

✅ Solution:

Designed custom UI system
Avoided default Streamlit look
Used card-based layout + gradients
🎯 What Makes This Project Stand Out

✔ Combines GNN + RL (advanced ML)
✔ Focus on Explainable AI (XAI)
✔ Strong UI/UX engineering effort
✔ Real-world fintech use case
✔ Recruiter-ready dashboard

🔮 Future Improvements
🔗 Graph visualization (network view)
📉 Time-series fraud trends
🤖 Model comparison panel
📊 Feature importance visualization
🌐 Deploy on cloud (Streamlit Cloud / Vercel)
👨‍💻 Author

Samyadeep Saha
M.Tech CSE | AI + Cybersecurity Enthusiast

⭐ If you like this project

Give it a ⭐ on GitHub — it helps a lot!