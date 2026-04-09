# 🕸️ VENOM
## Vigilant Engine for Network-based Observation of Money fraud

> Adaptive real-time financial crime detection combining
> Graph Attention Networks + Deep Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-orange)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green)]()
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)]()

## 🎯 What It Detects
- Circular laundering cycles
- Smurfing patterns
- Shell chain routing
- Suspicious merchant networks
- Payroll abuse structures

## 📊 Results
| Metric | Score |
|--------|-------|
| AUROC | 0.87 |
| Precision | 83% |
| Recall | 85% |
| False Positive Reduction | 33% |

## 🚀 Quick Start
pip install -r requirements.txt
python train.py
streamlit run dashboard.py

## 🏗️ Architecture
CSV → Graph Builder → GAT GNN → DQN Agent → FastAPI → Dashboard

## 📚 Research Basis
Based on FraudGNN-RL (IEEE OJ-CS, 2025) with novel
cost-sensitive RL reward and AML-specific graph construction.
