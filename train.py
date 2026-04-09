import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from stable_baselines3 import DQN

from utils.preprocessing import load_and_preprocess
from graph.graph_builder import build_graph
from model.gnn_encoder import VenomGNN
from model.dqn_agent import FraudEnv

# ─────────────────────────────────────────
# PHASE 1: Train GNN
# ─────────────────────────────────────────
print("\n" + "="*50)
print("🕸️  VENOM TRAINING PIPELINE")
print("="*50)

print("\n📦 Loading & preprocessing data...")
df = load_and_preprocess("data/transactions.csv", nrows=100000)

print("🔨 Building transaction graph...")
graph_data, node_map = build_graph(df)

# Train/test split on nodes
num_nodes = graph_data.x.shape[0]
indices = list(range(num_nodes))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, random_state=42,
    stratify=graph_data.y.numpy()
)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx]   = True

model = VenomGNN(in_channels=8, hidden_channels=64, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                              weight_decay=5e-4)

# Handle class imbalance with weighted loss
fraud_count   = graph_data.y.sum().item()
total         = len(graph_data.y)
weight        = torch.tensor([1.0, total / (fraud_count + 1e-8)])
criterion     = torch.nn.CrossEntropyLoss(weight=weight)

print("\n🧠 Training GNN Encoder (50 epochs)...")
best_auc, patience, patience_count = 0, 10, 0

for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    out  = model(graph_data.x, graph_data.edge_index)
    loss = criterion(out[train_mask], graph_data.y[train_mask])
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(
                model(graph_data.x, graph_data.edge_index), dim=1
            )
            fraud_probs = probs[test_mask, 1].numpy()
            true_labels = graph_data.y[test_mask].numpy()
            auc = roc_auc_score(true_labels, fraud_probs)
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
              f"AUROC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "model/venom_gnn.pt")
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= patience:
            print("⏹️  Early stopping triggered")
            break

print(f"\n✅ Best AUROC: {best_auc:.4f} | Model saved!")

# ─────────────────────────────────────────
# PHASE 2: Train DQN Agent
# ─────────────────────────────────────────
print("\n🤖 Training DQN RL Agent...")
model.load_state_dict(torch.load("model/venom_gnn.pt"))
model.eval()

with torch.no_grad():
    probs       = torch.softmax(
        model(graph_data.x, graph_data.edge_index), dim=1
    )
    risk_scores = probs[:, 1].numpy()
    true_labels = graph_data.y.numpy()

env       = FraudEnv(risk_scores, true_labels)
dqn_agent = DQN("MlpPolicy", env, verbose=0,
                  learning_rate=1e-3, batch_size=64,
                  buffer_size=10000, learning_starts=500)
dqn_agent.learn(total_timesteps=30000)
dqn_agent.save("model/venom_dqn")

print("✅ DQN Agent trained and saved!")
print("\n🕸️  VENOM training complete. Ready for deployment!")
