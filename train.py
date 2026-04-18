import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import DQN
from torch_geometric.utils import degree

from utils.preprocessing import load_and_preprocess
from graph.graph_builder import build_graph
from model.gnn_encoder import VenomGNN
from model.dqn_agent import FraudEnv

# ─────────────────────────
# LOAD DATA
# ─────────────────────────
print("\n" + "="*60)
print("🕸️ VENOM: FRAUD DETECTION PIPELINE")
print("="*60)

df = load_and_preprocess("data/transactions.csv", nrows=100000)

graph_data, node_map = build_graph(df)
num_nodes = graph_data.x.shape[0]

# ─────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────
scaler = StandardScaler()
graph_data.x = torch.tensor(
    scaler.fit_transform(graph_data.x.numpy()),
    dtype=torch.float32
)

# Add node degree (IMPORTANT)
deg = degree(graph_data.edge_index[0], num_nodes=num_nodes).view(-1, 1)
graph_data.x = torch.cat([graph_data.x, deg], dim=1)

# 🔥 SAVE FEATURE DIM (CRITICAL)
FEATURE_DIM = graph_data.x.shape[1]
print(f"✅ FINAL FEATURE DIM: {FEATURE_DIM}")

# Save feature dim for API
with open("model/feature_dim.txt", "w") as f:
    f.write(str(FEATURE_DIM))

# ─────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────
indices = list(range(num_nodes))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=graph_data.y.numpy()
)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_idx] = True
test_mask[test_idx]   = True

# ─────────────────────────
# MODEL
# ─────────────────────────
model = VenomGNN(
    in_channels=FEATURE_DIM,   # ✅ ALWAYS CORRECT
    hidden_channels=64,
    out_channels=2
)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# ─────────────────────────
# CLASS IMBALANCE
# ─────────────────────────
fraud_count = graph_data.y.sum().item()
total = len(graph_data.y)

weight = torch.tensor([
    1.0,
    (total - fraud_count) / (fraud_count + 1e-6)
])

criterion = torch.nn.CrossEntropyLoss(weight=weight)

print(f"Fraud count: {fraud_count} | Total: {total}")

# ─────────────────────────
# TRAIN GNN
# ─────────────────────────
print("\n🧠 Training GNN...")

best_auc = 0.0

for epoch in range(1, 101):

    model.train()
    optimizer.zero_grad()

    logits = model(graph_data.x, graph_data.edge_index)

    loss = criterion(
        logits[train_mask],
        graph_data.y[train_mask]
    )

    loss.backward()
    optimizer.step()

    # ───── EVAL
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(
            model(graph_data.x, graph_data.edge_index),
            dim=1
        )

        fraud_probs = probs[test_mask, 1].cpu().numpy()
        true_labels = graph_data.y[test_mask].cpu().numpy()

        auc = roc_auc_score(true_labels, fraud_probs)

    print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), "model/venom_gnn.pt")

print(f"\n✅ Best GNN AUC: {best_auc:.4f}")

# ─────────────────────────
# RL TRAINING
# ─────────────────────────
print("\n🤖 Training RL Agent...")

model.load_state_dict(torch.load("model/venom_gnn.pt"))
model.eval()

with torch.no_grad():
    probs = torch.softmax(
        model(graph_data.x, graph_data.edge_index),
        dim=1
    )

    risk_scores = probs[:, 1].cpu().numpy()
    embeddings = model.get_embeddings(
        graph_data.x,
        graph_data.edge_index
    ).cpu().numpy()

    true_labels = graph_data.y.cpu().numpy()

env = FraudEnv(
    risk_scores=risk_scores.astype(np.float32),
    true_labels=true_labels.astype(np.int32)
)

dqn_agent = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-3,
    batch_size=64,
    buffer_size=10000
)

dqn_agent.learn(total_timesteps=30000)
dqn_agent.save("model/venom_dqn")

print("✅ RL training complete!")

# ─────────────────────────
# FINAL EVAL
# ─────────────────────────
print("\n📊 Final Evaluation")

preds = (risk_scores[:10000] > 0.3).astype(int)

print(classification_report(
    true_labels[:10000],
    preds,
    zero_division=0
))

print("\n🚀 VENOM pipeline complete!")