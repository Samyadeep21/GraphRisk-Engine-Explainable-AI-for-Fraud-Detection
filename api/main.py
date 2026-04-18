import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import torch
import io
import numpy as np

from stable_baselines3 import DQN
from torch_geometric.utils import degree

from utils.preprocessing import _preprocess_df
from graph.graph_builder import build_graph
from model.gnn_encoder import VenomGNN

# ───────────────────────── INIT
app = FastAPI(
    title="🕸️ VENOM API",
    version="4.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ───────────────────────── LOAD FEATURE DIM
with open("model/feature_dim.txt", "r") as f:
    FEATURE_DIM = int(f.read().strip())

print(f"✅ Loaded FEATURE_DIM: {FEATURE_DIM}")

# ───────────────────────── LOAD MODELS
gnn = VenomGNN(
    in_channels=FEATURE_DIM,
    hidden_channels=64,
    out_channels=2
)

gnn.load_state_dict(torch.load("model/venom_gnn.pt", map_location="cpu"))
gnn.eval()

print("✅ GNN model loaded")

dqn = DQN.load("model/venom_dqn")
print("✅ DQN model loaded")

# ───────────────────────── ROUTES
@app.get("/")
def root():
    return {"message": "VENOM API Live"}

# ───────────────────────── ANALYZE
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), nrows=50000)

        df_clean = _preprocess_df(df)
        graph, _ = build_graph(df_clean)

        # SAME FEATURE ENGINEERING AS TRAINING
        deg = degree(graph.edge_index[0], num_nodes=graph.x.size(0)).view(-1, 1)
        graph.x = torch.cat([graph.x, deg], dim=1)

        # GNN
        with torch.no_grad():
            logits = gnn(graph.x, graph.edge_index)
            probs = torch.softmax(logits, dim=1)

            risk_scores = probs[:, 1].cpu().numpy()

        results = []

        for i in range(min(200, len(risk_scores))):

            score = float(risk_scores[i])

            # RL INPUT (correct)
            state = np.array([score], dtype=np.float32)

            action, _ = dqn.predict(state, deterministic=True)

            decision = "FLAGGED" if int(action) == 1 else "APPROVED"

            results.append({
                "node_id": int(i),
                "risk_score": round(score, 4),
                "decision": decision
            })

        # ✅ FIXED SUMMARY
        total = len(results)
        flagged = sum(r["decision"] == "FLAGGED" for r in results)
        flagged_nodes = [r for r in results if r["decision"] == "FLAGGED"]

        return {
            "summary": {
                "total_analyzed": total,
                "flagged": flagged,
                "approved": total - flagged,
                "flag_rate": round(flagged / total * 100, 2)
            },
            "all_results": results,
            "flagged_nodes": flagged_nodes
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))