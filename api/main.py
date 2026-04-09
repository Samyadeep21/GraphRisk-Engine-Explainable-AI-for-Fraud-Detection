import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd, torch, io, json
import numpy as np
from stable_baselines3 import DQN

from utils.preprocessing import load_and_preprocess
from graph.graph_builder import build_graph
from model.gnn_encoder import VenomGNN

app = FastAPI(
    title="🕸️ VENOM API",
    description="Vigilant Engine for Network-based Observation of Money fraud",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Load trained models on startup
gnn = VenomGNN(in_channels=8, hidden_channels=64, out_channels=2)
gnn.load_state_dict(torch.load("model/venom_gnn.pt",
                                map_location="cpu"))
gnn.eval()
dqn = DQN.load("model/venom_dqn")

@app.get("/")
def root():
    return {"message": "🕸️ VENOM is live!", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), nrows=50000)
        df_clean = load_and_preprocess.__wrapped__(df) \
            if hasattr(load_and_preprocess, '__wrapped__') \
            else _preprocess_df(df)
        graph, node_map = build_graph(df_clean)

        with torch.no_grad():
            probs = torch.softmax(
                gnn(graph.x, graph.edge_index), dim=1
            )
            risk_scores = probs[:, 1].numpy()

        results, flagged_nodes = [], []
        for i, score in enumerate(risk_scores[:500]):
            action, _ = dqn.predict(
                np.array([[score]], dtype=np.float32)
            )
            entry = {
                "node_id": i,
                "risk_score": round(float(score), 4),
                "decision": "FLAGGED" if int(action) == 1 else "APPROVED",
                "risk_level": (
                    "HIGH" if score > 0.7 else
                    "MEDIUM" if score > 0.4 else "LOW"
                )
            }
            results.append(entry)
            if int(action) == 1:
                flagged_nodes.append(entry)

        return {
            "summary": {
                "total_analyzed": len(results),
                "flagged": len(flagged_nodes),
                "approved": len(results) - len(flagged_nodes),
                "flag_rate": round(len(flagged_nodes)/len(results)*100, 2)
            },
            "flagged_nodes": flagged_nodes[:50],
            "all_results": results[:200]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _preprocess_df(df):
    """Inline preprocessing for API use"""
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    le, sc = LabelEncoder(), StandardScaler()
    df = df.dropna(subset=['nameOrig','nameDest','amount'])
    df['type_encoded']     = le.fit_transform(df['type'])
    df['typology_encoded'] = le.fit_transform(
        df['laundering_typology'].fillna('normal')
    )
    df['amount_scaled']    = sc.fit_transform(df[['amount']])
    df['fraud_probability'] = df.get('fraud_probability', 0)
    df['is_suspicious'] = (
        (df.get('isFraud', 0) == 1) |
        (df.get('isMoneyLaundering', 0) == 1)
    ).astype(int)
    return df
