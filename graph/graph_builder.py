import torch
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

def build_graph(df):
    # All unique accounts = nodes
    all_nodes = pd.concat([df['nameOrig'], df['nameDest']]).unique()
    node_map = {name: idx for idx, name in enumerate(all_nodes)}
    num_nodes = len(node_map)
    print(f"✅ Total nodes (accounts): {num_nodes}")

    # Edges: each transaction = a directed edge
    src = [node_map[n] for n in df['nameOrig']]
    dst = [node_map[n] for n in df['nameDest']]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Edge features: amount, type, hour, day
    edge_features = df[[
        'amount_scaled', 'type_encoded',
        'hour', 'day_of_week'
    ]].values
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    # Node features: initialize with zeros (will learn via GNN)
    x = torch.zeros((num_nodes, 8), dtype=torch.float)

    # Node labels: is any transaction from/to this node suspicious?
    y = torch.zeros(num_nodes, dtype=torch.long)
    for i, row in df.iterrows():
        if row['is_suspicious'] == 1:
            y[node_map[row['nameOrig']]] = 1
            y[node_map[row['nameDest']]] = 1

    graph = Data(x=x, edge_index=edge_index,
                 edge_attr=edge_attr, y=y)
    print(f"✅ Graph built: {graph}")
    print(f"✅ Suspicious nodes: {y.sum().item()}")
    return graph, node_map

if __name__ == "__main__":
    from utils.preprocessing import load_and_preprocess
    df = load_and_preprocess("data/transactions.csv", nrows=100000)
    graph, node_map = build_graph(df)
