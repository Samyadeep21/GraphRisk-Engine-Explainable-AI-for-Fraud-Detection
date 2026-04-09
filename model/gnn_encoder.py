import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class VenomGNN(torch.nn.Module):
    def __init__(self, in_channels=8, hidden_channels=64, out_channels=2):
        super(VenomGNN, self).__init__()
        # Layer 1: 4-head attention — captures diverse neighbor patterns
        self.gat1 = GATConv(in_channels, hidden_channels,
                             heads=4, dropout=0.3)
        # Layer 2: Single head — final node representation
        self.gat2 = GATConv(hidden_channels * 4, hidden_channels,
                             heads=1, dropout=0.3)
        # Output: fraud probability per node
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        return self.classifier(x)

    def get_embeddings(self, x, edge_index):
        """Returns node embeddings before classification — used by DQN"""
        with torch.no_grad():
            x = self.gat1(x, edge_index)
            x = F.elu(x)
            x = self.gat2(x, edge_index)
            return F.elu(x)

if __name__ == "__main__":
    model = VenomGNN()
    print("✅ VENOM GNN Architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Total parameters: {total_params:,}")
