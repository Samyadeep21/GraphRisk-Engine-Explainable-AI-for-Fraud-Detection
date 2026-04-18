import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class VenomGNN(torch.nn.Module):
    def __init__(self, in_channels=8, hidden_channels=64, out_channels=2):
        super(VenomGNN, self).__init__()

        self.gat1 = GATConv(in_channels, hidden_channels, heads=4, dropout=0.3)
        self.gat2 = GATConv(hidden_channels * 4, hidden_channels, heads=1, dropout=0.3)

        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    # ✅ REQUIRED forward function
    def forward(self, x, edge_index, return_attention=False):

        x = F.dropout(x, p=0.3, training=self.training)

        x, attn1 = self.gat1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)

        x = F.dropout(x, p=0.3, training=self.training)

        x, attn2 = self.gat2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)

        out = self.classifier(x)

        if return_attention:
            return out, attn2

        return out

    # ✅ embeddings for RL
    def get_embeddings(self, x, edge_index):
        with torch.no_grad():
            x = self.gat1(x, edge_index)
            x = F.elu(x)

            x = self.gat2(x, edge_index)
            x = F.elu(x)

            return x