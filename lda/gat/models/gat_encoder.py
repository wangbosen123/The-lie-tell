import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, pool, global_mean_pool
from torch_geometric.data import Batch

class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim=None, num_heads_list=[4, 4], dropout=0.1):
        super(GATEncoder, self).__init__()

        self.num_layers = len(num_heads_list)
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()

        current_dim = in_channels
        for i, heads in enumerate(num_heads_list):
            if i == len(num_heads_list) - 1:
                out_dim = out_channels
                concat = False
                actual_heads = 1
            else:
                out_dim = 2 * out_channels // heads
                concat = True
                actual_heads = heads

            self.convs.append(GATConv(
                current_dim,
                out_dim,
                heads=actual_heads,
                dropout=dropout,
                concat=concat,
                edge_dim=edge_dim,
            ))

            if concat:
                current_dim = out_dim * actual_heads
            else:
                current_dim = out_dim

            self.pools.append(
                pool.SAGPooling(in_channels=current_dim, ratio=0.25)
            )

        self.dropout = dropout


    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for i, (conv, pool) in enumerate(zip(self.convs, self.pools)):
            x = conv(x, edge_index, edge_attr=edge_attr)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            x, edge_index, edge_attr, batch, _, _ = pool(x, edge_index, edge_attr, batch)
        # x = global_mean_pool(x, batch)

        batch_value = batch.max().item() + 1
        x = x.view(batch_value, -1, x.size(-1))
        return x