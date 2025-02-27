import os
import sys
sys.path.append('./')
sys.path.append('../')
from models.MSG3D.msgcn import MultiScale_GraphConv as MS_GCN
from models.MSG3D.gcn import GraphConv as GraphConv


import torch
import torch.nn as nn
import torch.nn.functional as F

# If you're using the same MS_GCN from msgcn.py, import it:
# from models.MSG3D.msgcn import MultiScale_GraphConv as MS_GCN

class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_nodes,
                 num_person,
                 num_gcn_scales,
                 adj_matrix,
                 dropout=0,
                 in_channels=1):
        """
        A simplified model that uses only MultiScale_GraphConv (MS_GCN) blocks.
        
        :param num_class: Number of output classes
        :param num_nodes: Number of skeleton joints (graph nodes)
        :param num_person: Max number of persons (bodies) in the sequence
        :param num_gcn_scales: Number of scales (K) to use in MS_GCN
        :param adj_matrix: The adjacency matrix for the skeleton graph
        :param dropout: Dropout probability (optional)
        :param in_channels: Number of input channels
        """
        super(Model, self).__init__()

        A = adj_matrix

        # Batch normalization on the input
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_nodes)
        print('Using dropout: {}'.format(dropout))

        # Example channel sizes
        c1 = 36 #36
        c2 = c1 * 2  # 72
        c3 = c2 * 2  # 144

        ## Three MS_GCN blocks
        # self.sgcn1 = MS_GCN(num_gcn_scales, in_channels, c1, A, 
        #                     disentangled_agg=True, dropout=dropout)
        # self.sgcn2 = MS_GCN(num_gcn_scales, c1, c2, A, 
        #                     disentangled_agg=True, dropout=dropout)
        # self.sgcn3 = MS_GCN(num_gcn_scales, c2, c3, A, 
        #                     disentangled_agg=True, dropout=dropout)

        # Three MGN blocks
        self.sgcn1 = GraphConv( in_channels, c1, A, dropout=dropout)
        self.sgcn2 = GraphConv( c1, c2, A, dropout=dropout)
        self.sgcn3 = GraphConv( c2, c3, A, dropout=dropout)

        # Final classification layer
        self.fc = nn.Linear(c3, num_class)

    def forward(self, x):
        """
        Forward pass.
        
        :param x: Input data of shape (N, C, T, V, M) 
                  where N=batch size, C=input channels, 
                  T=frames, V=number of joints, M=number of persons
        :return: Tuple of (logits, features).
        """
        N, C, T, V, M = x.size()

        # (N, C, T, V, M) --> (N, M*V*C, T)
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)

        # Reshape to (N*M, C, T, V) for MS_GCN
        x = x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()
        # x shape is now (N*M, in_channels, T, V) initially

        # Layer 1
        x = self.sgcn1(x)  # (N*M, c1, T, V)
        x = F.relu(x, inplace=False)

        # Layer 2
        x = self.sgcn2(x)  # (N*M, c2, T, V)
        x = F.relu(x, inplace=False)

        # Layer 3
        x = self.sgcn3(x)  # (N*M, c3, T, V)
        x = F.relu(x, inplace=False)

        # Global Average Pooling over time (T) and joints (V)
        x = x.mean(-1).mean(-1)  # (N*M, c3)

        # Reshape back to (N, M, c3) then average over the M persons
        x = x.view(N, M, -1).mean(1)  # (N, c3)

        # Final classification
        logits = self.fc(x)  # (N, num_class)

        return logits, x