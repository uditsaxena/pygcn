import torch.nn as nn
import torch.nn.functional as F

from pygcn.layers import GraphConvolution


# Kipf et. al.
class Vanilla_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Vanilla_GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x)


# GCN with two layers and one fully connected layer
class GCN2_FC1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN2_FC1, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, int(nhid / 2) )
        self.linear = nn.Linear(int(nhid / 2), nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.dropout(F.relu(self.gc2(x, adj)), self.dropout, training=self.training)
        x = self.linear(x)
        return F.log_softmax(x)
