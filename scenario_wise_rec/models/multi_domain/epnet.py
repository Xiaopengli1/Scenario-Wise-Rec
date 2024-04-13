import torch
from torch import nn
from ...basic.layers import MLP, EmbeddingLayer, GateNU


class EPNet(nn.Module):
    """
    Star

    """
    def __init__(self, sce_features, agn_features, fcn_dims):
        super().__init__()
        self.sce_features = sce_features
        self.agn_features = agn_features
        self.sce_embedding = EmbeddingLayer(sce_features)
        self.agn_embedding = EmbeddingLayer(agn_features)
        self.sce_dims = sum([fea.embed_dim for fea in sce_features])
        self.agn_dims = sum([fea.embed_dim for fea in agn_features])
        self.dims = self.sce_dims + self.agn_dims
        self.gatenu = GateNU(self.dims, self.agn_dims)
        self.mlp = MLP(self.agn_dims, fcn_dims)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        sce_x = self.sce_embedding(x, self.sce_features, squeeze_dim=True)
        agn_x = self.agn_embedding(x, self.agn_features, squeeze_dim=True)
        gate_input = torch.cat((sce_x, agn_x.detach()), dim=1)
        gate_output = self.gatenu(gate_input)
        agn_x = agn_x * gate_output
        output = self.mlp(agn_x)
        output = self.sigmoid(output).squeeze()
        return output