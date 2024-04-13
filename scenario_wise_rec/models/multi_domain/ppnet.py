import torch
import torch.nn as nn

from ...basic.activation import activation_layer
from ...basic.layers import MLP, EmbeddingLayer, GateNU


class PPTowerBlock(nn.Module):
    def __init__(self, input_dim, fcn_dims):
        super().__init__()
        self.input_dim = input_dim
        self.dims = [self.input_dim] + fcn_dims
        self.gate_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.mlp_layers.append(MLP(input_dim=self.dims[i], dims=[self.dims[i + 1]], output_layer=False))
            self.gate_layers.append(GateNU(self.dims[0], self.dims[i + 1]))
        self.final_layer = nn.Linear(self.dims[-1], 1)
        self.sig = activation_layer("sigmoid")

    def forward(self, agn_emb, gate_input_emb):
        hidden = gate_input_emb
        for i in range(len(self.mlp_layers)):
            gate_out = self.gate_layers[i](gate_input_emb)
            hidden = self.mlp_layers[i](hidden)
            hidden = hidden * gate_out
        hidden = self.final_layer(hidden)
        hidden = self.sig(hidden)
        return hidden


class PPNet(nn.Module):
    def __init__(self, id_features, agn_features, domain_num, fcn_dims):
        super().__init__()
        self.id_features = id_features
        self.agn_features = agn_features
        self.domain_num = domain_num
        self.id_embedding = EmbeddingLayer(id_features)
        self.agn_embedding = EmbeddingLayer(agn_features)
        self.id_dims = sum([fea.embed_dim for fea in id_features])
        self.agn_dims = sum([fea.embed_dim for fea in agn_features])
        self.input_dims = self.id_dims + self.agn_dims
        self.domain_tower = nn.ModuleList()
        for i in range(domain_num):
            self.domain_tower.append(PPTowerBlock(self.input_dims, fcn_dims))

    def forward(self, x):
        domain_id = x["domain_indicator"].clone().detach()
        mask = []
        out = []

        id_x = self.id_embedding(x, self.id_features, squeeze_dim=True)
        agn_x = self.agn_embedding(x, self.agn_features, squeeze_dim=True)
        gate_input = torch.cat((id_x, agn_x.detach()), dim=1)

        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)
            domain_input_agn_emb = agn_x
            domain_input_gate_emb = gate_input
            domain_out = self.domain_tower[d](domain_input_agn_emb, domain_input_gate_emb)
            out.append(domain_out)

        final = torch.zeros_like(out[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), out[d], final)
        return final.squeeze(1)
