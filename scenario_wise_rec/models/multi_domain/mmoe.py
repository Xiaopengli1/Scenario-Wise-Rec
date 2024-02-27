import torch
import torch.nn as nn
from ...basic.layers import MLP, EmbeddingLayer


class MMOE(nn.Module):
    """Multi-gate Mixture-of-Experts model.

    Args:
        features (list): the list of `Feature Class`, training by the expert and tower module.
        domain_num (int): number of domains.
        n_expert (int): the number of expert nets.
        expert_params (dict): the params of all the expert modules, keys include:`{"dims":list, "activation": str, "dropout":float}.
        tower_params_list (list): the list of tower params dict, the keys same as expert_params.
    """

    def __init__(self, features, domain_num, n_expert, expert_params, tower_params):
        super().__init__()
        self.features = features
        self.domain_num = domain_num
        self.n_expert = n_expert
        self.embedding = EmbeddingLayer(features)
        self.input_dims = sum([fea.embed_dim for fea in features])
        self.experts = nn.ModuleList(
            MLP(self.input_dims, output_layer=False, **expert_params) for i in range(self.n_expert))
        self.gates = nn.ModuleList(
            MLP(self.input_dims, output_layer=False, **{
                "dims": [self.n_expert],
                "activation": "softmax"
            }) for i in range(self.domain_num))  # n_gate = n_domains
        self.towers = nn.ModuleList(MLP(expert_params["dims"][-1], **tower_params) for i in range(self.domain_num))

    def forward(self, x):
        domain_id = x["domain_indicator"].clone().detach()

        embed_x = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size, input_dims]
        expert_outs = [expert(embed_x).unsqueeze(1) for expert in self.experts
                       ]  # expert_out[i]: [batch_size, 1, expert_dims[-1]]
        expert_outs = torch.cat(expert_outs, dim=1)  # [batch_size, n_expert, expert_dims[-1]]
        gate_outs = [gate(embed_x).unsqueeze(-1) for gate in self.gates]  # gate_out[i]: [batch_size, n_expert, 1]
        mask = []
        ys = []
        d = 0
        for gate_out, tower in zip(gate_outs, self.towers):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)
            d += 1
            expert_weight = torch.mul(gate_out, expert_outs)  # [batch_size, n_expert, expert_dims[-1]]
            expert_pooling = torch.sum(expert_weight, dim=1)  # [batch_size, expert_dims[-1]]
            tower_out = tower(expert_pooling)  # [batch_size, 1]
            y = torch.sigmoid(tower_out)
            ys.append(y)
        final = torch.zeros_like(ys[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), ys[d], final)
        return final.squeeze(1)
