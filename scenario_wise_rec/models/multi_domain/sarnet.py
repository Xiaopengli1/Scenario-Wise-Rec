import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
from ...basic.layers import MLP, EmbeddingLayer


class DebiasExpertNet(torch.nn.Module):
    """
    DebiasExpertNet

    """
    def __init__(self, input_size):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.linear = nn.Linear(input_size, 16)

    def forward(self, x):
        x = self.bn(x)
        x = self.linear(x)
        return x


class Sarnet(nn.Module):
    """
    Sarnet

    """
    def __init__(self, features, domain_num, domain_shared_expert_num=8, domain_specific_expert_num = 2):
        super().__init__()
        self.features = features
        self.embedding = EmbeddingLayer(features)
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.domain_num = domain_num
        self.domain_shared_expert_num = domain_shared_expert_num
        self.domain_specific_expert_num = domain_specific_expert_num
        self.domain_weight = nn.ParameterList()
        self.domain_bias = nn.ParameterList()
        for d in range(self.domain_num):
            self.domain_weight.append(Parameter(torch.empty(1, self.input_dim), requires_grad=True))
            self.domain_bias.append(Parameter(torch.empty(self.input_dim), requires_grad=True))
        self.shared_expert = nn.ModuleList([DebiasExpertNet(self.input_dim)
                                            for _ in range(self.domain_shared_expert_num)])
        self.domain_specific_expert = nn.ModuleList()
        for d in range(domain_num):
            domain_d_expert = nn.ModuleList()
            for i in range(self.domain_specific_expert_num):
                domain_d_expert.append(DebiasExpertNet(self.input_dim))
            self.domain_specific_expert.append(domain_d_expert)

        self.gate_net = torch.nn.Linear(self.input_dim, self.domain_shared_expert_num+self.domain_specific_expert_num)
        self.final_mlp = MLP(input_dim=16, output_layer=True, dims=[32, 32])
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.domain_weight)):
            self.domain_weight[i] = init.xavier_uniform_(self.domain_weight[i])
            self.domain_bias[i] = init.uniform_(self.domain_bias[i], 0, 1)

    def forward(self, x):
        domain_id = x["domain_indicator"].clone().detach()
        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]
        mask = []
        share = []
        specific = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)
            domain_input = emb
            domain_input = torch.mul(domain_input, self.domain_weight[d])
            domain_input = domain_input+self.domain_bias[d]
            share.append(domain_input)
            domain_specific_model_d = self.domain_specific_expert[d]
            specific_expert_out = torch.stack([domain_specific_model_d[i](domain_input)
                                               for i in range(self.domain_specific_expert_num)], dim=1)
            specific.append(specific_expert_out)

        shared_emb = torch.zeros_like(share[0])
        for d in range(self.domain_num):
            shared_emb = torch.where(mask[d].unsqueeze(1), share[d], shared_emb)
        shared_expert_out = torch.stack([self.shared_expert[i](shared_emb)
                                         for i in range(self.domain_shared_expert_num)], dim=1)

        specific_emb = torch.zeros_like(specific[0])
        for d in range(self.domain_num):
            specific_emb = torch.where(mask[d].unsqueeze(1).unsqueeze(2).expand(-1, specific_emb.shape[1], specific_emb.shape[2]), specific[d], specific_emb)

        expert_out = torch.concat([shared_expert_out, specific_emb], dim=1)
        gate_value = torch.softmax(self.gate_net(shared_emb), dim=-1)
        expert_out = torch.mul(expert_out, gate_value.unsqueeze(2))
        expert_out = torch.sum(expert_out, dim=1)
        final = torch.sigmoid(self.final_mlp(expert_out))
        return final.squeeze(1)
