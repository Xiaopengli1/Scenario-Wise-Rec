import torch
import torch.nn as nn
from ...basic.layers import MLP, EmbeddingLayer


class M2M(nn.Module):
    """
    M2M

    """
    def __init__(self, features, domain_feature, domain_num, num_experts = 4, expert_output_size = 16,
                 transformer_dims={"num_encoder_layers": 2, "num_decoder_layers": 2, "dim_feedforward": 16}):
        super().__init__()
        self.embedding = EmbeddingLayer(features)
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in self.features])
        self.domain_feature = domain_feature
        self.num_experts = num_experts
        self.expert_output_size = expert_output_size
        self.domain_num = domain_num
        self.transformer = nn.Transformer(d_model=self.input_dim, nhead=4, **transformer_dims)
        self.experts = nn.ModuleList([
            MLP(self.input_dim, output_layer = False, dims = [self.expert_output_size], activation = "leakyrelu")
            for _ in range(self.num_experts)])
        self.task_mlp = MLP(self.domain_feature[0].embed_dim, output_layer= False,
                            dims = [self.expert_output_size], activation = "leakyrelu")
        self.scenario_mlp = MLP(self.domain_feature[0].embed_dim, output_layer= False,
                                dims = [self.expert_output_size], activation = "leakyrelu")
        self.vw_mlp = MLP(self.expert_output_size, output_layer = False,
                             dims = [4 * self.expert_output_size * self.expert_output_size], activation = "leakyrelu")
        self.vb_mlp = MLP(self.expert_output_size, output_layer=False,
                          dims=[2 * self.expert_output_size], activation="leakyrelu")
        self.v = nn.Parameter(torch.ones(2 * self.expert_output_size, 1))
        self.meta_tower_w_mlp = MLP(self.expert_output_size, output_layer = False,
                                  dims = [self.expert_output_size * self.expert_output_size], activation = "leakyrelu")
        self.meta_tower_b_mlp = MLP(self.expert_output_size, output_layer=False,
                                    dims=[self.expert_output_size], activation="leakyrelu")
        self.relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
        self.output_mlp = MLP(self.expert_output_size, dims = [64, 32])

    def forward(self, x):
        domain_id_emb = self.embedding(x, self.domain_feature, squeeze_dim=True)
        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]
        b_size = emb.shape[0]

        # transformer
        transformer_out = self.transformer(emb, emb)
        scenario_out = self.scenario_mlp(domain_id_emb)  # b, expert_output
        task_output = self.task_mlp(domain_id_emb)  # b, expert_output
        expert_output = [expert(transformer_out) for expert in self.experts]  # 8; b, expert_output
        expert_output = torch.stack(expert_output, dim=1)  # b, num_experts, expert_output

        # meta_attention_module
        meta_input = torch.cat([expert_output, task_output.unsqueeze(1).repeat(1, self.num_experts, 1)], dim=2).reshape(
            b_size, self.num_experts, -1)
        meta_weight = self.vw_mlp(scenario_out).reshape(b_size, 2 * self.expert_output_size, 2 * self.expert_output_size)
        meta_bias = self.vb_mlp(scenario_out).unsqueeze(1)
        meta_output = self.relu(torch.matmul(meta_input, meta_weight).squeeze(2) + meta_bias)
        meta_output = torch.matmul(meta_output, self.v.unsqueeze(0)).squeeze(2)  # b, num_experts
        alpha = torch.softmax(meta_output, dim=1).unsqueeze(2)  # b, num_experts, 1
        rt = torch.sum(alpha * expert_output, dim=1)

        # meta_tower_module
        tower_weight = self.meta_tower_w_mlp(scenario_out).reshape(b_size, self.expert_output_size, self.expert_output_size)
        tower_bias = self.meta_tower_b_mlp(scenario_out).reshape(b_size, self.expert_output_size)
        output = self.relu(torch.matmul(rt.unsqueeze(1), tower_weight).squeeze(1) + tower_bias + rt)
        output = self.sigmoid(self.output_mlp(output))  # b, 1
        return output.squeeze(1)
