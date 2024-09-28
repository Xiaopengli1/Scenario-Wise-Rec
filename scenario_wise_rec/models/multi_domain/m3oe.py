import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ...basic.layers import EmbeddingLayer


class Weights(torch.nn.Module):

    def __init__(self, weight_shape, tau, tau_step, initial_deep, softmax_type=2):
        super().__init__()
        assert isinstance(weight_shape, (int, list))
        norm = weight_shape[-1] if isinstance(weight_shape, list) else weight_shape

        if initial_deep is None:
            initial_deep = np.ones(weight_shape, dtype=np.float32) / norm
            print(f'initial_deep: {initial_deep}')
        else:
            initial_deep = np.ones(weight_shape, dtype=np.float32) * initial_deep
        self.deep_weights = torch.nn.Parameter(torch.from_numpy(initial_deep), requires_grad=True)
        self.softmax_type = softmax_type
        self.tau = tau
        self.tau_step = tau_step

    def forward(self, ):
        if self.tau > 0.01:
            self.tau -= self.tau_step

        if self.softmax_type == 0:
            assert 0
            return F.softmax(self.deep_weights, dim=-1)
        elif self.softmax_type == 1:
            assert 0
            output = F.softmax(self.deep_weights / self.tau, dim=-1)
            return output
        elif self.softmax_type == 2:
            assert 0
            output = F.gumbel_softmax(self.deep_weights, tau=self.tau, hard=False, dim=-1)
            return output
        elif self.softmax_type == 3:
            output = F.sigmoid(self.deep_weights)
            return output


class Mlp_N(nn.Module):

    """
    fcn_dim: list of dimensions of mlp layers, e.g., [1024, 512, 512, 256, 256, 64]
    return [linear, bn1d, relu]*n
    """

    def __init__(self, fcn_dim):
        super().__init__()
        self.fcn_dim = fcn_dim
        self.n = len(fcn_dim)

        self.domain_specific = nn.ModuleList()
        for (i) in range(self.n - 1):
            self.domain_specific.append(nn.Linear(self.fcn_dim[i], self.fcn_dim[i + 1]))
            self.domain_specific.append(nn.LayerNorm(self.fcn_dim[i + 1]))
            self.domain_specific.append(nn.ReLU())
            # self.domain_specific.append(nn.Linear(self.fcn_dim[-1], 1))

    def forward(self, x):
        output = x
        for f in self.domain_specific:
            output = f(output)
        return output


class M3oE(nn.Module):
    # add domain-specific expert based on MTMD
    # gate(shared expert) + task-specific + domain-specifics
    def __init__(self, features, domain_num, fcn_dims, expert_num, exp_d, exp_t, bal_d, bal_t, tau=1, task_num=1,
                 tau_step=0.00005, softmax_type=3, device = "cpu"):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1
        self.fcn_dim = [self.input_dim] + fcn_dims
        self.domain_num = domain_num
        self.task_num = task_num
        self.expert_num = expert_num
        self.embedding = EmbeddingLayer(features)
        self.device = device
        self._weight_exp_d = Weights(1, tau, tau_step, exp_d, softmax_type)
        self._weight_exp_t = Weights(1, tau, tau_step, exp_t, softmax_type)
        self._weight_bal_d = Weights(1, tau, tau_step, bal_d, softmax_type)
        self._weight_bal_t = Weights(1, tau, tau_step, bal_t, softmax_type)

        assert len(
            self.fcn_dim) > 3, f'too few layers assigned, must larger than 3. Star owns 3 layers, mmoe owns the rest.'
        self.star_dim = self.fcn_dim[:3]
        self.fcn_dim = self.fcn_dim[3:]
        self.skip_conn = Mlp_N([self.star_dim[0], self.star_dim[2]])
        self.shared_weight = nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1]))
        self.shared_bias = nn.Parameter(torch.zeros(self.star_dim[1]))

        # multi-domain fusion: star
        self.slot_weight = nn.ParameterList(
            [nn.Parameter(torch.empty(self.star_dim[0], self.star_dim[1])) for i in range(self.domain_num)])
        self.slot_bias = nn.ParameterList([nn.Parameter(torch.zeros(self.star_dim[1])) for i in range(self.domain_num)])

        self.star_mlp = Mlp_N([self.star_dim[1], self.star_dim[2]])

        # for m in ([self.shared_weight]+ self.slot_weight):
        torch.nn.init.xavier_uniform_(self.shared_weight.data)
        for m in (self.slot_weight):
            torch.nn.init.xavier_uniform_(m.data)

        # multi-task balance: mmoe
        self.expert = nn.ModuleList()
        for d in range(expert_num):
            self.expert.append(Mlp_N(self.fcn_dim))

        self.domain_expert = nn.ModuleList()
        for d in range(domain_num):
            self.domain_expert.append(Mlp_N(self.fcn_dim))

        self.gate = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Linear(self.fcn_dim[0], expert_num), torch.nn.Softmax(dim=1)) for i in
             range(domain_num)])

        self.tower = nn.ModuleList()
        for d in range(domain_num):
            domain_specific = nn.Sequential(
                nn.Linear(self.fcn_dim[-1], self.fcn_dim[-1]),
                # nn.BatchNorm1d(self.fcn_dim[-1]),
                nn.LayerNorm(self.fcn_dim[-1]),
                nn.ReLU(),
                nn.Linear(self.fcn_dim[-1], 1)
            )
            self.tower.append(domain_specific)

    def forward(self, x, test_flag=False):
        # _device = x.device
        domain_id = x["domain_indicator"].clone().detach()
        mask = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

        input_emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        skip = self.skip_conn(input_emb)
        emb = torch.zeros((input_emb.shape[0], self.star_dim[1])).to(self.device)
        for i, (_weight, _bias) in enumerate(zip(self.slot_weight, self.slot_bias)):
            _output = torch.matmul(input_emb, torch.multiply(_weight, self.shared_weight)) + _bias + self.shared_bias
            emb = torch.where(mask[i].unsqueeze(1).to(self.device), _output, emb)
        emb = self.star_mlp(emb) + skip

        gate_value = [self.gate[i](emb.detach()).unsqueeze(1) for i in
                      range(self.domain_num)]  # [domain_num, batch_size, 1, expert_num]

        out = []  # batch_size, expert_num, embedding_size
        for i in range(self.expert_num):
            domain_input = self.expert[i](emb)
            out.append(domain_input)
        domain_exp_out = []
        for i in range(self.domain_num):
            domain_input = self.domain_expert[i](emb)
            domain_exp_out.append(domain_input)
        # task_exp_out = []
        # for i in range(self.task_num):
        #     task_input = self.task_expert[i](emb)
        #     task_exp_out.append(task_input)
        fea = torch.cat([out[i].unsqueeze(1) for i in range(self.expert_num)], dim=1)  # batch_size, expert_num, 1
        domain_fea = torch.cat([domain_exp_out[i].unsqueeze(1) for i in range(self.domain_num)],
                               dim=1)  # batch_size, domain_num, 1
        # task_fea = torch.cat([task_exp_out[i].unsqueeze(1) for i in range(self.task_num)],
        #                      dim=1)  # batch_size, task_num, 1

        weighted_domain_fea = []
        for i in range(self.domain_num):
            temp_ = self._weight_bal_d() * domain_fea[:, i, :]
            for j in range(self.domain_num):
                if j != i:
                    temp_ += (1 - self._weight_bal_d()) / (self.domain_num - 1) * domain_fea[:, j, :]
            weighted_domain_fea.append(temp_)
        weighted_task_fea = []
        # for i in range(self.task_num):
        #     temp_ = self._weight_bal_t() * task_fea[:, i, :]
        #     for j in range(self.task_num):
        #         if j != i:
        #             temp_ += (1 - self._weight_bal_t()) / (self.task_num - 1) * task_fea[:, j, :]
        #     weighted_task_fea.append(temp_)

        fused_fea = [torch.bmm(gate_value[i], fea).squeeze(1) +
                     self._weight_exp_d() * weighted_domain_fea[i % self.domain_num]
                     for i in range(self.domain_num)]

        output = [torch.sigmoid(self.tower[i](fused_fea[i]).squeeze(1)) for i in
                   range(self.domain_num)]

        result = torch.zeros_like(output[0])
        for d in range(self.domain_num):
            result = torch.where(mask[d], output[d], result)
        return result
