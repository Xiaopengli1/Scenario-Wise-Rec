
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
from ...basic.layers import MLP, EmbeddingLayer
from ...basic.activation import activation_layer


class Star(nn.Module):
    """
    Star

    """
    def __init__(self, features, num_domains, fcn_dims, aux_dims):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1  # 生成的主网络层数+一层最后输出
        self.fcn_dim = [self.input_dim] + fcn_dims + [1]  # 把这个input_dim加进来，并把最后的一写出来，方便生成参数
        self.num_domains = num_domains
        self.aux_dims = aux_dims
        self.embedding = EmbeddingLayer(features)

        # domain Norm
        self.dn_share_gamma = Parameter(torch.ones(self.input_dim))
        self.dn_share_bias = Parameter(torch.zeros(self.input_dim))
        self.eps = 1e-6

        # auxiliary network
        self.auxnet = MLP(self.input_dim, dims=self.aux_dims)
        self.relu = activation_layer("relu")
        self.sig = activation_layer("sigmoid")

        # shared FCN
        self.share_parm_w = nn.ParameterList()
        self.share_parm_b = nn.ParameterList()
        for i in range(self.layer_num):
            self.share_parm_w.append(Parameter(torch.empty((self.fcn_dim[i], self.fcn_dim[i + 1])), requires_grad=True))
            self.share_parm_b.append(Parameter(torch.empty(self.fcn_dim[i + 1]), requires_grad=True))

        # domain specific FCN

        self.domain_specific_dn_gamma = nn.ParameterList()
        self.domain_specific_dn_bias = nn.ParameterList()
        self.domain_specific_w = nn.ParameterList()
        self.domain_specific_b = nn.ParameterList()
        self.domain_specific_bn = nn.ModuleList()

        for d in range(self.num_domains):
            self.domain_specific_dn_gamma.append(Parameter(torch.ones(self.input_dim)))
            self.domain_specific_dn_bias.append(Parameter(torch.zeros(self.input_dim)))

            lay_weight = nn.ParameterList()
            lay_bias = nn.ParameterList()
            lay_domain_specific_bn = nn.ModuleList()
            for i in range(self.layer_num):
                lay_weight.append(Parameter(torch.empty((self.fcn_dim[i], self.fcn_dim[i + 1])), requires_grad=True))
                lay_bias.append(Parameter(torch.empty(self.fcn_dim[i + 1]), requires_grad=True))
                lay_domain_specific_bn.append(nn.BatchNorm1d(self.fcn_dim[i + 1]))
            self.domain_specific_w.append(lay_weight)
            self.domain_specific_b.append(lay_bias)
            self.domain_specific_bn.append(lay_domain_specific_bn)

        # Parameters Init
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.share_parm_w)):
            self.share_parm_w[i] = init.kaiming_uniform_(self.share_parm_w[i])
            self.share_parm_b[i] = init.uniform_(self.share_parm_b[i], 0, 1)

        for d in range(len(self.domain_specific_w)):
            for i in range(len(self.domain_specific_w[d])):
                self.domain_specific_w[d][i] = init.kaiming_uniform_(self.domain_specific_w[d][i])
                self.domain_specific_b[d][i] = init.uniform_(self.domain_specific_b[d][i], 0, 1)

    def forward(self, x):
        domain_id = x["domain_indicator"].clone().detach()

        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]
        # mask 存储每个batch中的domain id
        mask = []
        # out 存储
        out = []

        aux_out = self.auxnet(emb)
        for d in range(self.num_domains):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = emb

            # dn
            mean = domain_input.mean(dim=0)
            # var = domain_input.var(dim=0)
            var = ((domain_input - mean) ** 2).mean(dim=0)
            domain_input = (domain_input - mean) / torch.sqrt(var + self.eps)  # Normalize input
            domain_input = ((self.dn_share_gamma * self.domain_specific_dn_gamma[d]) * domain_input
                            + self.dn_share_bias + self.domain_specific_dn_bias[d])  # Scale and shift

            for layer_i in range(self.layer_num):
                w_tmp = self.share_parm_w[layer_i] * self.domain_specific_w[d][layer_i]

                b_tmp = self.share_parm_b[layer_i] + self.domain_specific_b[d][layer_i]

                domain_input = domain_input @ w_tmp + b_tmp

                domain_input = self.domain_specific_bn[d][layer_i](domain_input)
                domain_input = self.relu(domain_input)

            out.append(domain_input)

        final = torch.zeros_like(out[0])
        for d in range(self.num_domains):
            final = torch.where(mask[d].unsqueeze(1), out[d], final)
        final = self.sig(final + aux_out)
        return final.squeeze(1)
