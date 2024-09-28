
import torch
from torch import nn
from torch.nn.parameter import Parameter
from ...basic.layers import EmbeddingLayer
from ...basic.activation import activation_layer


class HamurLarge(nn.Module):
    # 7 layers MLP with 2 adapter cells
    def __init__(self, features, domain_num, fcn_dims, hyper_dims, k ):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1
        self.fcn_dim = [self.input_dim] + fcn_dims
        self.domain_num = domain_num
        self.embedding = EmbeddingLayer(features)

        self.relu = activation_layer("relu")
        self.sig = activation_layer("sigmoid")

        self.layer_list = nn.ModuleList()

        # backbone network architecture
        for d in range(domain_num):
            domain_specific = nn.ModuleList()
            domain_specific.append(nn.Linear(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[2], self.fcn_dim[3]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[3]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[3], self.fcn_dim[4]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[4]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[4], self.fcn_dim[5]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[5]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[5], self.fcn_dim[6]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[6]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[6], self.fcn_dim[7]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[7]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[7], 1))

            self.layer_list.append(domain_specific)

        # instance representation matrix initiation
        self.k = k
        self.u = nn.ParameterList()
        self.v = nn.ParameterList()

        # u,v matrix initiation
        self.u.append(Parameter(torch.ones((self.fcn_dim[6], self.k)), requires_grad=True))
        self.u.append(Parameter(torch.ones((32, self.k)), requires_grad=True))
        self.u.append(Parameter(torch.ones((self.fcn_dim[7], self.k)), requires_grad=True))
        self.u.append(Parameter(torch.ones((32, self.k)), requires_grad=True))

        self.v.append(Parameter(torch.ones((self.k, 32)), requires_grad=True))
        self.v.append(Parameter(torch.ones((self.k, self.fcn_dim[6])), requires_grad=True))
        self.v.append(Parameter(torch.ones((self.k, 32)), requires_grad=True))
        self.v.append(Parameter(torch.ones((self.k, self.fcn_dim[7])), requires_grad=True))

        # hyper-network design
        hyper_dims += [self.k * self.k]
        input_dim = self.input_dim
        hyper_layers = []
        for i_dim in hyper_dims:
            hyper_layers.append(nn.Linear(input_dim, i_dim))
            hyper_layers.append(nn.BatchNorm1d(i_dim))
            hyper_layers.append(nn.ReLU())
            hyper_layers.append(nn.Dropout(p=0))
            input_dim = i_dim
        self.hyper_net = nn.Sequential(*hyper_layers)

        # adapter initiation
        self.b_list = nn.ParameterList() # bias
        self.b_list.append(Parameter(torch.zeros((32)), requires_grad=True))
        self.b_list.append(Parameter(torch.zeros((self.fcn_dim[6])), requires_grad=True))
        self.b_list.append(Parameter(torch.zeros((32)), requires_grad=True))
        self.b_list.append(Parameter(torch.zeros((self.fcn_dim[7])), requires_grad=True))

        self.gamma1 = nn.Parameter(torch.ones(self.fcn_dim[6])) # domain norm parameters
        self.bias1 = nn.Parameter(torch.zeros(self.fcn_dim[6]))
        self.gamma2 = nn.Parameter(torch.ones(self.fcn_dim[7]))
        self.bias2 = nn.Parameter(torch.zeros(self.fcn_dim[7]))
        self.eps = 1e-5

    def forward(self, x):

        domain_id = x["domain_indicator"].clone().detach()

        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        mask = []

        out_l = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = emb

            # hyper_network_out
            hyper_out_full = self.hyper_net(domain_input)  # B * (k * k)
            # Representation matrix
            hyper_out = hyper_out_full.reshape(-1, self.k, self.k)  # B * k * k

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear

            domain_input = model_list[1](domain_input)  # bn

            domain_input = model_list[2](domain_input)  # relu    B * m



            domain_input = model_list[3](domain_input)  # linear

            domain_input = model_list[4](domain_input)  # bn

            domain_input = model_list[5](domain_input)  # relu




            domain_input = model_list[6](domain_input)  # linear

            domain_input = model_list[7](domain_input)  # bn

            domain_input = model_list[8](domain_input)  # relu





            domain_input = model_list[9](domain_input)  # linear

            domain_input = model_list[10](domain_input)  # bn

            domain_input = model_list[11](domain_input)  # relu




            domain_input = model_list[12](domain_input)  # linear

            domain_input = model_list[13](domain_input)  # bn

            domain_input = model_list[14](domain_input)  # relu




            domain_input = model_list[15](domain_input)  # linear

            domain_input = model_list[16](domain_input)  # bn

            domain_input = model_list[17](domain_input)  # relu

            # First Adapter-cell

            # Adapter layer-1: Down projection
            w1 = torch.einsum('mi,bij,jn->bmn',self.u[0] , hyper_out,self.v[0])
            b1 = self.b_list[0]
            tmp_out = torch.einsum('bf,bfj->bj',domain_input,w1)
            tmp_out += b1

            # Adapter layer-2: non-linear
            tmp_out = self.sig(tmp_out)

            # Adapter layer-3: Up - projection
            w2 = torch.einsum('mi,bij,jn->bmn',self.u[1] , hyper_out,self.v[1])
            b2 = self.b_list[1]
            tmp_out = torch.einsum('bf,bfj->bj',tmp_out,w2)
            tmp_out += b2

            # Adpater layer-4: Domain norm
            mean = tmp_out.mean(dim=0)
            var = tmp_out.var(dim=0)
            x_norm = (tmp_out - mean) / torch.sqrt(var + self.eps)
            out = self.gamma1 * x_norm + self.bias1

            # Adapter: short-cut
            domain_input = out+domain_input


            domain_input = model_list[18](domain_input)  # linear

            domain_input = model_list[19](domain_input)  # bn

            domain_input = model_list[20](domain_input)  # relu

            # Second Adapter-cell

            # Adapter layer-1: Down projection
            w1 = torch.einsum('mi,bij,jn->bmn', self.u[2], hyper_out, self.v[2])
            b1 = self.b_list[2]
            tmp_out = torch.einsum('bf,bfj->bj', domain_input, w1)
            tmp_out += b1

            # Adapter layer-2: non-linear
            tmp_out = self.sig(tmp_out)

            # Adapter layer-3: Up - projection
            w2 = torch.einsum('mi,bij,jn->bmn', self.u[3], hyper_out, self.v[3])
            b2 = self.b_list[3]
            tmp_out = torch.einsum('bf,bfj->bj', tmp_out, w2)
            tmp_out += b2

            # Adpater layer-4: Domain norm
            mean = tmp_out.mean(dim=0)
            var = tmp_out.var(dim=0)
            x_norm = (tmp_out - mean) / torch.sqrt(var + self.eps)
            out = self.gamma2 * x_norm + self.bias2

            # Adapter: short-cut
            domain_input = out + domain_input


            domain_input = model_list[21](domain_input) # linear

            domain_input = self.sig(domain_input)       # relu

            out_l.append(domain_input)

        final = torch.zeros_like(out_l[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), out_l[d], final)

        return final.squeeze(1)


class HamurSmall(nn.Module):
    # 2 layers MLP with 1 adapter cells
    def __init__(self, features, domain_num, fcn_dims, hyper_dims, k ):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1
        self.fcn_dim = [self.input_dim] + fcn_dims
        self.domain_num = domain_num
        self.embedding = EmbeddingLayer(features)

        self.relu = activation_layer("relu")
        self.sig = activation_layer("sigmoid")

        self.layer_list = nn.ModuleList()
        for d in range(domain_num):
            domain_specific = nn.ModuleList()
            domain_specific.append(nn.Linear(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())
            domain_specific.append(nn.Linear(self.fcn_dim[2], 1))

            self.layer_list.append(domain_specific)

        # instance matrix initiation
        self.k = k
        self.u = nn.ParameterList()
        self.v = nn.ParameterList()

        # u,v initiation
        self.u.append(Parameter(torch.ones((self.fcn_dim[2], self.k)), requires_grad=True))
        self.u.append(Parameter(torch.ones((32, self.k)), requires_grad=True))

        self.v.append(Parameter(torch.ones((self.k, 32)), requires_grad=True))
        self.v.append(Parameter(torch.ones((self.k, self.fcn_dim[2])), requires_grad=True))

        # hypernwt work
        hyper_dims += [self.k * self.k]
        input_dim = self.input_dim
        hyper_layers = []
        for i_dim in hyper_dims:
            hyper_layers.append(nn.Linear(input_dim, i_dim))
            hyper_layers.append(nn.BatchNorm1d(i_dim))
            hyper_layers.append(nn.ReLU())
            hyper_layers.append(nn.Dropout(p=0))
            input_dim = i_dim
        self.hyper_net = nn.Sequential(*hyper_layers)

        # Adapter parameters
        self.b_list = nn.ParameterList()
        self.b_list.append(Parameter(torch.zeros((32)), requires_grad=True))
        self.b_list.append(Parameter(torch.zeros((self.fcn_dim[2])), requires_grad=True))

        self.gamma1 = nn.Parameter(torch.ones(self.fcn_dim[2]))
        self.bias1 = nn.Parameter(torch.zeros(self.fcn_dim[2]))
        self.eps = 1e-5

    def forward(self, x):

        domain_id = x["domain_indicator"].clone().detach()

        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        mask = []

        out_l = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = emb

            # hyper-network output
            hyper_out_full = self.hyper_net(domain_input)  # B * (k * k)
            # Representation matrix
            hyper_out = hyper_out_full.reshape(-1, self.k, self.k)  # B * k * k

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear

            domain_input = model_list[1](domain_input)  # bn

            domain_input = model_list[2](domain_input)  # relu



            domain_input = model_list[3](domain_input)  # linear

            domain_input = model_list[4](domain_input)  # bn

            domain_input = model_list[5](domain_input)  # relu

            # Adapter cell
            # Adapter layer-1: Down projection
            w1 = torch.einsum('mi,bij,jn->bmn', self.u[0], hyper_out, self.v[0])
            b1 = self.b_list[0]
            tmp_out = torch.einsum('bf,bfj->bj', domain_input, w1)
            tmp_out += b1

            # Adapter layer-2: Non-linear
            tmp_out = self.sig(tmp_out)

            # Adapter layer-3: Up projection
            w2 = torch.einsum('mi,bij,jn->bmn', self.u[1], hyper_out, self.v[1])
            b2 = self.b_list[1]
            tmp_out = torch.einsum('bf,bfj->bj', tmp_out, w2)
            tmp_out += b2

            # Adapter layer-4: Domain norm
            mean = tmp_out.mean(dim=0)
            var = tmp_out.var(dim=0)
            x_norm = (tmp_out - mean) / torch.sqrt(var + self.eps)
            out = self.gamma1 * x_norm + self.bias1

            # Adapter: Short-cut
            domain_input = out + domain_input

            domain_input = model_list[6](domain_input)
            domain_input = self.sig(domain_input)

            out_l.append(domain_input)

        final = torch.zeros_like(out_l[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), out_l[d], final)

        return final.squeeze(1)

class Mlp_2_Layer(nn.Module):
    # 2-layres Mlp model
    def __init__(self, features, domain_num, fcn_dims):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1
        self.fcn_dim = [self.input_dim] + fcn_dims
        self.domain_num = domain_num
        self.embedding = EmbeddingLayer(features)

        self.relu = activation_layer("relu")
        self.sig = activation_layer("sigmoid")

        self.layer_list = nn.ModuleList()
        for d in range(domain_num):
            domain_specific = nn.ModuleList()

            domain_specific.append(nn.Linear(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[2], 1))
            self.layer_list.append(domain_specific)

    def forward(self, x):

        domain_id = x["domain_indicator"].clone().detach()

        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        mask = []

        out = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = emb

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear
            domain_input = model_list[1](domain_input)  # bn
            domain_input = model_list[2](domain_input)  # relu

            domain_input = model_list[3](domain_input)  # linear
            domain_input = model_list[4](domain_input)  # bn
            domain_input = model_list[5](domain_input)  # relu

            domain_input = model_list[6](domain_input)
            domain_input = self.sig(domain_input)

            out.append(domain_input)

        final = torch.zeros_like(out[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), out[d], final)
        return final.squeeze(1)


class Mlp_7_Layer(nn.Module):
    # 7-layers Mlp model
    def __init__(self, features, domain_num, fcn_dims):
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.layer_num = len(fcn_dims) + 1
        self.fcn_dim = [self.input_dim] + fcn_dims
        self.domain_num = domain_num
        self.embedding = EmbeddingLayer(features)

        self.relu = activation_layer("relu")
        self.sig = activation_layer("sigmoid")

        self.layer_list = nn.ModuleList()
        for d in range(domain_num):
            domain_specific = nn.ModuleList()

            domain_specific.append(nn.Linear(self.fcn_dim[0], self.fcn_dim[1]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[1]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[1], self.fcn_dim[2]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[2]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[2], self.fcn_dim[3]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[3]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[3], self.fcn_dim[4]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[4]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[4], self.fcn_dim[5]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[5]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[5], self.fcn_dim[6]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[6]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[6], self.fcn_dim[7]))
            domain_specific.append(nn.BatchNorm1d(self.fcn_dim[7]))
            domain_specific.append(nn.ReLU())

            domain_specific.append(nn.Linear(self.fcn_dim[7], 1))
            self.layer_list.append(domain_specific)

    def forward(self, x):

        domain_id = x["domain_indicator"].clone().detach()

        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        mask = []

        out = []
        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            domain_input = emb

            model_list = self.layer_list[d]

            domain_input = model_list[0](domain_input)  # linear
            domain_input = model_list[1](domain_input)  # bn
            domain_input = model_list[2](domain_input)  # relu

            domain_input = model_list[3](domain_input)  # linear
            domain_input = model_list[4](domain_input)  # bn
            domain_input = model_list[5](domain_input)  # relu

            domain_input = model_list[6](domain_input)  # linear
            domain_input = model_list[7](domain_input)  # bn
            domain_input = model_list[8](domain_input)  # relu

            domain_input = model_list[9](domain_input)  # linear
            domain_input = model_list[10](domain_input)  # bn
            domain_input = model_list[11](domain_input)  # relu

            domain_input = model_list[12](domain_input)  # linear
            domain_input = model_list[13](domain_input)  # bn
            domain_input = model_list[14](domain_input)  # relu

            domain_input = model_list[15](domain_input)  # linear
            domain_input = model_list[16](domain_input)  # bn
            domain_input = model_list[17](domain_input)  # relu

            domain_input = model_list[18](domain_input)  # linear
            domain_input = model_list[19](domain_input)  # bn
            domain_input = model_list[20](domain_input)  # relu

            domain_input = model_list[21](domain_input)
            domain_input = self.sig(domain_input)

            out.append(domain_input)

        final = torch.zeros_like(out[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), out[d], final)
        return final.squeeze(1)