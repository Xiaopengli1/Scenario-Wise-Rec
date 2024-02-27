"""
Date: create on 01/05/2023
References: 
    paper: AdaptDHM: Adaptive Distribution Hierarchical Model for Multi-Domain CTR Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from ...basic.layers import EmbeddingLayer


class AdaptDHM(nn.Module):
    """adasparse

        Args:
            features (list[Feature Class]): features.
            fcn_dims (list): the params of the MLP/FCN module
            cluster_num (int): the number of the cluster
            beta (float): hyperparameters for a cluster center update
            device (string): the device used in the computer
        """
    def __init__(self, features, fcn_dims, cluster_num, beta, device):
        super(AdaptDHM, self).__init__()
        self.features = features
        self.beta = beta
        self.cluster_num = cluster_num
        self.layer_num = len(fcn_dims)+1
        self.dims = sum([fea.embed_dim for fea in features])
        self.fcn_dims = [self.dims] + fcn_dims + [1]
        self.embedding = EmbeddingLayer(self.features)
        self.center = F.normalize(torch.randn(cluster_num, self.dims), p=2, dim=1).to(device)

        # shared FCN
        self.domain_w = nn.ParameterList()
        self.domain_b = nn.ParameterList()
        w_list = nn.ParameterList()
        b_list = nn.ParameterList()
        for i in range(self.layer_num):
            w = Parameter(torch.empty((self.fcn_dims[i], self.fcn_dims[i + 1])).to(device), requires_grad=True)
            b = Parameter(torch.empty(self.fcn_dims[i + 1]).to(device), requires_grad=True)
            # nn.init.normal_(w, 0, 1.0)
            # nn.init.normal_(b, 0, 1e-7)
            nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
            nn.init.normal_(b, 0, 1e-7)
            w_list.append(w)
            b_list.append(b)
        self.domain_w.append(w_list)
        self.domain_b.append(b_list)

        # domain specific FCN
        for d in range(self.cluster_num):
            w_list = nn.ParameterList()
            b_list = nn.ParameterList()
            for i in range(self.layer_num):
                w = Parameter(torch.empty((self.fcn_dims[i], self.fcn_dims[i + 1])).to(device), requires_grad=True)
                b = Parameter(torch.empty(self.fcn_dims[i + 1]).to(device), requires_grad=True)
                # nn.init.normal_(w, 0, 1.0)
                # nn.init.normal_(b, 0, 1e-7)
                nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
                nn.init.normal_(b, 0, 1e-7)
                w_list.append(w)
                b_list.append(b)
            self.domain_w.append(w_list)
            self.domain_b.append(b_list)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def cal_router(self, input, times=3):
        # x:n*embedding, y:cluster_num*embedding
        x = input.clone().detach()
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)
            x = x.unsqueeze(1).repeat(1, self.center.shape[0], 1)
            for i in range(times):
                y = self.center
                y = y.unsqueeze(0)
                # n*cluster_num
                sij = torch.mul(x, y).sum(dim=2).squeeze()
                rij = F.softmax(sij, dim=1)
                if not self.training:
                    break
                # cluster_num*embedding
                cij = torch.mul(rij.unsqueeze(2), x).sum(dim=0).squeeze()
                self.center = F.normalize(self.beta * self.center + (1 - self.beta) * cij, p=2, dim=1)
            if self.training:
                y = self.center
                y = y.unsqueeze(0)
                # n*cluster_num
                sij = torch.mul(x, y).sum(dim=2).squeeze()
                rij = F.softmax(sij, dim=1)
            router_value = torch.argmax(rij, dim=1).unsqueeze(1)
            return router_value

    def forward(self, x):
        x = self.embedding(x, self.features, squeeze_dim=True)
        router_value = self.cal_router(x)

        for d in range(1, self.cluster_num+1):
            present = x
            for i in range(self.layer_num-1):
                present = torch.matmul(present, torch.mul(self.domain_w[0][i], self.domain_w[d][i]))
                present = self.relu(present)
            present = torch.matmul(present, torch.mul(self.domain_w[0][i+1], self.domain_w[d][i+1]))
            present = self.sigmoid(present)

            if d == 1:
                out = present
            else:
                out = torch.cat((out, present), 1)
        out = torch.gather(out, 1, router_value).squeeze()
        return out
