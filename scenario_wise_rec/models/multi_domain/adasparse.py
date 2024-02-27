"""
Date: create on 01/05/2023
References: 
    paper: AdaSparse: Learning Adaptively Sparse Structures for Multi-Domain Click-Through Rate Prediction
    Authors: Xuanhua Yang, xuanhua.yxh@alibaba-inc.com
"""

import torch
import torch.nn as nn
from ...basic.layers import EmbeddingLayer, Pruner, activation_layer


class AdaSparse(nn.Module):
    """adasparse

        Args:
            sce_features (list[Feature Class]): scenario features.
            agn_features (list[Feature Class]): agnostic features.
            mlp_params (dict): the params of the MLP module, keys include:`{"dims":list, "activation": str,
            "dropout": float, "output_layer":bool`}
            form: form of Weighting Factors mentioned in the picture. one of the 'Binary', 'Scaling' and 'Fusion'
            epsilon (float), beta (float), alpha (float): hyperparameters for Weighting Factors
            delta_alpha (float): hyperparameter to control the update value of alpha in each update process.
        """
    def __init__(self, sce_features, agn_features, mlp_params, form='Fusion', epsilon=1e-2,
                 beta=2.0, alpha=1.0, delta_alpha=1e-4):
        super(AdaSparse, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.delta_alpha = torch.tensor(delta_alpha)

        self.sce_features = sce_features
        self.agn_features = agn_features

        self.sce_dims = sum([fea.embed_dim for fea in sce_features])
        self.agn_dims = sum([fea.embed_dim for fea in agn_features])
        self.dims = self.sce_dims+self.agn_dims

        self.sce_embedding = EmbeddingLayer(sce_features)
        self.agn_embedding = EmbeddingLayer(agn_features)
        mlp_dims = mlp_params["dims"]
        if mlp_dims is None:
            mlp_dims = []
        input_dim = self.sce_dims+self.agn_dims
        self.mlp_layers = nn.ModuleList([])
        self.pruner_layers = nn.ModuleList([])
        self.pruner_layers.append(Pruner(self.sce_dims, self.agn_dims, form=form, epsilon=epsilon, beta=beta))
        for i_dim in mlp_dims:
            layers = list()
            layers.append(nn.Linear(input_dim, i_dim))
            layers.append(nn.BatchNorm1d(i_dim))
            layers.append(activation_layer(mlp_params["activation"]))
            layers.append(nn.Dropout(p=mlp_params["dropout"]))
            input_dim = i_dim

            self.pruner_layers.append(Pruner(self.sce_dims, input_dim, form=form, epsilon=epsilon, beta=beta))
            self.mlp_layers.append(nn.Sequential(*layers))
        self.mlp_layers.append(nn.Linear(input_dim, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        with torch.no_grad():
            self.alpha += self.delta_alpha
        sce_x = self.sce_embedding(x, self.sce_features, squeeze_dim=True)
        agn_x = self.agn_embedding(x, self.agn_features, squeeze_dim=True)
        agn_x = self.pruner_layers[0](sce_x, agn_x, self.alpha) * agn_x
        x = torch.cat((sce_x, agn_x), 1)
        for i in range(len(self.mlp_layers)-1):
            x = self.mlp_layers[i](x)
            x = self.pruner_layers[i+1](sce_x, x, self.alpha)*x
        x = self.mlp_layers[-1](x).squeeze()
        return self.sigmoid(x)
