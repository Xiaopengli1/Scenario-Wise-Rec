
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
from ...basic.layers import MLP, EmbeddingLayer
from ...basic.activation import activation_layer


class Base(nn.Module):
    """
    base

    """
    def __init__(self, features, num_domains, **kwargs):
        """

        features: input features
        num_domains: number of domains

        """
        super().__init__()
        self.features = features
        self.input_dim = sum([fea.embed_dim for fea in features])
        self.num_domains = num_domains
        self.embedding = EmbeddingLayer(features)
        ###############################################
        # Part1: Initialize your scenario-shared model.
        #
        # example:
        # self.main_model = MLP(self.input_dim)
        ###############################################

        # Part2: Initialize your domain specific model.
        for d in range(self.num_domains):
            ###############################################
            # Part2: Initialize your scenario-specific model.
            #
            # example:
            # self.specific_model = MLP(self.input_dim)
            ###############################################
            pass

        # Parameters Init
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x):
        domain_id = x["domain_indicator"].clone().detach()
        emb = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size,total_dims]

        # Backbone model process

        # domain mask
        mask = []
        # out
        out = []

        for d in range(self.num_domains):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)
            domain_input = emb

            # domain specific input process


            out.append(domain_input)

        # Out put formatting
        final = torch.zeros_like(out[0])
        for d in range(self.num_domains):
            final = torch.where(mask[d].unsqueeze(1), out[d], final)

        # squeeze the output into 2-d tensor
        return final.squeeze(1)