import torch
import torch.nn as nn
from ...basic.layers import MLP, EmbeddingLayer


class SharedBottom(nn.Module):
    """Shared Bottom multi-task model.

    Args:
        features (list): the list of `Feature Class`, training by the bottom and tower module.
        domain_num (int): the number of domains.
        bottom_params (dict): the params of the last MLP module, keys include:`{"dims": list, "activation": str, "dropout":float}, keep `{"output_layer":False}`.
        tower_params (list): the list of tower params dict, the keys same as bottom_params.
    """

    def __init__(self, features, domain_num, bottom_params, tower_params):
        super().__init__()
        self.features = features
        self.embedding = EmbeddingLayer(features)
        self.bottom_dims = sum([fea.embed_dim for fea in features])
        self.domain_num = domain_num

        self.bottom_mlp = MLP(self.bottom_dims, **{**bottom_params, **{"output_layer": False}})
        self.towers = nn.ModuleList(
            MLP(bottom_params["dims"][-1], **tower_params) for i in range(self.domain_num))
        # self.predict_layers = nn.ModuleList(PredictionLayer(task_type) for i in range(self.domain_num))

    def forward(self, x):
        domain_id = x["domain_indicator"].clone().detach()
        input_bottom = self.embedding(x, self.features, squeeze_dim=True)
        x = self.bottom_mlp(input_bottom)

        mask = []

        ys = []

        for d in range(self.domain_num):
            domain_mask = (domain_id == d)
            mask.append(domain_mask)

            tower = self.towers[d]

            tower_out = tower(x)
            y = torch.sigmoid(tower_out)  # regression->keep, binary classification->sigmoid
            ys.append(y)

        final = torch.zeros_like(ys[0])
        for d in range(self.domain_num):
            final = torch.where(mask[d].unsqueeze(1), ys[d], final)
        return final.squeeze(1)
