import torch.nn as nn


def build_mlp(
    layer_sizes=[1, 256, 128, 64, 32, 3],
    activation="tanh"
):

    layers = []

    for i in range(len(layer_sizes) - 2):

        linear_layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
        nn.init.xavier_uniform_(linear_layer.weight)

        layers.append(linear_layer)

        if activation == "tanh":
            layers.append(nn.Tanh())
        elif activation == "relu":
            layers.append(nn.ReLU())
        else:
            layers.append(nn.LeakyReLU())

    output_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])
    nn.init.xavier_uniform_(output_layer.weight)

    layers.append(output_layer)

    model = nn.Sequential(*layers)

    return model