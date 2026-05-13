import torch.nn as nn
#rho(t) -approximation

def build_mlp(
    layer_sizes=[1, 256, 128, 64, 32, 3],
    activation="tanh"
):
#1 -> input dimension (time)
#256 -> hidden layer 
#128 -> hidden layer (compression)
#... (more compression)
#3 ->output dimensions
#rho00, Re(rho01), Im(rho01)
    layers = []
#build hidden layers
    for i in range(len(layer_sizes) - 2):

        linear_layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
        nn.init.xavier_uniform_(linear_layer.weight)                #xavier initialization (helps to stabilize training)

        layers.append(linear_layer)
#add activation function
        if activation == "tanh":
            layers.append(nn.Tanh())
        elif activation == "relu":
            layers.append(nn.ReLU())
        else:
            layers.append(nn.LeakyReLU())
#final output layer 
    output_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])
    nn.init.xavier_uniform_(output_layer.weight)

    layers.append(output_layer)
#combining all layers into one model 
    model = nn.Sequential(*layers)

    return model