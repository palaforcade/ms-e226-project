from torch import nn

HIDDEN_LAYER_SIZE = 28


class OneLayerModel(nn.Module):
    def __init__(self, input_size):
        super(OneLayerModel, self).__init__()

        self.fc1 = nn.Linear(input_size, HIDDEN_LAYER_SIZE)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, 1)
        self.standardize = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.standardize(out)
        return out
