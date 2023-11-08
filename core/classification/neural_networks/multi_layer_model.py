from torch import nn

HIDDEN_LAYER_SIZE_FIRST = 15
HIDDEN_LAYER_SIZE_SECOND = 4


class MultiLayerModel(nn.Module):
    def __init__(self, input_size):
        super(MultiLayerModel, self).__init__()

        self.fc1 = nn.Linear(input_size, HIDDEN_LAYER_SIZE_FIRST)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE_FIRST, HIDDEN_LAYER_SIZE_SECOND)
        self.activation2 = nn.Sigmoid()
        self.fc3 = nn.Linear(HIDDEN_LAYER_SIZE_SECOND, 1)
        self.standardize = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation2(out)
        out = self.fc3(out)
        out = self.standardize(out)
        return out
