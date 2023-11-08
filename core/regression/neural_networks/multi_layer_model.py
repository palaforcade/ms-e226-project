from torch import nn

HIDDEN_LAYER_SIZE_FIRST = 30
HIDDEN_LAYER_SIZE_SECOND = 10


class MultiLayerModel(nn.Module):
    def __init__(self, input_size):
        super(MultiLayerModel, self).__init__()

        self.fc1 = nn.Linear(input_size, HIDDEN_LAYER_SIZE_FIRST)
        self.activation = nn.LeakyReLU()
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE_FIRST, HIDDEN_LAYER_SIZE_SECOND)
        self.activation2 = nn.Tanh()
        self.fc3 = nn.Linear(HIDDEN_LAYER_SIZE_SECOND, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation2(out)
        out = self.fc3(out)
        return out
