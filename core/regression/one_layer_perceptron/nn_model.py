from torch import nn


class NNModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NNModel, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out
