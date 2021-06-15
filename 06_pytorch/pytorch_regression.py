import torch
import torch.nn as nn

class Regression(torch.nn.Module):

    def __init__(self, K):
        super(Regression, self).__init__()

        self.W1 = nn.Linear(1, K, bias=True)
        self.activation = nn.Sigmoid()
        self.w2 = nn.Linear(K, 1)

    def forward(self, x):
        a = self.W1(x)
        h = self.activation(a)
        z = self.w2(h)

        return z
