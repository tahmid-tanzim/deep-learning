import torch
import numpy as np


class MyDenseLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyDenseLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim, requires_grad=True))
        self.biases = nn.Parameter(torch.randn(1, output_dim, requires_grad=True))

    def forward(self, inputs):
        z = torch.matmul(inputs, self.weights) + self.biases
        output = torch.sigmoid(z)
        return output