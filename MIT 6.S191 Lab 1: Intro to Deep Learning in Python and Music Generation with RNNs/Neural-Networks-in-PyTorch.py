import torch

# import mitdeeplearning as mdl

import numpy as np
import matplotlib.pyplot as plt


### Defining a dense layer ###

# num_inputs: number of input nodes
# num_outputs: number of output nodes
# x: input to the layer

class OurDenseLayer(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(OurDenseLayer, self).__init__()
        # Define and initialize parameters: a weight matrix W and bias b
        # Note that the parameter initialize is random!
        self.W = torch.nn.Parameter(torch.randn(num_inputs, num_outputs))
        self.bias = torch.nn.Parameter(torch.randn(num_outputs))

    def forward(self, x):
        z = torch.matmul(x, self.W) + self.bias
        y = torch.sigmoid(z)
        return y

if __name__ == "__main__":
    # Define a layer and test the output!
    num_inputs = 5
    num_outputs = 2
    layer = OurDenseLayer(num_inputs, num_outputs)
    x_input = torch.tensor([[1, 2.5, 3.75, 4.24, 5.67]])
    y = layer(x_input)

    print(f"input shape: {x_input.shape}")
    print(f"output shape: {y.shape}")
    print(f"output result: {y}")