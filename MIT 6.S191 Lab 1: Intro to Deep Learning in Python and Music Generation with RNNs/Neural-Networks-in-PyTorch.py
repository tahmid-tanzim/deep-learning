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

### Defining a model using subclassing ###
class LinearWithSigmoidActivation(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearWithSigmoidActivation, self).__init__()
        self.linear = torch.nn.Linear(num_inputs, num_outputs)
        self.activation = torch.nn.Sigmoid()

    def forward(self, inputs):
        linear_output = self.linear(inputs)
        output = self.activation(linear_output)
        return output

### Custom behavior with subclassing nn.Module ###
class LinearButSometimesIdentity(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearButSometimesIdentity, self).__init__()
        self.linear = torch.nn.Linear(num_inputs, num_outputs)

    def forward(self, inputs, isidentity=False):
        if isidentity:
            return inputs
        else:
            return self.linear(inputs)


if __name__ == "__main__":
    # Define a layer and test the output!
    num_inputs = 2
    num_outputs = 3
    layer = OurDenseLayer(num_inputs, num_outputs)
    x_input = torch.tensor([[4.56, 5.67]])
    y = layer(x_input)

    print(f"1. Layer input shape: {x_input.shape}")
    print(f"1. Layer output shape: {y.shape}")
    print(f"1. Layer output result: {y}")

    ### Defining a neural network using the PyTorch Sequential API ###

    # define the number of inputs and outputs
    n_input_nodes = 2
    n_output_nodes = 3

    # Define the model
    model = torch.nn.Sequential(
        torch.nn.Linear(n_input_nodes, n_output_nodes),
        torch.nn.Sigmoid()
    )

    # Test the model
    x_input = torch.tensor([[4.56, 5.67]])
    y = model(x_input)

    print(f"2. Sequential Model input shape: {x_input.shape}")
    print(f"2. Sequential Model output shape: {y.shape}")
    print(f"2. Sequential Model output result: {y}")

    n_input_nodes = 2
    n_output_nodes = 3
    model = LinearWithSigmoidActivation(n_input_nodes, n_output_nodes)
    x_input = torch.tensor([[4.56, 5.67]])
    y = model(x_input)
    print(f"3. LinearWithSigmoidActivation Model input shape: {x_input.shape}")
    print(f"3. LinearWithSigmoidActivationModel output shape: {y.shape}")
    print(f"3. LinearWithSigmoidActivation Model output result: {y}")

    # Test the IdentityModel
    model = LinearButSometimesIdentity(num_inputs=2, num_outputs=3)
    x_input = torch.tensor([[4.5678, 5.6789]])

    out_with_linear = model(x_input, isidentity=False)

    out_with_identity = model(x_input, isidentity=True)

    print(f"input: {x_input}")
    print("Network linear output: {}; network identity output: {}".format(out_with_linear, out_with_identity))
    