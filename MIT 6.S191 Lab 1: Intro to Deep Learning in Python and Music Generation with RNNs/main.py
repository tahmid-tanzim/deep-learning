import torch
import torch.nn as nn

# import mitdeeplearning as mdl

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 1. Let’s start by creating some tensors and inspecting their properties:
    integer_tensor = torch.tensor(1234)
    decimal_tensor = torch.tensor(3.14159265359)

    print(f"`integer` is a {integer_tensor.ndim}-d Tensor: {integer_tensor}")
    print(f"`decimal` is a {decimal_tensor.ndim}-d Tensor: {decimal_tensor}")

    # 2. Vectors and lists can be used to create 1-d tensors:
    fibonacci = torch.tensor([1, 1, 2, 3, 5, 8])
    count_to_100 = torch.tensor(range(100))

    print(f"`fibonacci` is a {fibonacci.ndim}-d Tensor with shape: {fibonacci.shape}")
    print(f"`count_to_100` is a {count_to_100.ndim}-d Tensor with shape: {count_to_100.shape}")

    # 3. Defining higher-order Tensors ###
    matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])

    print("matrix must be a torch Tensor object - ",isinstance(matrix, torch.Tensor))
    print("matrix must be a 2-d Tensor - ", matrix.ndim)

    '''TODO: Define a 4-d Tensor.'''
    # Use torch.zeros to initialize a 4-d Tensor of zeros with size 10 x 3 x 256 x 256.
    #   You can think of this as 10 images where each image is RGB 256 x 256.
    images = torch.zeros(10, 3, 256, 256)

    print("images must be a torch Tensor object - ", isinstance(images, torch.Tensor))
    print(f"images is a {images.ndim}-d Tensor with shape: {images.shape}")

    # 4. use slicing to access subtensors within a higher-rank tensor:
    row_vector = matrix[1]
    column_vector = matrix[:, 1]
    scalar = matrix[0, 1]

    print(f"`row_vector`: {row_vector}")
    print(f"`column_vector`: {column_vector}")
    print(f"`scalar`: {scalar}")
    