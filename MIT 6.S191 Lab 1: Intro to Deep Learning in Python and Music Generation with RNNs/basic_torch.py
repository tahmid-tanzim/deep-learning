import torch


def basic_tensor_dimension_shape():
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

    # 5. random tensors
    random_tensor = torch.randn(5, 7)
    print(f"random_tensor is a {random_tensor.ndim}-d Tensor with shape: {random_tensor.shape}")
    for i in random_tensor:
        print(i)
        
def func(a, b):
    '''TODO: Define the operation for c, d, e.'''
    c = a + b
    d = b - 1
    e = c * d
    return e

def computations_on_tensors():
    # Create the nodes in the graph and initialize values
    a = torch.tensor(15)
    b = torch.tensor(61)

    # Add them!
    c1 = torch.add(a, b)
    c2 = a + b  # PyTorch overrides the "+" operation so that it is able to act on Tensors
    print(f"c1: {c1}")
    print(f"c2: {c2}")

    # Consider example values for a,b
    a, b = 1.5, 2.5
    # Execute the computation
    e_out = func(a, b)
    print(f"e_out: {e_out}")

if __name__ == "__main__":
    basic_tensor_dimension_shape()
    computations_on_tensors()   

    