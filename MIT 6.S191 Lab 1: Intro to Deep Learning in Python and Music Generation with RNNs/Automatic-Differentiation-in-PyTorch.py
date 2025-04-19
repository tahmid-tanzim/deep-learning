import torch

# import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ### Gradient computation ###

    # y = x^2
    # Example: x = 3.0
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2
    y.backward()  # Compute the gradient

    dy_dx = x.grad
    print("dy_dx of y=x^2 at x=3.0 is: ", dy_dx)
    assert dy_dx == 6.0

    ### Function minimization with autograd and gradient descent ###

    # Initialize a random value for our intial x
    x = torch.randn(1)
    print(f"Initializing x={x.item()}")

    learning_rate = 1e-2  # Learning rate
    history = []
    x_f = 5  # Target value


    # We will run gradient descent for a number of iterations. At each iteration, we compute the loss,
    #   compute the derivative of the loss with respect to x, and perform the update.
    for i in range(500):
        x = torch.tensor([x], requires_grad=True)

        # TODO: Compute the loss as the square of the difference between x and x_f
        loss = (x - x_f) ** 2

        # Backpropagate through the loss to compute gradients
        loss.backward()

        # Update x with gradient descent
        x = x.item() - learning_rate * x.grad

        history.append(x.item())

    # Plot the evolution of x as we optimize toward x_f!
    plt.plot(history)
    plt.plot([0, 500], [x_f, x_f])
    plt.legend(('Predicted', 'True'))
    plt.xlabel('Iteration')
    plt.ylabel('x value')
    plt.show()