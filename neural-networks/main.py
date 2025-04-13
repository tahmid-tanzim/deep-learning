import math
import numpy as np
from enum import Enum

"""
# A simple drug dosage efficacy
Input As x-axis: 0 <= dosage <= 1 
Output as y-axis: 0 <= efficacy <= 1; No ~> 0 & Yes ~> 1 
"""


# class syntax
class ActivationFunctionType(Enum):
    SIGMOID = "sigmoid"
    RELU = "ReLU"  # Rectified Linear Unit
    SOFT_PLUS = "softplus"
    HYPERBOLIC = "hyperbolic_tangent"


def activation_functions(x: float, types: str) -> float:
    y = 0.0
    match types:
        case ActivationFunctionType.SIGMOID.value:
            y = math.exp(x) / (math.exp(x) + 1)
        case ActivationFunctionType.RELU.value:
            y = max(0.0, x)
        case ActivationFunctionType.SOFT_PLUS.value:
            y = math.log(1 + math.exp(x))
        case ActivationFunctionType.HYPERBOLIC.value:
            y = (math.exp(x) - math.exp(x * -1)) / (math.exp(x) + math.exp(x * -1))
    return y


# Hidden Layer 1 Node 1
def node_1(x):
    m = -34.4
    b = 2.14
    x1 = x * m + b
    y1 = activation_functions(x1, ActivationFunctionType.SOFT_PLUS.value)
    return y1


# Hidden Layer 1 Node 2
def node_2(x):
    m = -2.52
    b = 1.29
    x1 = x * m + b
    y1 = activation_functions(x1, ActivationFunctionType.SOFT_PLUS.value)
    return y1


def neural_network(dosage):
    b = -0.58
    o1 = node_1(dosage) * -1.3
    o2 = node_2(dosage) * 2.28
    efficacy = o1 + o2 + b
    print(f"Dosage: {dosage}, Efficacy: {efficacy}")


if __name__ == "__main__":
    for d in np.arange(0.0, 1.1, 0.1):
        neural_network(d)
