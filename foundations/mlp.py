import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        x = np.array(x)
        for i, (W, b) in enumerate(zip(weights, biases)):
            x = x @ W + b

            # ReLU only for hidden layers
            if i < len(weights) - 1:
                x = np.maximum(0, x)
        # h1 =np.maximum(0,x@W1 + b1)
        # FP=h1@W2+b2
        return np.round(x, 4)


