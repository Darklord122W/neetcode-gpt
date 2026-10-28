import numpy as np
from typing import List

def relu_gradient(z):
    return (z > 0).astype(float)

class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2) #this is a row matrix
        b2 = np.array(b2)
        y_true = np.array(y_true)
        z1 = W1 @ x + b1

        # FP=np.maximum(0,W2@(np.maximum(0,W1@x+b1))+b2)
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)
        FP = W2 @ a1 + b2
        MSE = np.mean((FP - y_true) ** 2)
        dz2=dL_dy = (2 / len(FP)) * (FP - y_true)
        # dW2=np.outer(dz2,np.maximum(0,W1@x+b1)> 0)
        dW2 = np.outer(dz2, a1)
        db2=dz2
        dz1 = (W2.T @ dz2) * relu_gradient(z1)
        dW1=np.outer(dz1,x)
        db1=dz1
        return {
            "loss": round(float(MSE), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist()
        }



        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        