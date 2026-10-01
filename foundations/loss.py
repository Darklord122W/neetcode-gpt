import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        lensofarray=len(y_true)
        loss=0.0
        for i in range(lensofarray):
            loss+=y_true[i]*np.log(y_pred[i])+(1-y_true[i])*np.log(1-y_pred[i])
        return round(-1/lensofarray*loss,4)


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss=0.0
        for i in range(len(y_true)):
            loss+=np.sum(y_true[i]*np.log(y_pred[i]))
        return round(-1/len(y_true)*loss,4)
  
