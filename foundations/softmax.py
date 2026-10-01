import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_val = np.max(z)    # or z.max()
        if max_val>=1000:
            z=z-max_val

        afterexp=np.exp(z)
        sumofallexp=np.sum(afterexp)
        out=afterexp/sumofallexp
        return np.round(out, 4)