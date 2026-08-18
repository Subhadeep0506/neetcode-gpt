import numpy as np
from numpy.typing import NDArray


class Solution:
    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_val = np.max(z)
        res = []
        exp_sum = sum([math.exp(item - max_val) for item in z])
        
        for item in z:
            _softmax = math.exp(item - max_val) / exp_sum
            res.append(_softmax)
        
        return np.round(res, 4)