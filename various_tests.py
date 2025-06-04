import numpy as np
from SCGAL import approx_min_evec
import random

def M_shape_other(M_data):
    """
        M_data: data
            Some encoding of the matrix M.
    """
    return (M_data.shape[0], M_data.shape[1])

def M_mv_other(M_data, v, A_shape):
    """
        M_data: data
            Some encoding of the matrix M.
        v: vector
            Vector to multiply with the matrix M.
    """
    return M_data @ v

if __name__ == "__main__":
    sum = 0
    num_iter = 100
    for i in range(num_iter):
        sign = np.ones(10)
        sign[:5] = -1
        skinny = np.random.normal(size=(100, 10))
        M = skinny @ np.diag(sign) @ skinny.T
        xi, v_sum = approx_min_evec(
            M,
            M_shape_other,
            M_mv_other,
            q=10
        )
        sum += xi
    print(sum/num_iter)