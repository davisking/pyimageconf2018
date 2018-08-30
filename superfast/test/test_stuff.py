import numpy as np
import superfast

def test_sum():
    a = np.array([[1,2,3],[2,3,4]]).astype('float64')
    assert(a.sum() == superfast.sum_row_major_order(a))
