import numpy as np

from helper_functions import l2_norm, l2_test

def test_l2norm():
    x = np.arange(10)
    assert l2_norm(x, x) == 0
    assert l2_test(x, x)