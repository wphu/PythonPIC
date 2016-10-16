import numpy as np
def array_info(array):
    shape = array.shape
    dtype = array.dtype
    min = array.min()
    max = array.max()
    ptp = array.ptp()
    mean = array.mean()
    std = array.std()
    return """
    shape: {}
    dtype: {}
    min: {}
    max: {}
    peak to peak: {}
    mean: {}
    std: {}
    """.format(shape, dtype, min, max, ptp, mean, std)

if __name__=="__main__":
    x = np.arange(10)
    print(array_info(x))
