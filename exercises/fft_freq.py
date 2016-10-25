import numpy as np
import scipy.fftpack as fft

def test(L=1, NG=8):
    print("\nL: ", L)
    x, dx = np.linspace(0, L, NG, retstep=True, endpoint=False)
    # print(x)
    print("dx: ", dx)
    k = fft.fftfreq(NG, dx)
    kr = fft.rfftfreq(NG, dx)
    # print(k)
    print(kr)
    first_mode = 1/L
    print("first mode: ", first_mode)
    scaling = k[1]/first_mode
    print("scaling: k1/first mode", scaling)

if __name__ == "__main__":
    test()
    test(2*np.pi)
    test(4*np.pi)
    test(7)
