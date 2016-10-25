import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt

def test(L=1, NG=64, A = 8):
    print("\nL: ", L)
    x, dx = np.linspace(0, L, NG, retstep=True, endpoint=False)
    print("dx: ", dx)
    k = 2*np.pi*fft.fftfreq(NG, dx)
    first_mode = 2*np.pi/L
    print("first mode: ", first_mode)
    scaling = k[1]/first_mode
    print("scaling: k1/first mode", scaling)

    func = A * np.sin(first_mode*x)
    func_fft = fft.fft(func)
    func_der_fft = fft.ifft(func_fft * 1j* k).real
    print(func_der_fft[0]/(A*first_mode))

    plt.plot(x, func)
    plt.plot(x, func_der_fft)
    plt.show()

if __name__ == "__main__":
    test()
    test(2*np.pi)
    test(4*np.pi)
    test(7)
