import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def plot_2d_animation(x, y, z):
    fig, ax = plt.subplots()
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(z.min(), z.max())
    line, = ax.plot([], [])
    title = ax.set_title("y = {}".format(y[0]))

    def animate(i):
        line.set_data(x, z[i, :])
        title.set_text("y = {}".format(y[i]))
        return [line, title]

    anim = animation.FuncAnimation(fig, animate, z.shape[0], interval=60)
    plt.show()

def plot_2d_animation(x, y, z):
    pass

tmax = 1
xmax = 12
t = np.linspace(0,tmax,128, endpoint=False)
dt = t[1] - t[0]
x = np.linspace(0,xmax,128, endpoint=False)
dx = x[1] - x[0]

k_resolution = np.pi * 2
k = np.fft.fftshift(np.fft.fftfreq(x.size, dx)) * k_resolution
wavevector = 3 * k_resolution

omega_resolution = 2 * np.pi
omega_vector = np.fft.fftshift(np.fft.fftfreq(t.size, dt)) * omega_resolution
omega = 5 * omega_resolution
print("omega", omega, "k", wavevector)

T, X = np.meshgrid(t, x, indexing='ij')
z = np.sin(wavevector*X+omega*T)


plt.contourf(T, X, z, 50)
plt.colorbar()
plt.xlabel("t")
plt.ylabel("x")
plt.show()


space_fft = np.fft.fftshift(np.fft.fft(z, axis=1), axes=1)
T_SPACE, K = np.meshgrid(t, k, indexing='ij')
plottable_space_fft = (np.abs(space_fft))
plt.contourf(T_SPACE, K, plottable_space_fft, 50)
plt.ylim(k.min(), k.max())
# import ipdb; ipdb.set_trace()
plt.hlines(wavevector, t.min(), t.max(), colors='w', linestyles="--")
plt.colorbar()
plt.xlabel("t")
plt.ylabel("k")
plt.show()


time_fft = np.fft.fftshift(np.fft.fft(z, axis=0), axes=0)
OMEGA, X_TIME = np.meshgrid(omega_vector, x, indexing='ij')
plottable_time_fft = (np.abs(time_fft))
plt.contourf(OMEGA, X_TIME, plottable_time_fft, 50)
plt.vlines(omega, x.min(), x.max(), colors='w', linestyles="--")
plt.colorbar()
plt.xlabel("omega")
plt.ylabel("x")
plt.show()

space_time_fft = np.fft.fftshift(np.fft.fft2(z))
plottable_space_time_fft = (np.abs(space_time_fft))
OMEGA, K = np.meshgrid(omega_vector, k, indexing='ij')
plt.contourf(OMEGA, K, plottable_space_time_fft, 50)
plt.colorbar()
plt.plot([omega], [wavevector], "wo")
plt.xlabel("omega")
plt.ylabel("k")
plt.show()
