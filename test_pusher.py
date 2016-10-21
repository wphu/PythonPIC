from Species import Species
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import l2_test

def test_constant_field():
	s = Species(1, 1, 1)
	s.x = np.array([0], dtype=float)
	print(s.x)
	uniform_field = lambda x: np.ones_like(x)
	t, dt = np.linspace(0, 10, 200, retstep=True, endpoint=False)
	x_analytical = 0.5*t**2 + 0
	x_data = []
	for i in range(t.size):
		x_data.append(s.x[0])
		s.push_particles(uniform_field, dt, np.inf)
	x_data = np.array(x_data)
	print(x_analytical-x_data)

	def plot():
		fig, (ax1, ax2) = plt.subplots(2, sharex=True)
		ax1.plot(t, x_analytical, "b-", label="analytical result")
		ax1.plot(t, x_data, "ro--", label="simulation result")
		ax1.legend()

		ax2.plot(t, x_data - x_analytical, label="difference")
		ax2.legend()

		ax2.set_xlabel("t")
		ax1.set_ylabel("x")
		ax2.set_ylabel("delta x")
		plt.show()	
		return None

	assert l2_test(x_analytical, x_data), plot()


if __name__=="__main__":
	test_constant_field()
