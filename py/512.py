import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import regressions as reg


theta = np.array([0, 10, 20, 30, 40, 50, 60,
                  70, 80, 90, 100, 110, 120])

N = np.array([829, 847, 817, 689, 637, 584, 580,
              533, 514, 464, 423, 422, 356])

N1 = 1 / N
N01 = 1 / N[0]

X = 1 - np.cos(theta)
Y = N1 - N01

plt.scatter(X, Y, s=10, color='black')

plt.grid()

plt.savefig('../images/512-2.png', dpi=300)
plt.show()
