import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import regressions as reg

theta = np.array([0, 10, 20, 30, 40, 50, 60,
                  70, 80, 90, 100, 110, 120]) * np.pi / 180

N = np.array([829, 847, 817, 689, 637, 584, 480,
              410, 378, 326, 297, 313, 349])

s_theta = 45 / 285 / 2

sN = np.array([6 / 37, 5 / 37, 10 / 37, 9 / 37, 8 / 37,
               17 / 27, 5 / 27, 5 / 27, 5 / 27,
               8 / 35, 7 / 35, 8 / 35, 12 / 35]) * 200

Y = 1 / N * 1e3
sY = sN / N ** 2 * 1e3
X = 1 - np.cos(theta)
sX = abs(np.sin(theta)) * s_theta

plt.scatter(X, Y, s=10, color='black')
plt.errorbar(X, Y, xerr=sX, yerr=sY, color='black', ls='')
cs = curve_fit(reg.lin1, X[:-2], Y[:-2], sigma=sY[:-2], maxfev=1000)
reg.plot_func(reg.lin1, X, cs[0])

plt.xticks(np.arange(0, 1.7, 0.2))
plt.xticks(np.arange(-0.1, 1.7, 0.05), minor=True)
plt.yticks(np.arange(1, 4.125, 0.5))
plt.yticks(np.arange(1, 4.125, 0.125), minor=True)
plt.grid(which='minor', alpha=0.4)
plt.grid(which='major', alpha=1.0)

plt.xlabel(r'$1-\cos{\theta}$')
plt.ylabel(r'$\frac{10^3}{N(\theta)}$', rotation=0)

plt.savefig('../images/512-2.png', dpi=300)
plt.show()

Eg = 662

x0 = cs[0][1]
sx0 = np.sqrt(np.diag(cs[1]))[1] + sX[7]
print(x0, sx0)

E = -Eg * x0
sE = Eg * sx0
print(E, sE)
