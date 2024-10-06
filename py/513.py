import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import regressions as reg

R = 1e5

U = np.array([10715, 10032, 9492, 9019, 8508, 8015,
              7500, 7006, 6494, 6017, 5495, 5001,
              4510, 4011, 3514, 2985]) * 1e-3

I = np.array([6706, 6416, 6405, 6520, 6782, 7149,
              7546, 7926, 8243, 8399, 8401, 8302,
              8063, 7623, 6994, 5780]) * 1e-5 / R * 1e6

sI = 1 * 1e-5 / R * 1e6

I = I[U.argsort()]
U.sort()

plt.scatter(U, I, s=10, color='black')

plt.xlabel(r'$U_c,\ \text{В}$')
plt.ylabel(r'$I_a,\ \text{мкА}$')

plt.xticks(np.arange(2, 13, 2))
plt.xticks(np.arange(1, 13, 0.5), minor=True)
plt.yticks(np.arange(0.5, 1, 0.1))
plt.yticks(np.arange(0.5, 0.95, 0.025), minor=True)
plt.grid(which='minor', alpha=0.4)
plt.grid(which='major', alpha=1.0)

plt.savefig('../images/513-4.png', dpi=300)
plt.show()

p1 = np.arange(3, 9)
p2 = np.arange(11, 16)
size = 10

plt.scatter(U, I, s=size, color='black', facecolors='none')

plt.scatter(U[p1], I[p1], s=size, color='black')
cs1 = curve_fit(reg.parabola_centered, U[p1], I[p1], sigma=sI,
                maxfev=100000, bounds=((-np.inf, -np.inf, -np.inf), (np.inf, np.inf, 0)))
reg.plot_func(reg.parabola_centered, U[p1], cs1[0])
plt.axvline(cs1[0][0], color='black')

plt.scatter(U[p2], I[p2], s=size, color='black')
cs2 = curve_fit(reg.parabola_centered, U[p2], I[p2], sigma=sI,
                maxfev=100000, bounds=((-np.inf, -np.inf, 0), (np.inf, np.inf, np.inf)))
reg.plot_func(reg.parabola_centered, U[p2], cs2[0])
plt.axvline(cs2[0][0], color='black')

plt.xticks(np.arange(2, 13, 2))
plt.xticks(np.arange(1, 13, 0.5), minor=True)
plt.yticks(np.arange(0.5, 1, 0.1))
plt.yticks(np.arange(0.5, 0.95, 0.025), minor=True)
plt.grid(which='minor', alpha=0.4)
plt.grid(which='major', alpha=1.0)

plt.savefig('../images/513-5.png', dpi=300)
plt.show()

E1 = cs1[0][0]
sE1 = np.sqrt(np.diag(cs1[1]))[0]
E2 = cs2[0][0]
sE2 = np.sqrt(np.diag(cs2[1]))[0]
print(E1, sE1)
print(E2, sE2)

h = 6.626e-34
q = 1.602e-19
m = 9.109e-31

d = h * np.sqrt(5 / 32 / (m * (E2 - E1) * q))
sd = d / 2 * (sE1 + sE2) / (E1 + E2)
print(d, sd)

U0 = 4 / 5 * E2 - 9 / 5 * E1
sU0 = 4 / 5 * sE2 + 9 / 5 * sE1
print(U0, sU0)
