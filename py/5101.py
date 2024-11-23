import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import sigma

import regressions as reg
from kurwa_fit import kurwa_fit


f0 = 130.3e6
f1 = 131.1e6
f2 = 129.7e6

Q = f0 / (f1 - f2)
print(f'Q = {Q}')

eds0 = 2.65
s_eds0 = 0.05 * eds0 + 0.01
eds0 *= 1e-3
s_eds0 *= 1e-3
d = 14.6e-3
sd = 0.1e-3
N = 46

B_mod = np.sqrt(2) * 2 * eds0 / np.pi ** 2 / d ** 2 / 46 / f0
s_B_mod = B_mod * np.sqrt((s_eds0 / eds0) ** 2 + (2 * sd / d) ** 2)
print(f'B_mod = {B_mod}')
print(f's_B_mod = {s_B_mod}')

U1 = np.array([26.33, 31.10, 36.05, 41.30, 46.60, 51.60,
               57.16, 62.00, 66.90, 71.80, 77.50])
E1 = np.array([4.61, 5.48, 6.34, 7.28, 8.20, 9.09,
               10.08, 11.00, 11.77, 12.70, 13.70])
U2 = np.array([26.20, 31.10, 35.80, 41.40, 46.70, 51.40,
               57.00, 62.00, 66.80, 70.90, 77.50])
E2 = np.array([4.90, 5.79, 6.67, 7.70, 8.66, 9.60,
               10.60, 11.50, 12.40, 13.10, 14.30])
sU1 = 0.05 * U1 + 0.01
sE1 = 0.05 * E1 + 0.01
sU2 = 0.05 * U2 + 0.01
sE2 = 0.05 * E2 + 0.01

size = 25

plt.scatter(U1, E1, s=size, color='black', label='Передняя сторона')
plt.errorbar(U1, E1, xerr=sU1, yerr=sE1, fmt='none', color='black')
cs1, cov1 = kurwa_fit(reg.lin, U1, E1, sigma=sE1)
reg.plot_func(reg.lin, U1, cs1)

plt.scatter(U2, E2, s=size, marker='^', color='black', label='Задняя сторона')
plt.errorbar(U2, E2, xerr=sU2, yerr=sE2, fmt='none', color='black')
cs2, cov2 = kurwa_fit(reg.lin, U2, E2, sigma=sE2)
reg.plot_func(reg.lin, U2, cs2)

plt.xlabel('U, мВ')
plt.ylabel(r'$\varepsilon$, мВ')
plt.grid()
plt.legend()
plt.savefig('../images/5101-1.png', dpi=300)
plt.show()
plt.cla()