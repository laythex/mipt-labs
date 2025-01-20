import numpy as np
import matplotlib.pyplot as plt
import regressions as reg
from kurwa_fit import kurwa_fit

f0 = 130.3e6
f1 = 131.1e6
f2 = 129.7e6
sf = 0.1e6

Q = f0 / (f1 - f2)
sQ = Q * sf * np.sqrt(1 / f0 ** 2 + 2 / (f1 - f2) ** 2)
print(f'Q = {Q}, {float(sQ)}')

eds0 = 2.65
s_eds0 = 5e-2 * eds0 + 0.01

d = 14.6e-3
sd = 0.1e-3
N = 46
v = 50

B_mod = np.sqrt(2) * 2 * eds0 / (np.pi ** 2 * d ** 2 * N * v)
s_B_mod = B_mod * np.sqrt((s_eds0 / eds0) ** 2 + (2 * sd / d) ** 2)
print(f'B_mod = {B_mod}, {s_B_mod}')

U1 = np.array([26.33, 31.10, 36.05, 41.30, 46.60, 51.60,
               57.16, 62.00, 66.90, 71.80, 77.50])
E1 = np.array([4.61, 5.48, 6.34, 7.28, 8.20, 9.09,
               10.08, 11.00, 11.77, 12.70, 13.70])
U2 = np.array([26.20, 31.10, 35.80, 41.40, 46.70, 51.40,
               57.00, 62.00, 66.80, 70.90, 77.50])
E2 = np.array([4.90, 5.79, 6.67, 7.70, 8.66, 9.60,
               10.60, 11.50, 12.40, 13.10, 14.30])
sU1 = 1e-2 * U1 + 0.01
sE1 = 1e-2 * E1 + 0.01
sU2 = 1e-2 * U2 + 0.01
sE2 = 1e-2 * E2 + 0.01

size = 20

cs1, cov1 = kurwa_fit(reg.lin, U1, E1, sigma=sE1)
reg.plot_func(reg.lin, U1, cs1)
plt.scatter(U1, E1, s=size, color='black',
            label=rf'Передняя сторона: $\varepsilon={round(cs1[0], 3)}\cdot U-{-round(cs1[1], 3)}$')
plt.errorbar(U1, E1, xerr=sU1, yerr=sE1, fmt='none', color='black')

cs2, cov2 = kurwa_fit(reg.lin, U2, E2, sigma=sE2)
reg.plot_func(reg.lin, U2, cs2)
plt.scatter(U2, E2, s=size, marker='^', color='black',
            label=rf'Задняя сторона: $\varepsilon={round(cs2[0], 3)}\cdot U+{round(cs2[1], 3)}$')
plt.errorbar(U2, E2, xerr=sU2, yerr=sE2, fmt='none', color='black')


u0 = 64.3
s_u0 = 4.4
plt.axvline(u0, ls='--', color='black')

plt.xlabel('U, мВ')
plt.ylabel(r'$\varepsilon$, мВ')
plt.xticks(np.arange(20, 85, 10))
plt.xticks(np.arange(20, 85, 2), minor=True)
plt.yticks(np.arange(4, 15, 2))
plt.yticks(np.arange(4, 15, 0.4), minor=True)
plt.grid(which='minor', alpha=0.4)
plt.grid(which='major', alpha=1.0)
plt.legend()
plt.savefig('../images/5101-1.png', dpi=400)
# plt.show()
plt.cla()

eds1 = cs1[0] * u0 + cs1[1]
eds2 = cs2[0] * u0 + cs2[1]
s_eds1 = np.sqrt(u0 ** 2 * cov2[0][0] + cov2[1][1] + 2 * u0 * cov2[0][1] + cs1[0] ** 2 * s_u0 ** 2)
s_eds2 = np.sqrt(u0 ** 2 * cov2[0][0] + cov2[1][1] + 2 * u0 * cov2[0][1] + cs2[0] ** 2 * s_u0 ** 2)
print(f'eds1 = {eds1}, {s_eds1}')
print(f'eds2 = {eds2}, {s_eds2}')

B1 = 2 * eds1 / (v * np.pi ** 2 * N * d ** 2)
B2 = 2 * eds2 / (v * np.pi ** 2 * N * d ** 2)
sB1 = B1 * np.sqrt((s_eds1 / eds1) ** 2 + (2 * sd / d) ** 2)
sB2 = B2 * np.sqrt((s_eds1 / eds1) ** 2 + (2 * sd / d) ** 2)
print(f'B1 = {B1}, {sB1}')
print(f'B2 = {B2}, {sB2}')

B0 = (B1 + B2) / 2
sB0 = np.sqrt(sB1 ** 2 + sB2 ** 2) / 2
print(f'B0 = {B0}, {sB0}')

h = 6.626e-34
mb = 927.4e-26
g = h * f0 / mb / B0 * 1e3
sg = g * np.sqrt((sf / f0) ** 2 + (sB0 / B0) ** 2)
print(f'g = {g}, {sg}')

fs = np.array([118.34, 124.34, 130.34, 136.34, 142.34])
etas = np.array([0.74, 0.78, 0.84, 0.88, 0.91])
s_fs = 0.01
s_etas = np.array([0.03, 0.04, 0.04, 0.03, 0.04])

cs, cov = kurwa_fit(reg.lin0, fs, etas, sigma=s_etas)
reg.plot_func(reg.lin0, fs, cs)
plt.scatter(fs, etas, s=size, color='black',
            label=rf'$\eta={round(cs[0], 4)}\cdot \nu$')
plt.errorbar(fs, etas, xerr=s_fs, yerr=s_etas, fmt='none', color='black')

plt.xlabel(r'$\nu$, МГц')
plt.ylabel(r'$\eta$')
plt.xticks(np.arange(120, 145, 5))
plt.xticks(np.arange(117, 145, 1), minor=True)
plt.yticks(np.arange(0.7, 0.96, 0.05))
plt.yticks(np.arange(0.7, 0.95, 0.01), minor=True)
plt.grid(which='minor', alpha=0.4)
plt.grid(which='major', alpha=1.0)
plt.legend(loc='upper left')
plt.savefig('../images/5101-8.png', dpi=400)
# plt.show()
plt.cla()

Bf = B0 / etas[2]
sBf = Bf * np.sqrt((sB0 / B0) ** 2 + (s_etas[2] / etas[2]) ** 2)
print(f'Bf = {Bf}, {sBf}')

k = cs[0] * 1e-6
sk = np.sqrt(cov[0][0]) * 1e-6
g1 = h / (k * mb * Bf) * 1e3
sg1 = g1 * np.sqrt((sBf / Bf) ** 2 + (sk / k) ** 2)
print(f'g1 = {g1}, {sg1}')
