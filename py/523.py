import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import regressions as reg


def hartmann_inv(d, _lam0, _k, _d0):
    return _lam0 + _k / (d - _d0)


def hartmann(lam, _lam0, _k, _d0):
    return _d0 + _k / (lam - _lam0)


angle1 = [1892, 1898, 1938, 2196, 2212,
          2246, 2258, 2288, 2306, 2318,
          2336, 2346, 2370, 2390, 2404,
          2420, 2438, 2444, 2482, 2492,
          2514, 2540, 2552, 2616, 2646]

angle2 = [344, 890, 1556, 1978, 2156, 2168, 2372, 2606]

lam1 = [5331, 5341, 5401, 5852, 5882,
        5945, 5976, 6030, 6074, 6096,
        6143, 6164, 6217, 6267, 6305,
        6334, 6383, 6402, 6507, 6533,
        6599, 6678, 6717, 6929, 7032]

lam2 = [4047, 4358, 4916, 5461, 5770, 5791, 6234, 6907]

s_angle = 2

angle_cal = angle1 + angle2
lam_cal = lam1 + lam2

cs_cal = curve_fit(hartmann, lam_cal, angle_cal, sigma=s_angle)
reg.plot_func(hartmann, lam_cal, cs_cal[0])

s_cs_cal = np.sqrt(np.diag(cs_cal[1]))

lam0, C, d0 = cs_cal[0][0], cs_cal[0][1], cs_cal[0][2]
s_lam0, s_C, s_d0 = s_cs_cal[0], s_cs_cal[1], s_cs_cal[2]

plt.scatter(lam1, angle1, color='black', s=15, label='Неоновая лампа')
plt.scatter(lam2, angle2, color='black', s=15, marker='^', label='Ртутная лампа')

plt.xlabel(r'$Длина\ волны\ \lambda,\ \AA$')
plt.ylabel(r'$Показания\ барабана,\ усл.\ ед.$')

plt.legend()
plt.grid()

plt.savefig('../images/523-1.png')
plt.show()

angle_H = np.array([446, 862, 1498, 2490])
lam_H = hartmann_inv(angle_H, *cs_cal[0])

s_lam_H = np.sqrt(s_lam0 ** 2 + ((lam_H - lam0) / C) ** 2 * (s_C ** 2 + (lam_H - lam0) ** 2 * s_d0 ** 2))

print(*lam_H)
print(*s_lam_H)

h = 6.626e-34
c = 299792458
e = 1.602e-19

energies_H = h * c / lam_H * 1e10 / e
s_energies_H = energies_H * s_lam_H / lam_H
coefficients = (1 / 4 - 1 / np.array([6, 5, 4, 3]) ** 2)

cs = curve_fit(reg.lin, coefficients, energies_H, sigma=s_energies_H)
reg.plot_func(reg.lin, coefficients, cs[0])
print(cs[0][0], np.sqrt(np.diag(cs[1]))[0])
print(energies_H, s_energies_H)
plt.scatter(coefficients, energies_H, color='black')

plt.xlabel(r'$1/2^2-1/m^2$')
plt.ylabel(r'$Энергия\ E,\ эВ$')

plt.grid()

plt.savefig('../images/523-2.png')
plt.show()

angle_I = np.array([2302, 2196, 1630])
lam_I = hartmann_inv(angle_I, *cs_cal[0])
s_lam_I = np.sqrt(s_lam0 ** 2 + ((lam_I - lam0) / C) ** 2 * (s_C ** 2 + (lam_I - lam0) ** 2 * s_d0 ** 2))

# print(*lam_I)
# print(*s_lam_I)

energies_I = h * c / lam_I * 1e10 / e
s_energies_I = energies_I * s_lam_I / lam_I

hn2 = (energies_I[1] - energies_I[0]) / 5
s_hn2 = (s_energies_I[1] + s_energies_I[0]) / 5
# print(energies_I[2], s_energies_I[2])

hn1 = 0.027
EA = 0.94

hnel = energies_I[0] - 0.5 * hn2 + 1.5 * hn1
s_hnel = s_energies_I[0] + 0.5 * s_hn2

# print(hnel)
# print(s_hnel)

# print(energies_I[2] - EA)
# print(s_energies_I[2])

print(energies_I[2] - hnel)
print(s_energies_I[2] + s_hnel)
