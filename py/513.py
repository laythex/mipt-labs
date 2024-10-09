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

sU = 1e-3 + 5e-3 * U
sI = 1e-5 / R * 1e6 + 5e-3 * I

I = I[U.argsort()]
U.sort()

size = 10

plt.scatter(U, I, s=size, color='black')
plt.errorbar(U, I, xerr=sU, yerr=sI, color='black', ls='')

plt.xlabel(r'$U_c,\ \text{В}$')
plt.ylabel(r'$I_a,\ \text{мА}$')

plt.xticks(np.arange(2, 13, 2))
plt.xticks(np.arange(1, 13.5, 0.5), minor=True)
plt.yticks(np.arange(0.5, 1, 0.1))
plt.yticks(np.arange(0.5, 0.95, 0.025), minor=True)
plt.grid(which='minor', alpha=0.4)
plt.grid(which='major', alpha=1.0)

plt.savefig('../images/513-4.png', dpi=300)
plt.show()

p1 = np.arange(3, 9)
p2 = np.arange(11, 16)

plt.scatter(U, I, s=size, color='black', facecolors='none')

plt.scatter(U[p1], I[p1], s=size, color='black')
plt.errorbar(U[p1], I[p1], xerr=sU[p1], yerr=sI[p1], color='black', ls='')

cs1 = curve_fit(reg.parabola_centered, U[p1], I[p1], sigma=sI[p1],
                maxfev=100000, bounds=((-np.inf, -np.inf, -np.inf), (np.inf, np.inf, 0)))
reg.plot_func(reg.parabola_centered, U[p1], cs1[0])

E1 = cs1[0][0]
sE1 = np.sqrt(np.diag(cs1[1]))[0]

plt.axvline(E1, color='black')
plt.text(E1 + 0.1, 0.505, fontsize=8,
         s=rf'$E_1=({"{:.3f}".format(E1)}\pm{"{:.3f}".format(sE1)})\ $' + r'$\text{В}$')

I0 = cs1[0][1]
sI0 = np.sqrt(np.diag(cs1[1]))[1]

plt.scatter(U[p2], I[p2], s=size, color='black')
plt.errorbar(U[p2], I[p2], xerr=sU[p2], yerr=sI[p2], color='black', ls='')

cs2 = curve_fit(reg.parabola_centered, U[p2], I[p2], sigma=sI[p2],
                maxfev=100000, bounds=((-np.inf, -np.inf, 0), (np.inf, np.inf, np.inf)))
reg.plot_func(reg.parabola_centered, U[p2], cs2[0])

E2 = cs2[0][0]
sE2 = np.sqrt(np.diag(cs2[1]))[0]

plt.axvline(E2, color='black')
plt.text(E2 + 0.1, 0.505, fontsize=8,
         s=rf'$E_2=({"{:.3f}".format(E2)}\pm{"{:.3f}".format(sE2)})\ $' + r'$\text{В}$')

plt.axhline(cs1[0][1], color='black', ls='--')
plt.text(1.5, I0 + 0.004,
         s=rf'$I_0=({"{:.1f}".format(I0 * 1e3)}\pm{"{:.1f}".format(sI0 * 1e3)})\ $' + r'$\text{мкА}$')

plt.xlabel(r'$U_c,\ \text{В}$')
plt.ylabel(r'$I_a,\ \text{мА}$')

plt.xticks(np.arange(2, 13, 2))
plt.xticks(np.arange(1, 13.5, 0.5), minor=True)
plt.yticks(np.arange(0.5, 1, 0.1))
plt.yticks(np.arange(0.5, 0.95, 0.025), minor=True)
plt.grid(which='minor', alpha=0.4)
plt.grid(which='major', alpha=1.0)

plt.savefig('../images/513-5.png', dpi=300)
plt.show()

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
print(I0, sI0)
w = -np.log(I / I0)
sw = np.sqrt((sI / I) ** 2 + (sI0 / I0) ** 2)

plt.scatter(U, w, s=size, color='black')
plt.errorbar(U, w, xerr=sU, yerr=sw, color='black', ls='')

plt.xlabel(r'$U_c,\ \text{В}$')
plt.ylabel(r'$w\cdot C$')

plt.xticks(np.arange(2, 13, 2))
plt.xticks(np.arange(1, 13.5, 0.5), minor=True)
plt.yticks(np.arange(0, 0.4, 0.1))
plt.yticks(np.arange(0, 0.4, 0.02), minor=True)
plt.grid(which='minor', alpha=0.4)
plt.grid(which='major', alpha=1.0)

plt.savefig('../images/513-6.png', dpi=300)
plt.show()

u0 = 2.5
e1, e2 = 6, 10
se1, se2 = 1, 1

l1 = 1 / 2 * h / np.sqrt(2 * m * (e1 + u0) * q)
l2 = 3 / 4 * h / np.sqrt(2 * m * (e2 + u0) * q)
sl1 = l1 / (2 * (e1 + u0)) * se1
sl2 = l2 / (2 * (e2 + u0)) * se2
print('Динам')
print(l1, l2)
print(sl1, sl2)

l3 = h * np.sqrt(5 / 32 / (m * (e2 - e1) * q))
sl3 = l3 / 2 * (se1 + se2) / (e1 + e2)
u01 = 4 / 5 * e2 - 9 / 5 * e1
su01 = 4 / 5 * se2 - 9 / 5 * se1
print(l3, sl3)
print(u01, su01)
