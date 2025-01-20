import numpy as np
import matplotlib.pyplot as plt
import regressions as reg
from kurwa_fit import kurwa_fit

pn = 760
Tn = 15 + 273.15
T = 25 + 273.15
rho = 101325 * 29 * 1e-3 / 8.31 / Tn * 1e-3 # г/см^3

P1 = np.array([730, 700, 675, 650, 625, 600, 575, 550, 525, 500,
               475, 450, 425, 400, 375, 350, 325, 300, 275, 250,
               225, 200, 175, 150, 125, 100, 75, 50, 25, 0])

P0 = 750.5
P = P0 - P1
sP = np.sqrt(0.8 ** 2 + 0.5 ** 2)

N = np.array([3597, 3459, 3159, 2877, 2614, 2285, 1862, 1353, 831, 370,
              78, 1, 1, 3, 3, 2, 2, 1, 2, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

n = N / 10
sn = np.sqrt(N) / 10

I = np.array([19, 63, 97, 134, 168, 209, 248, 289, 329, 367,
              409, 454, 496, 547, 590, 640, 691, 735, 783, 833,
              870, 893, 900, 899, 895, 890, 885, 875, 870, 865])

sI = 1

d = np.array([0, 1, 2, 3, 4, 5, 6, 6.5, 6.75,
              7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75,
              9, 9.25, 9.5, 9.75, 10, 10.25, 10.5]) + 10
d /= 10

t = np.array([59.892, 59.700, 60.075, 60.049, 60.104, 59.975, 60.168, 59.896,
              60.202, 60.038, 60.102, 59.608, 60.278, 60.115, 59.945, 60.440,
              65.841, 81.176, 61.018, 60.087, 60.013, 60.341, 59.931, 60.056])

M = np.array([847, 1002, 981, 827, 880, 856, 849, 809,
              752, 683, 649, 540, 434, 367, 235, 104,
              72, 56, 27, 16, 16, 12, 22, 12])

m = M / t
sm = np.sqrt(M) / t

size = 10

m1, m2 = 7, 16

plt.scatter(d, m, s=size, color='black')
plt.errorbar(d, m, yerr=sm, fmt='none', color='black')
cse, cove = kurwa_fit(reg.lin, d[m1:m2], m[m1:m2], sigma=sm[m1:m2])
k, b = cse
x1e = float(-b / k)
sx1e = float(1 / k ** 2 * np.sqrt(k ** 2 * cove[1][1] + b ** 2 * cove[0][0] - 2 * k * b * cove[0][1]))
reg.plot_func(reg.lin, np.append(d[m1:m2], x1e), cse)
x1m = float((d[m1] + d[m2]) / 2)
sx1m = float(0.25 / np.sqrt(2))
plt.axvline(x1e, ls='dashdot', color='black', label=f'{round(x1e, 2)} см')
plt.axvline(x1m, ls='--', color='black', label=f'{round(x1m, 2)} см')

x1e_norm = x1e * (P0 / pn) * (Tn / T)
sx1e_norm = sx1e * (P0 / pn) * (Tn / T)
x1m_norm = x1m * (P0 / pn) * (Tn / T)
sx1m_norm = sx1m * (P0 / pn) * (Tn / T)

print(f'x1e = {x1e, sx1e}')
print(f'x1m = {x1m, sx1m}')
print(f'x1e_n = {x1e_norm, sx1e_norm}')
print(f'x1m_n = {x1m_norm, sx1m_norm}')
print(f'x1e_s = {x1e_norm * rho, sx1e_norm * rho}')
print(f'x1m_s = {x1m_norm * rho, sx1m_norm * rho}')

plt.xlabel(f'x, мм')
plt.ylabel(f'N, 1/с')
plt.grid()
plt.legend()

plt.savefig(f'../images/541-4.png', dpi=300)
# plt.show()
plt.cla()

n_max = 14
n1, n2 = 5, 10
n0 = 7

plt.scatter(P[:n_max], n[:n_max], s=size, color='black')
plt.errorbar(P[:n_max], n[:n_max], xerr=sP, yerr=sn[:n_max], fmt='none', color='black')
cse, cove = kurwa_fit(reg.lin, P[n1:n2], n[n1:n2], sigma=sn[n1:n2])
k, b = cse
p2e = float(-b / k)
sp2e = float(1 / k ** 2 * np.sqrt(k ** 2 * cove[1][1] + b ** 2 * cove[0][0] - 2 * k * b * cove[0][1]))
reg.plot_func(reg.lin, np.append(P[n1:n2], p2e), cse)

ns = [0]
sns = [0]
for i in range(1, n_max - 1):
    na = n[i + 1] - n[i - 1]
    Pa = P[i + 1] - P[i - 1]
    ns.append(na / Pa)
    sna = sn[i + 1] ** 2 + sn[i - 1] ** 2
    spa = sP ** 2 + sP ** 2
    sns.append(np.sqrt(sna / na ** 2 + spa / Pa ** 2))
ns.append(0)
sns.append(0)

csm, covm = kurwa_fit(reg.parabola_centered,
                      P[n0 - 2:n0 + 3], ns[n0 - 2:n0 + 3], sigma=sns[n0 - 2:n0 + 3])
p2m = float(csm[0])
sp2m = float(np.sqrt(covm[0][0]))

plt.axvline(p2e, ls='dashdot', color='black', label=f'{round(p2e, 1)} мм рт. ст.')
plt.axvline(p2m, ls='--', color='black', label=f'{round(p2m, 1)} мм рт. ст.')

x0 = 9
x2m_norm = x0 * p2m / pn * Tn / T
x2e_norm = x0 * p2e / pn * Tn / T
sx2m_norm = x0 * sp2m / pn * Tn / T
sx2e_norm = x0 * sp2e / pn * Tn / T

print(f'x2e_n = {x2e_norm, sx2e_norm}')
print(f'x2m_n = {x2m_norm, sx2m_norm}')
print(f'x2e_s = {x2e_norm * rho, sx2e_norm * rho}')
print(f'x2m_s = {x2m_norm * rho, sx2m_norm * rho}')

plt.xlabel(f'P, мм рт. ст.')
plt.ylabel(f'N, 1/с')
plt.grid()
plt.legend()

plt.savefig(f'../images/541-5.png', dpi=300)
# plt.show()
plt.cla()

plt.scatter(P[1:n_max - 2], ns[1:n_max - 2], s=size, color='black')
plt.errorbar(P[1:n_max - 2], ns[1:n_max - 2], xerr=sP, yerr=sns[1:n_max - 2], fmt='none', color='black')
reg.plot_func(reg.parabola_centered, P[n0 - 2:n0 + 3], csm)

plt.axvline(p2m, ls='--', color='black')

plt.xlabel('P, мм рт. ст.')
plt.ylabel(r'dN/dP, 1/(с $\cdot$ мм рт. ст.)')
plt.grid()

plt.savefig(f'../images/541-6.png', dpi=300)
# plt.show()
plt.cla()

i1, i2 = 6, 20
j1, j2 = 23, 30
plt.scatter(P, I, s=size, color='black')
plt.errorbar(P, I, xerr=sP, yerr=sI, fmt='none', color='black')
cse1, cove1 = kurwa_fit(reg.lin, P[i1:i2], I[i1:i2], sigma=sI)
cse2, cove2 = kurwa_fit(reg.lin, P[j1:j2], I[j1:j2], sigma=sI)
reg.plot_func(reg.lin, P[i1:i2 + 3], cse1)
reg.plot_func(reg.lin, P[j1 - 3:j2], cse2)
k1, b1 = cse1
k2, b2 = cse2
p3e = float((b2 - b1) / (k1 - k2))
sp3e = float(1 / (k1 - k2) ** 2 * np.sqrt(cove1[1][1] + cove2[1][1] +
                                    p3e ** 2 * (cove1[0][0] + cove2[0][0]) +
                                    2 * p3e * (cove1[0][1] + cove2[0][1])))
plt.axvline(p3e, ls='--', color='black', label=f'{round(p3e, 1)} мм рт. ст.')

R = 5
x3e_norm = R * p3e / pn * Tn / T
sx3e_norm = R * sp3e / pn * Tn / T

print(f'x3e_n = {x3e_norm, sx3e_norm}')
print(f'x3e_s = {x3e_norm * rho, sx3e_norm * rho}')

plt.xlabel('P, мм рт. ст.')
plt.ylabel('I, пА')
plt.grid()
plt.legend()

plt.savefig(f'../images/541-7.png', dpi=300)
# plt.show()
plt.cla()

s = (x2e_norm - x1e_norm) / 1.2 * 10
ss = float(np.sqrt(sx2e_norm ** 2 + sx1e_norm ** 2) / 1.2 * 10)
print(f'd = {s, ss}')

a = 0.32
Ee = (x2e_norm / a) ** (2 / 3)
Em = (x2m_norm / a) ** (2 / 3)
sEe = 2 / 3 / a * (a / x2e_norm) ** (1 / 3) * sx2e_norm
sEm = 2 / 3 / a * (a / x2m_norm) ** (1 / 3) * sx2m_norm

print(f'Ee = {Ee, sEe}')
print(f'Em = {Em, sEm}')

Nd = n[0]
sNd = sn[0]
print(f'Nd = {Nd, sNd}')

Na = 6.02e23
Thl = 2.311e4 * 365 * 24 * 3600
nu = float(Nd / Na * 4 * np.pi / 0.04 * Thl / np.log(2))
s_nu = float(nu * sNd / Nd)
print(f'nu = {nu, s_nu}')

mass = 239 * nu * 1e3
s_mass = mass * s_nu / nu
print(f'mass = {mass, s_mass}')


