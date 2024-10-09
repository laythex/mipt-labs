import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import regressions as reg

U1 = np.array([6029, 5797, 5570, 5416, 5191, 5009, 4797, 4588,
               4410, 4204, 3991, 3604, 3412, 3210, 3010, 2793,
               2400, 2208, 2003, 2610, 2300, 1709, 1397, 1005,
               701, 397, 235, 2253, 2355, 2449, 2221, 2048,
               1951, 2106, 1904, 1810, 3700, 3799, 3905]) * 1e-2

I1 = np.array([2700, 2590, 2442, 2320, 2132, 2000, 1887, 1843,
               1889, 2032, 2260, 2450, 2284, 1967, 1628, 1234,
               600, 1527, 1749, 855, 661, 1562, 1256, 847,
               521, 237, 119, 914, 607, 633, 1505, 1765,
               1772, 1708, 1758, 1682, 2527, 2532, 2455]) * 1e-1

U2 = np.array([5997, 5502, 5003, 4502, 3998, 3503, 2998, 2495,
               2010, 1488, 1001, 498, 236, 2104, 2204, 2307,
               1898, 1807, 2263, 2403, 2601, 2545, 3406, 3301,
               3599, 3713, 3802, 3923, 4752, 4801, 4906, 5106, 4704]) * 1e-2

I2 = np.array([1709, 1432, 1131, 1252, 1703, 1743, 841, 173,
               1580, 1133, 620, 132, 13, 1587, 1501, 911,
               1512, 1446, 1400, 229, 224, 190, 1612, 1431,
               1833, 1847, 1858, 1800, 1134, 1123, 1116, 1180, 1152]) * 1e-1

U3 = np.array([6006, 5515, 4998, 4496, 4002, 3500, 3007, 2512,
               2015, 1504, 1002, 493, 236, 2101, 2200, 2308,
               1903, 1809, 2255, 2407, 2602, 2556, 2455, 2698,
               2802, 2909, 2846, 3901, 3812, 3696, 3504, 4899,
               5113, 5200, 5308, 5400]) * 1e-2

I3 = np.array([776, 567, 443, 755, 1127, 1062, 177, 30,
               1359, 906, 255, 14, 15, 1389, 1382, 1273,
               1288, 1209, 1360, 189, -56, 21, 101, -90,
               -52, 65, 0, 1220, 1240, 1203, 1074, 481,
               430, 441, 472, 517]) * 1e-1

sU1 = 1e-2 + 5e-3 * U1
sU2 = 1e-2 + 5e-3 * U2
sU3 = 1e-2 + 5e-3 * U3

sI1 = 1e-1 + 5e-3 * I1
sI2 = 1e-1 + 5e-3 * I2
sI3 = 1e-1 + 5e-3 * I3

I1 = I1[U1.argsort()]
U1.sort()
I2 = I2[U2.argsort()]
U2.sort()
I3 = I3[U3.argsort()]
U3.sort()

size = 15
plt.scatter(U1, I1, s=size, color='black', marker='o', label=r'$U_{зад}=4\ \text{В}$')
plt.errorbar(U1, I1, xerr=sU1, yerr=sI1, color='black', ls='')
plt.scatter(U2, I2, s=size, color='black', marker='^', label=r'$U_{зад}=6\ \text{В}$')
plt.errorbar(U2, I2, xerr=sU2, yerr=sI2, color='black', ls='')
plt.scatter(U3, I3, s=size, color='black', marker='s', label=r'$U_{зад}=8\ \text{В}$')
plt.errorbar(U3, I3, xerr=sU3, yerr=sI3, color='black', ls='')

plt.legend()
plt.xticks(np.arange(0, 65, 10))
plt.xticks(np.arange(0, 65, 2), minor=True)
plt.yticks(np.arange(0, 275, 50))
plt.yticks(np.arange(-25, 275, 12.5), minor=True)
plt.grid(which='minor', alpha=0.4)
plt.grid(which='major', alpha=1.0)

plt.xlabel(r'$U_{а},\ \text{В}$')
plt.ylabel(r'$I_{к},\ \text{мкА}$')

plt.savefig('../images/521-4.png', dpi=300)
plt.show()

p11 = np.arange(5, 14)
p12 = np.arange(23, 29)
p21 = np.arange(4, 10)
p22 = np.arange(18, 24)
p31 = np.arange(5, 11)
p32 = np.arange(21, 27)

U = [U1, U2, U3]
I = [I1, I2, I3]
sU = [sU1, sU2, sU3]
sI = [sI1, sI2, sI3]
p = [[p11, p12], [p21, p22], [p31, p32]]

d = [[[2, 2], [2, 2]],
     [[2, 2], [2, 2]],
     [[3, 5], [2, 2]]]

h = [[260, 155], [110, 85], [90, 85]]
m = [275, 200, 150]
s = [50, 25, 25]
U_z = [4, 6, 8]

u = [0, 0, 0]
su = [0, 0, 0]

size = 10
for i in range(3):
    plt.scatter(U[i], I[i], s=size, color='black', facecolors='none')
    plt.scatter(U[i][p[i][0]], I[i][p[i][0]], s=size, color='black')
    plt.scatter(U[i][p[i][1]], I[i][p[i][1]], s=size, color='black')
    plt.errorbar(U[i][p[i][0]], I[i][p[i][0]],
                 xerr=sU[i][p[i][0]], yerr=sI[i][p[i][0]], color='black', ls='')
    plt.errorbar(U[i][p[i][1]], I[i][p[i][1]],
                 xerr=sU[i][p[i][1]], yerr=sI[i][p[i][1]], color='black', ls='')

    cs1 = curve_fit(reg.parabola_centered, U[i][p[i][0]], I[i][p[i][0]], sigma=sI[i][p[i][0]],
                    maxfev=100000, bounds=((-np.inf, -np.inf, -np.inf), (np.inf, np.inf, 0)))
    reg.plot_func(reg.parabola_centered, U[i][p[i][0]], cs1[0])
    cs2 = curve_fit(reg.parabola_centered, U[i][p[i][1]], I[i][p[i][1]], sigma=sI[i][p[i][1]],
                    maxfev=100000, bounds=((-np.inf, -np.inf, -np.inf), (np.inf, np.inf, 0)))
    reg.plot_func(reg.parabola_centered, U[i][p[i][1]], cs2[0])

    u1 = cs1[0][0]
    u2 = cs2[0][0]
    su1 = np.sqrt(np.diag(cs1[1]))[0]
    su2 = np.sqrt(np.diag(cs2[1]))[0]
    u[i] = u2 - u1
    su[i] = su1 + su2

    plt.axvline(u1, color='black')
    plt.axvline(u2, color='black')
    plt.text(u1 + 1, h[i][0], fontsize=8,
             s=rf'$U_1={"{:.2f}".format(u1)}\pm{"{:.2f}".format(su1)}\ $' + r'$\text{В}$')
    plt.text(u2 + 1, h[i][1], fontsize=8,
             s=rf'$U_2={"{:.2f}".format(u2)}\pm{"{:.2f}".format(su2)}\ $' + r'$\text{В}$')

    plt.text(50.5, 5, s=r'$U_{зад}=$' + rf'${U_z[i]}\ $' + r'$\text{В}$')

    plt.xlabel(r'$U_{а},\ \text{В}$')
    plt.ylabel(r'$I_{к},\ \text{мкА}$')

    plt.xticks(np.arange(0, 65, 10))
    plt.xticks(np.arange(0, 65, 2), minor=True)
    plt.yticks(np.arange(0, m[i], s[i]))
    plt.yticks(np.arange(-25, m[i], s[i] / 4), minor=True)
    plt.grid(which='minor', alpha=0.4)
    plt.grid(which='major', alpha=1.0)

    plt.savefig(f'../images/521-{5 + i}.png', dpi=300)

    plt.show()

print(*u)
print(*su)
um, s_um = np.mean(u), np.mean(su)
print(um, s_um)
q = 1.602e-19
print(um * q, s_um * q)

