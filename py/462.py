import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import regressions as reg

'''
l1 = np.array([0, 1, 2, 3, 4, 5])
I1 = np.array([92, 83, 79, 56, 45, 36])

plt.scatter(l1, I1, color='black')
ax = curve_fit(reg.lin, l1, I1)[0]
reg.plot_func(reg.lin, l1, ax)

plt.xlabel(r'$l,\ мм$')
plt.ylabel(r'$I, мА$')
plt.grid()

plt.savefig('../images/462_1.png')
plt.show()

l2 = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0])
I2 = np.array([95, 86, 68, 62, 60, 50, 31, 6, 0])

plt.scatter(l2, I2, color='black')
ax = curve_fit(reg.lin, l2, I2)[0]
reg.plot_func(reg.lin, l2, ax)

plt.xlabel(r'$l,\ мм$')
plt.ylabel(r'$I, мА$')
plt.grid()

plt.savefig('../images/462_2.png')
plt.show()

z = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0])
Ir = np.array([95, 86, 68, 62, 60, 50, 31, 6, 0])
It = np.array([5, 9, 11, 15, 21, 31, 47, 67, 81])

R = Ir / Ir[0]
T = It / It[-1]

plt.scatter(z, R, label='R')
plt.scatter(z, T, label='T')
plt.scatter(z, R + T, color='black', label='R+T')
ax = curve_fit(reg.lin, z, R + T)[0]
reg.plot_func(reg.lin, z, ax)

plt.xlabel(r'$l,\ мм$')
plt.ylabel(r'$R, T$')
plt.grid()
plt.legend()

plt.savefig('../images/462_3.png')
plt.show()

plt.scatter(z, np.log(T), color='black', label='R+T')
ax = curve_fit(reg.lin, z, np.log(T))[0]
reg.plot_func(reg.lin, z, ax)

plt.xlabel(r'$l,\ мм$')
plt.ylabel(r'$\ln{T}$')
plt.grid()

plt.savefig('../images/462_4.png')
plt.show()

c = 299792458
f = 37.08e9
la = c / f

La = -1 / ax[0] * 1e-3
nsf1 = 1 + (la / 4 / np.pi / La) ** 2

print(La)
print(nsf1 * np.sqrt(2))
'''
x = np.arange(0, 8.5, 0.5)
Ix = [9, 11, 12, 13, 14, 15, 14, 14, 13, 12, 11, 9, 8, 6, 5, 4, 3]

plt.scatter(x, Ix, color='black', label='R+T')

plt.xlabel(r'$x,\ мм$')
plt.ylabel(r'$I,\ мА$')
plt.grid()

plt.savefig('../images/462_5.png')
plt.show()
