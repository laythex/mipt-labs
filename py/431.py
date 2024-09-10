import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import regressions as reg

lam = 5461e-10

'''
x = np.array([60.2, 60.5, 60.8, 61.2])
# x = np.array([48.8, 48.2, 47.6, 47])
x -= 66
x = -1 / x
m = np.array([1, 2, 3, 4])

plt.scatter(x, m, color='black')
ax = curve_fit(reg.lin, x, m)[0]
reg.plot_func(reg.lin, x, ax)

print(ax[0])
b = 2 * np.sqrt(ax[0] * 1e-2 * lam)
print(b * 1e3)

plt.grid()
plt.ylabel('m', rotation=0)
plt.xlabel(r'$1/z,\ 1/см$')
plt.savefig('../images/431m_6.png')
plt.show()

'''
x = np.array([2.6, 3.5, 4.2]) - 2.6
m = np.array([0, 1, 2])

plt.scatter(m, x, color='black')
ax = curve_fit(reg.lin, m, x)[0]
reg.plot_func(reg.lin, m, ax)

f = 10 * 1e-2
b = f * lam / ax[0] * 1e3
print(ax[0])
print(b * 1e3)

plt.grid()
plt.xlabel('m', rotation=0)
plt.ylabel(r'$x,\ мм$')
plt.savefig('../images/431m_7.png')
plt.show()
