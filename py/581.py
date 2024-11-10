import numpy as np
import matplotlib.pyplot as plt
import regressions as reg
from kurwa_fit import kurwa_fit

T = np.array([915, 1074, 1108, 1204, 1303, 1373,
              1450, 1498, 1598, 1700, 1815, 1899])

I = np.array([536.2, 588.1, 613.4, 668.0, 707.5, 741.8,
              782.6, 809.3, 871.0, 971.4, 1066.4, 1078.0]) * 1e-3

U = np.array([1.71, 2.17, 2.40, 2.92, 3.32, 3.67,
              4.12, 4.41, 5.13, 6.37, 7.66, 7.74])

sT = 5
sI = 2e-3 * I + 0.1 * 1e-3
sU = 1e-3 * U + 0.01

T1 = np.arange(800, 2100, 100)
E1 = np.array([0.460, 0.458, 0.456, 0.454, 0.452, 0.450, 0.448,
               0.446, 0.443, 0.441, 0.439, 0.437, 0.435])

size = 5

plt.scatter(T1, E1, s=size, color='black')
cs, cov = kurwa_fit(reg.lin, T1, E1)
reg.plot_func(reg.lin, T1, cs)

plt.text(1425, 0.4555,
         s=r'$\varepsilon_{\lambda,\ T}=-2.11\cdot10^{-5}\frac{1}{К}\cdot T+0.48$')

plt.xlabel('T, К')
plt.ylabel(r'$\varepsilon_{\lambda,\ T}\ (\lambda=6500\ \AA)$')
plt.grid()
plt.savefig('../images/581-1.png', dpi=300)
# plt.show()
plt.cla()

E = cs[0] * T + cs[1]
sE = np.sqrt(abs(T ** 2 * cov[0][0] + 2 * T * cov[0][1] + cov[1][1] ** 2))

lnW = np.log(U * I / E)
s_lnW = np.sqrt((sU / U) ** 2 + (sI / I) ** 2 + (sE / E) ** 2)
lnT = np.log(T)
s_lnT = sT / T

plt.scatter(lnT, lnW, s=size, color='black')
plt.errorbar(lnT, lnW, xerr=s_lnT, yerr=s_lnW, fmt='none', color='black')
cs, cov = kurwa_fit(reg.lin, lnT, lnW)
reg.plot_func(reg.lin, lnT, cs)

cov = np.sqrt(np.diag(cov))

S = 0.36e-4
sigma1 = np.exp(cs[1]) / S
s_sigma1 = sigma1 * cov[1]

print('n =', cs[0], cov[0])
print('sigma =', sigma1, s_sigma1)

plt.text(6.91, 2.1,
         s=r'$\ln{\frac{IU}{\varepsilon_{\lambda,\ T}}}=3.25\cdot\ln{T}-14.7$')

plt.xlabel(r'$\ln{(T,\ [К])}$')
plt.ylabel(r'$\ln{(IU/\varepsilon_{\lambda,\ T}, [Вт])}$')
plt.grid()
plt.savefig('../images/581-2.png', dpi=300)
# plt.show()
plt.cla()

Y = U * I
X = E * S * T ** 4
sY = Y * np.sqrt((sU / U) ** 2 + (sI / I) ** 2)
sX = X * np.sqrt((sE / E) ** 2 + (4 * sT / T) ** 2)

plt.scatter(X, Y, s=size, color='black')
plt.errorbar(X, Y, xerr=sX, yerr=sY, fmt='none', color='black')
cs, cov = kurwa_fit(reg.lin0, X, Y)
reg.plot_func(reg.lin0, X, cs)
print(cs, np.sqrt(np.diag(cov)))
plt.text(0.3e8, 6.2,
         s=r'$W=4.46\cdot10^{-8}\frac{Вт}{м^2\cdot K^4}\cdot\varepsilon_{\lambda,\ T}ST^4$')

plt.xlabel(r'$\varepsilon_{\lambda,\ T}ST^4,\ м^2\cdot К^4$')
plt.ylabel(r'$UI,\ Вт$')
plt.grid()
plt.savefig('../images/581-3.png', dpi=300)
plt.show()
plt.cla()
