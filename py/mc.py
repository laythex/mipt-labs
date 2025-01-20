import numpy as np
import matplotlib.pyplot as plt


def u1(r):
    global k
    return -1 / r + k ** 2 / r ** 2 - k ** 2 / r ** 3


def u2(r):
    global k
    return -1 / r + k ** 2 / r ** 2


k = 2
X = np.linspace(1.4, 16, 1000)
Y1 = u1(X)
Y2 = u2(X)

r_newt = 2 * k ** 2
r_outer = k ** 2 * (1 + np.sqrt(1 - 3 / k ** 2))
r_inner = k ** 2 * (1 - np.sqrt(1 - 3 / k ** 2))

plt.plot(X, Y1, color='black', label='ОТО')
plt.plot(X, Y2, color='black', ls='--', label='Ньютон')

plt.axvline(r_newt, color='black', ls='--')
plt.axvline(r_outer, color='black')
plt.axvline(r_inner, color='black')

plt.text(r_newt, 0.012, r'$r_{класс}$')
plt.text(r_outer, 0.012, r'$r_{внешн}$')
plt.text(r_inner, 0.012, r'$r_{внутр}$')

plt.xlabel('Безразмерный радиус $r/r_s$')
plt.ylabel(r'Безразмерная потенциальная энергия $U/(\frac{mc^2}{2})$')

plt.xlim(-0.5, 15)
plt.ylim(-0.09, 0.01)

plt.xticks(np.arange(0, 16, 2))
plt.xticks(np.arange(0, 16, 0.5), minor=True)
plt.yticks(np.arange(-0.1, 0.01, 0.02))
plt.yticks(np.arange(-0.1, 0.01, 0.005), minor=True)
plt.grid(which='minor', alpha=0.4)
plt.grid(which='major', alpha=1.0)
plt.legend()

plt.savefig('../images/mc-2.png', dpi=300)
plt.show()
