import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import regressions as reg

U_in1 = np.array([10, 30, 60, 90, 120,
                  150, 200, 250, 300,
                  350, 400, 450, 500])

U_out1 = np.array([31.8, 93.4, 181.8, 280.1, 371.5,
                   464.8, 611.8, 760.8, 888.0,
                   1000, 1080, 1130, 1170]) / 2 * 10

K = U_out1 / U_in1

plt.scatter(U_in1, K, color='black')

plt.xlabel(r'$U_{m\_in},\ мВ$')
plt.ylabel(r'$K=U_{m\_out}/U_{m\_in}$')
plt.grid()

plt.savefig('../images/rt1-1.png', dpi=300)
plt.show()


w = np.array([1, 1.02, 1.04, 1.01, 1.03, 1.05,
              0.99, 0.98, 0.97, 0.97])

A = np.array([646, 507, 340, 588, 408, 282, 586, 484, 393, 323]) / 2 * 10 / 1e3

A = A[w.argsort()]
w.sort()

plt.scatter(w, A, color='black')

resA = A[4] / np.sqrt(2)
plt.axhline(resA, color='black', ls='--')

a, b = 1, 6
plt.plot(w[a:a + 2], A[a:a + 2], color='black')
plt.plot(w[b:b + 2], A[b:b + 2], color='black')

x1 = (resA - (A[a + 1] * w[a] - A[a] * w[a + 1]) / (w[a] - w[a + 1])) * (w[a] - w[a + 1]) / (A[a] - A[a + 1])
x2 = (resA - (A[b + 1] * w[b] - A[b] * w[b + 1]) / (w[b] - w[b + 1])) * (w[b] - w[b + 1]) / (A[b] - A[b + 1])
plt.scatter([x1, x2], [resA, resA], color='black', marker='x')
plt.plot([x1, x2], [resA, resA], color='black')

print(x2 - x1)
print(1 / (x2 - x1))

plt.xlabel(r'$\omega,\ МГц$')
plt.ylabel('$U_{out},\ В$')
plt.grid()

plt.savefig('../images/rt1-2.png', dpi=300)
plt.show()
