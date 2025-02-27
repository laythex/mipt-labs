import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import regressions as reg

I0 = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
               2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])

I1 = np.array([3.05, 3.10, 3.15, 3.25, 3.30, 3.35, 3.45, 3.50, 3.55])

N0 = np.array([1.130, 1.310, 1.300, 1.540, 2.529, 5.658, 8.687, 11.297, 12.956,
               14.856, 15.525, 13.816, 10.967, 6.748, 4.069, 4.329,
               20.954, 12.766, 2.359, 0.930, 0.780, 0.720])

N1 = np.array([7.378, 12.746, 18.874, 24.113, 22.633, 18.964, 9.987, 6.538, 3.829])

s_N0 = np.sqrt(N0 / 100)
s_N1 = np.sqrt(N1 / 100)

# plt.errorbar(I0, N0, yerr=s_N0, fmt='none', color='black')
# plt.scatter(I0, N0, s=5, color='black')
#
# plt.xlabel(r'$I,\ А$')
# plt.ylabel(r'$N,\ с^{-1}$')
# plt.xticks(np.arange(0, 5, 1))
# plt.xticks(np.arange(-0.2, 4.4, 0.2), minor=True)
# plt.yticks(np.arange(0, 22, 5))
# plt.yticks(np.arange(-1, 23, 1), minor=True)
# plt.grid(which='minor', alpha=0.4)
# plt.grid(which='major', alpha=1.0)
#
# plt.savefig('../images/542-1.png', dpi=300)
# plt.show()

I = np.concatenate((I0, I1))
N = np.concatenate((N0, N1))
s_N = np.sqrt(N / 100)

N = N[I.argsort()]
I.sort()

Nf = (N[0] + N[-1]) / 2
s_Nf = (s_N[0] + s_N[-1]) / 2
NmNf = N - Nf
s_NmNf = s_N + s_Nf

ax1 = plt.gca()
ax1.errorbar(I, NmNf, yerr=s_NmNf, fmt='none', color='black')
ax1.scatter(I, NmNf, s=5, color='black')

ax1.set_xlabel(r'$I,\ А$')
ax1.set_ylabel(r'$N-N_{ф},\ с^{-1}$')
ax1.set_xlim(0, I[-1] * 1.05)

ax1.set_xticks(np.arange(0, 5, 1))
ax1.set_xticks(np.arange(-0.2, 4.4, 0.2), minor=True)
ax1.set_yticks(np.arange(0, 26, 5))
ax1.set_yticks(np.arange(-1, 26, 1), minor=True)
ax1.grid(which='minor', alpha=0.4)
ax1.grid(which='major', alpha=1.0)

plt.savefig('../images/542-2.png', dpi=300)
plt.show()

i_e = 15
i_f = 27
cs_conv = curve_fit(reg.gauss, I[i_e:i_f], NmNf[i_e:i_f], sigma=s_NmNf[i_e:i_f])
# reg.plot_func(reg.gauss, I[i_e:i_f], cs_conv[0])
# plt.scatter(I, NmNf, s=5, color='black')

I_conv = cs_conv[0][1]
s_I_conv = np.sqrt(np.diag(cs_conv[1]))[1]

# plt.axvline(I_conv, color='black')
#
# plt.xlabel(r'$I,\ А$')
# plt.ylabel(r'$N-N_{ф},\ с^{-1}$')
# plt.xlabel(r'$I,\ А$')
# plt.ylabel(r'$N-N_{ф},\ с^{-1}$')
# plt.xticks(np.arange(2.8, 4, 0.2))
# plt.xticks(np.arange(2.8, 4, 0.05), minor=True)
# plt.yticks(np.arange(0, 26, 5))
# plt.yticks(np.arange(-1, 26, 1), minor=True)
# plt.grid(which='minor', alpha=0.4)
# plt.grid(which='major', alpha=1.0)
# plt.xlim(2.8, 3.8)
#
# plt.savefig('../images/542-4.png', dpi=300)
# plt.show()

p_conv = 1013.5
E0 = 511

k = p_conv / I_conv
s_k = k * s_I_conv / I_conv
print(k, s_k)
p = k * I
s_p = p * s_k / k
T = np.sqrt(p ** 2 + E0 ** 2) - E0
print(p)
print(T)
s_T = p / (T + E0) * s_p

# Тройной график

plt.subplots(layout='constrained')
ax1 = plt.gca()
ax1.errorbar(I, NmNf, yerr=s_NmNf, fmt='none', color='black')
ax1.scatter(I, NmNf, s=5, color='black')

ax1.set_xlabel(r'$I,\ А$')
ax1.set_ylabel(r'$N-N_{ф},\ с^{-1}$')
ax1.set_xlim(0, I[-1] * 1.05)

ax2 = ax1.twiny()
ax2.set_xlabel(r'$p_e,\ \frac{кэВ}{c}$')
ax2.set_xlim(0, p[-1] * 1.05)
ax2.spines['bottom'].set_position(('outward', 40))
ax2.set_xscale('function', functions=(lambda x: x / k, lambda x: x * k))
ax2.spines['bottom'].set_visible(True)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.xaxis.set_label_position('bottom')
ax2.xaxis.set_ticks_position('bottom')

ax3 = ax1.twiny()
ax3.set_xlabel(r'$T,\ кэВ$')
ax3.set_xlim(0, np.sqrt((p[-1] * 1.05) ** 2 + E0 ** 2) - E0)
ax3.spines['bottom'].set_position(('outward', 80))
ax3.set_xscale('function', functions=(lambda x: np.sqrt((x + E0) ** 2 - E0 ** 2),
                                      lambda x: np.sqrt(x ** 2 + E0 ** 2) - E0))
ax3.spines['bottom'].set_visible(True)
ax3.spines['top'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.xaxis.set_label_position('bottom')
ax3.xaxis.set_ticks_position('bottom')

ax1.set_xticks(np.arange(0, 5, 1))
ax1.set_xticks(np.arange(0, 4.4, 0.2), minor=True)
ax1.set_yticks(np.arange(0, 26, 5))
ax1.set_yticks(np.arange(-1, 26, 1), minor=True)
ax1.grid(which='minor', alpha=0.4)
ax1.grid(which='major', alpha=1.0)

plt.savefig('../images/542-6.png', dpi=300)
plt.show()

i_c = 5
i_d = -3

NmNf = NmNf[i_c:i_d]
p = p[i_c:i_d]
s_NmNf = s_NmNf[i_c:i_d]
s_p = s_p[i_c:i_d]
T = T[i_c:i_d]
s_T = s_T[i_c:i_d]

yFC = np.sqrt(NmNf) / p ** 1.5 * 1e6
s_yFC = yFC * np.sqrt((0.5 * s_NmNf / NmNf) ** 2 + (1.5 * s_p / p) ** 2)

i_a = 1
i_b = i_a + 9

# plt.errorbar(xFC, yFC, xerr=s_xFC, yerr=s_yFC, fmt='none', color='black')
# plt.scatter(xFC, yFC, s=5, color='black')
# cs = curve_fit(reg.lin1, xFC[i_a:i_b], yFC[i_a:i_b], maxfev=10000)
# T_e = cs[0][1]
# reg.plot_func(reg.lin1, [xFC[i_a - 1], T_e], cs[0])
# plt.scatter(T_e, 0, color='black', marker='x')
# print(T_e, np.sqrt(np.diag(cs[1]))[1])
#
# plt.xlabel(r'$T,\ кэВ$')
# plt.ylabel(r'$\sqrt{(N-N_{ф})/p^3}\cdot 10^6$')
# plt.xticks(np.arange(100, 800, 100))
# plt.xticks(np.arange(60, 740, 20), minor=True)
# plt.yticks(np.arange(0, 500, 50))
# plt.yticks(np.arange(0, 500, 25), minor=True)
# plt.grid(which='minor', alpha=0.4)
# plt.grid(which='major', alpha=1.0)
#
# plt.savefig('../images/542-3.png', dpi=300)
# plt.show()
