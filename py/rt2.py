import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


def Q(x):
    return 0.5 * (1 - erf(x / np.sqrt(2)))


def snr2mus(snr):
    return 10 ** (snr * 0.1)


def qerr(snr, m=1.0):
    return Q(np.sqrt(m * snr2mus(snr)))


def berr(snr):
    return 0.5 * np.exp(-snr2mus(snr) * 0.5)


size = 40

# -------------------- 1 --------------------

snr1 = np.arange(0, 9)
Pe1 = np.array([0.077, 0.055, 0.037, 0.023, 0.013,
                0.0062, 0.0023, 0.00072, 0.00016])
Pe1th = qerr(snr1, 2)

plt.scatter(snr1, Pe1, s=size, color='black', label='Эксперимент')
plt.scatter(snr1, Pe1th, s=size, color='black', facecolors='none', label='Теория')

plt.xlabel('Соотношение сигнал/шум')
plt.ylabel('Вероятность ошибки')
plt.legend()
plt.grid()
plt.savefig('../images/rt2-1.png', dpi=300)
# plt.show()
plt.cla()

# -------------------- 2 --------------------

snr2 = np.arange(2, 12)
Pe2 = np.array([0.11, 0.084, 0.060, 0.040, 0.024, 0.013,
                0.0061, 0.0024, 0.00078, 0.00017])
Pe2th = qerr(snr2)

plt.scatter(snr2, Pe2, s=size, color='black', label='Эксперимент')
plt.scatter(snr2, Pe2th, s=size, color='black', facecolors='none', label='Теория')

plt.xlabel('Соотношение сигнал/шум')
plt.ylabel('Вероятность ошибки')
plt.legend()
plt.grid()
plt.savefig('../images/rt2-2.png', dpi=300)
# plt.show()
plt.cla()

# -------------------- 3 --------------------

snr3 = np.arange(0, 10)
Pe3 = np.array([0.1582, 0.1290, 0.1036, 0.0783, 0.0567,
                0.0382, 0.0231, 0.0129, 0.0061, 0.0023])
Pe3nc = np.array([0.3458, 0.3010, 0.2552, 0.2074, 0.1600,
                  0.1147, 0.0767, 0.0451, 0.0231, 0.0102])
Pe3th = qerr(snr3)
Pe3nc_th = 0.5 * qerr(snr3) + berr(snr3)

plt.scatter(snr3, Pe3, s=size, color='black', label='$P_e$, Эксперимент')
plt.scatter(snr3, Pe3th, s=size, color='black', facecolors='none', label='$P_e$, Теория')

plt.scatter(snr3, Pe3nc, s=size, color='black', marker='^', label='$P_e^{[nc]}$, Эксперимент')
plt.scatter(snr3, Pe3nc_th, s=size, color='black', marker='^', facecolors='none', label='$P_e^{[nc]}$, Теория')

plt.xlabel('Соотношение сигнал/шум')
plt.ylabel('Вероятность ошибки')
plt.legend()
plt.grid()
plt.savefig('../images/rt2-3.png', dpi=300)
# plt.show()
plt.cla()

# -------------------- 4 --------------------

snr4 = np.arange(3, 13)
Pe4 = np.array([0.152, 0.111, 0.075, 0.046, 0.025,
                0.012, 0.0047, 0.0015, 0.0004, 0.000068])
Pb4 = np.array([0.079, 0.057, 0.038, 0.023, 0.012,
                0.006, 0.00024, 0.0007, 0.0002, 0.000035])
Pb4s = np.array([0.112, 0.082, 0.056, 0.034, 0.018,
                 0.0085, 0.0033, 0.0012, 0.0003, 0.00006])
Pe4th = 2 * qerr(snr4) - qerr(snr4) ** 2
Pb4th = qerr(snr4)
Pb4s_th = 1.5 * qerr(snr4) - qerr(snr4) ** 2

plt.scatter(snr4, Pe4, s=size, color='black', label='$P_e$, Эксперимент')
plt.scatter(snr4, Pe4th, s=size, color='black', facecolors='none', label='$P_e$, Теория')

plt.scatter(snr4, Pb4, s=size, color='black', marker='^', label='$P_{b,\ Gray}$, Эксперимент')
plt.scatter(snr4, Pb4th, s=size, color='black', marker='^', facecolors='none', label='$P_{b,\ Gray}$, Теория')

plt.scatter(snr4, Pb4s, s=size, color='black', marker='s', label='$P_{b,\ Binary}$, Эксперимент')
plt.scatter(snr4, Pb4s_th, s=size, color='black', marker='s', facecolors='none', label='$P_{b,\ Binary}$, Теория')

plt.xlabel('Соотношение сигнал/шум')
plt.ylabel('Вероятность ошибки')
plt.legend()
plt.grid()
plt.savefig('../images/rt2-4.png', dpi=300)
# plt.show()
plt.cla()

# -------------------- 5 --------------------

snr5 = np.arange(12, 23)
Pe5p8 = np.array([0.031, 0.015, 0.0065, 0.0024, 0.0006,
                  0.00016, 0.00004, 0, 0, 0, 0])
Pe5a8 = np.array([0.192, 0.147, 0.106, 0.073, 0.045, 0.026,
                  0.013, 0.0052, 0.0018, 0.0004, 0.0001])
Pe5a16 = np.array([0.509, 0.462, 0.412, 0.362, 0.311, 0.262,
                   0.210, 0.163, 0.117, 0.080, 0.050])
Pe5q16 = np.array([0.111, 0.068, 0.037, 0.017, 0.007,
                   0.0024, 0.0006, 0.0014, 0, 0, 0])
mp = 2 * (np.sin(np.pi / 8)) ** 2
ma = 6 / 63
Pe5p8th = 2 * qerr(snr5, mp)
Pe5a8th = 1.75 * qerr(snr5, ma)
mq = 0.2
ma = 6 / 225
Pe5a16th = 1.875 * qerr(snr5, ma)
Pe5q16th = 3 * qerr(snr5, mq) - 2.25 * qerr(snr5, mq) ** 2

plt.scatter(snr5, Pe5p8, s=size, color='black', label='$P_{e,\ 8PSK}$, Эксперимент')
plt.scatter(snr5, Pe5p8th, s=size, color='black', facecolors='none', label='$P_{e,\ 8PSK}$, Теория')

plt.scatter(snr5, Pe5a8, s=size, color='black', marker='^', label='$P_{e,\ 8ASK}$, Эксперимент')
plt.scatter(snr5, Pe5a8th, s=size, color='black', marker='^', facecolors='none', label='$P_{e,\ 8ASK}$, Теория')

plt.scatter(snr5, Pe5a16, s=size, color='black', marker='s', label='$P_{e,\ 16ASK}$, Эксперимент')
plt.scatter(snr5, Pe5a16th, s=size, color='black', marker='s', facecolors='none', label='$P_{e,\ 16ASK}$, Теория')

plt.scatter(snr5, Pe5q16, s=size, color='black', marker='d', label='$P_{e,\ 16QSK}$, Эксперимент')
plt.scatter(snr5, Pe5q16th, s=size, color='black', marker='d', facecolors='none', label='$P_{e,\ 16QSK}$, Теория')

plt.xlabel('Соотношение сигнал/шум')
plt.ylabel('Вероятность ошибки')
plt.legend()
plt.grid()
plt.savefig('../images/rt2-5.png', dpi=300)
# plt.show()
plt.cla()

# -------------------- 8 --------------------

snr8 = np.arange(0, 9)
Pe8c = np.array([0.16, 0.13, 0.1, 0.079, 0.057,
                 0.038, 0.023, 0.013, 0.006])
Pe8nc = np.array([0.3, 0.27, 0.23, 0.18, 0.14,
                  0.1, 0.068, 0.041, 0.021])
Pe8c_th = qerr(snr8)
Pe8nc_th = berr(snr8)

plt.scatter(snr8, Pe8c, s=size, color='black', label='$P_e^{[c]}$, Эксперимент')
plt.scatter(snr8, Pe8c_th, s=size, color='black', facecolors='none', label='$P_e^{[c]}$, Теория')

plt.scatter(snr8, Pe8nc, s=size, color='black', marker='^', label='$P_e^{[nc]}$, Эксперимент')
plt.scatter(snr8, Pe8nc_th, s=size, color='black', marker='^', facecolors='none', label='$P_e^{[nc]}$, Теория')

plt.xlabel('Соотношение сигнал/шум')
plt.ylabel('Вероятность ошибки')
plt.legend()
plt.grid()
plt.savefig('../images/rt2-8.png', dpi=300)
plt.show()
plt.cla()