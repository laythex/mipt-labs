import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regressions as reg
from curva_fit import curva_fit

ch = np.arange(2048) + 1

co60 = np.array(pd.read_excel('555/co60.xlsx')[5:])[:, 1]
na22 = np.array(pd.read_excel('555/na22.xlsx')[5:])[:, 1]
cs137 = np.array(pd.read_excel('555/cs137.xlsx')[5:])[:, 1]
eu152 = np.array(pd.read_excel('555/eu152.xlsx')[5:])[:, 1]
am241 = np.array(pd.read_excel('555/am241.xlsx')[5:])[:, 1]
bg = np.array(pd.read_excel('555/bg.xlsx')[5:])[:, 1]

co60 -= bg
na22 -= bg
cs137 -= bg
eu152 -= bg
am241 -= bg

# ------------------------------ Общий план ------------------------------

size = 3
mark = 10

plt.plot(ch, co60, 'o', ls='', ms=size, label=r'$^{60}Co$', markevery=(0, mark))
plt.plot(ch, na22, 'o', ls='', ms=size, label=r'$^{22}Na$', markevery=(1, mark))
plt.plot(ch, cs137, 'o', ls='', ms=size, label=r'$^{137}Cs$', markevery=(2, mark))
plt.plot(ch, eu152, 'o', ls='', ms=size, label=r'$^{152}Eu$', markevery=(3, mark))
plt.plot(ch, am241, 'o', ls='', ms=size, label=r'$^{241}Am$', markevery=(4, mark))

plt.ylim(0, 350)
plt.grid()
plt.legend()

# plt.show()
plt.cla()

# ------------------------------ Поиск фотопиков ------------------------------

size = 5
width = 2

# ------------------------------ Co60 ------------------------------

p1 = np.arange(1490, 1690)
p2 = np.arange(1700, 1900)

cs1 = curva_fit(reg.gauss, ch[p1], co60[p1],
                maxfev=1000000, p0=(30, 1600, 100))
reg.plot_func(reg.gauss, ch[p1], cs1[0], lw=width)
cs2 = curva_fit(reg.gauss, ch[p2], co60[p2],
                maxfev=1000000, p0=(30, 1800, 100))
reg.plot_func(reg.gauss, ch[p2], cs2[0], lw=width)

N1_co60, N2_co60 = cs1[0][1], cs2[0][1]
sN1_co60, sN2_co60 = np.sqrt(np.diag(cs1[1]))[1], np.sqrt(np.diag(cs2[1]))[1]

plt.axvline(cs1[0][1], color='black', lw=width)
plt.axvline(cs2[0][1], color='black', lw=width)
plt.text(N1_co60 + 15, 46, fontsize=10,
         s=rf'$N_1={"{:.1f}".format(N1_co60)}\pm{"{:.1f}".format(sN1_co60)}$')
plt.text(N2_co60 + 15, 46, fontsize=10,
         s=rf'$N_1={"{:.1f}".format(N2_co60)}\pm{"{:.1f}".format(sN2_co60)}$')

plt.scatter(ch, co60, s=size, color='black')

plt.xlim(1400, 2000)
plt.ylim(0, 50)
plt.xlabel('Номер канала')
plt.ylabel('Скорость счета, отн. ед.')

plt.grid()
# plt.show()
plt.cla()

# ------------------------------ Na22 ------------------------------

p = np.arange(1600, 1800)

cs = curva_fit(reg.gauss, ch[p], na22[p],
               maxfev=1000000, p0=(10, 1700, 100))
reg.plot_func(reg.gauss, ch[p], cs[0], lw=width)

N_na22 = cs[0][1]
sN_na22 = np.sqrt(np.diag(cs[1]))[1]

plt.axvline(cs[0][1], color='black', lw=width)
plt.text(N_na22 + 15, 11, fontsize=10,
         s=rf'$N={"{:.1f}".format(N_na22)}\pm{"{:.1f}".format(sN_na22)}$')

plt.scatter(ch, na22, s=size, color='black')

plt.xlim(1400, 2000)
plt.ylim(-5, 15)
plt.xlabel('Номер канала')
plt.ylabel('Скорость счета, отн. ед.')

plt.grid()
# plt.show()
plt.cla()

# ------------------------------ Cs137 ------------------------------

p = np.arange(800, 1000)

cs = curva_fit(reg.gauss, ch[p], cs137[p],
               maxfev=1000000, p0=(150, 900, 100))
reg.plot_func(reg.gauss, ch[p], cs[0], lw=width)

N_cs137 = cs[0][1]
sN_cs137 = np.sqrt(np.diag(cs[1]))[1]

plt.axvline(cs[0][1], color='black', lw=width)
plt.text(N_cs137 + 15, 161, fontsize=10,
         s=rf'$N={"{:.1f}".format(N_cs137)}\pm{"{:.1f}".format(sN_cs137)}$')

plt.scatter(ch, cs137, s=size, color='black')

plt.xlim(750, 1050)
plt.ylim(-5, 175)
plt.xlabel('Номер канала')
plt.ylabel('Скорость счета, отн. ед.')

plt.grid()
# plt.show()
plt.cla()

# ------------------------------ Калибровка ------------------------------

E1_co60, E2_co60 = 1.1732, 1.3325
E_na22 = 1.274
E_cs137 = 0.6617

deltax = 0.03
deltay = 15
plt.text(E1_co60 + deltax / 1.5, N1_co60 - deltay, fontsize=10, s=r'$^{60}Co$')
plt.text(E2_co60 - deltax * 2, N2_co60 - deltay, fontsize=10, s=r'$^{60}Co$')
plt.text(E_na22 - deltax * 2, N_na22 - deltay, fontsize=10, s=r'$^{22}Na$')
plt.text(E_cs137 + deltax / 1.5, N_cs137 - deltay, fontsize=10, s=r'$^{137}Cs$')

Es = [E1_co60, E2_co60, E_na22, E_cs137]
Ns = [N1_co60, N2_co60, N_na22, N_cs137]
sNs = [sN1_co60, sN2_co60, sN_na22, sN_cs137]

size = 10

cs = curva_fit(reg.lin_inv, Es, Ns, sigma=sNs)
reg.plot_func(reg.lin_inv, Es, cs[0])
plt.scatter(Es, Ns, color='black', s=size)
plt.errorbar(Es, Ns, yerr=sNs, color='black', ls='')

a, b = cs[0] * 1e3
sa, sb = np.sqrt(np.diag(cs[1])) * 1e3
print(a, b)
print(sa, sb)

plt.text(0.705, 1550, fontsize=10, s=r'$E=aN+b$')
plt.text(0.705, 1500, fontsize=10,
         s=rf'$a=({"{:.4f}".format(a)}\pm{"{:.4f}".format(sa)})$ эВ')
plt.text(0.705, 1450, fontsize=10,
         s=rf'$b=({"{:.1f}".format(b)}\pm{"{:.1f}".format(sb)})$ эВ')

plt.xlabel(r'Энергия $\gamma$-кванта E, кэВ')
plt.ylabel('Номер канала N')

plt.grid()
# plt.show()
plt.cla()

# ------------------------------ Eu152 ------------------------------

p = np.arange(430, 560)

cs = curva_fit(reg.gauss, ch[p], eu152[p],
               maxfev=1000000, p0=(300, 500, 50))
reg.plot_func(reg.gauss, ch[p], cs[0], lw=width)

N_eu152 = cs[0][1]
sN_eu152 = np.sqrt(np.diag(cs[1]))[1]

plt.axvline(cs[0][1], color='black', lw=width)
plt.text(N_eu152 + 15, 351, fontsize=10,
         s=rf'$N={"{:.1f}".format(N_eu152)}\pm{"{:.1f}".format(sN_eu152)}$')

plt.scatter(ch, eu152, s=size, color='black')

plt.xlim(400, 600)
plt.ylim(0, 400)
plt.xlabel('Номер канала')
plt.ylabel('Скорость счета, отн. ед.')

plt.grid()
# plt.show()
plt.cla()

# ------------------------------ Am241 ------------------------------

p = np.arange(430, 560)

cs = curva_fit(reg.gauss, ch[p], eu152[p],
               maxfev=1000000, p0=(300, 500, 50))
reg.plot_func(reg.gauss, ch[p], cs[0], lw=width)

N_eu152 = cs[0][1]
sN_eu152 = np.sqrt(np.diag(cs[1]))[1]

plt.axvline(cs[0][1], color='black', lw=width)
plt.text(N_eu152 + 15, 351, fontsize=10,
         s=rf'$N={"{:.1f}".format(N_eu152)}\pm{"{:.1f}".format(sN_eu152)}$')

plt.scatter(ch, eu152, s=size, color='black')

plt.xlim(400, 600)
plt.ylim(0, 400)
plt.xlabel('Номер канала')
plt.ylabel('Скорость счета, отн. ед.')

plt.grid()
# plt.show()
plt.cla()

# ------------------------------ Am241 ------------------------------

p = np.arange(90, 160)

cs = curva_fit(reg.gauss, ch[p], am241[p],
               maxfev=1000000, p0=(1800, 125, 40))
reg.plot_func(reg.gauss, ch[p], cs[0], lw=width)

N_am241 = cs[0][1]
sN_am241 = np.sqrt(np.diag(cs[1]))[1]

plt.axvline(cs[0][1], color='black', lw=width)
plt.text(N_am241 + 5, 1800, fontsize=10,
         s=rf'$N={"{:.1f}".format(N_am241)}\pm{"{:.1f}".format(sN_am241)}$')

plt.scatter(ch, am241, s=size, color='black')

plt.xlim(50, 200)
plt.ylim(0, 1900)
plt.xlabel('Номер канала')
plt.ylabel('Скорость счета, отн. ед.')

plt.grid()
plt.show()
plt.cla()
