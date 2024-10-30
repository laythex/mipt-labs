import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regressions as reg
from kurwa_fit import kurwa_fit

ch = np.arange(2048) + 1

co60 = np.array(pd.read_excel('555/co60.xlsx')[5:])[:, 1]
na22 = np.array(pd.read_excel('555/na22.xlsx')[5:])[:, 1]
cs137 = np.array(pd.read_excel('555/cs137.xlsx')[5:])[:, 1]
eu152 = np.array(pd.read_excel('555/eu152.xlsx')[5:])[:, 1]
am241 = np.array(pd.read_excel('555/am241.xlsx')[5:])[:, 1]
bg = np.array(pd.read_excel('555/bg.xlsx')[5:])[:, 1]

# co60 -= bg
# na22 -= bg
# cs137 -= bg
# eu152 -= bg
# am241 -= bg

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

plt.savefig('../images/555-common.png', dpi=300)
# plt.show()
plt.cla()

plt.plot(ch, bg, 'o', ls='', ms=size, color='black')
plt.grid()
plt.savefig('../images/555-bg.png', dpi=300)
# plt.show()
plt.cla()

# ------------------------------ Поиск фотопиков ------------------------------

size = 1
width = 2
fwhm = 2 * np.sqrt(2 * np.log(2))

# ------------------------------ Co60 ------------------------------

p1 = np.arange(1490, 1690)
p2 = np.arange(1700, 1900)

cs1 = kurwa_fit(reg.gauss, ch[p1], co60[p1],
                maxfev=1000000, p0=(30, 1600, 100))
reg.plot_func(reg.gauss, ch[p1], cs1[0], lw=width)
cs2 = kurwa_fit(reg.gauss, ch[p2], co60[p2],
                maxfev=1000000, p0=(30, 1800, 100))
reg.plot_func(reg.gauss, ch[p2], cs2[0], lw=width)

sig1, sig2 = np.sqrt(np.diag(cs1[1])), np.sqrt(np.diag(cs2[1]))

N1_co60, N2_co60 = cs1[0][1], cs2[0][1]
sN1_co60, sN2_co60 = sig1[1], sig2[1]

dN1_co60, dN2_co60 = cs1[0][2] * fwhm, cs2[0][2] * fwhm
sdN1_co60, sdN2_co60 = sig1[2] * fwhm, sig2 * fwhm

plt.axvline(N1_co60, color='black', lw=width)
plt.axvline(N2_co60, color='black', lw=width)
plt.text(1000, 66, fontsize=10,
         s=rf'$N_1={"{:.1f}".format(N1_co60)}\pm{"{:.1f}".format(sN1_co60)}$')
plt.text(1000, 46, fontsize=10,
         s=rf'$N_2={"{:.1f}".format(N2_co60)}\pm{"{:.1f}".format(sN2_co60)}$')

plt.scatter(ch, co60, s=size, color='black')

# plt.xlim(1400, 2000)
# plt.ylim(0, 50)
plt.xlabel('Номер канала')
plt.ylabel('Скорость счета, отн. ед.')
plt.grid()

plt.savefig('../images/555-co60.png', dpi=300)
# plt.show()
plt.cla()

# ------------------------------ Na22 ------------------------------

p1 = np.arange(1600, 1800)
p2 = np.arange(650, 800)

cs1 = kurwa_fit(reg.gauss, ch[p1], na22[p1],
                maxfev=1000000, p0=(10, 1700, 100))
reg.plot_func(reg.gauss, ch[p1], cs1[0], lw=width)
cs2 = kurwa_fit(reg.gauss, ch[p2], na22[p2],
                maxfev=1000000, p0=(60, 715, 50))
reg.plot_func(reg.gauss, ch[p2], cs2[0], lw=width)

sig1, sig2 = np.sqrt(np.diag(cs1[1])), np.sqrt(np.diag(cs2[1]))

N1_na22, N2_na22 = cs1[0][1], cs2[0][1]
sN1_na22, sN2_na22 = sig1[1], sig2[1]

dN1_na22, dN2_na22 = cs1[0][2] * fwhm, cs2[0][2] * fwhm
sdN1_na22, sdN2_na22 = sig1[2] * fwhm, sig2[2] * fwhm

plt.axvline(N1_na22, color='black', lw=width)
plt.text(1000, 61, fontsize=10,
         s=rf'$N_1={"{:.1f}".format(N1_na22)}\pm{"{:.1f}".format(sN1_na22)}$')
plt.axvline(N2_na22, color='black', lw=width)
plt.text(1000, 71, fontsize=10,
         s=rf'$N_2={"{:.1f}".format(N2_na22)}\pm{"{:.1f}".format(sN2_na22)}$')

plt.scatter(ch, na22, s=size, color='black')

# plt.xlim(1400, 2000)
# plt.ylim(-5, 15)
plt.xlabel('Номер канала')
plt.ylabel('Скорость счета, отн. ед.')
plt.grid()

plt.savefig('../images/555-na22.png', dpi=300)
# plt.show()
plt.cla()

# ------------------------------ Cs137 ------------------------------

p = np.arange(800, 1000)

cs = kurwa_fit(reg.gauss, ch[p], cs137[p],
               maxfev=1000000, p0=(150, 900, 100))
reg.plot_func(reg.gauss, ch[p], cs[0], lw=width)

sig = np.sqrt(np.diag(cs[1]))

N_cs137 = cs[0][1]
sN_cs137 = sig[1]

dN_cs137 = cs[0][2] * fwhm
sdN_cs137 = sig[2] * fwhm

plt.axvline(N_cs137, color='black', lw=width)
plt.text(1000, 161, fontsize=10,
         s=rf'$N={"{:.1f}".format(N_cs137)}\pm{"{:.1f}".format(sN_cs137)}$')

plt.scatter(ch, cs137, s=size, color='black')

# plt.xlim(750, 1050)
# plt.ylim(-5, 175)
plt.xlabel('Номер канала')
plt.ylabel('Скорость счета, отн. ед.')
plt.grid()

plt.savefig('../images/555-cs137.png', dpi=300)
# plt.show()
plt.cla()

# ------------------------------ Калибровка ------------------------------

E1_co60, E2_co60 = 1.1732, 1.3325
E1_na22, E2_na22 = 1.274, 0.511
E_cs137 = 0.6617

deltax = 0.03
deltay = 15
plt.text(E1_co60 + deltax / 1.5, N1_co60 - deltay, fontsize=10, s=r'$^{60}Co$')
plt.text(E2_co60 - deltax * 2.5, N2_co60 - deltay, fontsize=10, s=r'$^{60}Co$')
plt.text(E1_na22 - deltax * 2.5, N1_na22 - deltay, fontsize=10, s=r'$^{22}Na$')
plt.text(E2_na22 + deltax, N2_na22 - deltay, fontsize=10, s=r'$^{22}Na$')
plt.text(E_cs137 + deltax / 1.5, N_cs137 - deltay, fontsize=10, s=r'$^{137}Cs$')

Es = [E1_co60, E2_co60, E1_na22, E2_na22, E_cs137]
Ns = [N1_co60, N2_co60, N1_na22, N2_na22, N_cs137]
sNs = [sN1_co60, sN2_co60, sN1_na22, sN2_na22, sN_cs137]

size = 10

cs = kurwa_fit(reg.lin, Es, Ns, sigma=sNs)
reg.plot_func(reg.lin, Es, cs[0])
plt.scatter(Es, Ns, color='black', s=size)
plt.errorbar(Es, Ns, yerr=sNs, color='black', ls='')

a, b = cs[0]
sa, sb = np.sqrt(np.diag(cs[1]))
# print(a, b)
# print(sa, sb)

cs = kurwa_fit(reg.lin_inv, Es, Ns, sigma=sNs)
a1, b1 = cs[0]
# print(a1, b1)

plt.text(0.705, 1550, fontsize=10, s=r'$N=aE+b$')
plt.text(0.705, 1500, fontsize=10,
         s=rf'$a=({"{:.0f}".format(a)}\pm{"{:.0f}".format(sa)})$ 1/МэВ')
plt.text(0.705, 1450, fontsize=10,
         s=rf'$b=({"{:.0f}".format(b)}\pm{"{:.0f}".format(sb)})$')

plt.xlabel(r'Энергия $\gamma$-кванта E, МэВ')
plt.ylabel('Номер канала N')
plt.grid()

plt.savefig('../images/555-cal.png', dpi=300)
# plt.show()
plt.cla()

# ------------------------------ Eu152 ------------------------------

p1 = np.arange(430, 560)
p2 = np.arange(40, 130)
p3 = np.arange(180, 240)
p4 = np.arange(320, 420)

cs1 = kurwa_fit(reg.gauss, ch[p1], eu152[p1],
                maxfev=1000000, p0=(300, 500, 50))
cs2 = kurwa_fit(reg.gauss, ch[p2], eu152[p2],
                maxfev=1000000, p0=(2500, 100, 20))
cs3 = kurwa_fit(reg.gauss, ch[p3], eu152[p3],
                maxfev=1000000, p0=(1500, 210, 30))
cs4 = kurwa_fit(reg.gauss, ch[p4], eu152[p4],
                maxfev=1000000, p0=(100, 360, 50))

reg.plot_func(reg.gauss, ch[p1], cs1[0], lw=width)
reg.plot_func(reg.gauss, ch[p2], cs2[0], lw=width)
reg.plot_func(reg.gauss, ch[p3], cs3[0], lw=width)
reg.plot_func(reg.gauss, ch[p4], cs4[0], lw=width)

sig1, sig2, sig3, sig4 = \
    (np.sqrt(np.diag(cs1[1])), np.sqrt(np.diag(cs2[1])), np.sqrt(np.diag(cs3[1])), np.sqrt(np.diag(cs4[1])))

N1_eu152, N2_eu152, N3_eu152, N4_eu152 = cs1[0][1], cs2[0][1], cs3[0][1], cs4[0][1]
sN1_eu152, sN2_eu152, sN3_eu152, sN4_eu152 = sig1[1], sig2[1], sig3[1], sig4[1]

dN1_eu152, dN2_eu152, dN3_eu152, dN4_eu152 = \
    cs1[0][2] * fwhm, cs2[0][2] * fwhm, cs3[0][2] * fwhm, cs4[0][2] * fwhm
sdN1_eu152, sdN2_eu152, sdN3_eu152, sdN4_eu152 = \
    sig1[2] * fwhm, sig2[2] * fwhm, sig3[2] * fwhm, sig4[2] * fwhm

plt.axvline(N1_eu152, color='black', lw=width)
plt.text(500, 1301, fontsize=10,
         s=rf'$N_1={"{:.1f}".format(N1_eu152)}\pm{"{:.1f}".format(sN1_eu152)}$')
plt.axvline(N2_eu152, color='black', lw=width)
plt.text(500, 1901, fontsize=10,
         s=rf'$N_2={"{:.1f}".format(N2_eu152)}\pm{"{:.1f}".format(sN2_eu152)}$')
plt.axvline(N3_eu152, color='black', lw=width)
plt.text(500, 1701, fontsize=10,
         s=rf'$N_3={"{:.1f}".format(N3_eu152)}\pm{"{:.1f}".format(sN3_eu152)}$')
plt.axvline(N4_eu152, color='black', lw=width)
plt.text(500, 1501, fontsize=10,
         s=rf'$N_4={"{:.1f}".format(N4_eu152)}\pm{"{:.1f}".format(sN4_eu152)}$')

plt.scatter(ch, eu152, s=size, color='black')

plt.xlim(0, 700)
# plt.ylim(0, 400)
plt.xlabel('Номер канала')
plt.ylabel('Скорость счета, отн. ед.')
plt.grid()

plt.savefig('../images/555-eu152.png', dpi=300)
# plt.show()
plt.cla()

# ------------------------------ Am241 ------------------------------

p1 = np.arange(90, 160)
p2 = np.arange(50, 90)

cs1 = kurwa_fit(reg.gauss, ch[p1], am241[p1],
                maxfev=1000000, p0=(1800, 125, 40))
cs2 = kurwa_fit(reg.gauss, ch[p2], am241[p2],
                maxfev=1000000, p0=(200, 75, 20))

reg.plot_func(reg.gauss, ch[p1], cs1[0], lw=width)
reg.plot_func(reg.gauss, ch[p2], cs2[0], lw=width)

sig1, sig2 = np.sqrt(np.diag(cs1[1])), np.sqrt(np.diag(cs2[1]))

N1_am241, N2_am241 = cs1[0][1], cs2[0][1]
sN1_am241, sN2_am241 = sig1[1], sig2[2]

dN1_am241, dN2_am241 = cs1[0][2] * fwhm, cs2[0][2] * fwhm
sdN1_am241, sdN2_am241 = sig1[2] * fwhm, sig2[2] * fwhm

plt.axvline(N1_am241, color='black', lw=width)
plt.text(200, 1510, fontsize=10,
         s=rf'$N_1={"{:.1f}".format(N1_am241)}\pm{"{:.1f}".format(sN1_am241)}$')
plt.axvline(N2_am241, color='black', lw=width)
plt.text(200, 1610, fontsize=10,
         s=rf'$N_2={"{:.1f}".format(N2_am241)}\pm{"{:.1f}".format(sN2_am241)}$')

plt.scatter(ch, am241, s=size, color='black')

plt.xlim(0, 500)
# plt.ylim(0, 1900)
plt.xlabel('Номер канала')
plt.ylabel('Скорость счета, отн. ед.')
plt.grid()

plt.savefig('../images/555-am241.png', dpi=300)
# plt.show()
plt.cla()

# ------------------------------ Таблица ------------------------------

Ns = np.array([N1_co60, N2_co60, N1_na22, N2_na22, N_cs137,
               N1_eu152, N2_eu152, N3_eu152, N4_eu152,
               N1_am241, N2_am241])
dNs = np.array([dN1_co60, dN2_co60, dN1_na22, dN2_na22, dN_cs137,
                dN1_eu152, dN2_eu152, dN3_eu152, dN4_eu152,
                dN1_am241, dN2_am241])

Es = a1 * Ns + b1
dEs = a1 * dNs
Rs = dEs / Es

# ------------------------------ График R ------------------------------

p = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
X = 1 / Es
Y = Rs ** 2

cs = kurwa_fit(reg.lin0, X[p], Y[p])
reg.plot_func(reg.lin0, X[p], cs[0])
plt.scatter(X, Y, color='black', s=size)
print(cs[0][0], np.sqrt(cs[1]))

plt.xlabel(r'$1/E,\ \text{МэВ}^{-1}$')
plt.ylabel('$R^2$')
plt.grid()

plt.savefig('../images/555-R.png', dpi=300)
# plt.show()
plt.cla()

# ------------------------------ Обратное рассеяние ------------------------------

p = np.arange(250, 450)
cs = kurwa_fit(reg.gauss, ch[p], co60[p] - bg[p],
               maxfev=1000000, p0=(40, 350, 50))
reg.plot_func(reg.gauss, ch[p], cs[0], lw=width)
Nbs_co60 = cs[0][1]
plt.scatter(ch, co60 - bg, color='black', s=1)
plt.axvline(Nbs_co60, color='black')
print(Nbs_co60 * a1 + b1)

plt.xlabel('Номер канала')
plt.ylabel('Скорость счета, отн. ед.')
plt.grid()

plt.savefig('../images/555-co60bs.png', dpi=300)
plt.show()
plt.cla()

p = np.arange(250, 450)
cs = kurwa_fit(reg.gauss, ch[p], cs137[p] - bg[p],
               maxfev=1000000, p0=(40, 350, 50))
reg.plot_func(reg.gauss, ch[p], cs[0], lw=width)
Nbs_cs137 = cs[0][1]
plt.scatter(ch, cs137 - bg, color='black', s=1)
plt.axvline(Nbs_cs137, color='black')
print(Nbs_cs137 * a1 + b1)

plt.xlabel('Номер канала')
plt.ylabel('Скорость счета, отн. ед.')
plt.grid()

plt.savefig('../images/555-cs137bs.png', dpi=300)
plt.show()
plt.cla()

# ------------------------------ Край Комптона ------------------------------

Emc2 = 0.511

Em_co60 = E2_co60 / (1 + Emc2 / E2_co60 / 2)
Nm_co60 = a * Em_co60 + b
Em_na22 = E1_na22 / (1 + Emc2 / E1_na22 / 2)
Nm_na22 = a * Em_na22 + b
Em_cs137 = E_cs137 / (1 + Emc2 / E_cs137 / 2)
Nm_cs137 = a * Em_cs137 + b

# plt.scatter(ch, co60, s=size, color='black')
# plt.axvline(Nm_co60, color='black')
# plt.xlabel('Номер канала')
# plt.ylabel('Скорость счета, отн. ед.')
# plt.grid()
# plt.savefig('../images/555-comp-co60.png', dpi=300)
# plt.show()
# plt.cla()
#
# plt.scatter(ch, na22, s=size, color='black')
# plt.axvline(Nm_na22, color='black')
# plt.xlabel('Номер канала')
# plt.ylabel('Скорость счета, отн. ед.')
# plt.grid()
# plt.savefig('../images/555-comp-na22.png', dpi=300)
# plt.show()
# plt.cla()
#
# plt.scatter(ch, cs137, s=size, color='black')
# plt.axvline(Nm_cs137, color='black')
# plt.xlabel('Номер канала')
# plt.ylabel('Скорость счета, отн. ед.')
# plt.grid()
# plt.savefig('../images/555-comp-cs137.png', dpi=300)
# plt.show()
# plt.cla()
