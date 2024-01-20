import scipy
from scipy import integrate
import scipy.optimize as opt
from scipy import stats
import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
import scipy.constants as const
import uncertainties as unc
import uncertainties.unumpy as unumpy
import uncertainties.umath as umath
import math
import pandas as pd
from scipy.odr import *
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Serif",
    "font.size": 20
})

###########################################################################################################
def kvadratna_funkcija(x, a, b):
    return a * x**(-2) + b


def linearna_funkcija(x, a, b):
    return a * x + b






##### Seznami podatkov

sun_b = unc.ufloat(315, np.sqrt(315))

akt_b = sun_b / (900 / 60)
print("Aktivnsot ozadja A_b:", akt_b, "min^-1")


d_1 = 1.2# * 10**(-2)
d_2 = 1.05# * 10**(-2)
t1 = 20 / 60
t2 = 50 / 60
t3 = 400 / 60


x_raz = np.array([d_1, d_1 + d_2, d_1 + d_2 + d_2, d_1 + d_2 + d_2 + d_2, d_1 + d_2 + d_2 + d_2 + d_2, d_1 + d_2 + d_2 + d_2 + d_2 + d_2, d_1 + d_2 + d_2 + d_2 + d_2 + d_2 + d_2, d_1 + d_2 + d_2 + d_2 + d_2 + d_2 + d_2 + d_2,
                  d_1 + d_2 + d_2 + d_2 + d_2 + d_2 + d_2 + d_2 + d_2, d_1 + d_2 + d_2 + d_2 + d_2 + d_2 + d_2 + d_2 + d_2 + d_2])
y_raz = np.array([1265, 751, 543, 311, 206, 200, 147, 112, 102, 85])


y_raz = unumpy.uarray(y_raz, np.sqrt(y_raz))
akt_raz = y_raz/t1 - akt_b

akt_raz_sqrt = 1 / (akt_raz)**(1/2)


optimizedParameters, pcov = opt.curve_fit(linearna_funkcija, x_raz, unumpy.nominal_values(akt_raz_sqrt), sigma=unumpy.std_devs(akt_raz_sqrt), absolute_sigma=True)

print(optimizedParameters)
plt.plot(np.concatenate((np.array([0]), x_raz), axis=None), linearna_funkcija(np.concatenate((np.array([0]), x_raz), axis=None), *optimizedParameters),
          color="black", linestyle="dashed", label="Fit")
# plt.plot(x_raz, kvadratna_funkcija(x_raz, *optimizedParameters))
plt.errorbar(x_raz, unumpy.nominal_values(akt_raz_sqrt), yerr=unumpy.std_devs(akt_raz_sqrt), linestyle='None', marker='.', capsize=3, label="Izmerjeno")

plt.ylabel('$1/\sqrt{A}$ [min$^{1/2}$]')
plt.xlabel('$r$ [cm]')
plt.title("$1/\sqrt{A}$ v odvisnosti od razdalje $r$")
plt.legend()

plt.savefig("GamaBeta_oddaljenost.png", dpi=300, bbox_inches="tight")
plt.show()

print("Vrednost r_{GM} iz fita: ", (1/(unc.ufloat(optimizedParameters[0], pcov[1, 1]))**2 / akt_raz[0])**(1/2), "cm")#optimizedParameters[1] * optimizedParameters[0])



A = 4.5
B = 6.5
G = 129
H = 161
I = 258
J = 258
A1 = 0.0095 * 2.70 * 1000
A2 = 0.009 * 2.70 * 1000
A3 = 0.005 * 2.70 * 1000
A4 = 0.0095 * 2.70 * 1000
A5 = 0.005 * 2.70 * 1000
Q = 0.8
R = 1.6
S = 3.2
T = 6.4




plos_dos_b = np.array([0, A, B, A+B, A + A1, B+A1, B+A1+A2, B+A1+A2+A3, B+A3, G, A+B+A1+A2+A3, B+A1+A2+A3+A4, A+B+A1+A2+A3+A4+A5, G+A, G+B, H, H+B, H+G, I, I+G, I+H, J, J+G, J+H])
sun_dos_b = np.array([1309, 1192, 1146, 1053, 715, 799, 513, 407, 888, 125, 438, 320, 220, 109, 116, 90, 66, 47, 61, 46, 54, 58, 76, 58])

sun_dos_b = unumpy.uarray(sun_dos_b, np.sqrt(sun_dos_b))

akt_dos_b = sun_dos_b / t2 - akt_b

akt_dos_b_ba = akt_dos_b - np.average(akt_dos_b[-7:])

print(np.average(akt_dos_b[-7:]), "min^-1")



plt.errorbar(plos_dos_b, unumpy.nominal_values(akt_dos_b), yerr=unumpy.std_devs(akt_dos_b), linestyle='None', marker='.', capsize=3, label="Izmerjeno")
# plt.errorbar(plos_dos_b[-7:], unumpy.nominal_values(akt_dos_b)[-7:], yerr=unumpy.std_devs(akt_dos_b)[-7:], linestyle='None', marker='.', capsize=3, label="Izmerjeno")
plt.plot(plos_dos_b, np.ones(np.shape(plos_dos_b)) * unc.nominal_value(np.average(akt_dos_b[-7:])), label="$\overline{A_{\gamma}}$")

plt.ylabel('$A$ [min$^{-1}$]')
plt.xlabel('$s$ [mg/cm$^2$]')
plt.title("Aktivnost v odvisnosti od debeline\nplasti aluminija")
plt.legend()

plt.savefig("GamaBeta_doseg_beta.png", dpi=300, bbox_inches="tight")
plt.show()

print("Doseg beta R_0: ", unc.ufloat(180, 20), "mg/cm^2")








plos_rzp = np.array([0, Q, R, R+Q, S, Q+S, S+R, T, T+Q, T+R, T+S])
sun_rzp = np.array([383, 299, 348, 350, 291, 242, 280, 249, 217, 236, 237])
sun_rzp = unumpy.uarray(sun_rzp, np.sqrt(sun_rzp))

akt_rzp = sun_rzp / t3 - akt_b



optimizedParameters2, pcov2 = opt.curve_fit(linearna_funkcija, plos_rzp, unumpy.nominal_values(unumpy.log(akt_rzp)), sigma=unumpy.std_devs(unumpy.log(akt_rzp)), absolute_sigma=True)

# print(optimizedParameters2)
plt.plot(np.concatenate((np.array([0]), plos_rzp), axis=None), linearna_funkcija(np.concatenate((np.array([0]), plos_rzp), axis=None), *optimizedParameters2),
          color="black", linestyle="dashed", label="Fit")

# plt.plot(plos_rzp, unumpy.nominal_values(unumpy.log(akt_rzp)))
plt.errorbar(plos_rzp, unumpy.nominal_values(unumpy.log(akt_rzp)), yerr=unumpy.std_devs(unumpy.log(akt_rzp)), linestyle='None', marker='.', capsize=3, label="Izmerjeno")

plt.ylabel('ln($A$) [min$^{-1}$]')
plt.xlabel('$d$ [mm]')
plt.title("Naravni logaritem aktivnosti v odvisnosti od\ndebeline plasti svinca")
plt.legend()

plt.savefig("GamaBeta_razpolovna.png", dpi=300, bbox_inches="tight")
plt.show()

# print(unumpy.log(akt_rzp))

print("Razpolovna debelina l_{1/2}", -np.log(2) / unc.ufloat(optimizedParameters2[0], np.sqrt(pcov2[0][0])), "cm")


# ##### Konstante, parametri in začetne vrednosti
# r = 9.5 / 100 #m
# d = 5.5 / 1000 #m

# ##### Definicije funkcij
# def lin_fun(x, a, b):
#     return a * x + b

# ##### Obdelava seznamov
# x1 = x1 ** 2
# y1 = (y1 / 1000) * const.g

# ##### Fitanje
# slope, intercept, r_value, p_value1, std_err = stats.linregress(x1, y1)

# optimizedParameters, pcov = opt.curve_fit(lin_fun, x1, y1)

# ##### Graf
# plt.plot(np.concatenate((np.array([0]), x1), axis=None), lin_fun(np.concatenate((np.array([0]), x1), axis=None), *optimizedParameters),
#          color="black", linestyle="dashed", label="Fit")
# plt.plot(x1, y1, "o", label="Izmerjeno", color="#0033cc")

# plt.xlabel('$U^2 [V^2]$')
# plt.ylabel('$F [N]$')

# xlim_spodnja = 0
# xlim_zgornja = 3600000
# ylim_spodnja = 0
# ylim_zgornja = 0.018
# plt.xlim(xlim_spodnja, xlim_zgornja)
# plt.ylim(ylim_spodnja, ylim_zgornja)
# # major_ticksx = np.arange(xlim_spodnja, xlim_zgornja + 0.0000001, (xlim_zgornja - xlim_spodnja)/6)
# # minor_ticksx = np.arange(xlim_spodnja, xlim_zgornja + 0.0000001, (xlim_zgornja - xlim_spodnja)/60)
# # major_ticksy = np.arange(ylim_spodnja, ylim_zgornja + 0.0000001, (ylim_zgornja - ylim_spodnja)/4)
# # minor_ticksy = np.arange(ylim_spodnja, ylim_zgornja + 0.0000001, (ylim_zgornja - ylim_spodnja)/40)
# # plt.xticks(major_ticksx)
# # plt.xticks(minor_ticksx, minor=True)
# # plt.yticks(major_ticksy)
# # plt.yticks(minor_ticksy, minor=True)
# # plt.grid(which='minor', alpha=0.2)
# # plt.grid(which='major', alpha=0.5)
# plt.legend()

# # plt.show()
# # plt.savefig("Vaja_47_graf_epsilon.png", dpi=300, bbox_inches="tight")

# ##### Izračuni
# print(slope * 2 * d**2 / (np.pi * r**2), ", relativna napaka brez d in S: ", std_err / slope, slope)

# napaka = 0.09 # Izračunana relativna napaka skupaj z d^2 in S