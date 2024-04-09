import scipy
from scipy import integrate
import scipy.optimize as opt
from scipy import stats
import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
import scipy.constants as const
import uncertainties as unc
import uncertainties.unumpy as unp
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


def obtezeno_povprecje(uarray):
    vrednosti = unp.nominal_values(uarray)
    negotovosti = unp.std_devs(uarray)

    obtezitev = 1/(negotovosti**2)
    obtezeno_povprecje = np.sum(vrednosti * obtezitev) / np.sum(obtezitev)

    obtezena_negotovost = np.sqrt(np.sum(negotovosti**2 * obtezitev**2) / (np.sum(obtezitev)**2))

    return unc.ufloat(obtezeno_povprecje, obtezena_negotovost)

#####################################################################################################################################################
def lin_fun(x, a, b):
    return a * x + b




#####################################################################################################################################################

# with open("/FP4/Boltz/Podatki/T_14.txt", "r") as dat:

a1 = np.genfromtxt(r"FP4\Boltz\Podatki\T_14_1.txt", delimiter='\t')

# print(a.T)

x1 = a1.T[0]
y1 = np.log(a1.T[1])

##### Fitanje
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x1, y1)

optimizedParameters1, pcov1 = opt.curve_fit(lin_fun, x1, y1)#, sigma=unumpy.std_devs(akt_raz_sqrt), absolute_sigma=True)

# print(optimizedParameters)
plt.plot([0.3, 0.8], lin_fun(np.array([0.3, 0.8]), *optimizedParameters1),
          color="black", linestyle="dashed")#, label="Fit")
# plt.plot(np.concatenate((np.array([0]), x1), axis=None), lin_fun(np.concatenate((np.array([0]), x1), axis=None), *optimizedParameters1),
#           color="black", linestyle="dashed")#, label="Fit")
plt.plot(x1, y1, "o", label="$14{,}1\ ^{\circ}$C")

# plt.show()

kb_Te1 = unc.ufloat(optimizedParameters1[0], np.sqrt(pcov1[0][0]))
kb1 = (kb_Te1 * ((273 + 14.1) / 1.6e-19))**(-1)
print(kb1)




####################################################################################################################################



a2 = np.genfromtxt(r"FP4\Boltz\Podatki\T_34_1.txt", delimiter='\t')

# print(a.T)

x2 = a2.T[0]
y2 = np.log(a2.T[1])

##### Fitanje
slope2, intercept2, r_value2, p_value12, std_err2 = stats.linregress(x2, y2)

optimizedParameters2, pcov2 = opt.curve_fit(lin_fun, x2, y2)#, sigma=unumpy.std_devs(akt_raz_sqrt), absolute_sigma=True)

# print(optimizedParameters)
plt.plot([0.3, 0.8], lin_fun(np.array([0.3, 0.8]), *optimizedParameters2),
          color="black", linestyle="dashed")#, label="Fit")
# plt.plot(np.concatenate((np.array([0]), x2), axis=None), lin_fun(np.concatenate((np.array([0]), x2), axis=None), *optimizedParameters2),
#           color="black", linestyle="dashed")#, label="Fit")
plt.plot(x2, y2, "o", label="$34{,}1\ ^{\circ}$C")

# plt.show()

kb_Te2 = unc.ufloat(optimizedParameters2[0], np.sqrt(pcov2[0][0]))
kb2 = (kb_Te2 * ((273 + 34.1) / 1.6e-19))**(-1)
print(kb2)



####################################################################################################################################



a3 = np.genfromtxt(r"FP4\Boltz\Podatki\T_53_2.txt", delimiter='\t')

# print(a.T)

x3 = a3.T[0]
y3 = np.log(a3.T[1])

##### Fitanje
slope3, intercept3, r_value3, p_value13, std_err3 = stats.linregress(x3, y3)

optimizedParameters3, pcov3 = opt.curve_fit(lin_fun, x3, y3)#, sigma=unumpy.std_devs(akt_raz_sqrt), absolute_sigma=True)

# print(optimizedParameters)
plt.plot([0.3, 0.8], lin_fun(np.array([0.3, 0.8]), *optimizedParameters3),
          color="black", linestyle="dashed")#, label="Fit")
# plt.plot(np.concatenate((np.array([0]), x3), axis=None), lin_fun(np.concatenate((np.array([0]), x3), axis=None), *optimizedParameters3),
#           color="black", linestyle="dashed")#, label="Fit")
plt.plot(x3, y3, "o", label="$53{,}2\ ^{\circ}$C")

plt.legend()
plt.xlim(0.36, 0.62)
plt.ylim(-18, -5)
plt.ylabel('$\ln(I_C / I_1)$')
plt.xlabel('$U_{BE}$ [V]')
plt.title("Aktivnost v odvisnosti od debeline\nplasti aluminija")



plt.show()

kb_Te3 = unc.ufloat(optimizedParameters3[0], np.sqrt(pcov3[0][0]))
kb3 = (kb_Te3 * ((273 + 53.2) / 1.6e-19))**(-1)
print(kb3)

####################################################################################################################################


print(obtezeno_povprecje([kb1, kb2, kb3]))








################################################################################################################################################

a4 = np.genfromtxt(r"FP4\Boltz\Podatki\U_0_5.txt", delimiter='\t')

x4 = a4.T[0]
# y4 = np.log(a4.T[1])
y4 = a4.T[1]


a5 = np.genfromtxt(r"FP4\Boltz\Podatki\U_0_58.txt", delimiter='\t')

x5 = a5.T[0]
# y4 = np.log(a4.T[1])
y5 = a5.T[1]

plt.plot(x4, y4)
plt.plot(x5, y5)

plt.show()


plt.plot(x4, np.log(y4))
plt.plot(x5, np.log(y5))
plt.show()