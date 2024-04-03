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

    obtezitev = 1/(negotovosti)
    obtezeno_povprecje = np.sum(vrednosti * obtezitev) / np.sum(obtezitev)

    obtezena_negotovost = np.sqrt(np.sum(negotovosti**2 * obtezitev**2) / (np.sum(obtezitev)**2))

    return unc.ufloat(obtezeno_povprecje, obtezena_negotovost)

############################################################################################################################

v = [16.648, 19.742, 32.591, 18.484, 18.883, 34.462, 14.854, 31.152, 20.799, 29.486, 16.893, 20.504, 45.501, 16.637, 28.688, 15.319, 17.365, 21.454, 27.607, 27.514, 21.609, 11.518, 34.431, 59.002, 20.478, 28.591, 12.475, 18.293, 16,226]
V = np.array([52, 66, 126, 48, 44, 68, 29, 74, 52, 111, 35, 54, 206, 56, 36, 54, 38, 50, 55, 99, 448, 42, 22, 146, 147, 61, 109, 19, 37, 48])


V_err = unp.uarray(V, np.ones(np.shape(V)) * 3)

rho = 973

d = unc.ufloat(5, 5*0.02) * 10**(-3)

ni = 18.3 * 10**(-6)
rho_z = 1.1929

r = np.sqrt((9 * ni * np.array(v) * 10**(-6)) / (2 * (rho - rho_z) * 9.81))


sez = []

for i in range(len(v)):
    ne = (4 * np.pi)/3 * ((9 * ni * v[i] * 10**(-6)) / (2 * (rho - rho_z) * 9.81))**(3/2) * (rho - rho_z) * 9.81 * d/V_err[i]
    sez.append(ne)
# print(sez)

n_e0 = []

print("$$$$$$$$$$$$$$$$$$$$$$", sez, "#####################")

sez = np.sort(np.array(sez))

sez = sez.tolist()

for i in sez[:-2]:
    if i.n < 1.0e-19:
        continue
    else:
        n_e0.append(i)

n_e0 = np.array(n_e0)
print("nnnnnnnnnn", n_e0, "nnnnnnn")


plt.scatter(np.ones(np.shape(unp.nominal_values(n_e0))), unp.nominal_values(n_e0))
plt.show()

# plt.hist(unp.nominal_values(n_e0))

# plt.show()

# print(n_e0 / 1.6e-19)

# print(np.average(unp.nominal_values(n_e0 / np.int16(unp.nominal_values(n_e0 / 1.6e-19)))))

# print(np.sort(n_e0))


j = 0
k = 0
temp = []

for i in np.linspace(0, np.max(unp.nominal_values(n_e0)) - 1e-20, 1000):
    if n_e0[k] < i:
        k += 1
        j += 1
        temp.append(j)
    else:
        temp.append(j)

plt.plot(np.linspace(0, np.max(unp.nominal_values(n_e0)), 1000), temp)
plt.xlabel(r'$e$ [A\,s]')
plt.ylabel('$N$')
plt.title('Kumulativni histogram števila kapljic,\nvečjih od $e$')
x_limits = plt.xlim()
y_limits = plt.ylim()
plt.ylim(y_limits)
plt.xlim(x_limits)

# plt.savefig('FP4/Milik/Slike/Histogram.png', dpi=300, bbox_inches="tight")
plt.show()


print(np.average(n_e0[-2:]), np.average(n_e0[-5:-2]), np.average(n_e0[-11:-5]), np.average(n_e0[:-11]))
print(np.average(n_e0[-2:]) - np.average(n_e0[-5:-2]), np.average(n_e0[-5:-2]) - np.average(n_e0[-11:-5]), np.average(n_e0[-11:-5]) - np.average(n_e0[:-11]))
print(np.average(np.array([np.average(n_e0[-2:]) - np.average(n_e0[-5:-2]), np.average(n_e0[-5:-2]) - np.average(n_e0[-11:-5]), np.average(n_e0[-11:-5]) - np.average(n_e0[:-11])])))

print(obtezeno_povprecje(np.array([np.average(n_e0[-2:]) - np.average(n_e0[-5:-2]), np.average(n_e0[-5:-2]) - np.average(n_e0[-11:-5]), np.average(n_e0[-11:-5]) - np.average(n_e0[:-11])])))

print(unc.ufloat(np.average(unp.nominal_values(n_e0)[-5:]), np.std(unp.nominal_values(n_e0)[-5:], ddof=1)) - unc.ufloat(np.average(unp.nominal_values(n_e0)[1:-5]), np.std(unp.nominal_values(n_e0)[1:-5], ddof=1)))
print(obtezeno_povprecje(n_e0[-5:]) - obtezeno_povprecje(n_e0[1:-5]))

# print(r.tolist())

e_01 = unc.ufloat(np.average(unp.nominal_values(n_e0[-2:])), np.std(unp.nominal_values(n_e0[-2:])))
e_02 = unc.ufloat(np.average(unp.nominal_values(n_e0[-7:-2])), np.std(unp.nominal_values(n_e0[-7:-2])))
e_03 = unc.ufloat(np.average(unp.nominal_values(n_e0[1:-7])), np.std(unp.nominal_values(n_e0[1:-7])))

print("#######", e_02 - e_01, e_03 - e_02, (e_03 - e_01) / 2)

print(obtezeno_povprecje(np.array([e_02 - e_01, e_03 - e_02, (e_03 - e_01) / 2])))