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

############################################################################################################################

def linearna_funkcija(x, a, b):
    return a * x + b

def linearna_funkcija_2(B, x):
    return B[0]*x + B[1]

def sigma(z_p, z_o):
    return (z_p**-1 + z_o**-1)**(-1)

z_p0 = unc.ufloat(80, 2) * 10**(-3)
z_o0 = unc.ufloat(830, 10) * 10**(-3)

lam = 633 * 10**(-9)


n = np.array([0, 1, 2, 3, 4, 5])
s = unp.uarray([94, 100.5, 111, 121, 131, 142.5], [0, 0.5, 0.5, 0.5, 0.5, 0.5]) * 10**(-3)
s = s - unc.ufloat(94, 0) * 10**(-3)

# print(s * 10**3)

sigma_list = []
for i in range(np.shape(n)[0]):
    sigma_list.append(sigma(z_p0 + s[i], z_o0 - s[i]))
    # print("zz", (z_p0 + s[i], z_o0 - s[i]))

sigme = np.array(sigma_list)
# print(sigme)

od_sigme = (lam * sigme)**(-1)


optimizedParameters, pcov = opt.curve_fit(linearna_funkcija, n, unp.nominal_values(od_sigme), sigma=unp.std_devs(od_sigme), absolute_sigma=True)

# print(optimizedParameters)
plt.plot(np.concatenate((np.array([0]), n), axis=None), linearna_funkcija(np.concatenate((np.array([0]), n), axis=None), *optimizedParameters),
          color="black", linestyle="dashed", label="Fit")
# plt.plot(x_raz, kvadratna_funkcija(x_raz, *optimizedParameters))
plt.errorbar(n, unp.nominal_values(od_sigme), yerr=unp.std_devs(od_sigme), linestyle='None', marker='.', capsize=3, label="Meritev")


plt.xlabel('$n - n_0$')
plt.ylabel('$1/\lambda \zeta$ [m$^{-2}$]')
# plt.title("Zaslon s petimi režami")
x_limits = plt.xlim()
y_limits = plt.ylim()
plt.ylim((y_limits))
plt.xlim(x_limits)
plt.legend()



# plt.savefig('FP4/UklSve/Radij.png', dpi=300, bbox_inches="tight")
plt.show()

r = ((unc.ufloat(np.abs(optimizedParameters[0]), np.sqrt(pcov[0, 0])))**(-1/2))
n_0 = ( - unc.ufloat(optimizedParameters[1], np.sqrt(pcov[1, 1])) / unc.ufloat(optimizedParameters[0], np.sqrt(pcov[0, 0])))

print("Premer okrogle odprtine:", r * 10**3, "mm")
print("Presečišče n_0:", n_0)