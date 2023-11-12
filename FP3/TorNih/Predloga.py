import scipy
from scipy import integrate
import scipy.optimize as opt
from scipy import stats
import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
import scipy.constants as const

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Serif",
    "font.size": 20
})

###########################################################

##### Seznami podatkov
x1 = np.array([640, 745, 1010, 1180, 1320, 1555, 1670, 1735, 1785, 1825])
y1 = np.array([0.24, 0.29, 0.49, 0.68, 0.87, 1.03, 1.19, 1.42, 1.54, 1.68])

##### Konstante, parametri in začetne vrednosti
r = 9.5 / 100 #m
d = 5.5 / 1000 #m

##### Definicije funkcij
def lin_fun(x, a, b):
    return a * x + b

##### Obdelava seznamov
x1 = x1 ** 2
y1 = (y1 / 1000) * const.g

##### Fitanje
slope, intercept, r_value, p_value1, std_err = stats.linregress(x1, y1)

optimizedParameters, pcov = opt.curve_fit(lin_fun, x1, y1)

##### Graf
plt.plot(np.concatenate((np.array([0]), x1), axis=None), lin_fun(np.concatenate((np.array([0]), x1), axis=None), *optimizedParameters),
         color="black", linestyle="dashed", label="Fit")
plt.plot(x1, y1, "o", label="Izmerjeno", color="#0033cc")

plt.xlabel('$U^2 [V^2]$')
plt.ylabel('$F [N]$')

xlim_spodnja = 0
xlim_zgornja = 3600000
ylim_spodnja = 0
ylim_zgornja = 0.018
plt.xlim(xlim_spodnja, xlim_zgornja)
plt.ylim(ylim_spodnja, ylim_zgornja)
# major_ticksx = np.arange(xlim_spodnja, xlim_zgornja + 0.0000001, (xlim_zgornja - xlim_spodnja)/6)
# minor_ticksx = np.arange(xlim_spodnja, xlim_zgornja + 0.0000001, (xlim_zgornja - xlim_spodnja)/60)
# major_ticksy = np.arange(ylim_spodnja, ylim_zgornja + 0.0000001, (ylim_zgornja - ylim_spodnja)/4)
# minor_ticksy = np.arange(ylim_spodnja, ylim_zgornja + 0.0000001, (ylim_zgornja - ylim_spodnja)/40)
# plt.xticks(major_ticksx)
# plt.xticks(minor_ticksx, minor=True)
# plt.yticks(major_ticksy)
# plt.yticks(minor_ticksy, minor=True)
# plt.grid(which='minor', alpha=0.2)
# plt.grid(which='major', alpha=0.5)
plt.legend()

# plt.show()
# plt.savefig("Vaja_47_graf_epsilon.png", dpi=300, bbox_inches="tight")

##### Izračuni
print(slope * 2 * d**2 / (np.pi * r**2), ", relativna napaka brez d in S: ", std_err / slope, slope)

napaka = 0.09 # Izračunana relativna napaka skupaj z d^2 in S