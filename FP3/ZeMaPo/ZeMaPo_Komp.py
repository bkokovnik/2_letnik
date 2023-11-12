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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Serif",
    "font.size": 18
})

###########################################################


##### Prvi set podatkov #####################################################################################################################################################

##### Seznami podatkov
Ux1 = np.array([10, 15, 20, 30, -10, -15, -20, -30]) #deg
Uy1 = np.array([184.2, 181.0, 183.8, 179.0, 188.6, 188.2, 188.6, 200.2]) #mA


##### Konstante, parametri in začetne vrednosti
N = 60
d = unc.ufloat(12.9, 0.2) * 10**(-2)   #m
l = unc.ufloat(60.0, 0.5) * 10**(-2)   #m
r = d/2

##### Definicije funkcij
def fit_fun1(x, a, b):
    return a * x + b


##### Obdelava seznamov
Ux1_err = unumpy.uarray(Ux1, np.ones(np.shape(Ux1)) * 0.5)
Uy1_err = unumpy.uarray(Uy1, np.ones(np.shape(Ux1)) * 5) * 10**(-3)

x1 = unumpy.nominal_values(Ux1_err)
y1 = (Uy1_err * N * const.mu_0) / (umath.sqrt(l**2 + (2*r)**2))
print(y1)


# Sample data as a uarray with values and absolute uncertainties
values = unumpy.nominal_values(y1)
abs_uncertainties = unumpy.std_devs(y1)

# Calculate relative uncertainties
relative_uncertainties = abs_uncertainties / values

# Calculate the weights based on relative error
weights = 1 / (relative_uncertainties ** 2)

# Calculate the weighted sum
weighted_sum = values.dot(weights)

# Calculate the total weight
total_weight = sum(weights)

# Calculate the uncertainty of the weighted average
uncertainty_of_weighted_average = 1 / math.sqrt(total_weight)
B_z_avg = weighted_sum / total_weight
print("Weighted Average: ", weighted_sum / total_weight)
print("Uncertainty of Weighted Average: ", np.std(unumpy.nominal_values(y1)))


##### Napake



##### Fitanje
# slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x1, y1)

# optimizedParameters1, pcov1 = opt.curve_fit(fit_fun1, x1, y1, sigma=unumpy.std_devs(Uy1_err), absolute_sigma=True)

##### Graf
# plt.plot(np.concatenate((np.array([0]), x1), axis=None), fit_fun1(np.concatenate((np.array([0]), x1), axis=None), *optimizedParameters1),
        #  color="black", linestyle="dashed", label="Fit")

# plt.plot(x1, y1, "o", label="Izmerjeno", color="#0033cc")

plt.errorbar(unumpy.nominal_values(x1), unumpy.nominal_values(y1), yerr=unumpy.std_devs(y1), fmt='.', color="#0033cc", ecolor='black', capsize=3, label="Izmerjeno")

plt.xlabel('$U$ [mV]')
plt.ylabel('$\Delta T$ [K]')

# xlim_spodnja = np.min(x1)
# xlim_zgornja = np.max(y1)
# ylim_spodnja = np.min(y1)
# ylim_zgornja = np.max(y1)

# plt.xlim(xlim_spodnja, xlim_zgornja)
# plt.ylim(ylim_spodnja, ylim_zgornja)

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
plt.title("Izmerjena napetost v odvisnosti od razlike v temperaturi")

# plt.savefig("TopPre_Umeritev.png", dpi=300, bbox_inches="tight")
plt.show()


##### Izračuni
# print(slope1 * 2 * d**2 / (np.pi * r**2), ", relativna napaka brez d in S: ", std_err1 / slope1, slope1)
# print("Naklon:", slope1, ", Negotovost: ", std_err1)
# print(f'The slope = {optimizedParameters1[0]}, with uncertainty {np.sqrt(pcov1[0][0])}')

# slope1 = unc.ufloat(optimizedParameters1[0], np.sqrt(pcov1[0][0]))