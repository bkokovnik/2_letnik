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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Serif",
    "font.size": 18
})

###########################################################


##### Prvi set podatkov #####################################################################################################################################################

##### Seznami podatkov
Uy1 = np.array([92.5, 89.5, 86.1, 81.5, 77.4, 73.5, 69.9, 65.7, 63.0, 58.8, 55.4, 52.1, 50.2, 46.2, 43.5, 39.2, 35.9, 33.0, 31.1, 28.7, 23.3]) #C
Ux1 = np.array([3.862, 3.737, 3.573, 3.384, 3.199, 3.031, 2.871, 2.696, 2.579, 2.394, 2.248, 2.110, 2.028, 1.862, 1.747, 1.568, 1.433, 1.311, 1.237, 1.137, 0.921]) #mV


##### Konstante, parametri in začetne vrednosti


##### Definicije funkcij
def fit_fun1(x, a, b):
    return a * x + b


##### Obdelava seznamov
Ux1_err = unumpy.uarray(Ux1, np.ones(np.shape(Ux1)) * 0.0005)
Uy1_err = unumpy.uarray(Uy1, np.array([2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))

x1 = unumpy.nominal_values(Ux1_err)
y1 = unumpy.nominal_values(Uy1_err)


##### Napake



##### Fitanje
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x1, y1)

optimizedParameters1, pcov1 = opt.curve_fit(fit_fun1, x1, y1, sigma=unumpy.std_devs(Uy1_err), absolute_sigma=True)

##### Graf
plt.plot(np.concatenate((np.array([0]), x1), axis=None), fit_fun1(np.concatenate((np.array([0]), x1), axis=None), *optimizedParameters1),
         color="black", linestyle="dashed", label="Fit")

plt.plot(x1, y1, "o", label="Izmerjeno", color="#0033cc")

# plt.errorbar(x1, y1, yerr=0.005/y1, fmt='.', color="#0033cc", ecolor='black', capsize=3, label="Izmerjeno")

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
print("Naklon:", slope1, ", Negotovost: ", std_err1)
print(f'The slope = {optimizedParameters1[0]}, with uncertainty {np.sqrt(pcov1[0][0])}')

slope1 = unc.ufloat(optimizedParameters1[0], np.sqrt(pcov1[0][0]))



##### Drugi set podatkov #####################################################################################################################################################

##### Seznami podatkov
Uy2 = np.array([30.4, 44.7, 59.9]) #
Ux2 = np.array([0.2315, 0.3185, 0.4230]) #


##### Konstante, parametri in začetne vrednosti
d_m = 44.4
r_m = d_m/2 * 10**(-3)
l = 56.0

d_m_err = unc.ufloat(d_m, 0.1) #mm
r_m_err = d_m_err/(2*1000)
l_err = unc.ufloat(l, 0.1)

##### Definicije funkcij
def fit_fun2(x, a, b):
    return a * x + b


##### Obdelava seznamov
Ux2_err = unumpy.uarray(Ux2, np.ones(np.shape(Ux2)) * 0.0005)
Uy2_err = unumpy.uarray(Uy2, np.ones(np.shape(Uy2)) * 0.1)

y2 = Uy2_err * l_err * 10**(-3) / (np.pi * r_m_err**2)
x2 = Ux2_err * slope1


##### Napake
d_y2 = np.array([0.005, 0.005, 0.005])


##### Fitanje
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(unumpy.nominal_values(x2), unumpy.nominal_values(y2))

optimizedParameters2, pcov2 = opt.curve_fit(fit_fun2, unumpy.nominal_values(x2), unumpy.nominal_values(y2), sigma=unumpy.std_devs(y2), absolute_sigma=True)

##### Graf
plt.plot(np.concatenate((np.array([0]), unumpy.nominal_values(x2)), axis=None), fit_fun2(np.concatenate((np.array([0]), unumpy.nominal_values(x2)), axis=None), *optimizedParameters2),
         color="black", linestyle="dashed", label="Fit")

plt.plot(unumpy.nominal_values(x2), unumpy.nominal_values(y2), "o", label="Izmerjeno", color="#0033cc")

# plt.errorbar(unumpy.nominal_values(x2), unumpy.nominal_values(y2), yerr=unumpy.std_devs(y2), fmt='.', color="#0033cc", ecolor='black', capsize=3, label="Izmerjeno")

plt.xlabel('$\Delta$$T$ [K]')
plt.ylabel('$Pl/S$ [W/m]')


# xlim_spodnja = np.min(x2)
# xlim_zgornja = np.max(y2)
# ylim_spodnja = np.min(y2)
# ylim_zgornja = np.max(y2)

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
plt.title("Gostota toplotnega toka pomnožena z razdaljo med luknjama\nv odvisnosti od spremembe temperature med tema luknjama")

# plt.savefig("TopPre_Prevodnost.png", dpi=300, bbox_inches="tight")
plt.show()


##### Izračuni
# print(slope1 * 2 * d**2 / (np.pi * r**2), ", relativna napaka brez d in S: ", std_err1 / slope1, slope1)
print("Naklon:", slope2, ", Negotovost: ", std_err2)
print(optimizedParameters2[0], np.sqrt(pcov2[0][0]))


print(f'The slope = {optimizedParameters2[0]}, with uncertainty {np.sqrt(pcov2[0][0])}')


print(r_m_err)

lam = unc.ufloat(optimizedParameters2[0], 8)
ro = 2710
c = 887
L = unc.ufloat(0.1, 0.02)

print(L**2 / (2*(lam / (ro * c))))