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

##### Drugi set podatkov #####################################################################################################################################################

##### Seznami podatkov
Ux2 = np.array([52, 48.0, 43.0, 40, 37, 35, 33, 30, 27.5, 25.5, 23.5, 21.5, 20.0, 18, 17, 15.5, 14, 12.5, 11]) #
# Ux2 = np.array([29, 24, 19, 44, 32, 53, 13]) #
Uy2_1 = np.array([6, 8, 11, 13, 18, 20, 24, 29, 35, 41, 47, 52, 56, 64, 66, 70, 74, 76, 78]) #
Uy2_2 = np.array([7, 9, 12, 14, 17, 21, 24, 31, 38, 44, 52, 59, 63, 70, 72, 77, 80, 82, 84]) #
Ut = np.array([19.09, 19.13, 19.27]) #s


##### Konstante, parametri in začetne vrednosti
d_p = unc.ufloat(1.57, 0.01) * 10**(-2)
r_p = d_p/2 * 10**(-2)
m_p = unc.ufloat(32.63, 0.01) * 10**(-3)
l_p = unc.ufloat(4.13, 0.02) * 10**(-2)

d_n = unc.ufloat(1.60, 0.01) * 10**(-2)
D_n = unc.ufloat(1.90, 0.01) * 10**(-2)
r_n = d_n/(2)
R_n = D_n/(2)
l_n = unc.ufloat(4.99, 0.01) * 10**(-2)
m_n = unc.ufloat(5.41, 0.01) * 10**(-3)

##### Definicije funkcij
def fit_fun2(x, a, b):
    return a * x + b


##### Obdelava seznamov
Ux2_err = unumpy.uarray(Ux2, np.ones(np.shape(Ux2)) * 0.2)
Ux2_err = Ux2_err * 10**(-2)
Uy2_1_err = unumpy.uarray(Uy2_1, np.ones(np.shape(Uy2_1)) * 0.5) * np.pi / 180
Uy2_2_err = unumpy.uarray(Uy2_2, np.ones(np.shape(Uy2_2)) * 0.5) * np.pi / 180


Uy2_err = unumpy.uarray(np.mean( np.array([Uy2_1, Uy2_2]), axis=0 ), np.ones(np.shape(Ux2)) * 0.5) * np.pi / 180
# Uy2_err = unumpy.uarray(np.array([29, 45, 60, 9, 24, 60, 78]), np.ones(np.shape(Ux2)) * 0.2) * np.pi / 180

y2 = Uy2_err
x2 = Ux2_err

Ut_err = unumpy.uarray(Ut, np.std(Ut))
om_arr = (2 * np.pi) / (Ut_err/10)
om = unc.ufloat(np.average(unumpy.nominal_values(om_arr)), unumpy.std_devs(om_arr)[1])
J_p = m_p * (((r_p ** 2) / 4) + ((l_p ** 2) / 12))
J_n = (m_n / 12) * (3 * ((r_n ** 2) + (R_n ** 2)) + 4 * (l_n ** 2))
J = J_n + J_p

print("Vztr. mom.: ", J, J_p, J_n)

p_krat_B = om**2 * J
print(p_krat_B)

B_z_p = const.mu_0 / (unumpy.tan(y2) * 4 * np.pi * (x2 ** 3))

# print(B_z_p)
# print(unumpy.std_devs(B_z_p))

# print(np.average(unumpy.nominal_values(B_z_p)), np.std(unumpy.nominal_values(B_z_p)))

# print(np.sqrt(-np.average(unumpy.nominal_values(B_z_p)) * unc.nominal_value(p_krat_B)))


# Sample data as a uarray with values and absolute uncertainties
values = unumpy.nominal_values(B_z_p)
abs_uncertainties = unumpy.std_devs(B_z_p)

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
B_z_p_avg = weighted_sum / total_weight
print("Weighted Average: ", weighted_sum / total_weight)
print("Uncertainty of Weighted Average: ", uncertainty_of_weighted_average)
B_z_p_nekaj = np.average(unumpy.nominal_values(B_z_p)[:-8])

print("B_z z omejenimi podatki: ", np.sqrt(B_z_p_nekaj * unc.nominal_value(p_krat_B)), ", negotovost: ", np.std(unumpy.nominal_values(B_z_p[:-8])))
print("B_z z oteženim povprečjem: ", np.sqrt(B_z_p_avg * unc.nominal_value(p_krat_B)))

print("Recimo da končni B_z: ", np.sqrt(weighted_sum / total_weight * unc.nominal_value(p_krat_B)), ", z relativno negotovostjo: ", np.sqrt(uncertainty_of_weighted_average ** 2 + (unc.std_dev(p_krat_B) / unc.nominal_value(p_krat_B)) ** 2), 
      " in absolutno negotovostjo: ", np.sqrt(uncertainty_of_weighted_average ** 2 + (unc.std_dev(p_krat_B) / unc.nominal_value(p_krat_B)) ** 2) * np.sqrt(weighted_sum / total_weight * unc.nominal_value(p_krat_B)))

print("p z omejenimi podatki: ", np.sqrt((B_z_p_avg ** (-1)) * unc.nominal_value(p_krat_B)))
print("B/p: ", B_z_p_avg, uncertainty_of_weighted_average * B_z_p_avg)






##### Seznami podatkov
# Ux2 = np.array([48.0, 43.0, 40, 37, 35, 33, 30, 27.5, 25.5, 23.5, 21.5, 20.0, 18, 17, 15.5, 14, 12.5, 11]) #
# Uy2_1 = np.array([8, 11, 13, 18, 20, 24, 29, 35, 41, 47, 52, 56, 64, 66, 70, 74, 76, 78]) #
# Uy2_2 = np.array([9, 12, 14, 17, 21, 24, 31, 38, 44, 52, 59, 63, 70, 72, 77, 80, 82, 84]) #
# Ut = np.array([19.09, 19.13, 19.27]) #s


# ##### Konstante, parametri in začetne vrednosti
# d_p = 1.57
# r_p = d_p/2 * 10**(-3)
# m_p = 32.63 * 10**(-3)
# l_p = 4.13 * 10**(-2)

# d_n = 1.60 * 10**(-2)
# D_n = 1.90 * 10**(-2)
# r_n = d_n/(2)
# R_n = D_n/(2)
# l_n = 4.99 * 10**(-2)
# m_n = 5.41 * 10**(-3)

# ##### Definicije funkcij
# def fit_fun2(x, a, b):
#     return a * x + b


# ##### Obdelava seznamov
# Uy2_1 = Uy2_1
# Uy2_2 = Uy2_2
# Ux2 = Ux2 * 10**(-2)

# Uy2 = np.average(np.array([Uy2_1, Uy2_2]), axis=0)

# y2 = Uy2
# x2 = Ux2

# Ut_err = unumpy.uarray(Ut, np.std(Ut))
# om_arr = 1/(Ut_err/10)
# om = unc.ufloat(np.average(unumpy.nominal_values(om_arr)), unumpy.std_devs(om_arr)[1])
# J = m_p * ((r_p ** 2) / 4 + (l_p ** 2) / 12) + (m_n / 12) * (3 * ((r_n ** 2) + (R_n ** 2)) + (l_n ** 2))

# p_krat_B = om**2 * J

# B_z_p = (1 / (np.tan(y2))) * const.mu_0 / (4 * np.pi * (x2 ** 3))



values = unumpy.nominal_values(B_z_p)[:-8]
abs_uncertainties = unumpy.std_devs(B_z_p)[:-8]

# print(values, abs_uncertainties)

uncertainties = - abs_uncertainties / values

# print(uncertainties, abs_uncertainties[0] / values[0])

weights = 1 / (uncertainties ** 2)
average = np.sum(weights * values) / (np.sum(weights))

print(average)





##### Napake


##### Fitanje
# slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(unumpy.nominal_values(x2), unumpy.nominal_values(y2_1))

# optimizedParameters2, pcov2 = opt.curve_fit(fit_fun2, unumpy.nominal_values(x2), unumpy.nominal_values(y2_1), sigma=unumpy.std_devs(y2_1), absolute_sigma=True)

##### Graf
# plt.plot(np.concatenate((np.array([0]), unumpy.nominal_values(x2)), axis=None), fit_fun2(np.concatenate((np.array([0]), unumpy.nominal_values(x2)), axis=None), *optimizedParameters2),
#          color="black", linestyle="dashed", label="Fit")

# plt.plot(np.nox2, B_z_p, "o", label="Izmerjeno", color="#0033cc")
plt.plot(unumpy.nominal_values(x2), np.ones(np.shape(unumpy.nominal_values(x2))) * average)

plt.errorbar(unumpy.nominal_values(x2), unumpy.nominal_values(B_z_p), yerr=unumpy.std_devs(B_z_p), fmt='.', color="#0033cc", ecolor='black', capsize=3, label="Izmerjeno")

plt.xlabel('$r$ [cm]')
plt.ylabel('$B/p$ [N/m]')

plt.ylim(0,1.5*10**(-5))
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
plt.title("B/p pri različnih razdaljah do magneta")

# plt.savefig("ZeMaPo.png", dpi=300, bbox_inches="tight")
plt.show()


##### Izračuni
# print(slope1 * 2 * d**2 / (np.pi * r**2), ", relativna napaka brez d in S: ", std_err1 / slope1, slope1)
# print("Naklon:", slope2, ", Negotovost: ", std_err2)
# print(optimizedParameters2[0], np.sqrt(pcov2[0][0]))


# print(f'The slope = {optimizedParameters2[0]}, with uncertainty {np.sqrt(pcov2[0][0])}')



# lam = unc.ufloat(optimizedParameters2[0], 8)
ro = 2710
c = 887
L = unc.ufloat(0.1, 0.02)

# print(L**2 / (2*(lam / (ro * c))))

print(J)