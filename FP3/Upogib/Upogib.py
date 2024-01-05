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
m_k = unc.ufloat(14, 1)
m_1 = unc.ufloat(10, 1)
m_2 = unc.ufloat(21, 1)
m_3 = unc.ufloat(20, 1)
m_4 = unc.ufloat(50, 1)
m_5 = unc.ufloat(100, 1)
m_6 = unc.ufloat(201, 1)
m_7 = unc.ufloat(201, 1)
m_8 = unc.ufloat(503, 1)
m_9 = unc.ufloat(1006, 1)

mase = np.array([m_k, m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8, m_9])

Ux2 = np.array([m_k, m_k + m_1, m_k + m_1 + m_2, m_k + m_1 + m_2 + m_3,  m_k + m_1 + m_2 + m_3 + m_4, m_k + m_1 + m_2 + m_3 + m_4 + m_5, 
                m_k + m_1 + m_2 + m_3 + m_4 + m_5 + m_6, m_k + m_1 + m_2 + m_3 + m_4 + m_5 + m_6 + m_7, 
                m_k + m_1 + m_2 + m_3 + m_4 + m_5 + m_6 + m_7 + m_8, m_k + m_1 + m_2 + m_3 + m_4 + m_5 + m_6 + m_7 + m_8 + m_9,
                m_k + m_1 + m_2 + m_3 + m_4 + m_5 + m_6 + m_7 + m_8, m_k + m_1 + m_2 + m_3 + m_4 + m_5 + m_6 + m_7,
                m_k + m_1 + m_2 + m_3 + m_4 + m_5 + m_6, m_k + m_1 + m_2 + m_3 + m_4 + m_5, m_k + m_1 + m_2 + m_3 + m_4,
                m_k + m_1 + m_2 + m_3, m_k + m_1 + m_2, m_k + m_1, m_k]) #

# Ux2 = np.array([29, 24, 19, 44, 32, 53, 13]) #
Uy2_1 = np.array([7.370, 7.345, 7.295, 7.250, 7.110, 6.830, 6.340, 5.780, 4.320, 1.430, 4.255, 5.625, 6.190, 6.725, 6.980, 7.125, 7.160, 7.250, 7.265]) #
Uy2_2 = np.array([9.285, 9.260, 9.220, 9.190, 9.100, 8.930, 8.585, 8.260, 7.405, 5.690, 7.400, 8.255, 8.590, 8.925, 9.100, 9.190, 9.220, 9.255, 9.270]) #

# Sila mikrometra
d = np.array([9.365, 8.295, 7.260, 4.970, 3.275, 2.750])
F = np.array([104, 87, 91, 74, 67, 71])


##### Konstante, parametri in začetne vrednosti
print(mase)


##### Definicije funkcij
def fit_fun2(x, a, b):
    return a * x + b


##### Obdelava seznamov
plt.plot(d, F)
# plt.plot(unumpy.nominal_values(Ux2), Uy2_1)
# plt.plot(unumpy.nominal_values(Ux2), Uy2_2)
plt.show()



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



##### Napake


##### Fitanje
# slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(unumpy.nominal_values(x2), unumpy.nominal_values(y2_1))

# optimizedParameters2, pcov2 = opt.curve_fit(fit_fun2, unumpy.nominal_values(x2), unumpy.nominal_values(y2_1), sigma=unumpy.std_devs(y2_1), absolute_sigma=True)

##### Graf
# plt.plot(np.concatenate((np.array([0]), unumpy.nominal_values(x2)), axis=None), fit_fun2(np.concatenate((np.array([0]), unumpy.nominal_values(x2)), axis=None), *optimizedParameters2),
#          color="black", linestyle="dashed", label="Fit")

# plt.plot(np.nox2, B_z_p, "o", label="Izmerjeno", color="#0033cc")
# # plt.plot(unumpy.nominal_values(x2), np.ones(np.shape(unumpy.nominal_values(x2))) * average)

# # plt.errorbar(unumpy.nominal_values(x2), unumpy.nominal_values(B_z_p), yerr=unumpy.std_devs(B_z_p), fmt='.', color="#0033cc", ecolor='black', capsize=3, label="Izmerjeno")

# # plt.xlabel('$r$ [cm]')
# # plt.ylabel('$B/p$ [enote]')


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
# plt.show()


##### Izračuni
# print(slope1 * 2 * d**2 / (np.pi * r**2), ", relativna napaka brez d in S: ", std_err1 / slope1, slope1)
# print("Naklon:", slope2, ", Negotovost: ", std_err2)
# print(optimizedParameters2[0], np.sqrt(pcov2[0][0]))


# print(f'The slope = {optimizedParameters2[0]}, with uncertainty {np.sqrt(pcov2[0][0])}')