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

Uy2_1_err = unumpy.uarray(Uy2_1, np.ones(np.shape(Uy2_1)) * 0.005)
Uy2_2_err = unumpy.uarray(Uy2_2, np.ones(np.shape(Uy2_2)) * 0.005)


# Sila mikrometra
d = np.array([9.365, 8.295, 7.260, 4.970, 3.275, 2.750])
F = np.array([104, 87, 91, 74, 67, 71])

d_err = unumpy.uarray(d, np.array([0.005 for i in d]))
F_err = unumpy.uarray(F, np.array([0.005 for i in F]))


##### Konstante, parametri in začetne vrednosti
# print(mase)


##### Definicije funkcij
# def fit_fun2(x, a, b):
#     return a * x + b

def fit_fun2(B, x):
    '''Linear function y = m*x + b'''
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0]*x + B[1]


##### Obdelava seznamov
# plt.plot(d, F)
# plt.plot(unumpy.nominal_values(Ux2), Uy2_1)
# plt.plot(unumpy.nominal_values(Ux2), Uy2_2)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import *


############# Mikrometrska ura

# Initiate some data, giving some randomness using random.random().
x_mik = unumpy.nominal_values(d_err)
y_mik = unumpy.nominal_values(F_err)

x_err_mik = unumpy.std_devs(d_err)
y_err_mik = unumpy.std_devs(F_err)

# Create a model for fitting.
lin_model_mik = Model(fit_fun2)

# Create a RealData object using our initiated data from above.
data_mik = RealData(x_mik, y_mik, sx=x_err_mik, sy=y_err_mik)

# Set up ODR with the model and data.
odr_mik = ODR(data_mik, lin_model_mik, beta0=[0., 1.])

# Run the regression.
out_mik = odr_mik.run()

# Use the in-built pprint method to give us results.
out_mik.pprint()
'''Beta: [ 1.01781493  0.48498006]
Beta Std Error: [ 0.00390799  0.03660941]
Beta Covariance: [[ 0.00241322 -0.01420883]
 [-0.01420883  0.21177597]]
Residual Variance: 0.00632861634898189
Inverse Condition #: 0.4195196193536024
Reason(s) for Halting:
  Sum of squares convergence'''

x_fit_mik = np.linspace(x_mik[0], x_mik[-1], 1000)
y_fit_mik = fit_fun2(out_mik.beta, x_mik)

plt.errorbar(x_mik, y_mik, xerr=x_err_mik, yerr=y_err_mik, linestyle='None', marker='x')
plt.plot(x_mik, y_fit_mik)

plt.show()

print(out_mik.beta)






############# Okrogla palica

# Initiate some data, giving some randomness using random.random().
x_ok = unumpy.nominal_values(Ux2)
y_ok = unumpy.nominal_values(Uy2_1_err)

x_err_ok = unumpy.std_devs(Ux2)
y_err_ok = unumpy.std_devs(Uy2_1_err)

# Create a model for fitting.
quad_model_ok = Model(fit_fun2)

# Create a RealData object using our initiated data from above.
data_ok = RealData(x_ok, y_ok, sx=x_err_ok, sy=y_err_ok)

# Set up ODR with the model and data.
odr_ok = ODR(data_ok, quad_model_ok, beta0=[0., 1.])

# Run the regression.
out_ok = odr_ok.run()

# Use the in-built pprint method to give us results.
out_ok.pprint()
'''Beta: [ 1.01781493  0.48498006]
Beta Std Error: [ 0.00390799  0.03660941]
Beta Covariance: [[ 0.00241322 -0.01420883]
 [-0.01420883  0.21177597]]
Residual Variance: 0.00632861634898189
Inverse Condition #: 0.4195196193536024
Reason(s) for Halting:
  Sum of squares convergence'''

x_fit_ok = np.linspace(x_ok[0], x_ok[-1], 1000)
y_fit_ok = fit_fun2(out_ok.beta, x_ok)

plt.errorbar(x_ok, y_ok, xerr=x_err_ok, yerr=y_err_ok, linestyle='None', marker='x')
plt.plot(x_ok, y_fit_ok)

plt.show()

print(out_ok.beta)



############ Kvadratna palica

# Initiate some data, giving some randomness using random.random().
x_kv = unumpy.nominal_values(Ux2)
y_kv = unumpy.nominal_values(Uy2_2_err)

x_err_kv = unumpy.std_devs(Ux2)
y_err_kv = unumpy.std_devs(Uy2_2_err)

# Create a model for fitting.
quad_model_kv = Model(fit_fun2)

# Create a RealData object using our initiated data from above.
data_kv = RealData(x_kv, y_kv, sx=x_err_kv, sy=y_err_kv)

# Set up ODR with the model and data.
odr_kv = ODR(data_kv, quad_model_kv, beta0=[0., 1.])

# Run the regression.
out_kv = odr_kv.run()

# Use the in-built pprint method to give us results.
out_kv.pprint()
'''Beta: [ 1.01781493  0.48498006]
Beta Std Error: [ 0.00390799  0.03660941]
Beta Covariance: [[ 0.00241322 -0.01420883]
 [-0.01420883  0.21177597]]
Residual Variance: 0.00632861634898189
Inverse Condition #: 0.4195196193536024
Reason(s) for Halting:
  Sum of squares convergence'''

x_fit_kv = np.linspace(x_kv[0], x_kv[-1], 1000)
y_fit_kv = fit_fun2(out_kv.beta, x_kv)

plt.errorbar(x_kv, y_kv, xerr=x_err_kv, yerr=y_err_kv, linestyle='None', marker='x')
plt.plot(x_kv, y_fit_kv)

plt.show()

print(out_kv.beta)



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