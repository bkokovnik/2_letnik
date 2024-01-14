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
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

################################################################
def fit_fun2(B, x):
    '''Linear function y = m*x + b'''
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0]*x + B[1]


tau_fit = unumpy.uarray([7.8972, 7.8579, 7.3579, 7.1702, 7.9263, 7.9022], [0.0003, 0.0005, 0.0004, 0.0005, 0.0004, 0.0004])
tau_tangenta = unumpy.uarray([7.6, 7.8, 7.5, 7.15, 8.0, 7.81], [0.4, 0.08, 0.4, 0.3, 1, 0.16])

R = unc.ufloat(5, 0.1) * 10**9
r = (unc.ufloat(38, 0.2) * 10**(-3))/2
h = unc.ufloat(6.5, 0.2) * 10**(-3)


# Sample data as a uarray with values and absolute uncertainties
values1 = unumpy.nominal_values(tau_fit)
abs_uncertainties1 = unumpy.std_devs(tau_fit)

# Calculate relative uncertainties
relative_uncertainties1 = abs_uncertainties1 / values1

# Calculate the weights based on relative error
weights1 = 1 / (relative_uncertainties1 ** 2)

# Calculate the weighted sum
weighted_sum1 = values1.dot(weights1)

# Calculate the total weight
total_weight1 = sum(weights1)

# Calculate the uncertainty of the weighted average
uncertainty_of_weighted_average1 = 1 / math.sqrt(total_weight1)
tau_fit_avg = unc.ufloat(weighted_sum1 / total_weight1, uncertainty_of_weighted_average1)




# Sample data as a uarray with values and absolute uncertainties
values2 = unumpy.nominal_values(tau_tangenta)
abs_uncertainties2 = unumpy.std_devs(tau_tangenta)

# Calculate relative uncertainties
relative_uncertainties2 = abs_uncertainties2 / values2

# Calculate the weights based on relative error
weights2 = 1 / (relative_uncertainties2 ** 2)

# Calculate the weighted sum
weighted_sum2 = values2.dot(weights2)

# Calculate the total weight
total_weight2 = sum(weights2)

# Calculate the uncertainty of the weighted average
uncertainty_of_weighted_average2 = 1 / math.sqrt(total_weight2)
tau_tangenta_avg = unc.ufloat(weighted_sum2 / total_weight2, uncertainty_of_weighted_average2)

# print(tau_fit_avg, tau_tangenta_avg)

C_fit = tau_fit_avg / R
C_tangenta = tau_tangenta_avg / R

epsilon_fit = C_fit * h / (const.epsilon_0 * np.pi * r**2)
epsilon_tangenta = C_tangenta * h / (const.epsilon_0 * np.pi * r**2)

# print(epsilon_fit, epsilon_tangenta)


# plt.plot(np.array([0.3270, 0.2378,0.1282, 0.12247,  0.1017, 0.0498]), np.array([1008, 1008, 504, 504, 196, 196]), ".")
U_0 = unumpy.uarray([1.316, -1.3136, 0.6597, -0.6615,  0.2717, -0.2325], np.ones(np.shape([1.316, -1.3136, 0.6597, -0.6615,  0.2717, -0.2325])) * 0.05)
m = unumpy.uarray([1008, -1008, 504, -504, 196, -196], np.ones(np.shape([1008, -1008, 504, -504, 196, -196])) * 1) * 10**(-3) * 9.81




# Initiate some data, giving some randomness using random.random().
x_mik = unumpy.nominal_values(m)
y_mik = unumpy.nominal_values(U_0)

x_err_mik = unumpy.std_devs(m)
y_err_mik = unumpy.std_devs(U_0)

# Create a model for fitting.
lin_model_mik = Model(fit_fun2)

# Create a RealData object using our initiated data from above.
data_mik = RealData(x_mik, y_mik, sx=x_err_mik, sy=y_err_mik)

# Set up ODR with the model and data.
odr_mik = ODR(data_mik, lin_model_mik, beta0=[0., 1.])

# Run the regression.
out_mik = odr_mik.run()

# Use the in-built pprint method to give us results.
# out_mik.pprint()
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

plt.plot(x_mik, y_fit_mik, label="Fit", color="black")
plt.errorbar(unumpy.nominal_values(m), unumpy.nominal_values(U_0), xerr=unumpy.std_devs(m), yerr=unumpy.std_devs(U_0), linestyle='None', marker='.', capsize=3, label="Izmerjeno")
plt.xlabel('$F$ (N)')
plt.ylabel('$U_0$ (V)')
plt.legend()
plt.title("$U_0$ v odvisnosti od obremenitve")
# plt.savefig("Piezo/Piezo_naklon.pgf")
plt.show()

k = unc.ufloat(out_mik.beta[0], out_mik.sd_beta[0])
d_fit = - k * C_fit
d_tangenta = - k * C_tangenta


print("Epsilon, določen s fitom: ", epsilon_fit)
print("Epsilon, določen s tangentami: ", epsilon_tangenta)
print("C, določen s fitom: ", C_fit, "F")
print("C, določen s tangentami: ", C_tangenta, "F")
print("Naklon fitane premice k: ", k, "V/N")
print("d, določen s fitom: ", d_fit, "m/V")
print("d, določen s tangentami: ", d_tangenta, "m/V")