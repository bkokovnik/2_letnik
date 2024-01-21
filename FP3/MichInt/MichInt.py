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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Serif",
    "font.size": 20
})

###########################################################################################################
def kvadratna_funkcija(x, a, b):
    return a * x**(-2) + b


def linearna_funkcija(x, a, b):
    return a * x + b



def lin_fun2(B, x):
    '''Linear function y = m*x + b'''
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0]*x + B[1]



def obtezeno_povprecje(uarray):
    vrednosti = unumpy.nominal_values(uarray)
    negotovosti = unumpy.std_devs(uarray)

    obtezitev = 1/(negotovosti**2)
    obtezeno_povprecje = np.sum(vrednosti * obtezitev) / np.sum(obtezitev)

    obtezena_negotovost = np.sqrt(np.sum(negotovosti**2 * obtezitev**2) / (np.sum(obtezitev)**2))

    return unc.ufloat(obtezeno_povprecje, obtezena_negotovost)

###########################################################################################################

### Izračun prestavnega razmerja

val_dol = unc.ufloat(632.8, 0.1) #nm
N_pres = unumpy.uarray([100, 100, 85, 100, 50], [2, 2, 1, 5, 1])
d_pres = unumpy.uarray([0.14, 0.16, 0.13, 0.15, 0.085], [0.005, 0.005, 0.005, 0.005, 0.005]) * 10**(6)

d_z_N_pres = d_pres / N_pres

print(np.average(d_z_N_pres) * 2/val_dol)


### Lomni količnik zraka

l = unc.ufloat(50, 1) * 10**6 #nm

dp = unumpy.uarray([0.5, 1, 0.75, 0.3, 1.5], [0.05, 0.05, 0.05, 0.05, 0.05]) #bar
N_tlak  =unumpy.uarray([32, 50, 39, 16, 75], [1, 1, 1, 1, 1])

N_lambda = N_tlak * val_dol / (2 * l)



# Initiate some data, giving some randomness using random.random().
x_mik = unumpy.nominal_values(dp)
y_mik = unumpy.nominal_values(N_lambda)

x_err_mik = unumpy.std_devs(dp)
y_err_mik = unumpy.std_devs(N_lambda)

# Create a model for fitting.
lin_model_mik = Model(lin_fun2)

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
y_fit_mik = lin_fun2(out_mik.beta, x_mik)

plt.errorbar(unumpy.nominal_values(dp), unumpy.nominal_values(N_lambda), xerr=unumpy.std_devs(dp), yerr=unumpy.std_devs(N_lambda), linestyle='None', marker='.', capsize=3, label="Izmerjeno")
plt.plot(x_mik, y_fit_mik, label="Fit", color="black")

plt.ylabel('$\Delta n$')
plt.xlabel('$\Delta p$ [bar]')
plt.title("Odvisnot spremembe lomnega količnika od\nabsolutnega tlaka")
plt.legend()

# plt.savefig("MichInt/MichInt_tlak.png", dpi=300, bbox_inches="tight")
plt.show()


n_od_p = unc.ufloat(out_mik.beta[0], np.sqrt(out_mik.cov_beta[0][0]))

print("Tlačna odvisnost lomnega količnika: ", n_od_p *10**4, " 10 ^{-4} bar^{-1}")

print("Lomni količnik n pri 1000 bar: ", 1000 * n_od_p + 1)

print("Koherenčna dolžina: ", unc.ufloat(4, 1) * 550, "nm")



### Natrijev duplet

d_majhno = unumpy.uarray([6.90, 6.56, 6.64, 6.99, 6.91], [0.05, 0.05, 0.05, 0.05, 0.05])
N_majhno = unc.ufloat(100, 5)

d_ekvi_e = unc.ufloat(6.77, 0.05)

d_majhno = (d_majhno - d_ekvi_e)

val_dol_avg = 2 * np.average(d_majhno) / (100 * np.average(d_z_N_pres) * 2/val_dol)

print(val_dol_avg * 10**6)


d_2_1 = unc.ufloat(6.67, 0.01)
d_2_2 = unc.ufloat(6.80, 0.01)
d_2_3 = unc.ufloat(6.83, 0.01)
d_2_4 = unc.ufloat(6.88, 0.01)
d_2_5 = unc.ufloat(6.92, 0.01)
d_2_6 = unc.ufloat(6.97, 0.01)



d_2 = unumpy.uarray([unc.nominal_value(d_2_2 - d_2_1), unc.nominal_value(d_2_3 - d_2_2), unc.nominal_value(d_2_4 - d_2_3), unc.nominal_value(d_2_5 - d_2_4), unc.nominal_value(d_2_6 - d_2_5)],
                    [unc.std_dev(d_2_2 - d_2_1), unc.std_dev(d_2_3 - d_2_2), unc.std_dev(d_2_4 - d_2_3), unc.std_dev(d_2_5 - d_2_4), unc.std_dev(d_2_6 - d_2_5)])

d_2 = d_2 / (np.average(d_z_N_pres) * 2/val_dol)

duplet = val_dol_avg**2 / (2 * np.average(d_2))

print(duplet * 10**6)