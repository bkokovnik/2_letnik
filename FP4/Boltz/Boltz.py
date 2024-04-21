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
from typing import Callable


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Serif",
    "font.size": 20
})


def obtezeno_povprecje(uarray):
    '''Izračuna obteženo povprečje unumpy arraya'''

    vrednosti = unp.nominal_values(uarray)
    negotovosti = unp.std_devs(uarray)

    obtezitev = 1/(negotovosti**2)
    obtezeno_povprecje = np.sum(vrednosti * obtezitev) / np.sum(obtezitev)

    obtezena_negotovost = np.sqrt(np.sum(negotovosti**2 * obtezitev**2) / (np.sum(obtezitev)**2))

    return unc.ufloat(obtezeno_povprecje, obtezena_negotovost)

#####################################################################################################################################################
def lin_fun(x, a, b):
    return a * x + b

def lin_fun2(B, x):
    '''Linearna funkcija y = m*x + b'''
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0]*x + B[1]


def fit_napake(a: unp.uarray, b: unp.uarray, funkcija: Callable[[np.array, any], np.ndarray], print=0):

    '''Sprejme 2 unumpy arraya skupaj z funkcijo in izračuna fit, upoštevajoč x in y napake'''
    # Podatki
    x_mik = unp.nominal_values(a)
    y_mik = unp.nominal_values(b)

    x_err_mik = unp.std_devs(a)
    y_err_mik = unp.std_devs(b)

    # Create a model for fitting.
    lin_model_mik = Model(funkcija)

    # Create a RealData object using our initiated data from above.
    data_mik = RealData(x_mik, y_mik, sx=x_err_mik, sy=y_err_mik)

    # Set up ODR with the model and data.
    odr_mik = ODR(data_mik, lin_model_mik, beta0=[0., 1.])

    # Run the regression.
    out_mik = odr_mik.run()

    if print == 1:
        out_mik.pprint()
    
    return out_mik

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


def graf_fit_napake(a, b, fit, fit_label="Fit", podatki_label="Izmerjeno"):
    '''Sprejme 2 unumpy arraya, fitane parametre in nariše črtkan fit, skupaj z errorabr prikazom podatkov'''
    # Podatki
    x_mik = unp.nominal_values(a)
    y_mik = unp.nominal_values(b)

    x_err_mik = unp.std_devs(a)
    y_err_mik = unp.std_devs(b)

    x_fit_mik = np.linspace(x_mik[0], x_mik[-1], 1000)
    y_fit_mik = lin_fun2(fit.beta, x_mik)

    plt.errorbar(x_mik, y_mik, xerr=x_err_mik, yerr=y_err_mik, linestyle='None', marker='.', capsize=3, label=podatki_label)
    plt.plot(x_mik, y_fit_mik, label=fit_label, color="black")




#####################################################################################################################################################

# with open("/FP4/Boltz/Podatki/T_14.txt", "r") as dat:

a1 = np.genfromtxt(r"FP4\Boltz\Podatki\T_14_1.txt", delimiter='\t')

# print(a.T)

x1 = a1.T[0]
y1 = np.log(a1.T[1])

y1_unc = unp.uarray(np.log(a1.T[1]), (np.ones(np.shape(a1.T[1])) * 0.001e-6) / a1.T[1])
# y1_unc = np.log((y1_unc))

x1_unc = unp.uarray(a1.T[0], np.ones(np.shape(a1.T[0])) * 0.005)

print(x1_unc)

# p = fit_napake(x1_unc, y1_unc, 1)

##### Fitanje
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x1, y1)

optimizedParameters1, pcov1 = opt.curve_fit(lin_fun, x1, y1)#, sigma=unp.std_devs(akt_raz_sqrt), absolute_sigma=True)


fit_1 = fit_napake(x1_unc, y1_unc, lin_fun2, 1)
graf_fit_napake(x1_unc, y1_unc, fit_1)
plt.show()
# print(optimizedParameters1, "AAAAAAAAAAAAAAAAAAAAAAAAAAAA")




# x_fit_mik = np.linspace(unp.nominal_values(x1_unc)[0], unp.nominal_values(x1_unc)[-1], 1000)
# y_fit_mik = lin_fun2(p.beta, unp.nominal_values(x1_unc))

# plt.plot(unp.nominal_values(x1_unc), y_fit_mik, label="Fit", color="black")







# print(optimizedParameters1, pcov1, "AAAAAAAAAAAAAAAAAAAAAAAAAAA")

# print(optimizedParameters)
# plt.plot([0.3, 0.8], lin_fun(np.array([0.3, 0.8]), *optimizedParameters1),
        #   color="black", linestyle="dashed")#, label="Fit")
# plt.plot(np.concatenate((np.array([0]), x1), axis=None), lin_fun(np.concatenate((np.array([0]), x1), axis=None), *optimizedParameters1),
#           color="black", linestyle="dashed")#, label="Fit")
plt.plot(x1, y1, "o", label="$14{,}1\ ^{\circ}$C")

# plt.show()

# kb_Te1 = unc.ufloat(optimizedParameters1[0], np.sqrt(pcov1[0][0]))
# kb1 = (kb_Te1 * ((273 + 14.1) / 1.6e-19))**(-1)
# print(kb1)




####################################################################################################################################



a2 = np.genfromtxt(r"FP4\Boltz\Podatki\T_34_1.txt", delimiter='\t')

# print(a.T)

x2 = a2.T[0]
y2 = np.log(a2.T[1])

##### Fitanje
slope2, intercept2, r_value2, p_value12, std_err2 = stats.linregress(x2, y2)

optimizedParameters2, pcov2 = opt.curve_fit(lin_fun, x2, y2)#, sigma=np.log(np.ones(np.shape(y2)) * 0.), absolute_sigma=True)
# optimizedParameters2_c, pcov2_c = opt.curve_fit(lin_fun, y2, x2, sigma=np.log(np.ones(np.shape(y2)) * 0.01), absolute_sigma=True)

# print(optimizedParameters2[0], pcov2[0][0])











# print(optimizedParameters)
plt.plot([0.3, 0.8], lin_fun(np.array([0.3, 0.8]), *optimizedParameters2),
          color="black", linestyle="dashed")#, label="Fit")
# plt.plot(np.concatenate((np.array([0]), x2), axis=None), lin_fun(np.concatenate((np.array([0]), x2), axis=None), *optimizedParameters2),
#           color="black", linestyle="dashed")#, label="Fit")
plt.plot(x2, y2, "o", label="$34{,}1\ ^{\circ}$C")

# plt.show()

kb_Te2 = unc.ufloat(optimizedParameters2[0], np.sqrt(pcov2[0][0]))
kb2 = (kb_Te2 * ((273 + 34.1) / 1.6e-19))**(-1)
print(kb2)



####################################################################################################################################



a3 = np.genfromtxt(r"FP4\Boltz\Podatki\T_53_2.txt", delimiter='\t')

# print(a.T)

x3 = a3.T[0]
y3 = np.log(a3.T[1])

##### Fitanje
slope3, intercept3, r_value3, p_value13, std_err3 = stats.linregress(x3, y3)

optimizedParameters3, pcov3 = opt.curve_fit(lin_fun, x3, y3)#, sigma=unp.std_devs(akt_raz_sqrt), absolute_sigma=True)

# print(optimizedParameters)
plt.plot([0.3, 0.8], lin_fun(np.array([0.3, 0.8]), *optimizedParameters3),
          color="black", linestyle="dashed")#, label="Fit")
# plt.plot(np.concatenate((np.array([0]), x3), axis=None), lin_fun(np.concatenate((np.array([0]), x3), axis=None), *optimizedParameters3),
#           color="black", linestyle="dashed")#, label="Fit")
plt.plot(x3, y3, "o", label="$53{,}2\ ^{\circ}$C")

plt.legend()
plt.xlim(0.36, 0.62)
plt.ylim(-18, -5)
plt.ylabel('$\ln(I_C / I_1)$')
plt.xlabel('$U_{BE}$ [V]')
# plt.title("Aktivnost v odvisnosti od debeline\nplasti aluminija")



plt.show()

kb_Te3 = unc.ufloat(optimizedParameters3[0], np.sqrt(pcov3[0][0]))
kb3 = (kb_Te3 * ((273 + 53.2) / 1.6e-19))**(-1)
print(kb3)

####################################################################################################################################


print(f"Obteženo povprečje vseh treh vrednosti: {obtezeno_povprecje([kb1, kb2, kb3])}")








################################################################################################################################################

a4 = np.genfromtxt(r"FP4\Boltz\Podatki\U_0_5.txt", delimiter='\t')

x4 = a4.T[0][3:]
# y4 = np.log(a4.T[1])
y4 = a4.T[1][3:]


a5 = np.genfromtxt(r"FP4\Boltz\Podatki\U_0_58.txt", delimiter='\t')

x5 = a5.T[0][1:]
# y4 = np.log(a4.T[1])
y5 = a5.T[1][1:]

plt.plot(x4, y4, "o", label="$0{,}5\ $V")
plt.plot(x5, y5, "o", label="$0{,}58\ $V")

plt.legend()
x_limits = plt.xlim()
y_limits = plt.ylim()
plt.ylim(y_limits)
plt.xlim(x_limits)
plt.ylabel('$I_C$ [mA]')
plt.xlabel('$T$ [$^\circ$ C]')
# plt.title("Aktivnost v odvisnosti od debeline\nplasti aluminija")
plt.show()







optimizedParameters4, pcov4 = opt.curve_fit(lin_fun, x4, np.log(y4))#, sigma=unp.std_devs(akt_raz_sqrt), absolute_sigma=True)
optimizedParameters5, pcov5 = opt.curve_fit(lin_fun, x5, np.log(y5))#, sigma=unp.std_devs(akt_raz_sqrt), absolute_sigma=True)

# print(optimizedParameters)







plt.plot(x4, np.log(y4), "o", label="$0{,}5\ $V")
plt.plot(x5, np.log(y5), "o", label="$0{,}58\ $V")


plt.legend()
# plt.xlim(0.36, 0.62)
# plt.ylim(-18, -5)
plt.ylabel('$\ln(I_C / I_1)$')
plt.xlabel('$T$ [$^\circ$ C]')
plt.plot([0.3, 0.8], lin_fun(np.array([0.3, 0.8]), *optimizedParameters3),
          color="black", linestyle="dashed")#, label="Fit")
# plt.title("Aktivnost v odvisnosti od debeline\nplasti aluminija")

plt.show()