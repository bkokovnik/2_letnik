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
    "font.size": 15
})

#####################################################################################################################################################

def obtezeno_povprecje(uarray):
    '''Izračuna obteženo povprečje unumpy arraya'''

    vrednosti = unp.nominal_values(uarray)
    negotovosti = unp.std_devs(uarray)

    obtezitev = 1/(negotovosti**2)
    obtezeno_povprecje = np.sum(vrednosti * obtezitev) / np.sum(obtezitev)

    obtezena_negotovost = np.sqrt(np.sum(negotovosti**2 * obtezitev**2) / (np.sum(obtezitev)**2))

    return unc.ufloat(obtezeno_povprecje, obtezena_negotovost)

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

def exp_fun(B, x):
    """Eksponentna funkcija y = A exp(Bx)"""
    return B[0] * np.exp(B[1] * x**B[2])

def tok_fun(B,x):
    return B[0] * x**B[1] * np.exp( - B[2] / x)

def fit_napake(x: unp.uarray, y: unp.uarray, funkcija=lin_fun2, print=0) -> np.ndarray:

    '''Sprejme 2 unumpy arraya skupaj z funkcijo in izračuna fit, upoštevajoč x in y napake'''
    # Podatki
    x_mik = unp.nominal_values(x)
    y_mik = unp.nominal_values(y)

    x_err_mik = unp.std_devs(x)
    y_err_mik = unp.std_devs(y)

    # Create a model for fitting.
    lin_model_mik = Model(funkcija)

    # Create a RealData object using our initiated data from above.
    data_mik = RealData(x_mik, y_mik, sx=x_err_mik, sy=y_err_mik)

    # Set up ODR with the model and data.
    odr_mik = ODR(data_mik, lin_model_mik, beta0=[0., 3., 1.], maxit=100000)

    # odr_mik.set_job(maxit=10000)

    # Run the regression.
    out_mik = odr_mik.run()

    if print == 1:
        out_mik.pprint()
    
    return out_mik

def fit_napake_x(x: unp.uarray, y: unp.uarray, funkcija=lin_fun2, print=0) -> tuple:
    '''Sprejme 2 unumpy arraya skupaj z funkcijo in izračuna fit, upoštevajoč y napake'''
    optimizedParameters, pcov = opt.curve_fit(lin_fun, unp.nominal_values(x), unp.nominal_values(y), sigma=unp.std_devs(x), absolute_sigma=True)
    return (optimizedParameters, pcov)

def graf_errorbar(x: unp.uarray, y: unp.uarray, podatki_label="Izmerjeno"):
    '''Sprejme 2 unumpy arraya, fitane parametre in nariše errorbar prikaz podatkov'''
    # Podatki
    x_mik = unp.nominal_values(x)
    y_mik = unp.nominal_values(y)

    x_err_mik = unp.std_devs(x)
    y_err_mik = unp.std_devs(y)

    plt.errorbar(x_mik, y_mik, xerr=x_err_mik, yerr=y_err_mik, linestyle='None', marker='.', capsize=3, label=podatki_label)

def graf_fit_tuple(x: unp.uarray, y: unp.uarray, fit: tuple, fit_label="Fit"):
    '''Sprejme 2 unumpy arraya, fitane parametre in nariše črtkan fit'''
    # Podatki
    x_mik = unp.nominal_values(x)
    y_mik = unp.nominal_values(y)

    x_fit_mik = np.linspace(x_mik[0] - x_mik[0] / 2, x_mik[-1] + x_mik[-1] / 2, 1000)

    # y_fit_mik = lin_fun(*(fit[0]), x_fit_mik)
    plt.plot(x_fit_mik, lin_fun(x_fit_mik, *(fit[0])),
          "--", label=fit_label, color="#5e5e5e", linewidth=1)

def graf_fit(x: unp.uarray, y: unp.uarray, fit: np.ndarray, fit_label="Fit"):
    '''Sprejme 2 unumpy arraya, fitane parametre in nariše črtkan fit'''
    # Podatki
    x_mik = unp.nominal_values(x)
    y_mik = unp.nominal_values(y)

    x_fit_mik = np.linspace(x_mik[0] - x_mik[0] / 2, x_mik[-1] + x_mik[-1] / 2, 1000)

    if type(fit) is tuple:
        y_fit_mik = lin_fun(*(fit_napake_x(x, y)[0]), x_fit_mik)
        plt.plot(x_fit_mik, y_fit_mik, "--", label=fit_label, color="#5e5e5e", linewidth=1)

    else:
        y_fit_mik = lin_fun2(fit.beta, x_fit_mik)
        plt.plot(x_fit_mik, y_fit_mik, "--", label=fit_label, color="#5e5e5e", linewidth=1)

def graf_oblika(Naslov: str, x_os: str, y_os: str, legenda=1):
    x_limits = plt.xlim()
    y_limits = plt.ylim()

    x_ticks_major = plt.gca().get_xticks()
    y_ticks_major = plt.gca().get_yticks()

    x_ticks_minor = np.concatenate([np.arange(start, stop, (stop - start) / 5)[1:] for start, stop in zip(x_ticks_major[:-1], x_ticks_major[1:])])
    y_ticks_minor = np.concatenate([np.arange(start, stop, (stop - start) / 5)[1:] for start, stop in zip(y_ticks_major[:-1], y_ticks_major[1:])])

    plt.xticks(x_ticks_major)
    plt.xticks(x_ticks_minor, minor=True)

    plt.yticks(y_ticks_major)
    plt.yticks(y_ticks_minor, minor=True)

    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)


    plt.legend()

    plt.xlabel(x_os)
    plt.ylabel(y_os)
    plt.title(Naslov, y=1.02)

#####################################################################################################################################################


# Podatki za 1. del naloge

a1 = np.genfromtxt(r"FP4\Boltz\Podatki\T_14_1.txt", delimiter='\t', encoding="utf-8")

y1_unc = unp.uarray(np.log(a1.T[1]), (np.ones(np.shape(a1.T[1])) * 0.001e-6) / a1.T[1])
x1_unc = unp.uarray(a1.T[0], np.ones(np.shape(a1.T[0])) * 0.005)



a2 = np.genfromtxt(r"FP4\Boltz\Podatki\T_34_1.txt", delimiter='\t')

y2_unc = unp.uarray(np.log(a2.T[1]), (np.ones(np.shape(a2.T[1])) * 0.001e-6) / a2.T[1])
x2_unc = unp.uarray(a2.T[0], np.ones(np.shape(a2.T[0])) * 0.005)



a3 = np.genfromtxt(r"FP4\Boltz\Podatki\T_53_2.txt", delimiter='\t')

y3_unc = unp.uarray(np.log(a3.T[1]), (np.ones(np.shape(a3.T[1])) * 0.001e-6) / a3.T[1])
x3_unc = unp.uarray(a3.T[0], np.ones(np.shape(a3.T[0])) * 0.005)




# Fitanje in izris grafa za 1. del vaje

fit1_alt = fit_napake_x(x1_unc, y1_unc)
fit2_alt = fit_napake_x(x2_unc, y2_unc)
fit3_alt = fit_napake_x(x3_unc, y3_unc)


fit1 = fit_napake(x1_unc, y1_unc)
fit2 = fit_napake(x2_unc, y2_unc)
fit3 = fit_napake(x3_unc, y3_unc)

# print("AAAAAAAAAAAAAAAA", unc.ufloat(fit1.beta[0], fit1.sd_beta[0]), unc.ufloat(fit1_alt[0], np.sqrt(pcov1_alt[0][0])))


graf_errorbar(x1_unc, y1_unc, "$14{,}1\ ^\circ$C")
graf_errorbar(x2_unc, y2_unc, "$34{,}1\ ^\circ$C")
graf_errorbar(x3_unc, y3_unc, "$53{,}2\ ^\circ$C")

x_limits = plt.xlim()
y_limits = plt.ylim()
plt.ylim(y_limits)
plt.xlim(x_limits)

# graf_fit_tuple(x1_unc, y1_unc, fit1_alt, "")
# graf_fit_tuple(x2_unc, y2_unc, fit2_alt, "")
# graf_fit_tuple(x3_unc, y3_unc, fit3_alt, "")
graf_fit(x1_unc, y1_unc, fit1, "")
graf_fit(x2_unc, y2_unc, fit2, "")
graf_fit(x3_unc, y3_unc, fit3, "")

graf_oblika(r"Diagram ln$(I_C / I_1)$ proti $U_{BE}$ pri različnih temperaturah", r"$U_{BE}$ [V]", r"ln$(I_C / I_1)$")

plt.savefig('FP4\Boltz\Grafi\log_Tok_Napetost.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()




# Izračun relevantnih količin za 1. del naloge

kb1 = (unc.ufloat(fit1.beta[0], fit1.sd_beta[0]) * ((273 + 14.1) / 1.6e-19))**(-1)
kb2 = (unc.ufloat(fit2.beta[0], fit2.sd_beta[0]) * ((273 + 34.1) / 1.6e-19))**(-1)
kb3 = (unc.ufloat(fit3.beta[0], fit3.sd_beta[0]) * ((273 + 53.2) / 1.6e-19))**(-1)

# kb1 = (unc.ufloat(fit1_alt[0][0], np.sqrt(fit1_alt[1][0][0])) * ((273 + 14.1) / 1.6e-19))**(-1)
# kb2 = (unc.ufloat(fit2_alt[0][0], np.sqrt(fit2_alt[1][0][0])) * ((273 + 14.1) / 1.6e-19))**(-1)
# kb3 = (unc.ufloat(fit3_alt[0][0], np.sqrt(fit3_alt[1][0][0])) * ((273 + 14.1) / 1.6e-19))**(-1)

k_B_arr = np.array([kb1, kb2, kb3])

print(f"Vse vrednosti k_B: {k_B_arr}")
print(f"Obteženo povprečje vrednosti k_B: {obtezeno_povprecje(k_B_arr)}")






a4 = np.genfromtxt(r"FP4\Boltz\Podatki\U_0_5.txt", delimiter='\t', encoding="utf-8")

y4_unc = unp.uarray(a4.T[1], (np.ones(np.shape(a4.T[1])) * 0.001e-6)) * 10**3
x4_unc = unp.uarray(a4.T[0], np.ones(np.shape(a4.T[0])) * 0.1) + 273


a5 = np.genfromtxt(r"FP4\Boltz\Podatki\U_0_58.txt", delimiter='\t', encoding="utf-8")

y5_unc = unp.uarray(a5.T[1], (np.ones(np.shape(a5.T[1])) * 0.001e-6)) * 10**3
x5_unc = unp.uarray(a5.T[0], np.ones(np.shape(a5.T[0])) * 0.1) + 273



sorted_indices = np.argsort(x4_unc)
x4_unc = x4_unc[sorted_indices]
y4_unc = y4_unc[sorted_indices]

sorted_indices = np.argsort(x5_unc)
x5_unc = x5_unc[sorted_indices]
y5_unc = y5_unc[sorted_indices]





# def lin_fun3(a, b, c, x):
#     return a * x**b * np.exp( - c / x)

# optimizedParameters5, pcov5 = opt.curve_fit(lin_fun3, unp.nominal_values(x4_unc), unp.nominal_values(y4_unc))#, sigma=unp.std_devs(akt_raz_sqrt), absolute_sigma=True)

# x_fit_mik = np.linspace(unp.nominal_values(x4_unc)[0], unp.nominal_values(x4_unc)[-1], 1000)
# y_fit_mik = lin_fun3(*optimizedParameters5, unp.nominal_values(x_fit_mik))

# plt.plot(unp.nominal_values(x4_unc), unp.nominal_values(y4_unc))
# plt.plot(x_fit_mik, y_fit_mik)

# opt1 = fit_napake(x4_unc, y4_unc, tok_fun, 1)

# x_fit_mik = np.linspace(unp.nominal_values(x4_unc)[0], unp.nominal_values(x4_unc)[-1], 1000)
# y_fit_mik = tok_fun(opt1.beta, unp.nominal_values(x_fit_mik))

# plt.plot(unp.nominal_values(x4_unc), unp.nominal_values(y4_unc))
# plt.plot(x_fit_mik, y_fit_mik)

# plt.show()

plt.plot(unp.nominal_values(x4_unc), unp.nominal_values(y4_unc), "o", label=r"$U = 0{,}5\ $V")
plt.plot(unp.nominal_values(x5_unc), unp.nominal_values(y5_unc), "o", label=r"$U = 0{,}58\ $V")
plt.plot(unp.nominal_values(x4_unc), unp.nominal_values(y4_unc), "--", color="#5e5e5e", linewidth=1)
plt.plot(unp.nominal_values(x5_unc), unp.nominal_values(y5_unc), "--", color="#5e5e5e", linewidth=1)

graf_oblika(r"Temperaturna odvisnost kolektorskega toka", r"$T\ $[K]", "$I_C\ $[mA]", 1)

plt.savefig('FP4\Boltz\Grafi\Tok_Temp.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()


# Graf logaritmiranega I_C

y4_unc_log = unp.uarray(np.log(unp.nominal_values(y4_unc[3:])), unp.std_devs(y4_unc[3:]) / unp.nominal_values(y4_unc[3:]))
y5_unc_log = unp.uarray(np.log(unp.nominal_values(y5_unc[2:])), unp.std_devs(y5_unc[2:]) / unp.nominal_values(y5_unc[2:]))

fit4 = fit_napake(x4_unc[3:], y4_unc_log, lin_fun2, 0)
fit5 = fit_napake(x5_unc[2:], y5_unc_log, lin_fun2, 0)

# graf_errorbar(x4_unc[2:], y4_unc_log, r"$U = 0{,}5\ $V")
# graf_errorbar(x5_unc[2:], y5_unc_log, r"$U = 0{,}58\ $V")

plt.plot(unp.nominal_values(x4_unc[3:]), unp.nominal_values(y4_unc_log), "o", label=r"$U = 0{,}5\ $V")
plt.plot(unp.nominal_values(x5_unc[2:]), unp.nominal_values(y5_unc_log), "o", label=r"$U = 0{,}58\ $V")


x_limits = plt.xlim()
y_limits = plt.ylim()
plt.ylim(y_limits)
plt.xlim(x_limits)

graf_oblika(r"Logaritmirana temperaturna odvisnost kolektorskega toka", r"$T\ $[K]", "log$(I_C / I_1)$", 1)

# graf_fit(x4_unc[3:], y4_unc_log, fit4, "")
# graf_fit(x5_unc[2:], y5_unc_log, fit5, "")

plt.savefig('FP4\Boltz\Grafi\log_Tok_temp.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()