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
    optimizedParameters, pcov = opt.curve_fit(lin_fun, unp.nominal_values(x), unp.nominal_values(y))#, sigma=unp.std_devs(y), absolute_sigma=True)
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

    x_fit_mik = np.linspace(x_mik[0] - 2 * x_mik[0], x_mik[-1] + x_mik[-1], 1000)

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
        y_fit_mik = lin_fun(fit[0][0], fit[0][1], x_fit_mik)
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

S189_t = [(-35.4, 1.008), (-30.4, 0.82), (-25.2, 0.5422), (-20.2, 0.3467), (-15.6, 0.1432)]
S160_t = [(-35.4, 0.93), (-30.4, 0.7714), (-25.4, 0.5222), (-20.4, 0.2967), (-15.6, 0.1482)]
S138_t = [(-31.4, 2.0733), (-26.0, 1.492), (-20.6, 1.06), (-15.2, 0.724), (-10.4, 0.4767)]
S119_t = [(-21.0, 2.1333), (-15.6, 1.5846), (-10.4, 0.8383), (-5.8, 0.5752)]

S189 = []
S160 = []
S138 = []
S119 = []

x1 = [1, 2, 3, 4, 5]
x2 = [2, 3, 4, 5]


i = 0
for U in S189_t:
    S189.append(U[0])

i = 0
for U in S160_t:
    S160.append(U[0])

i = 0
for U in S138_t:
    S138.append(U[0])

i = 0
for U in S119_t:
    S119.append(U[0])


# print(S189, S160, S138, S119)

S189 = unp.uarray(S189, [0.1 for i in range(len(S189))])
S160 = unp.uarray(S160, [0.1 for i in range(len(S189))])
S138 = unp.uarray(S138, [0.1 for i in range(len(S189))])
S119 = unp.uarray(S119, [0.1 for i in range(len(S119))])

fit_189 = fit_napake_x(x1, S189, print=1)
fit_160 = fit_napake_x(x1, S160, print=1)
fit_138 = fit_napake_x(x1, S138, print=1)
fit_119 = fit_napake_x(x2, S119, print=1)

graf_errorbar(x1, S189, "$T = 189\ $ [$^\circ$C]")
graf_errorbar(x1, S160, "$T = 160\ $ [$^\circ$C]")
graf_errorbar(x1, S138, "$T = 138\ $ [$^\circ$C]")
graf_errorbar(x2, S119, "$T = 119\ $ [$^\circ$C]")

x_limits = plt.xlim()
y_limits = plt.ylim()
plt.ylim(y_limits)
plt.xlim(x_limits)

graf_fit_tuple(x1, S189, fit_189, "")
graf_fit_tuple(x1, S160, fit_160, "")
graf_fit_tuple(x1, S138, fit_138, "")
graf_fit_tuple(x2, S119, fit_119, "")

graf_oblika(r"Napetosti zaporednih vrhov", r"$N$", r"$U_1$ [V]")


plt.savefig('FP4\FraHertz\Slike\Delte.pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()

de189 = unc.ufloat(fit_189[0][0], np.sqrt(fit_189[1][0][0]))
de160 = unc.ufloat(fit_160[0][0], np.sqrt(fit_160[1][0][0]))
de138 = unc.ufloat(fit_138[0][0], np.sqrt(fit_138[1][0][0]))
de119 = unc.ufloat(fit_119[0][0], np.sqrt(fit_119[1][0][0]))

print(f"T = 189: {de189}")
print(f"T = 160: {de160}")
print(f"T = 138: {de138}")
print(f"T = 119: {de119}")

print(f"Obteženo povprečje vseh: {obtezeno_povprecje(np.array([de189, de160, de138, de119]))}")



# plt.plot(x1, S189)
# plt.plot(x1, S160)
# plt.plot(x1, S138)
# plt.plot(x2, S119)

# plt.show()
