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
    "font.size": 18
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

def fit_fun(B, x):
    a = 1.1 / (8.617 * 10 ** (-5))
    return (1 - 15 / (np.pi) ** 4 * (-(a / x) ** 3 * np.log(1 - np.e ** ( - (a / x))) + (6 + 6 * (a / x) + 3 * (a / x) ** 2) * np.e ** (- (a / x)))) * B[0]

def fit_fun1(b, x):
    a = 1.1 / (8.617 * 10 ** (-5))
    return (1 - 15 / (np.pi) ** 4 * (-(a / x) ** 3 * np.log(1 - np.e ** ( - (a / x))) + (6 + 6 * (a / x) + 3 * (a / x) ** 2) * np.e ** (- (a / x)))) * b

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
    optimizedParameters, pcov = opt.curve_fit(funkcija, unp.nominal_values(x), unp.nominal_values(y))#, sigma=unp.std_devs(y), absolute_sigma=True)
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

    x_fit_mik = np.linspace(np.min(x_mik) - np.abs(np.min(x_mik)) / 2, np.max(x_mik) + np.abs(np.max(x_mik)), 5000)

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

    if legenda == 1:
        plt.legend()

    plt.xlabel(x_os)
    plt.ylabel(y_os)
    plt.title(Naslov, y=1.02)

#####################################################################################################################################################

P_el = unp.uarray([31.7, 30, 28.5, 27, 25.5, 24, 22.5, 21, 19.5, 18, 16.5, 15, 13.5, 12, 10.5, 9, 7.5, 6, 4.5, 3, 1.5, 0],
                  np.ones(np.shape(np.array([31.7, 30, 28.5, 27, 25.5, 24, 22.5, 21, 19.5, 18, 16.5, 15, 13.5, 12, 10.5, 9, 7.5, 6, 4.5, 3, 1.5, 0]))) * 0.2)

U = unp.uarray([230, 223, 215, 208, 200, 193, 185, 177.2, 168, 159.1, 150.5, 142.3, 132.8, 123.1, 113.1, 101.6, 90.7, 78.1, 63.9, 49.5, 31.6, 0],
               np.ones(np.shape(np.array([230, 223, 215, 208, 200, 193, 185, 177.2, 168, 159.1, 150.5, 142.3, 132.8, 123.1, 113.1, 101.6, 90.7, 78.1, 63.9, 49.5, 31.6, 0]))) * 0.3)

I = unp.uarray([137.3, 134.7, 132.2, 129.7, 127, 124.6, 121.9, 119.3, 115.8, 112.9, 109.1, 105.9, 102.1, 98.1, 93.7, 88.5, 83.3, 77, 69.3, 60.9, 48.8, 0],
               np.ones(np.shape(np.array([137.3, 134.7, 132.2, 129.7, 127, 124.6, 121.9, 119.3, 115.8, 112.9, 109.1, 105.9, 102.1, 98.1, 93.7, 88.5, 73.3, 77, 69.3, 60.9, 48.8, 0]))) * 0.2) * 10**(-3)

P_m = unp.uarray([2, 1.89, 1.78, 1.68, 1.58, 1.49, 1.39, 1.28, 1.18, 1.07, 0.965, 0.875, 0.769, 0.679, 0.583, 0.481, 0.390, 0.298, 0.209, 0.144, 0.060, 0.03],
                 np.ones(np.shape(np.array([2, 1.89, 1.78, 1.68, 1.58, 1.49, 1.39, 1.28, 1.18, 1.07, 0.965, 0.875, 0.769, 0.679, 0.583, 0.481, 0.390, 0.298, 0.209, 0.144, 0.060, 0.03]))) * 0.02) * 10**(-3)

P_m = P_m - P_m[-1]

P_cel = (4 * np.pi * (unc.ufloat(34, 0.5) + unc.ufloat(1.95, 0.01)) ** 2) * P_m / 1

graf_errorbar(P_el, P_cel, "Neovirano")

fit1 = fit_napake_x(P_el, P_cel, lin_fun)


U_ok = unp.uarray([230.2, 222.2, 215.2, 208, 200, 192.6, 184.6, 177, 168.4, 160, 151.0, 142, 132, 123, 113, 102, 91, 78, 65, 49, 32, 0],
                  np.ones(np.shape(np.array([230, 223, 215, 208, 200, 193, 185, 177.2, 168, 159.1, 150.5, 142.3, 132.8, 123.1, 113.1, 101.6, 90.7, 78.1, 63.9, 49.5, 31.6, 0]))) * 0.3)

I_ok = unp.uarray([137.7, 134.9, 132.7, 130, 127.2, 124.6, 121.7, 119.0, 115.9, 112.7, 109.4, 105.8, 101.9, 98.0, 93.7, 88.7, 83.3, 77.1, 69.8, 60.6, 48.8, 0],
                  np.ones(np.shape(np.array([137.3, 134.7, 132.2, 129.7, 127, 124.6, 121.9, 119.3, 115.8, 112.9, 109.1, 105.9, 102.1, 98.1, 93.7, 88.5, 73.3, 77, 69.3, 60.9, 48.8, 0]))) * 0.2) * 10**(-3)

P_m_ok = unp.uarray([605, 591, 583, 573, 542, 520, 493, 469, 430, 404, 375, 346, 302, 268, 232, 190, 152, 117, 75, 35, 0, -70],
                    np.ones(np.shape(np.array([2, 1.89, 1.78, 1.68, 1.58, 1.49, 1.39, 1.28, 1.18, 1.07, 0.965, 0.875, 0.769, 0.679, 0.583, 0.481, 0.390, 0.298, 0.209, 0.144, 0.060, 0.03]))) * 3) * 10**(-6)

P_m_ok = P_m_ok - P_m_ok[-1]

P_cel_ok = (4 * np.pi * (unc.ufloat(34, 0.5) + unc.ufloat(1.95, 0.01)) ** 2) * P_m_ok / 1



graf_errorbar(P_el, P_cel_ok, "Skozi silicijevo okno")

x_limits = plt.xlim()
y_limits = plt.ylim()

graf_oblika(r"Oddana moč žarnice v odvisnosti od električne moči", r"$P_{el}$ [W]", r"$P_{oddana}$ [W]")

plt.ylim(y_limits)
plt.xlim(x_limits)

graf_fit(P_el, P_cel, fit1, "")

plt.savefig(r"FP4\SevCT\Slike\Moc.pdf", bbox_inches='tight', pad_inches=0.1)
plt.show()

print(f"izkoristek: {unc.ufloat(fit1[0][0], np.sqrt(fit1[1][0][0]))}")



### Upornost nitke v odvisnosti od temperature

P_0 = 30
T_0 = 2700

T_x = (P_cel / P_0) ** (1/4) * T_0

T_x_s = np.concatenate((T_x[0:-1], unp.uarray([22.8 + 273], [0.1])))

R = np.concatenate((U[0:-1] / I[0:-1], unp.uarray([121.5], [0.5])))

graf_errorbar(T_x_s, R)

x_limits = plt.xlim()
y_limits = plt.ylim()
plt.ylim(y_limits)
plt.xlim(x_limits)

fit_R_T = fit_napake(T_x_s, R)

graf_fit(T_x_s, R, fit_R_T)

graf_oblika(r"Upornost nitke v odvisnosti od temperature", r"$T$ [K]", r"$R$ [$\Omega$]")

plt.savefig(r"FP4\SevCT\Slike\Upornost.pdf", bbox_inches='tight', pad_inches=0.1)
plt.show()

print(f"Upor/T: {unc.ufloat(fit_R_T.beta[0], fit_R_T.sd_beta[0])}")




### Zadnji del vaje

graf_errorbar(T_x[0:-2], P_cel_ok[0:-2] / P_cel[0:-2])

x_limits = plt.xlim()
y_limits = plt.ylim()
plt.ylim(y_limits)
plt.xlim(x_limits)


plt.plot(unp.nominal_values(T_x)[0:-1], fit_fun(np.array([1]), unp.nominal_values(T_x)[0:-1]), label="Model brez odbojnosti")

# optimizedParameters, pcov = opt.curve_fit(fit_fun1, unp.nominal_values(T_x[0:-2]), unp.nominal_values(P_cel_ok[0:-2] / P_cel[0:-2]), sigma=unp.std_devs(P_cel_ok[0:-2] / P_cel[0:-2]), absolute_sigma=True)

# print(optimizedParameters)

# plt.plot(unp.nominal_values(T_x)[0:-1], fit_fun(np.array([1.1 / (8.617 * 10 ** (-5)), 1]), unp.nominal_values(T_x)[0:-1]) * optimizedParameters[0])

# plt.plot(unp.nominal_values(T_x), fit_fun1(unp.nominal_values(T_x), *fit))

fit_P = fit_napake(T_x[0:-2], P_cel_ok[0:-2] / P_cel[0:-2], fit_fun)

x_fit_mik = np.linspace(np.min(unp.nominal_values(T_x)[0:-2]) - np.abs(np.min(unp.nominal_values(T_x)[0:-2])) / 2, np.max(unp.nominal_values(T_x)[0:-2]) + np.abs(np.max(unp.nominal_values(T_x)[0:-2])), 5000)

print(fit_P.beta, fit_P.sd_beta)

y_fit_mik = fit_fun(fit_P.beta, x_fit_mik)
plt.plot(x_fit_mik, y_fit_mik, "--", label="Model z odbojnostjo", color="#5e5e5e", linewidth=1)

nu = unc.ufloat(fit_P.beta[0], fit_P.sd_beta[0])

# plt.plot(unp.nominal_values(T_x)[0:-1], fit_fun(np.array([2 * 3.54 / (1 + 3.54 ** 2)]), unp.nominal_values(T_x)[0:-1]))

print((2 + (4 - 4 * nu ** 2) ** (0.5)) / (2 * nu))

graf_oblika("Delež prepuščene moči skozi silicijevo steklo\nvodvisnosti od temperature", r"$T$ [K]", r"$P_{Si} / P$")

plt.savefig(r"FP4\SevCT\Slike\Prepusceno.pdf", bbox_inches='tight', pad_inches=0.1)
plt.show()