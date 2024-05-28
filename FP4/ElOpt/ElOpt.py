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
    odr_mik = ODR(data_mik, lin_model_mik, beta0=[0.25 * 10 ** (-6), 15.75 * 10 ** (-6), 2.850 * 10 ** (-6), -0.74, 1], maxit=100000)

    # odr_mik.set_job(maxit=10000)

    # Run the regression.
    out_mik = odr_mik.run()

    if print == 1:
        out_mik.pprint()
    
    return out_mik

def fit_napake2(x: unp.uarray, y: unp.uarray, funkcija=lin_fun2, print=0) -> np.ndarray:

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
    odr_mik = ODR(data_mik, lin_model_mik, beta0=[0 * 10**(-6), 8 * 10**(-5), 1.5], maxit=100000)

    # odr_mik.set_job(maxit=10000)

    # Run the regression.
    out_mik = odr_mik.run()

    if print == 1:
        out_mik.pprint()
    
    return out_mik

def fit_napake_x(x: unp.uarray, y: unp.uarray, funkcija=lin_fun2, print=0) -> tuple:
    '''Sprejme 2 unumpy arraya skupaj z funkcijo in izračuna fit, upoštevajoč y napake'''
    optimizedParameters, pcov = opt.curve_fit(funkcija, unp.nominal_values(x), unp.nominal_values(y), sigma=unp.std_devs(x), absolute_sigma=True)
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

def graf_fit(x: unp.uarray, y: unp.uarray, fit: np.ndarray, funkcija=lin_fun2, fit_label="Fit"):
    '''Sprejme 2 unumpy arraya, fitane parametre in nariše črtkan fit'''
    # Podatki
    x_mik = unp.nominal_values(x)
    y_mik = unp.nominal_values(y)

    x_fit_mik = np.linspace(np.min(x_mik) - np.abs(np.max(x_mik)) / 2, np.max(x_mik) + np.abs(np.max(x_mik)) / 2, 5000)

    if type(fit) is tuple:
        y_fit_mik = funkcija(x_fit_mik, *fit[0])
        plt.plot(x_fit_mik, y_fit_mik, "--", label=fit_label, color="#5e5e5e", linewidth=1)

    else:
        y_fit_mik = funkcija(fit.beta, x_fit_mik)
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

### 1. del vaje

phi1 = np.array([-12, -7, -2, 3, 8, 13, 18, 23, 28, 32, 38, 43, 48, 53, 58, 63, 68, 73, 78])
I1 = np.array([202.8, 201, 198.5, 196.8, 189.6, 173.6, 162.3, 147.7, 132.8, 115, 88.2, 73.4, 59.5, 44.1, 28.3, 15.8, 7.4, 2.3, 0.3])

phi1 = unp.uarray(phi1, np.ones(np.shape(phi1))) * np.pi / 180
I1 = unp.uarray(I1, np.ones(np.shape(I1)) * 0.2) * 10 ** (-6)


def fun1(B, x):
    return B[0] + B[1] * (np.sin(x + B[2])) ** 2
# def fun1_1(x, a, b, c):
#     return a + b * (np.sin(x + c)) ** 2

# optimizedParameters, pcov = opt.curve_fit(fun1_1, unp.nominal_values(phi1), unp.nominal_values(I1), np.array([1, 251, ]), sigma=unp.std_devs(I1),  absolute_sigma=True)


# fit1 = fit_napake(phi1, I1, fun1, 1)

# # graf_errorbar(phi1, I1)
# plt.plot(unp.nominal_values(phi1), unp.nominal_values(I1), "o", label="Izmerjeno")

# x_limits = plt.xlim()
# y_limits = plt.ylim()
# plt.ylim(y_limits)
# plt.xlim(x_limits)

# graf_fit(phi1, I1, fit1, fun1)
# graf_oblika("Intenziteta v odvisnosti od kota polarizatorja", r"$\phi$ [rad]", r"$I$ [$\mu$A]")

# plt.show()



### 2. del vaje

phi2 = np.array([90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0])
phi2 = np.concatenate((phi2, - np.flip(phi2)[1:]))

I2 = np.array([4.7, 1.2, 0.2, 1.9, 5.9, 10.3, 18.9, 26.2, 33.8, 39.4, 42.6, 42.8, 39.9, 37.0, 30.7, 23.1, 17.4, 10.6, 5.1, 1.5, 0.3, 1.6, 4.9, 9.9, 17, 24.2, 32.2, 37.5, 39.3, 37.0, 36.9, 36.1, 33.4, 27.5, 21.1, 15.5, 9.3])

phi2 = unp.uarray(phi2, np.ones(np.shape(phi2))) * np.pi / 180
I2 = unp.uarray(I2, np.ones(np.shape(I2)) * 0.2) * 10 ** (-6)

def fun2(B, x):
    return B[0] + B[1] * (np.sin(2 * x + B[2])) ** 2

# fit2 = fit_napake(phi2, I2, fun2, 1)

# graf_errorbar(phi2, I2)

# x_limits = plt.xlim()
# y_limits = plt.ylim()
# plt.ylim(y_limits)
# plt.xlim(x_limits)

# # graf_fit(phi2, I2, fit2, fun2)
# plt.plot(unp.nominal_values(phi2), unp.nominal_values(I2), "o", label="Izmerjeno")
# graf_oblika("Intenziteta v odvisnosti od kota polarizatorja", r"$\phi$ [rad]", r"$I$ [$\mu$A]")

# plt.show()

# plt.plot(phi2, I2)
# plt.show()



### 3. del vaje

U3 = np.array([0, 0.06, 0.12, 0.18, 0.24, 0.3, 0.36, 0.42, 0.48, 0.54, 0.6, 0.66, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.8, 0.78, 0.76, 0.74, 0.72, 0.7, 0.66, 0.6, 0.54, 0.48, 0.42, 0.36, 0.3, 0.24, 0.18, 0.12, 0.06, 0]) * 10 ** 3

I3 = np.array([0.3, 0.3, 0.28, 0.26, 0.24, 0.24, 0.28, 0.44, 0.92, 2.1, 4.4, 8.6, 12.4, 13.7, 14.6, 15.4, 15.8, 15.9, 15.7, 15.8, 15.6, 15.1, 14.4, 13.5, 12.4, 10, 6.2, 3.2, 1.6, 0.67, 0.35, 0.22, 0.19, 0.19, 0.19, 0.18, 0.17])

I3 = I3 * 10 ** (-6)

U3_unc = unp.uarray(U3, np.ones(np.shape(U3)) * 10)
I3_unc = unp.uarray(I3, np.ones(np.shape(U3)) * 0.1 * 10 ** (-6))

def fun3(B, x):
    return B[0] + B[1] * (np.sin(B[2] * x ** 2 + B[3] / 2)) ** 2
# def fun3(x, a, b, c, d):
#     return a + b * (np.sin(c * x ** 2 + d / 2)) ** 2

# fit3 = fit_napake(U3_unc, I3_unc, fun3, 1)

# # # graf_errorbar(U3, I3)
# plt.plot(unp.nominal_values(U3), unp.nominal_values(I3), "o", label="Izmerjeno")

# # x_limits = plt.xlim()
# # y_limits = plt.ylim()
# # plt.ylim(y_limits)
# # plt.xlim(x_limits)

# graf_fit(U3_unc, I3_unc, fit3, fun3)
# graf_oblika("Intenziteta v odvisnosti od kota polarizatorja", r"$\phi$ [rad]", r"$I$ [$\mu$A]")

# # plt.show()

# # plt.plot(U3, I3)
# # plt.show()

# # optimizedParameters, pcov = opt.curve_fit(fun3, unp.nominal_values(U3), unp.nominal_values(I3),
# #                                           np.array([0.25 * 10 ** (-6), 15.75 * 10 ** (-6), 2.850 * 10 ** (-6), -0.74]),
# #                                           bounds=(np.array([0.20 * 10 ** (-6), 15.5 * 10 ** (-6), 2.70 * 10 ** (-6), -0.8]),
# #                                                   np.array([0.30 * 10 ** (-6), 16 * 10 ** (-6), 3 * 10 ** (-6), -0.7])))

# # a = np.linspace(-1, -0.5, 50)
# # plt.plot(U3, fun3(U3, *optimizedParameters))


# # print(optimizedParameters)
# # for i in a:
# #     plt.plot(U3, fun3(U3, 0.25 * 10 ** (-6), 15.8 * 10 ** (-6), 2.850 * 10 ** (-6), i))
# # plt.plot(U3, fun3(U3, 0.25 * 10 ** (-6), 15.75 * 10 ** (-6), 2.850 * 10 ** (-6), -0.74))
# plt.plot(U3, I3, ".")
# plt.show()


### 4. del vaje

phi4 = np.array([90, 80, 70, 60, 50, 40, 30, 20, 10, 0, -10, -20, -30, -40, -50, -60, -70, -80, -90])
I4 = np.array([25.1, 15.6, 8.3, 4.1, 3.6, 6.6, 13.2, 22.8, 32.2, 43.4, 52.9, 61.3, 64.2, 60.8, 58.3, 56.3, 43.9, 33.9, 23.1])

phi4 = unp.uarray(phi4, np.ones(np.shape(phi4))) * np.pi / 180
I4 = unp.uarray(I4, np.ones(np.shape(I4)) * 0.2) * 10 ** (-6)


# fit4 = fit_napake(phi4, I4, fun1)

# # graf_errorbar(phi4, I4)
# plt.plot(unp.nominal_values(phi4), unp.nominal_values(I4), "o", label="Izmerjeno")

# x_limits = plt.xlim()
# y_limits = plt.ylim()
# plt.ylim(y_limits)
# plt.xlim(x_limits)

# graf_fit(phi4, I4, fit4, fun1)

# graf_oblika("Amplituda v odvisnosti od\nkota polarizatorja za tekočim kristalom", r"$\phi$ [rad]", r"$I$ [A]")



# # plt.plot(phi4, I4)
# plt.show()



### 5. del vaje

phi5 = np.array([110, 105, 100, 98, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30, -35, -40, -45, -50, -55, -60, -65, -68, -70, -73, -75, -78, -80, -83, -85, -88, -90, -93, -95, -98, -100, -105, -110])
I5 = np.array([6.4, 12.2, 29, 40, 45.1, 45.6, 42, 38, 33.7, 27.8, 24.1, 19.1, 15.3, 11.7, 9.8, 7.9, 6.2, 4.5, 3.9, 3.8, 2.7, 2.8, 2.1, 2.6, 2.1, 2.9, 2.4, 2.9, 4.5, 4.9, 6.7, 7.8, 8.5, 10.5, 13.2, 16, 19.7, 21.7, 23.7, 25.9, 28.4, 31.8, 33.5, 36, 38.1, 40.6, 43.9, 46.7, 48.7, 50.4, 52.2, 54, 55.2])

phi5 = unp.uarray(phi5, np.ones(np.shape(phi5))) * np.pi / 180
I5 = unp.uarray(I5, np.ones(np.shape(I5)) * 0.2) * 10 ** (-6)


def fun5(B, x):
    n1 = 1.532
    n2 = 1.706
    return B[0] + B[1] * (np.sin(18.1 * (np.sqrt(n1**2 - (np.sin((x/2) * np.pi/B[2]))**2 ) - np.sqrt(n2**2 - (np.sin((x/2) * np.pi/B[2]))**2 ))))**2
# def fun5(B, x):
#     n1 = 1
#     n2 = 1
#     return B[0] + B[1] * (np.sin(B[2] * (np.sqrt(n1**2 - (np.sin(np.pi/B[3] * (x - B[4])))**2 ) - np.sqrt(n2**2 - (np.sin(np.pi/B[3] * (x - B[4])))**2 ))))**2

fit5 = fit_napake2(phi5, I5, fun5)

fit5.pprint()

graf_fit(phi5, I5, fit5, fun5, 1)
# plt.plot(unp.nominal_values(phi5), fun5([0 * 10**(-6), 8 * 10**(-5), 18.2, 0, 1.5], unp.nominal_values(phi5)))
plt.plot(unp.nominal_values(phi5), unp.nominal_values(I5), "o")
plt.show()