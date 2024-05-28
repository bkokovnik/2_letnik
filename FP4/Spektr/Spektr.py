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
from scipy.optimize import fsolve


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Serif",
    "font.size": 20
})

#####################################################################################################################################################

def custom_fit(a, b, c, x):
    return a + b * x + c * np.sqrt(x)

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
    optimizedParameters, pcov = opt.curve_fit(funkcija, unp.nominal_values(x), unp.nominal_values(y), sigma=unp.std_devs(y), absolute_sigma=True)
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

# Hg_koti = unp.uarray([114.8, 114.9, 115.1, 117.0], [0.1, 0.1, 0.1, 0.1])
# Hg_dejanske = np.array([577, 579, 546, 436])

# umeritev_fit, pcov = fit_napake_x(Hg_dejanske, Hg_koti, custom_fit)

# print(umeritev_fit, pcov)

# plt.plot(unp.nominal_values(Hg_dejanske), unp.nominal_values(Hg_koti), ".")

# x_fit_mik = np.linspace(np.min(unp.nominal_values(Hg_dejanske)) + np.min(unp.nominal_values(Hg_dejanske)), np.max(unp.nominal_values(Hg_dejanske)) - 2 * np.max(unp.nominal_values(Hg_dejanske)), 1000)

# plt.plot(x_fit_mik, custom_fit(*umeritev_fit, x_fit_mik))

# plt.show()



# Define the function kot(x) = a + b * x + c * sqrt(x)
def kot(x, a, b, c):
    return a + b * x + c * np.sqrt(x)

# Given data
Hg_koti = unp.uarray([114.9, 114.8, 115.1, 117.0], [0.05, 0.05, 0.05, 0.05])
Hg_dejanske = np.array([577, 579, 546, 436])

# Extract nominal values and standard deviations
Hg_koti_nominal = unp.nominal_values(Hg_koti)
Hg_koti_std = unp.std_devs(Hg_koti)

# Fit the curve
popt, pcov = curve_fit(kot, Hg_dejanske, Hg_koti_nominal, sigma=Hg_koti_std, absolute_sigma=True)

# Extract the fitted parameters
a, b, c = popt

# Generate points for the fitted curve
x_fit = np.linspace(min(Hg_dejanske) - min(Hg_dejanske), max(Hg_dejanske) + max(Hg_dejanske), 2000)
y_fit = kot(x_fit, *popt)

# Calculate the uncertainty in the fit
perr = np.sqrt(np.diag(pcov))

print(f"fit: {popt}\nNegotovost fita: {perr}")

# Plotting

plt.errorbar(Hg_dejanske, Hg_koti_nominal, yerr=Hg_koti_std, fmt='o', label='Meritve', capsize=5)

x_limits = plt.xlim()
y_limits = plt.ylim()

plt.plot(x_fit, y_fit, "--", label=f'Fit', color="#5e5e5e", linewidth=1)
# # plt.fill_between(x_fit, y_fit_lower, y_fit_upper, color='gray', alpha=0.5, label='95% Confidence Interval')
# plt.xlabel(r"$\lambda$ [nm]")
# plt.ylabel(r"$\varphi$ [$^{\circ}$]")
# plt.legend()
# plt.title('Umeritvena krivulja za pretvorbo med koti in valovnimi dolžinami')

plt.ylim(y_limits)
plt.xlim(x_limits)
graf_oblika('Umeritvena krivulja za pretvorbo med koti in valovnimi dolžinami',  r"$\lambda$ [nm]", r"$\varphi$ [$^{\circ}$]",1)


plt.savefig('FP4/Spektr/Slike/Umeritev.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()



# Given function kot(x) = a + b * x + c * sqrt(x)
def kot(x, a, b, c):
    return a + b * x + c * np.sqrt(x)

# Function to find x given y analytically for NumPy arrays
def find_x_analytical_array(y_array):
    # Define a function that represents the quadratic equation
    def equation(x):
        return (a + b * x + c * np.sqrt(x))**2 - y_array**2
    
    # Use fsolve to find the roots of the equation
    x_roots = fsolve(equation, np.zeros_like(y_array))  # Start with initial guess of zeros
    return x_roots

# # Main function to convert y to x
# def convert_y_to_x(uarray):
#     vrednosti = unp.nominal_values(uarray)
#     napake = unp.std_devs(uarray)

#     x_values = find_x_analytical_array(vrednosti)
#     x_plus = find_x_analytical_array(vrednosti + napake)
#     return unp.uarray(x_values, np.abs(x_plus - x_values))


def convert_y_to_x(y_array):
    # Define a function that represents the difference between kot(x) and the target y value
    def difference(x, y):
        return kot(x, a, b, c) - y
    
    # Use fsolve to find the x value that corresponds to the target y value for each y in the array
    x_values = np.zeros_like(y_array)
    for i, y in enumerate(y_array):
        x_values[i] = fsolve(difference, 0, args=(y,))[0]
    return x_values





varcna = np.array([114.6, 115.2, 117, 116, 114.9])
zelena = np.array([115, 114.8, 115.1])
rdeca = np.array([114.4, 114.2, 114.7])
rumena = np.array([114.7, 114.6, 114.9])
modra = np.array([116.3, 115.6, 116.8])
wolfram = np.array([115, 113.9, 114.4, 114.8, 115.6, 116.5, 117.9])
absorpcija = np.array([115.4, 115.7, 115.8, 115.9, 115.0, 114.7, 114.6, 114.9, 115.2, 115.5])


print(f"Varčna: {convert_y_to_x(varcna)}")
print(f"Zelena: {convert_y_to_x(zelena)}")
print(f"Rdeča: {convert_y_to_x(rdeca)}")
print(f"Rumena: {convert_y_to_x(rumena)}")
print(f"Modra: {convert_y_to_x(modra)}")
print(f"Wolfram: {convert_y_to_x(wolfram)}")
print(f"Absorpcija: {convert_y_to_x(absorpcija)}")

# print(convert_y_to_x(unp.nominal_values(varcna)))
# print(convert_y_to_x(unp.nominal_values(varcna)) - convert_y_to_x(unp.nominal_values(varcna) + unp.std_devs(varcna)))