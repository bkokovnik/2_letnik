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

    if legenda == 1:
        plt.legend()

    plt.xlabel(x_os)
    plt.ylabel(y_os)
    plt.title(Naslov, y=1.02)

#####################################################################################################################################################

### Umeritev

# Dolga
l_dolga = unc.ufloat(100, 0.1) * 10**(-3)
vrhovi_dolga = (unp.uarray([1.63, 5.02, 8.37], [0.05, 0.05, 0.05]) + unc.ufloat(1.78, 0.05)) * 10**(-5)
casi_dolga = []

i = 1
for element in vrhovi_dolga:
    element = element / i
    casi_dolga.append(element)
    i += 1

casi_dolga = np.asarray(casi_dolga)

hitrosti_dolga = (2 * l_dolga) / casi_dolga

print(hitrosti_dolga)




# Kratka
l_kratka = unc.ufloat(25, 0.1) * 10**(-3)
vrhovi_kratka = (unp.uarray([-0.89, -0.07, 0.80, 1.61, 2.46, 3.31, 4.15], [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]) + unc.ufloat(1.78, 0.05)) * 10**(-5)
casi_kratka = []

i = 1
for element in vrhovi_kratka:
    element = element / i
    casi_kratka.append(element)
    i += 1

casi_kratka = np.asarray(casi_kratka)

hitrosti_kratka = (2 * l_kratka) / casi_kratka

print(hitrosti_kratka)




# Zareza
l_zareza = unc.ufloat(90, 0.1) * 10**(-3)
vrhovi_zareza = (unp.uarray([1.32, 4.40, 7.44, 10.54], [0.05, 0.05, 0.05, 0.05]) + unc.ufloat(1.78, 0.05)) * 10**(-5)
casi_zareza = []

i = 1
for element in vrhovi_zareza:
    element = element / i
    casi_zareza.append(element)
    i += 1

casi_zareza = np.asarray(casi_zareza)

hitrosti_zareza = (2 * l_zareza) / casi_zareza

print(hitrosti_zareza)


c_jeklo = obtezeno_povprecje(np.concatenate((hitrosti_dolga, hitrosti_kratka, hitrosti_zareza), axis=0))

print(f"Hitrost zvoka v jeklu: {c_jeklo}")





### Meritev zarez

vrhovi_zareze = (unp.uarray([1.11, 1.32, 1.63], [0.05, 0.05, 0.05]) + unc.ufloat(1.76, 0.05)) * 10**(-5)

tri_zareze = vrhovi_zareze * c_jeklo / 2

print(f"Dimenzije treh zarez, ene zraven druge: {tri_zareze * 1000}")



### Hitrost v materialih

## Longitudinalno

c_vode = 1483.1 + 2.5 * (unc.ufloat(0.6, 0.2))
rho_jekla = 7800
rho_aluminija = 2700




# Aluminij

l_aluminij = unc.ufloat(20.04, 0.01) / 1000

cas_aluminij = (unc.ufloat(-7.8, 1) - unc.ufloat(-16.5, 1)) * 10**(-6)

cas_aluminij = np.asarray(cas_aluminij)

d_voda_alu = cas_aluminij * c_vode / 2

print(f"razdalja v vodi pri aluminiju: {d_voda_alu * 1000}")


# Jeklo

l_jeklo = unc.ufloat(20.45, 0.01) / 1000

cas_jeklo = (unc.ufloat(15.9, 1) - unc.ufloat(7.5, 1)) * 10**(-6)

cas_jeklo = np.asarray(cas_jeklo)

d_voda_jek = cas_jeklo * c_vode / 2

print(f"razdalja v vodi pri jeklu: {d_voda_jek * 1000}")



## Transverzalno

#Aluminij

cas_aluminij_trans = (unc.ufloat(-0.7, 1) - unc.ufloat(-15.9, 1)) * 10**(-6)

cas_aluminij_trans = np.asarray(cas_aluminij_trans)

d_voda_alu_trans = cas_aluminij_trans * c_vode / 2

print(f"razdalja v vodi pri aluminiju: {d_voda_alu_trans * 1000}")


# Jeklo

cas_jeklo_trans = (unc.ufloat(15.9, 1) - unc.ufloat(0.06, 1)) * 10**(-6)

cas_jeklo_trans = np.asarray(cas_jeklo_trans)

d_voda_jek_trans = cas_jeklo_trans * c_vode / 2

print(f"razdalja v vodi pri jeklu: {d_voda_jek_trans * 1000}")

c_a_l = l_aluminij * c_vode / d_voda_alu
c_j_l = l_jeklo * c_vode / d_voda_jek
c_a_t = l_aluminij * c_vode / d_voda_alu_trans
c_j_t = l_jeklo * c_vode / d_voda_jek_trans

print(f"Longitudinalna, aluminij: {l_aluminij * c_vode / d_voda_alu}")
print(f"Longitudinalna, jeklo: {l_jeklo * c_vode / d_voda_jek}")
print(f"Transverzalna, aluminij: {l_aluminij * c_vode / d_voda_alu_trans}")
print(f"Transverzalna, jeklo: {l_jeklo * c_vode / d_voda_jek_trans}")


p_j = (2 * c_j_t**2 - c_j_l**2) / (2 * (c_j_t**2 - c_j_l**2))
p_a = (2 * c_a_t**2 - c_a_l**2) / (2 * (c_a_t**2 - c_a_l**2))

print(f"Poisson jeklo: {(2 * c_j_t**2 - c_j_l**2) / (2 * (c_j_t**2 - c_j_l**2))}")
print(f"Poisson aluminij: {(2 * c_a_t**2 - c_a_l**2) / (2 * (c_a_t**2 - c_a_l**2))}")

print(f"E jeklo: {(c_j_l**2 * rho_jekla * (1 + p_j) * (1 - 2 * p_j)) / (1 - p_j)}")
print(f"E aluminij: {(c_a_l**2 * rho_aluminija * (1 + p_a) * (1 - 2 * p_a)) / (1 - p_a)}")


print(f"G jeklo: {rho_jekla * c_j_t**2}")
print(f"G aluminij: {rho_aluminija * c_a_t**2}")