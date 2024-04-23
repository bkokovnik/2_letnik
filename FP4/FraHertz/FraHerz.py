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

with open("FP4\FraHertz\Podatki\TEK0008.CSV", "r") as dat:
    x = []
    cas = []
    i = 1
    for vrstica in dat:
        while i < 19:
            i +=1
            break

        else:
            G = vrstica.strip("\n").replace(",", "").split(" ")
            t = float(G[0])
            U = float(G[-1])
            # print(U, t)
            x.append(U)
            cas.append(t)

# plt.plot(cas, x)
# plt.show()



with open("FP4\FraHertz\Podatki\TEK0009.CSV", "r") as dat:
    y = []
    cas = []
    i = 1
    for vrstica in dat:
        while i < 19:
            i +=1
            break

        else:
            G = vrstica.strip("\n").replace(",", "").split(" ")
            t = float(G[0])
            U = float(G[-1])
            # print(U, t)
            y.append(U)
            cas.append(t)


# plt.plot(x, y)
# plt.show()

x = np.array(x)
y = np.array(y) / (1e6)

sorted_indices = np.argsort(x)
x = x[sorted_indices]
y = y[sorted_indices]

# plt.plot(x, y)
# plt.show()

def average_y_at_unique_x(x, y):
    unique_x = np.unique(x)
    averaged_y = np.zeros_like(unique_x, dtype=float)
    counts = np.zeros_like(unique_x, dtype=int)

    for i, ux in enumerate(unique_x):
        indices = np.where(x == ux)[0]
        averaged_y[i] = np.mean(y[indices])
        counts[i] = len(indices)

    return unique_x, averaged_y, counts

x, y, n = average_y_at_unique_x(x, y)

# plt.plot(x, y)


def find_and_plot_peaks(x, y, sensitivity=0.1):
    # Find peaks
    peaks, _ = find_peaks(y, height=sensitivity*np.max(y))

    # Plot original graph
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Original Data')

    # Plot peaks
    peak_x = [x[i] for i in peaks]
    peak_y = [y[i] for i in peaks]
    plt.plot(peak_x, peak_y, 'ro', markersize=8, label='Peaks')

    # Annotate peak values
    for i, (px, py) in enumerate(zip(peak_x, peak_y)):
        plt.text(px, py, f'{py:.2f}', ha='center', va='bottom')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Peaks in Graph')
    plt.legend()
    plt.grid(True)
    plt.show()

# find_and_plot_peaks(x, y, 0.5)



def find_and_plot_peaks_smooth(x, y, sensitivity=0.12, window_length=12, polyorder=3):
    # Smooth the data
    y_smooth = savgol_filter(y, window_length, polyorder)

    # Find peaks
    peaks, _ = find_peaks(y_smooth, height=sensitivity*np.max(y_smooth))

    # Plot original graph
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, "o", label='Podatki')

    # Plot smoothed data
    plt.plot(x, y_smooth, "--", label=r"Savitzky-Golay filter", color="#5e5e5e", linewidth=1)

    # Plot peaks
    peak_x = [x[i] for i in peaks]
    peak_y = [y[i] for i in peaks]
    plt.plot(peak_x, peak_y, 'ro', markersize=8, label='Vrhovi')

    # Annotate peak values
    for i, (px, py) in enumerate(zip(peak_x, peak_y)):
        plt.text(px, py, f'{px:.2f}', ha='center', va='bottom')

    plt.xlabel("$I_2\ $ [A]")
    plt.ylabel("$U_1\ $ [V]")
    plt.title(r'$T = 119\ ^\circ$C')
    plt.legend()
    
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

    plt.show()

    return peak_x, peak_y

peak_x, peak_y = find_and_plot_peaks_smooth(x, y, 0.09, 8, 3)

print([(xp, float(f'{yp:.4f}')) for xp, yp in zip(peak_x, peak_y)])