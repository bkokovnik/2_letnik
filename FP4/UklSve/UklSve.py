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

##################################################################################################
# def fit_krivulja(x, j_0, D, d):
#     return (j_0 ) * (np.sin(k * (x + o) * D / 2) / (k * (x + o) * D / 2)) ** 2 * (np.sin(n * k * (x + o) * d / 2) / np.sin((k * (x + o) * d / 2))) ** 2

def fit_krivulja(x, j_0, D, d):
    result = np.empty_like(x)
    result[x == 0] = 1
    result[x != 0] = (j_0) * (np.sin(k * (x[x != 0] + o) * D / 2) / (k * (x[x != 0] + o) * D / 2)) ** 2 * (np.sin(n * k * (x[x != 0] + o) * d / 2) / np.sin((k * (x[x != 0] + o) * d / 2))) ** 2
    return result

def fit_krivulja_1(x, j_0, D):
    result = np.empty_like(x)
    result[x == 0] = 1
    # result[x != 0] = (j_0) * (np.sin(k * (x[x != 0] + o) * D / 2) / (k * (x[x != 0] + o) * D / 2)) ** 2 * (np.sin(n * k * (x[x != 0] + o) * d / 2) / np.sin((k * (x[x != 0] + o) * d / 2))) ** 2
    result[x != 0] = (j_0 ) * (np.sin(k * (x[x != 0] + o) * D / 2) / (k * (x[x != 0] + o) * D / 2)) ** 2
    return result


# def fit_krivulja_1(x, j_0, D):
#     return (j_0 ) * (np.sin(k * x * D / 2) / (k * x * D / 2)) ** 2
##################################################################################################

# Konstante
k = 2 * np.pi / (633e-9)






Reza_5 = pd.read_csv("FP4/UklSve/UklSve/Reza_5.dat", sep='\t', header=None).to_numpy()

x_5 = Reza_5[:, 0]
y_5 = Reza_5[:, 1]

x_5 = (np.arctan(x_5 / 2000))
# y_5 = y_5 / np.max(y_5)

# # Zamik grafa
height_threshold = 0.2  # Adjust this threshold as needed
peaks_5, _ = find_peaks(y_5, height=height_threshold)

# plt.plot(x_5[peaks_5], y_5[peaks_5] + 0.03, 'rv', label='Vrhovi')

x_5 = x_5 - x_5[np.where(np.max(y_5) == y_5)] - 0.00012 + 6e-5

x_trim_5 = x_5[20:-20]
y_trim_5 = y_5[20:-20]


# maska_1 = x_5 < 0.02
# maska_2 = x_5 > -0.02
# maska = maska_1 * maska_2

# x_trim_5 = x_5[maska]
# y_trim_5 = y_5[maska]

n = 5
o = 0

x_smooth_5 = np.linspace(np.min(x_5), np.max(x_5), 4000)

optimized_params_5, covariance_5 = curve_fit(fit_krivulja, x_trim_5, y_trim_5, p0=[0.031, 16.5e-6, 90e-6])#, bounds=([-5, -5, -5], [3.3e-2, 5, 5]), ftol=1e-12, xtol=1e-12)
print(optimized_params_5)

# plt.plot(x_smooth_5, fit_krivulja(x_smooth_5, *optimized_params_5), label='5 rež (fit)', zorder=4)
plt.plot(x_smooth_5, fit_krivulja(x_smooth_5, *[0.031, 2.03446e-5, 9.01957e-5]), label='5 rež (fit)', zorder=3)
# plt.plot(x_5, y_5, ".", markersize=3, label="Izmerjeno za 5 rež", zorder=2)
# plt.plot(x_5, fit_krivulja(x_5, 0.037, 16.5e-6, 90e-6, 0))



#####################################



Reza_3 = pd.read_csv("FP4/UklSve/UklSve/Reza_3.dat", sep='\t', header=None).to_numpy()

x_3 = Reza_3[:, 0]
y_3 = Reza_3[:, 1]

x_3 = (np.arctan(x_3 / 2000))
# y_3 = y_3 / np.max(y_3)

# # Zamik grafa
height_threshold = 0.2  # Adjust this threshold as needed
peaks_3, _ = find_peaks(y_3, height=height_threshold)

# plt.plot(x_5[peaks_5], y_5[peaks_5] + 0.03, 'rv', label='Vrhovi')

x_3 = x_3 - x_3[np.where(np.max(y_3) == y_3)] - 0.00022

x_trim_3 = x_3[20:-20]
y_trim_3 = y_3[20:-20]


# maska_1 = x_5 < 0.025
# maska_2 = x_5 > -0.025
# maska = maska_1 * maska_2

# x_5 = x_5[maska]
# y_5 = y_5[maska]

n = 3
o = 0

x_smooth_3 = np.linspace(np.min(x_3), np.max(x_3), 4000)

optimized_params_3, covariance_3 = curve_fit(fit_krivulja, x_trim_3, y_trim_3, p0=[0.037, 16.5e-6, 90e-6])#, bounds=([0.000000000001, 0.000000000001, 0.000000000001], [np.inf, np.inf, np.inf]), ftol=1e-12, xtol=1e-12)
print(optimized_params_3)

plt.plot(x_smooth_3, fit_krivulja(x_smooth_3, *optimized_params_3), label='3 reže (fit)', zorder=4)
# plt.plot(x_3, y_3, ".", markersize=3, label="3 reže (izmerjeno)", zorder=1)

plt.xlabel(r'$\phi$ [rad]')
plt.ylabel('$j/j_0$')
plt.title("Primerjava zaslona s petimi in tremi režami")
x_limits = plt.xlim()
y_limits = plt.ylim()
plt.ylim(y_limits)
plt.xlim(x_limits)
plt.legend()



# plt.savefig('FP4/UklSve/2_rezi_fit.png', dpi=300, bbox_inches="tight")
plt.show()


#####################################################################################################


Reza_2 = pd.read_csv("FP4/UklSve/UklSve/Reza_2.dat", sep='\t', header=None).to_numpy()

x_2 = Reza_2[:, 0]
y_2 = Reza_2[:, 1]

x_2 = (np.arctan(x_2 / 2000))
# y_5 = y_5 / np.max(y_5)

# # Zamik grafa
height_threshold = 0.2  # Adjust this threshold as needed
peaks_2, _ = find_peaks(y_2, height=height_threshold)

# plt.plot(x_5[peaks_5], y_5[peaks_5] + 0.03, 'rv', label='Vrhovi')

x_2 = x_2 - x_2[np.where(np.max(y_2) == y_2)] - 0.00012 + 6e-5

x_trim_2 = x_2[20:-20]
y_trim_2 = y_2[20:-20]


# maska_1 = x_5 < 0.025
# maska_2 = x_5 > -0.025
# maska = maska_1 * maska_2

# x_5 = x_5[maska]
# y_5 = y_5[maska]

n = 2
o = 0

x_smooth_2 = np.linspace(np.min(x_2), np.max(x_2), 4000)

optimized_params_2, covariance_2 = curve_fit(fit_krivulja, x_trim_2, y_trim_2, p0=[0.037, 16.5e-6, 90e-6])#, bounds=([0.000000000001, 0.000000000001, 0.000000000001], [np.inf, np.inf, np.inf]), ftol=1e-12, xtol=1e-12)
print(optimized_params_2)

plt.plot(x_smooth_2, fit_krivulja(x_smooth_2, *optimized_params_2), label='2 reži (fit)', zorder=4)
# plt.plot(x_5, y_5, ".", markersize=3, label="Izmerjeno za 2 reži", zorder=2)
# plt.plot(x_5, fit_krivulja(x_5, 0.037, 16.5e-6, 90e-6, 0))


#####################################



Reza_3 = pd.read_csv("FP4/UklSve/UklSve/Reza_3.dat", sep='\t', header=None).to_numpy()

x_3 = Reza_3[:, 0]
y_3 = Reza_3[:, 1]

x_3 = (np.arctan(x_3 / 2000))
# y_3 = y_3 / np.max(y_3)

# # Zamik grafa
height_threshold = 0.2  # Adjust this threshold as needed
peaks_3, _ = find_peaks(y_3, height=height_threshold)

# plt.plot(x_5[peaks_5], y_5[peaks_5] + 0.03, 'rv', label='Vrhovi')

x_3 = x_3 - x_3[np.where(np.max(y_3) == y_3)] - 0.00022

x_trim_3 = x_3[20:-20]
y_trim_3 = y_3[20:-20]


# maska_1 = x_5 < 0.025
# maska_2 = x_5 > -0.025
# maska = maska_1 * maska_2

# x_5 = x_5[maska]
# y_5 = y_5[maska]

n = 3
o = 0

x_smooth_3 = np.linspace(np.min(x_3), np.max(x_3), 4000)

optimized_params_3, covariance_3 = curve_fit(fit_krivulja, x_trim_3, y_trim_3, p0=[0.037, 16.5e-6, 90e-6])#, bounds=([0.000000000001, 0.000000000001, 0.000000000001], [np.inf, np.inf, np.inf]), ftol=1e-12, xtol=1e-12)
print(optimized_params_3)

plt.plot(x_smooth_3, fit_krivulja(x_smooth_3, *optimized_params_3), label='3 reže (fit)', zorder=3)
# plt.plot(x_3, y_3, ".", markersize=3, label="3 reže (izmerjeno)", zorder=1)

plt.xlabel(r'$\phi$ [rad]')
plt.ylabel('$j/j_0$')
plt.title("Primerjava zaslona s tremi in dvema režama")
x_limits = plt.xlim()
y_limits = plt.ylim()
plt.ylim(y_limits)
plt.xlim(x_limits)
plt.legend()



# plt.savefig('FP4/UklSve/2_rezi_fit.png', dpi=300, bbox_inches="tight")
plt.show()


#####################################################################################################


Reza_2 = pd.read_csv("FP4/UklSve/UklSve/Reza_2.dat", sep='\t', header=None).to_numpy()

x_2 = Reza_2[:, 0]
y_2 = Reza_2[:, 1]

x_2 = (np.arctan(x_2 / 2000))
# y_5 = y_5 / np.max(y_5)

# # Zamik grafa
height_threshold = 0.2  # Adjust this threshold as needed
peaks_2, _ = find_peaks(y_2, height=height_threshold)

# plt.plot(x_5[peaks_5], y_5[peaks_5] + 0.03, 'rv', label='Vrhovi')

x_2 = x_2 - x_2[np.where(np.max(y_2) == y_2)] - 0.00012 + 6e-5

x_trim_2 = x_2[20:-20]
y_trim_2 = y_2[20:-20]


# maska_1 = x_5 < 0.025
# maska_2 = x_5 > -0.025
# maska = maska_1 * maska_2

# x_5 = x_5[maska]
# y_5 = y_5[maska]

n = 2
o = 0

x_smooth_2 = np.linspace(np.min(x_2), np.max(x_2), 4000)

optimized_params_2, covariance_2 = curve_fit(fit_krivulja, x_trim_2, y_trim_2, p0=[0.037, 16.5e-6, 90e-6])#, bounds=([0.000000000001, 0.000000000001, 0.000000000001], [np.inf, np.inf, np.inf]), ftol=1e-12, xtol=1e-12)
print(optimized_params_2)

plt.plot(x_smooth_2, fit_krivulja(x_smooth_2, *optimized_params_2), label='2 reži (fit)', zorder=4)
# plt.plot(x_5, y_5, ".", markersize=3, label="Izmerjeno za 2 reži", zorder=2)
# plt.plot(x_5, fit_krivulja(x_5, 0.037, 16.5e-6, 90e-6, 0))


#####################################



Reza_1 = pd.read_csv("FP4/UklSve/UklSve/Reza_1.dat", sep='\t', header=None).to_numpy()

x_1 = Reza_1[:, 0]
y_1 = Reza_1[:, 1]

x_1 = (np.arctan(x_1 / 2000))
# y_1 = y_1 / np.max(y_3)

# # Zamik grafa
height_threshold = 0.2  # Adjust this threshold as needed
peaks_1, _ = find_peaks(y_1, height=height_threshold)

# plt.plot(x_5[peaks_5], y_5[peaks_5] + 0.03, 'rv', label='Vrhovi')

x_1 = x_1 - x_1[np.where(np.max(y_1) == y_1)] - 0.00022

x_trim_1 = x_1[20:-20]
y_trim_1 = y_1[20:-20]


# maska_1 = x_5 < 0.025
# maska_2 = x_5 > -0.025
# maska = maska_1 * maska_2

# x_5 = x_5[maska]
# y_5 = y_5[maska]

n = 1
o = 0

x_smooth_1 = np.linspace(np.min(x_1), np.max(x_1), 4000)

optimized_params_1, covariance_1 = curve_fit(fit_krivulja_1, x_trim_1, y_trim_1, p0=[0.037, 16.5e-6])#, bounds=([0.000000000001, 0.000000000001, 0.000000000001], [np.inf, np.inf, np.inf]), ftol=1e-12, xtol=1e-12)
print(optimized_params_1)

plt.plot(x_smooth_1, fit_krivulja_1(x_smooth_1, *optimized_params_1), label='1 reža (fit)', zorder=3)
# plt.plot(x_1, y_1, ".", markersize=3, label="1 reža (izmerjeno)", zorder=1)

plt.xlabel(r'$\phi$ [rad]')
plt.ylabel('$j$')
plt.title("Primerjava zaslona z dvema in eno režo")
x_limits = plt.xlim()
y_limits = plt.ylim()
plt.ylim(y_limits)
plt.xlim(x_limits)
plt.legend()



# plt.savefig('FP4/UklSve/2_rezi_fit.png', dpi=300, bbox_inches="tight")
plt.show()