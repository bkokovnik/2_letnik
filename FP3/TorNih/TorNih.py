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

Arr_brez_20_t0 = np.array([41.090, 40.860, 41.106])
Arr_valj_10_t0 = np.array([57.002, 56.731, 56.873])
Arr_kocka_10_t0 = np.array([39.189, 39.099, 39.146])
Arr_zobnik_10_t0 = np.array([47.858, 47.794, 47.810])

brez_t0_unc = unc.ufloat(np.average(Arr_brez_20_t0) / 20, np.std(Arr_brez_20_t0) / 20)
valj_t0_unc = unc.ufloat(np.average(Arr_valj_10_t0) / 10, np.std(Arr_valj_10_t0) / 10)
kvader_t0_unc = unc.ufloat(np.average(Arr_kocka_10_t0) / 10, np.std(Arr_kocka_10_t0) / 10)
zobnik_t0_unc = unc.ufloat(np.average(Arr_zobnik_10_t0) / 15, np.std(Arr_zobnik_10_t0) / 15)

l_zic = unc.ufloat(18.2, 0.1) * 10 ** (-2)
d_zic = unc.ufloat(0.54, 0.01) * 10 ** (-3)
r_zic = d_zic / 2

# Valj:
r_1 = unc.ufloat(1.44, 0.01) * 10 ** (-2) / 2
r_2 = unc.ufloat(8.72, 0.01) * 10 ** (-2) / 2
h_v = unc.ufloat(4.94, 0.01) * 10 ** (-2)
m_v = unc.ufloat(2490, 1) * 10 **(-3)

# Kocka:
a_1 = unc.ufloat(6.00, 0.01) * 10 ** (-2)
a_2 = a_1
h_k = unc.ufloat(6.01, 0.01) * 10 ** (-2)
r_k = unc.ufloat(3.98, 0.01) * 10 ** (-2) / 2
m_k = unc.ufloat(1193, 1) * 10 **(-3)

# Zobnik:
m_z = unc.ufloat(0.756, 0.001)
r_z = unc.ufloat(25.1, 0.1) * 10 ** (-3) / 2
R_z1 = unc.ufloat(70.1, 0.1) * 10 ** (-3) / 2
R_z2 = unc.ufloat(80.0, 0.1) * 10 ** (-3) / 2


# Vztrajnostni moment valja
J_v = m_v / 2 * (r_1 ** 2 + r_2 ** 2)


# Izračun torzijskega koeficienta D
D = J_v * (2 * np.pi / valj_t0_unc) ** 2 * (1 - (brez_t0_unc / valj_t0_unc) ** 2) ** (-1)


# Izračun J podstavka
J_b = D * (brez_t0_unc / (2 * np.pi)) ** 2


# Izračun vztrajnostnega momenta kvadra z luknjo
V_k_b = a_1 * a_2 * h_k
V_k_l = np.pi * r_k ** 2 * h_k
V_k = V_k_b - V_k_l
rho_k = m_k / V_k
J_k = 1 / 12 * rho_k * V_k_b * (a_1 ** 2 + a_2 ** 2) - 1 / 2 * V_k_l * rho_k * r_k ** 2

# Izračun J kvadra z luknjo preko nihajnih časov
J_k_i = D * (kvader_t0_unc / (2 * np.pi)) ** 2 - J_b


# Izračun vztrajnostnega radija
vztr_r_k = umath.sqrt(J_k_i / m_k)


# Izračun vztrajnostnega momenta zobnika
J_z_i = D * (zobnik_t0_unc / (2 * np.pi)) ** 2 - J_b
rho_z = m_z / ((unc.ufloat((unc.nominal_value(R_z1) + unc.nominal_value(R_z2)) / 2, 0.002)) ** 2 * np.pi * R_z1 - np.pi * r_z ** 2 * R_z1)
J_z = rho_z *  ((unc.ufloat((unc.nominal_value(R_z1) + unc.nominal_value(R_z2)) / 2, 0.002)) ** 2 * np.pi * R_z1) / 2 * (unc.ufloat((unc.nominal_value(R_z1) + unc.nominal_value(R_z2)) / 2, 0.002)) ** 2 - (rho_z * r_z ** 2 * np.pi * R_z1) / 2 * (r_z ** 2)

print("J podstavka: ", J_b * 10 **(3))
print("J valja: ", J_v * 10 **(3))
print("J kvadra: ", J_k * 10 **(3), ", izmerjen J kvadra: ", J_k_i * 10 **(3))
print("Ocena J zobnika: ", J_z * 10 **(3), ", izmerjen J zobnika: ", J_z_i * 10 **(3))
print("Torzijski koeficient D: ", D * 10 **(3))
print("Strižni modul G:", (D * 2 * l_zic) / (np.pi * r_zic ** 4))
print("Vztrajnostni radij kvadra: ", vztr_r_k * 10 ** 2)