import scipy
from scipy import integrate
import scipy.optimize as opt
from scipy import stats
import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
import scipy.constants as const

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Serif",
    "font.size": 18
})

###########################################################

##### Seznami podatkov
h = np.array([45, 43, 41, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4]) #cm
U = np.array([4.78, 6.1, 6.5, 6.8, 8.0, 8.7, 9.7, 11.3, 13.2, 15.3, 18.4, 22.0, 27.0, 31.9, 38.9, 48.4, 60.2, 77.5, 97.0, 122.8, 153.6, 188.5]) #mV

I2 = np.array([0.49, 0.76, 1.00, 1.26, 1.50, 1.75, 1.99, 2.24, 2.49, 3.01, 3.24, 3.51, 3.76, 4.01, 4.24, 4.51, 4.75, 4.99]) #A
U2 = np.array([561, 805, 1030, 1269, 1488, 1712, 1938, 2182, 2406, 2900, 3094, 3358, 3565, 3796, 4000, 4256, 4455, 4668])   #mV

##### Konstante, parametri in začetne vrednosti
I0 = 4
R = 10000
C = 1 * 10 ** (-6)
r1 = 9 * 10 ** (-3)
r2 = 11.5 * 10 ** (-3)
N1 = 2000
N2 = 200
r0 = 0.125
N3 = 200

Ss = np.pi * (r2 ** 2 + r1 ** 2) / 2

##### Definicije funkcij
def lin_fun(x, a, b):
    return a * x + b


##### Obdelava seznamov
h = h + 1.2
x1 = h * 10 ** (-2)
y1 = (U * (10 ** (-3)) * R * C)/(N1*Ss)

seznam_h = np.arange(np.min(h), np.max(h), 1000)

B_teoreticna = (N3 * (1.257e-6) * I0 * r0 ** 2)/(2*(r0 ** 2 + (h * 10 ** (-2)) ** 2) ** (3/2))

x2 = I2
y2 = (U2 * (10 ** (-3)) * R * C)/(N2*Ss)

##### Napake



##### Fitanje
slope, intercept, r_value, p_value1, std_err = stats.linregress(x2, y2)

optimizedParameters, pcov = opt.curve_fit(lin_fun, x2, y2)

##### Graf
# plt.plot(np.concatenate((np.array([0]), x1), axis=None), lin_fun(np.concatenate((np.array([0]), x1), axis=None), *optimizedParameters),
        #  color="black", linestyle="dashed", label="Fit")
# plt.plot(x1, y1, "o", label="Izmerjeno", color="#0033cc")

plt.errorbar(x1 * 10 ** 2, y1 * 10 ** 3, yerr=np.sqrt(0.02**2 + (0.5/10)**2 + (0.1/1)**2 + (0.3/U)**2)*y1 * 10 ** 3, fmt='.', color="#0033cc", ecolor='black', capsize=3, label="Izmerjeno") #, xerr=0.1 * np.ones(np.shape(x1))

print(np.sqrt(0.02**2 + (0.5/10)**2 + (0.1/1)**2 + (0.3/U)**2))

plt.plot(x1 * 10 ** 2, B_teoreticna * 10 ** 3, label="Izračunano", color="gray")

plt.xlabel('$h [cm]$')
plt.ylabel('$B [mT]$')

xlim_spodnja = 0
xlim_zgornja = 52
ylim_spodnja = 0
ylim_zgornja = 3.4
plt.xlim(xlim_spodnja, xlim_zgornja)
plt.ylim(ylim_spodnja, ylim_zgornja)
major_ticksx = np.array([0, 10, 20, 30, 40, 50])   #np.arange(xlim_spodnja, xlim_zgornja + 0.0000001, (xlim_zgornja - xlim_spodnja)/6)
# minor_ticksx = np.arange(xlim_spodnja, xlim_zgornja + 0.0000001, (xlim_zgornja - xlim_spodnja)/60)
major_ticksy = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])   # np.arange(ylim_spodnja, ylim_zgornja + 0.0000001, (ylim_zgornja - ylim_spodnja)/4)
# minor_ticksy = np.arange(ylim_spodnja, ylim_zgornja + 0.0000001, (ylim_zgornja - ylim_spodnja)/40)
plt.xticks(major_ticksx)
# plt.xticks(minor_ticksx, minor=True)
plt.yticks(major_ticksy)
# plt.yticks(minor_ticksy, minor=True)
# plt.grid(which='minor', alpha=0.2)
# plt.grid(which='major', alpha=0.5)
plt.legend()
plt.title("Gostota magnetnega polja v odvisnosti od višine na osi")

# plt.show()
plt.savefig("MagInd_A.png", dpi=300, bbox_inches="tight")
plt.show()

napaka = 0.09 # Izračunana relativna napaka skupaj z d^2 in S

print(np.max(B_teoreticna), np.max(y1), Ss)


# plt.plot(x2, y2, "o", label="Izmerjeno", color="#0033cc")
# plt.plot(x1 * 10 ** 2, B_teoreticna * 10 ** 3, label="Izračunano", color="gray")

plt.plot(np.concatenate((np.array([0]), x2), axis=None), lin_fun(np.concatenate((np.array([0]), x2), axis=None), *optimizedParameters),
         color="black", linestyle="dashed", label="Fit")

plt.errorbar(x2, y2, yerr=np.sqrt(0.02**2 + (0.5/10)**2 + (0.1/1)**2 + (5/U2)**2)*y2, fmt='.', color="#0033cc", ecolor='black', capsize=3, label="Izmerjeno") #, xerr=0.1 * np.ones(np.shape(x1))

major_ticksx = np.array([0, 1, 2, 3, 4, 5])
plt.xticks(major_ticksx)
# major_ticksy = np.array([])

plt.xlabel('$I [A]$')
plt.ylabel('$B [T]$')
plt.legend()
plt.title("Gostota magnetnega polja v reži v odvisnoti od napajalnega toka")

# plt.show()
plt.savefig("MagInd_B.png", dpi=300, bbox_inches="tight")


##### Izračuni
# print(slope * 2 * d**2 / (np.pi * r**2), ", relativna napaka brez d in S: ", std_err / slope, slope)
print(slope)

