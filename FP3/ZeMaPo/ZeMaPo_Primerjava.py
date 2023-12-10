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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Serif",
    "font.size": 18
})


p = 3.9770790269552316
l = 4.13 / 2 * 10 ** (-2)

d = np.arange(10, 60, 0.5) * 10 ** (-2)

plt.plot(d, (const.mu_0 * p) / (4 * np.pi * (d ** 2 + l ** 2) ** (3/2)), label="Prava vrednost")
plt.plot(d, (const.mu_0 * p) / (4 * np.pi * (d) ** (3)), label="Idealizacija")


plt.xlabel('$r$ [cm]')
plt.ylabel('$B$ [T]')


plt.legend()
plt.title("Primerjava med magnetnim poljem paličastega magneta in idealizacijo")

plt.savefig("ZeMaPo_primerjava.png", dpi=300, bbox_inches="tight")
plt.show()