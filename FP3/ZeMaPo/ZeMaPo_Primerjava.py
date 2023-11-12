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

p = 3.9770790269552316
l = 4.13 / 2 * 10 ** (-2)

d = np.arange(10, 60, 0.5) * 10 ** (-2)

plt.plot(d, (const.mu_0 * p) / (4 * np.pi * (d ** 2 + l ** 2) ** (3/2)))
plt.plot(d, (const.mu_0 * p) / (4 * np.pi * (d) ** (3)))

plt.show()