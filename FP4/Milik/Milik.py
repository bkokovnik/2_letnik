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


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Serif",
    "font.size": 20
})

############################################################################################################################

v = [16.648, 19.742, 32.591, 18.484, 18.883, 34.462, 14.854, 31.152, 20.799, 29.486, 16.893, 20.504, 45.501, 16.637, 28.688, 15.319, 17.365, 21.454, 27.607, 27.514, 21.609, 11.518, 34.431, 59.002, 20.478, 28.591, 12.475, 18.293, 16,226]
V = np.array([52, 66, 126, 48, 44, 68, 29, 74, 52, 111, 35, 54, 206, 56, 36, 54, 38, 50, 55, 99, 448, 42, 22, 146, 147, 61, 109, 19, 37, 48])

V_err = unp.uarray(V, np.ones(np.shape(V)) * 5)