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
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

###########################################################
def odrez(df, x_min, x_max):
    """
    Filter data based on specified x-value range.

    Parameters:
    - df: DataFrame, input data
    - x_min: float, minimum x-value to include
    - x_max: float, maximum x-value to include

    Returns:
    - DataFrame, filtered data
    """
    filtered_df = df[(df[df.columns[0]] < x_min) | (df[df.columns[0]] > x_max)]
    return filtered_df

def lin_fun(x, a, b):
    return a * x + b

def polnenje(x, U_b, U_0, t_0, tau):
    return U_b + U_0 * np.exp(- (x - t_0) / tau)

def exponential_function(x, U_b, U_0, t_0, tau):
    """
    Exponential function: f(x) = a * exp(b * x) + c

    Parameters:
    - x: input array
    - a, b, c: parameters to be optimized during curve fitting

    Returns:
    - Array of function values
    """
    # return a * np.exp(b * x) + c
    return U_b + U_0 * np.exp(- (x - t_0) / tau)# * (1/2) * ((np.abs(x - t_0) / (x - t_0)) + 1)

def fit_exponential(x, y):
    """
    Fit an exponential function to the given data.

    Parameters:
    - x: input array
    - y: output array (data to be fitted)

    Returns:
    - Tuple containing optimized parameters (a, b, c) and the covariance matrix
    """
    initial_guess = [1.0, 1.0, 1.0, 1.0]  # Initial guess for parameters (a, b, c)
    optimized_params, covariance = curve_fit(exponential_function, x, y, p0=initial_guess)
    return optimized_params, covariance

def moving_average(array, index, window_size=300):
    """
    Calculate the moving average of 100 values centered around the given index.

    Parameters:
    - array: NumPy array
    - index: Index around which to calculate the moving average
    - window_size: Size of the moving average window (default is 100)

    Returns:
    - Average of the values in the specified window
    """
    start = max(0, index - window_size // 2)
    end = min(len(array), index + window_size // 2 + 1)
    values_in_window = array[start:end]
    return np.average(values_in_window)

###########################################################

df = pd.read_csv(f"Piezo/PIEZO/SDS00006.csv", skiprows=12)

x1 = df[df.columns[0]]
y1 = df[df.columns[1]]

    # Plot the data
print("Column names:", df.columns)
plt.plot(df[df.columns[0]], df[df.columns[1]])
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('CSV Data Plot')
plt.show()


# x_min = float(input("Enter the minimum x-value: "))
x_max = float(input("Enter the maximum x-value: "))
# t_1 = float(input("Enter the value of t_1: "))
x_min = -20
# x_max = -12.35
t_1_0 = x_max + 0.5


x_data = odrez(df, x_min, x_max)[df.columns[0]]
y_data = odrez(df, x_min, x_max)[df.columns[1]]


# Smooth the data (Savitzky-Golay filter)
y_smooth = savgol_filter(y_data, 5000, 1)  # Adjust window size and polynomial order as needed


y = y_smooth
x = x_data.to_numpy()


# Fit the exponential function to the data
params, covariance = fit_exponential(x_data, y_data)

# Plot the original data and the fitted exponential function
plt.plot(df[df.columns[0]], df[df.columns[1]], ".", label='Podatki', markersize=4)
plt.plot(df[df.columns[0]], exponential_function(df[df.columns[0]], *params), label='Fit')
# plt.plot(x_data, y_smooth)
plt.xlabel('$t$ (s)')
plt.ylabel('$U$ (V)')
plt.legend()
plt.title("Razbremenitev, $m = 1008$ g")

plt.savefig("Piezo/Piezo6.pgf")
plt.show()

print("Optimized Parameters (U_b, U_0, t_0, tau):", params, covariance)


t_1 = t_1_0
sez = []

for n in range(11):
    t_1 = t_1_0 + n * 0.5
    index = np.argmin(np.abs(x - t_1))

    offset = 3000

    dx_dy = (moving_average(y, index + offset) - moving_average(y, index - offset)) / (moving_average(x, index + offset) - moving_average(x, index - offset))

    # Calculate the tangent line
    tangent_line_x = x
    tangent_line_y = dx_dy * (x - x[index]) + y[index]

    x_0 = (dx_dy * x[index] - y[index] + params[0]) / dx_dy

    # Plot the data, the chosen point, and the estimated tangent line
    # plt.plot(x, y, label='Noisy Data')
    # plt.scatter(x[index], y[index], color='red', label='Point of Interest')
    # plt.scatter(x[index + offset], y[index + offset], color='green')
    # plt.scatter(x[index - offset], y[index - offset], color='green')
    # plt.scatter(x_0, 0, color='red')
    # plt.plot(tangent_line_x, tangent_line_y, '--', label='Tangent Line (LOESS)')
    # plt.title(f'Closest Point to {t_1}: ({x[index]:.2f}, {y[index]:.2f})')
    # plt.legend()
    # plt.show()
    # print("tau_2 =", x_0 - x[index])

    sez.append(x_0 - x[index])


print("Pravi tau_2: ", unc.ufloat(np.average(np.asarray(sez)), np.std(np.asarray(sez))))