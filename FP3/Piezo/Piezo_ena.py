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






###########################################################







def polnenje(x, U_b, U_0, t_0, tau):
    return U_b + U_0 * np.exp(- (x - t_0) / tau)


df = pd.read_csv(f"Piezo/PIEZO/SDS00006.csv", skiprows=12)

x1 = df[df.columns[0]]
y1 = df[df.columns[1]]

    # Plot the data
print("Column names:", df.columns)
plt.plot(df[df.columns[0]], df[df.columns[1]])
plt.xlabel('Time (s)')  # Adjust the label based on your actual data
plt.ylabel('Voltage (V)')  # Adjust the label based on your actual data
plt.title('CSV Data Plot')
plt.show()

# x_min = float(input("Enter the minimum x-value: "))
# x_max = float(input("Enter the maximum x-value: "))
# t_1 = float(input("Enter the value of t_1: "))
x_min = -20
x_max = -12.35
t_1 = -8



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

# Example data
x_data = odrez(df, x_min, x_max)[df.columns[0]]
y_data = odrez(df, x_min, x_max)[df.columns[1]]

# Smooth the data (Savitzky-Golay filter)
y_smooth = savgol_filter(y_data, 5000, 1)  # Adjust window size and polynomial order as needed


# Fit the exponential function to the data
params, covariance = fit_exponential(x_data, y_data)

# Plot the original data and the fitted exponential function
plt.plot(df[df.columns[0]], df[df.columns[1]], ".", label='Original Data')
plt.plot(df[df.columns[0]], exponential_function(df[df.columns[0]], *params), label='Fitted Exponential')
plt.plot(x_data, y_smooth)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()

print("Optimized Parameters (U_b, U_0, t_0, tau):", params, covariance)


# Generate noisy data following a curve
x = x_data.to_numpy()
y = y_smooth

# Choose a value and find the closest point
target_x = t_1  # Change this to the x-value you're interested in
# print(x)
index = np.argmin(np.abs(x - t_1))
# print(index)

# min_value = np.inf
# min_index = None

# for i, value in enumerate(index):
#     if value < min_value:
#         min_value = value
#         min_index = i


# print(index[min_index], x[min_index], t_1)
# print(np.min(index))
# index = min_index

# plt.plot(x, np.abs(x - target_x), ".")
# plt.scatter(x[index], np.abs(x - target_x)[index], color="red")
# plt.show()

offset = 3000


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


# Use LOESS regression for a smooth estimate of the tangent
# loess = interp1d(x, y, kind='quadratic', fill_value='extrapolate')
dx_dy = (moving_average(y, index + offset) - moving_average(y, index - offset)) / (moving_average(x, index + offset) - moving_average(x, index - offset))
# dx_dy = (y_smooth[index + offset] - y_smooth[index - offset]) / (x[index + offset] - x[index - offset])

# Calculate the tangent line
tangent_line_x = x
tangent_line_y = dx_dy * (x - x[index]) + y[index]

x_0 = (dx_dy * x[index] - y[index]) / dx_dy

# Plot the data, the chosen point, and the estimated tangent line
plt.plot(x, y, label='Noisy Data')
plt.scatter(x[index], y[index], color='red', label='Point of Interest')
plt.scatter(x[index + offset], y[index + offset], color='green')
plt.scatter(x[index - offset], y[index - offset], color='green')
plt.scatter(x_0, 0, color='red')
plt.plot(tangent_line_x, tangent_line_y, '--', label='Tangent Line (LOESS)')
plt.title(f'Closest Point to {target_x}: ({x[index]:.2f}, {y[index]:.2f})')
plt.legend()
plt.show()

print("tau_2 =", x_0 - x[index])


































#     #### Fitanje
# slope, intercept, r_value, p_value1, std_err = stats.linregress(x1, y1)

# optimizedParameters, pcov = opt.curve_fit(polnenje, odrez(df, x_min, x_max)[df.columns[0]], odrez(df, x_min, x_max)[df.columns[1]])

# plt.plot(np.concatenate((np.array([0]), odrez(df, x_min, x_max)[df.columns[0]]), axis=None), polnenje(np.concatenate((np.array([0]), odrez(df, x_min, x_max)[df.columns[0]]), axis=None), *optimizedParameters),
#         color="black", linestyle="dashed", label="Fit")

##################################################################

# x_mik = odrez(df, x_min, x_max)[df.columns[0]]
# y_mik = odrez(df, x_min, x_max)[df.columns[1]]



# # Create a model for fitting.
# lin_model_mik = Model(polnenje)

# # Create a RealData object using our initiated data from above.
# data_mik = Data(x_mik, y_mik)

# # Set up ODR with the model and data.
# odr_mik = ODR(data_mik, lin_model_mik, beta0=[0., 1.])

# # Run the regression.
# out_mik = odr_mik.run()

# # Use the in-built pprint method to give us results.
# # out_mik.pprint()
# '''Beta: [ 1.01781493  0.48498006]
# Beta Std Error: [ 0.00390799  0.03660941]
# Beta Covariance: [[ 0.00241322 -0.01420883]
#  [-0.01420883  0.21177597]]
# Residual Variance: 0.00632861634898189
# Inverse Condition #: 0.4195196193536024
# Reason(s) for Halting:
#   Sum of squares convergence'''

# x_fit_mik = np.linspace(x_mik[0], x_mik[-1], 1000)
# y_fit_mik = polnenje(out_mik.beta, x_mik)

# plt.plot(x_mik, y_mik, linestyle='None', marker='.', capsize=3, label="Izmerjeno")
# plt.plot(x_mik, y_fit_mik, label="Fit", color="black")
# plt.title("Sila mikrometra v odvisnosti od njegovega položaja")
# plt.legend()
# plt.savefig("Upogib/Upogib_Mikrometer.png", dpi=300, bbox_inches="tight")

# plt.show()

###################################################################

    # Plot the data
# print("Column names:", df.columns)
# plt.plot(odrez(df, x_min, x_max)[df.columns[0]], odrez(df, x_min, x_max)[df.columns[1]])
# plt.xlabel('Time (s)')  # Adjust the label based on your actual data
# plt.ylabel('Voltage (V)')  # Adjust the label based on your actual data
# plt.title('CSV Data Plot')
# plt.show()



#     maks = np.argmax(y1)
#     mini = np.argmin(y1)
#     print(maks, mini)

#     if maks < mini:
#         x1 = df[df.columns[0]][maks:]
#         y1 = df[df.columns[1]][maks:]

#     print(np.shape(x1))
#     # # Plot the data
#     # print("Column names:", df.columns)
#     # plt.plot(df[df.columns[0]], df[df.columns[1]])
#     # plt.xlabel('Time (s)')  # Adjust the label based on your actual data
#     # plt.ylabel('Voltage (V)')  # Adjust the label based on your actual data
#     # plt.title('CSV Data Plot')
#     # plt.show()

#     ##### Fitanje
#     slope, intercept, r_value, p_value1, std_err = stats.linregress(x1, y1)

#     optimizedParameters, pcov = opt.curve_fit(polnenje, x1, y1)

#     plt.plot(np.concatenate((np.array([0]), x1), axis=None), polnenje(np.concatenate((np.array([0]), x1), axis=None), *optimizedParameters),
#          color="black", linestyle="dashed", label="Fit")
#     plt.plot(x1, y1, ".", label="Izmerjeno", color="#0033cc")
#     plt.show()
    

#     # podatki = pd.read_csv(f"Piezo/PIEZO/SDS0000{ime}.csv", sep='\t', header=None)
#     # podatki = podatki.to_numpy()
#     # podatki = podatki.transpose()
#     # print(podatki[12:])
#     # # plt.plot(podatki[0], podatki[1])
#     # # plt.show()

# # # Read the text file with pandas
# # file_path = "UkLec/UkLec_BK/Meritev_x_os.txt"
# # podatki_x = pd.read_csv(file_path, sep='\t', header=None)

# # # Convert to NumPy array
# # Meritev_x = podatki_x.to_numpy()


# # # Read the text file with pandas
# # file_path = "UkLec/UkLec_BK/Izmaknjen_po_x_osi.txt"
# # podatki_x_5cm = pd.read_csv(file_path, sep='\t', header=None)

# # # Convert to NumPy array
# # Meritev_x_5cm = podatki_x_5cm.to_numpy()