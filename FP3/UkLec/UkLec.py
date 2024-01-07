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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Serif",
    "font.size": 20
})

###########################################################
Imena = ["0", "0_5", "1", "1_5", "2", "inf_"]
Skupno = []

# Read the text file with pandas
file_path = "UkLec/UkLec_BK/Meritev_x_os.txt"
podatki_x = pd.read_csv(file_path, sep='\t', header=None)

# Convert to NumPy array
Meritev_x = podatki_x.to_numpy()


# Read the text file with pandas
file_path = "UkLec/UkLec_BK/Izmaknjen_po_x_osi.txt"
podatki_x_5cm = pd.read_csv(file_path, sep='\t', header=None)

# Convert to NumPy array
Meritev_x_5cm = podatki_x_5cm.to_numpy()




plt.plot(np.arange(8.6, -8.2, -0.2), Meritev_x_5cm, ".-", label="$\sigma = 5$ cm")
plt.plot(np.arange(8, -8.2, -0.2), Meritev_x, ".-", label="$\sigma = 0$ cm")
plt.xlabel('$x$ [cm]')
plt.ylabel('$U$ [V]')
plt.title("Prečni profil amplitude")
plt.legend()

# plt.savefig("UkLec/x_os.png", dpi=300, bbox_inches="tight")
plt.show()

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Example data
x4 = np.arange(8, -8.2, -0.2)
y4 = np.transpose(Meritev_x)[0]


# Find local maxima
peaks4, _4 = find_peaks(y4)

# Plot original data with detected maxima
plt.figure(figsize=(8, 6))
plt.plot(x4, y4, label='Data')
plt.plot(x4[peaks4], y4[peaks4], 'ro', label='Local Maxima')
plt.legend()
plt.title('Local Maxima Detection')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# Print the x and y values at detected maxima
print("Detected local maxima:")
for idx in peaks4:
    print(f"x = {x4[idx]:.2f}, y = {y4[idx]:.2f}")







# Read the text file with pandas
file_path = "UkLec/UkLec_BK/Meritev_z_os.txt"
podatki_z = pd.read_csv(file_path, sep='\t', header=None)

# Convert to NumPy array
Meritev_z = podatki_z.to_numpy()


# Read the text file with pandas
file_path = "UkLec/UkLec_BK/Izmaknjen_po_z_osi.txt"
podatki_z_5cm = pd.read_csv(file_path, sep='\t', header=None)

# Convert to NumPy array
Meritev_z_5cm = podatki_z_5cm.to_numpy()


# plt.plot(np.arange(8.4, -9.2, -0.2), Meritev_z_5cm, ".")
# plt.xlabel('$z$ [cm]')
# plt.ylabel('$U$ [V]')
# plt.title("Profil prečne amplitude")

# plt.show()

import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Generate noisy data
x = np.arange(8, -8.6, -0.2)
y = np.transpose(Meritev_z)[0]

# print(np.shape(x), np.shape(y))
# print(x, y)

# Smooth the data (Savitzky-Golay filter)
y_smooth = savgol_filter(y, 25, 1)  # Adjust window size and polynomial order as needed

# Find peaks in smoothed data
peaks, _ = find_peaks(y_smooth)

plt.plot(x, y, ".", label="$\sigma = 0$ cm")
plt.plot(x, y_smooth)#, label='Zglajene meritve')
# plt.plot(x[peaks], y_smooth[peaks], 'ro', label='Peaks')
plt.xlabel('$x$ [cm]')
plt.ylabel('$U$ [V]')
plt.title("Vzdolžni profil amplitude")
print("GLEJ TO!!!!!!", x[np.argmax(np.transpose(Meritev_z)[0])])


###############################


# Generate noisy data
x1 = np.arange(8.4, -9.2, -0.2)
y1 = np.transpose(Meritev_z_5cm)[0]

print(np.shape(x1), np.shape(y1))
print(x1, y1)

# Smooth the data (Savitzky-Golay filter)
y_smooth1 = savgol_filter(y1, 25, 1)  # Adjust window size and polynomial order as needed

# Find peaks in smoothed data
peaks1, _1 = find_peaks(y_smooth)

plt.plot(x1, y1, ".", label="$\sigma = 5$ cm")
plt.plot(x1, y_smooth1)#, label='Zglajene meritve, izmaknjen vir za 5 cm')
# plt.plot(x1[peaks1], y_smooth1[peaks1], 'ro', label='Peaks')
plt.legend()


# plt.savefig("UkLec/z_os.png", dpi=300, bbox_inches="tight")
plt.show()




import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# Example data
x = np.arange(8, -8.6, -0.2)
y = np.transpose(Meritev_z)[0]

# Apply Savitzky-Golay filter for smoothing
window_length = 25
poly_order = 1
y_smooth = savgol_filter(y, window_length=window_length, polyorder=poly_order)

# Find the maximum value of the smoothed function
max_smoothed_value = np.max(y_smooth)

# Define the threshold (70% of the maximum smoothed value)
threshold_value = 0.75 * max_smoothed_value

# Find the index where the smoothed function crosses the threshold
index_threshold = np.where(y_smooth >= threshold_value)[0][0]

# Corresponding x-value where the smoothed function reaches 70% of max
x_at_threshold = x[index_threshold]

# Plotting the original data and smoothed function
plt.figure(figsize=(8, 6))
plt.plot(x, y, ".", label='Izmerjeno')
plt.plot(x, y_smooth, label='Zglajene meritve')
plt.axhline(threshold_value, color='r', linestyle='--', label='75 \% maksimuma')

# Highlight the point where the smoothed function crosses the threshold
plt.plot(x_at_threshold, threshold_value, 'ro')

plt.legend()
plt.show()

print(f"The point where the smoothed function is 75% of the maximum is x = {x_at_threshold:.2f}")





val_dol = unumpy.uarray([35.5, 34, 35.5], [0.5, 0.5, 0.5])
val_dol = val_dol/4

# Sample data as a uarray with values and absolute uncertainties
values = unumpy.nominal_values(val_dol)
abs_uncertainties = unumpy.std_devs(val_dol)

# Calculate relative uncertainties
relative_uncertainties = abs_uncertainties / values

# Calculate the weights based on relative error
weights = 1 / (relative_uncertainties ** 2)

# Calculate the weighted sum
weighted_sum = values.dot(weights)

# Calculate the total weight
total_weight = sum(weights)

# Calculate the uncertainty of the weighted average
uncertainty_of_weighted_average = 1 / math.sqrt(total_weight)
B_z_p_avg = weighted_sum / total_weight
print("Weighted Average: ", weighted_sum / total_weight)
print("Uncertainty of Weighted Average: ", uncertainty_of_weighted_average)

val_dol_unc = unc.ufloat(weighted_sum / total_weight, uncertainty_of_weighted_average * weighted_sum / total_weight)
print("Valovna dolžina zvoka: ", val_dol_unc)

f = 48.7**2 / (val_dol_unc)

print("Izračunana goriščna razdalja f: ", f)
print("Izračunan b: ", (1/f - 1/(550))**(-1))
    
# plt.plot(numpy_array[:, 0], numpy_array[:, 1], "-o", label="Izmerjeno", color="#0033cc")
# # plt.ylim(top=-0.575)

# plt.show()

# n=0
# Naslovi = ["0 cm", "0,5 cm", "1 cm", "1,5 cm", "2 cm", "$\infty$ cm"]
# fig, axs = plt.subplot_mosaic([['left']], layout='constrained')

# for graf in Skupno:
#     axs['left'].plot(graf[:, 0], graf[:, 1], "-o", label=Naslovi[n], markersize=5)
#     n += 1

# axs['left'].set_ylim(-0.65, 0.2)
# axs['left'].set_xlabel("$U$ [V]")
# axs['left'].set_ylabel("$I$ [mA]")
# axs["left"].spines[['right', 'top']].set_visible(False)
# # fig.ylim(-0.65, 0.2)
# fig.legend(loc='outside center right')
# plt.title("Primerjava karakteristik napajane fotodiode\npri različnih osvetlitvah")
# plt.savefig("SiDiod_Napajana.png", dpi=300, bbox_inches="tight")
# plt.show()