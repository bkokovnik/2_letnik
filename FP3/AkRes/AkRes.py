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
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.interpolate import interp1d
import matplotlib


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Serif",
    "font.size": 20
})

##########################################################################

def obtezeno_povprecje(uarray):
    vrednosti = unumpy.nominal_values(uarray)
    negotovosti = unumpy.std_devs(uarray)

    obtezitev = 1/(negotovosti**2)
    obtezeno_povprecje = np.sum(vrednosti * obtezitev) / np.sum(obtezitev)

    obtezena_negotovost = np.sqrt(np.sum(negotovosti**2 * obtezitev**2) / (np.sum(obtezitev)**2))

    return unc.ufloat(obtezeno_povprecje, obtezena_negotovost)


##########################################################################
c = 340
A = unc.ufloat(56.7, 0.1) * 10**(-2)
B = unc.ufloat(38.5, 0.1) * 10**(-2)
C = unc.ufloat(24.0, 0.1) * 10**(-2)

# Specify the file path
file_path = "AkRes/Podatki/Odziv_brez.txt" 

# Read data using Pandas with tab delimiter
data = pd.read_csv(file_path, sep='\t', header=None)

# Ensure that the column indices are within the range of the number of columns
num_columns = data.shape[1]

# Convert Pandas DataFrame columns to 1D NumPy arrays
column1 = np.squeeze(np.array(data.iloc[:, 0])) if num_columns > 0 else np.array([])
column2 = np.squeeze(np.array(data.iloc[:, 1])) if num_columns > 1 else np.array([])
column3 = np.squeeze(np.array(data.iloc[:, 2])) if num_columns > 2 else np.array([])
column4 = np.squeeze(np.array(data.iloc[:, 3])) if num_columns > 3 else np.array([])

# Find peaks in column 3 with a minimum height threshold
height_threshold = 0.34  # Adjust this threshold as needed
peaks, _ = find_peaks(column2, height=height_threshold)
peaks = peaks[:-4]

# Plot the data with identified peaks
plt.plot(column1, column2, label='Meritev')
plt.plot(column1[peaks], column2[peaks] + 0.03, 'rv', label='Vrhovi')
plt.plot([882], np.array([0.33]) + 0.03, 'rv')
plt.xlabel(r'$\nu$ [Hz]')
plt.ylabel('$U_{amp}$ [V]')
plt.title('Resonančni odziv z označenimi izmerjenimi\nin teoretičnimi vrhovi')
x_limits = plt.xlim()
y_limits = plt.ylim()
plt.ylim(y_limits)
plt.xlim(x_limits)

# Print the indices and corresponding values of the peaks
# print("Indices of Peaks:", peaks)

n_x = [0, 1, 2]
n_y = [0, 1, 2]
n_z = [0, 1, 2]

Resonance_unsorted = []

for x in n_x:
    for y in n_y:
        for z in n_z:
            if x == 0 and y == 0 and z == 0:
                continue
            # print(x, y, z)
            frek1 = c / 2 * umath.sqrt((x/A)**2 + (y/B)**2 + (z/C)**2)
            if unc.nominal_value(frek1) < 1000:
                Resonance_unsorted.append([(x, y, z), frek1])

# print(Resonance_unsorted)

# Sort the list based on the ufloat values
Resonance = sorted(Resonance_unsorted, key=lambda x: x[1])

# Print the sorted list

za_label = 1
for sez in Resonance:
    if za_label == 1:
        plt.plot([unc.nominal_value(sez[1]), unc.nominal_value(sez[1])], (-200, 200), "-", color="gray", label="Izračunane vrednosti")
    else:   
        plt.plot([unc.nominal_value(sez[1]), unc.nominal_value(sez[1])], (-200, 200), "-", color="gray")
    za_label += 1

plt.legend()
# plt.savefig('AkRes/Resonanca_primerjava_duseno.png', dpi=300, bbox_inches="tight")
plt.show()
c_izrac_sez = []

for i in range(3):
    ni_temp = unc.ufloat(column1[peaks[i]], column2[peaks[i]])
    x_temp = Resonance[i][0][0]
    y_temp = Resonance[i][0][1]
    z_temp = Resonance[i][0][2]
    # print(x_temp, y_temp, z_temp)
    c_temp = 2 * ni_temp / (umath.sqrt((x_temp/A)**2 + (y_temp/B)**2 + (z_temp/C)**2))
    c_izrac_sez.append(c_temp)
    # print(c_temp)


c_avg = obtezeno_povprecje(c_izrac_sez)
print(c_avg)





# Plot the data with identified peaks
plt.plot(column1, column2, label='Meritev')
plt.xlabel(r'$\nu$ [Hz]')
plt.ylabel('$U_{amp}$ [V]')
plt.title('Resonančni odziv')
x_limits = plt.xlim()
y_limits = plt.ylim()
plt.ylim(y_limits)
plt.xlim(x_limits)

# plt.savefig('AkRes/Resonanca.png', dpi=300, bbox_inches="tight")
plt.show()






def calculate_fwhm(x, y, prominence_threshold=0.01):
    # Find peaks in the spectrum with a prominence threshold
    peaks, _ = find_peaks(y, prominence=prominence_threshold)

    fwhm_data = []

    # Iterate over detected peaks
    for peak_index in peaks:
        # Find the width of the peak at half maximum
        widths, *_ = peak_widths(y, [peak_index], rel_height=0.5)

        # Get the x value corresponding to the peak
        x_peak = x[peak_index]

        # Append FWHM and corresponding x value to the list
        fwhm_data.append((widths[0], x_peak))

    return fwhm_data

# Example usage:
# Replace the following with your actual data


# Adjust the prominence_threshold as needed
fwhm_data_example = calculate_fwhm(column1, column2, prominence_threshold=0.01)


# print("###########################################")
# print("Izračunane resoncance: ", Resonance)
# print("###########################################")
# print("Column 1 values at Peaks:", column1[peaks])
# print("###########################################")
# print("FWHM Data:", fwhm_data_example)
# print("###########################################")
# print("Column 2 values at Peaks:", column2[peaks])




data_primerjava = pd.read_csv("AkRes/Podatki/Odziv_duseno.txt", sep='\t', header=None)

# Ensure that the column indices are within the range of the number of columns
num_columns_primerjava = data_primerjava.shape[1]

# Convert Pandas DataFrame columns to 1D NumPy arrays
column1_primerjava = np.squeeze(np.array(data_primerjava.iloc[:, 0])) if num_columns_primerjava > 0 else np.array([])
column2_primerjava = np.squeeze(np.array(data_primerjava.iloc[:, 1])) if num_columns_primerjava > 1 else np.array([])
column3_primerjava = np.squeeze(np.array(data_primerjava.iloc[:, 2])) if num_columns_primerjava > 2 else np.array([])
column4_primerjava = np.squeeze(np.array(data_primerjava.iloc[:, 3])) if num_columns_primerjava > 3 else np.array([])


# Plot the data with identified peaks
plt.plot(column1, column2, label='Nedušeno')
plt.plot(column1_primerjava, column2_primerjava, label='Dušeno')
plt.xlabel(r'$\nu$ [Hz]')
plt.ylabel('$U_{amp}$ [V]')
plt.title('Primerjava nedušenega in dušenega resonančnega odziva')
x_limits = plt.xlim()
y_limits = plt.ylim()
plt.ylim(y_limits)
plt.xlim(x_limits)

plt.legend()
# plt.savefig('AkRes/Primerjava.png', dpi=300, bbox_inches="tight")
plt.show()








data3 = pd.read_csv("AkRes/Podatki/Polozaj_304.txt", sep='\t', header=None)

# Ensure that the column indices are within the range of the number of columns
num_columns3 = data3.shape[1]

# Convert Pandas DataFrame columns to 1D NumPy arrays
column1_3 = np.squeeze(np.array(data3.iloc[:, 0])) if num_columns3 > 0 else np.array([])
column2_3 = np.squeeze(np.array(data3.iloc[:, 1])) if num_columns3 > 1 else np.array([])
column3_3 = np.squeeze(np.array(data3.iloc[:, 2])) if num_columns3 > 2 else np.array([])
koordinata_x_3 = np.arange(len(column1_3))



plt.plot(koordinata_x_3, column1_3, "-o")
plt.xlabel(r'$x$ [cm]')
plt.ylabel('$U_{amp}$ [V]')
plt.title(r'Odvisnost signala od položaja mikrofona pri $\nu = 304$ Hz')
x_limits = plt.xlim()
y_limits = plt.ylim()
plt.ylim(y_limits)
plt.xlim(x_limits)


# plt.savefig('AkRes/Polozaj_1.png', dpi=300, bbox_inches="tight")
plt.show()


#####################################################################################################################

data4 = pd.read_csv("AkRes/Podatki/Polozaj_446.txt", sep='\t', header=None)

# Ensure that the column indices are within the range of the number of columns
num_columns4 = data4.shape[1]

# Convert Pandas DataFrame columns to 1D NumPy arrays
column1_4 = np.squeeze(np.array(data4.iloc[:, 0])) if num_columns4 > 0 else np.array([])
column2_4 = np.squeeze(np.array(data4.iloc[:, 1])) if num_columns4 > 1 else np.array([])
column3_4 = np.squeeze(np.array(data4.iloc[:, 2])) if num_columns4 > 2 else np.array([])
koordinata_x_4 = np.arange(len(column1_4))



plt.plot(koordinata_x_4, column1_4, "-o")
plt.xlabel(r'$x$ [cm]')
plt.ylabel('$U_{amp}$ [V]')
plt.title(r'Odvisnost signala od položaja mikrofona pri $\nu = 446$ Hz')
x_limits = plt.xlim()
y_limits = plt.ylim()
plt.ylim(y_limits)
plt.xlim(x_limits)


# plt.savefig('AkRes/Polozaj_2.png', dpi=300, bbox_inches="tight")
plt.show()


##############################################################################################################################

data5 = pd.read_csv("AkRes/Podatki/Polozaj_540.txt", sep='\t', header=None)

# Ensure that the column indices are within the range of the number of columns
num_columns5 = data5.shape[1]

# Convert Pandas DataFrame columns to 1D NumPy arrays
column1_5 = np.squeeze(np.array(data5.iloc[:, 0])) if num_columns5 > 0 else np.array([])
column2_5 = np.squeeze(np.array(data5.iloc[:, 1])) if num_columns5 > 1 else np.array([])
column3_5 = np.squeeze(np.array(data5.iloc[:, 2])) if num_columns5 > 2 else np.array([])
koordinata_x_5 = np.arange(len(column1_5))



plt.plot(koordinata_x_5, column1_5, "-o")
plt.xlabel(r'$x$ [cm]')
plt.ylabel('$U_{amp}$ [V]')
plt.title(r'Odvisnost signala od položaja mikrofona pri $\nu = 540$ Hz')
x_limits = plt.xlim()
y_limits = plt.ylim()
plt.ylim(y_limits)
plt.xlim(x_limits)


# plt.savefig('AkRes/Polozaj_3.png', dpi=300, bbox_inches="tight")
plt.show()


#########################################################################################################################

data6 = pd.read_csv("AkRes/Podatki/Polozaj_604.txt", sep='\t', header=None)

# Ensure that the column indices are within the range of the number of columns
num_columns6 = data6.shape[1]

# Convert Pandas DataFrame columns to 1D NumPy arrays
column1_6 = np.squeeze(np.array(data6.iloc[:, 0])) if num_columns6 > 0 else np.array([])
column2_6 = np.squeeze(np.array(data6.iloc[:, 1])) if num_columns6 > 1 else np.array([])
column3_6 = np.squeeze(np.array(data6.iloc[:, 2])) if num_columns6 > 2 else np.array([])
koordinata_x_6 = np.arange(len(column1_6))



plt.plot(koordinata_x_6, column1_6, "-o")
plt.xlabel(r'$x$ [cm]')
plt.ylabel('$U_{amp}$ [V]')
plt.title(r'Odvisnost signala od položaja mikrofona pri $\nu = 604$ Hz')
x_limits = plt.xlim()
y_limits = plt.ylim()
plt.ylim(y_limits)
plt.xlim(x_limits)


# plt.savefig('AkRes/Polozaj_4.png', dpi=300, bbox_inches="tight")
plt.show()


