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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Serif",
    "font.size": 20
})

###########################################################
Imena = ["1", "2", "3", "4", "5", "6"]
Skupno = []

def polnenje(x, U_b, U_0, t_0, tau):
    return U_b + U_0 * np.exp(- (x - t_0) / tau)


for ime in Imena:
    df = pd.read_csv(f"Piezo/PIEZO/SDS0000{ime}.csv", skiprows=12)

    x1 = df[df.columns[0]][200:]
    y1 = df[df.columns[1]][200:]

    maks = np.argmax(y1)
    mini = np.argmin(y1)
    print(maks, mini)

    if maks < mini:
        x1 = df[df.columns[0]][maks:]
        y1 = df[df.columns[1]][maks:]

    print(np.shape(x1))
    # # Plot the data
    # print("Column names:", df.columns)
    # plt.plot(df[df.columns[0]], df[df.columns[1]])
    # plt.xlabel('Time (s)')  # Adjust the label based on your actual data
    # plt.ylabel('Voltage (V)')  # Adjust the label based on your actual data
    # plt.title('CSV Data Plot')
    # plt.show()

    ##### Fitanje
    slope, intercept, r_value, p_value1, std_err = stats.linregress(x1, y1)

    optimizedParameters, pcov = opt.curve_fit(polnenje, x1, y1)

    plt.plot(np.concatenate((np.array([0]), x1), axis=None), polnenje(np.concatenate((np.array([0]), x1), axis=None), *optimizedParameters),
         color="black", linestyle="dashed", label="Fit")
    plt.plot(x1, y1, ".", label="Izmerjeno", color="#0033cc")
    plt.show()
    

    # podatki = pd.read_csv(f"Piezo/PIEZO/SDS0000{ime}.csv", sep='\t', header=None)
    # podatki = podatki.to_numpy()
    # podatki = podatki.transpose()
    # print(podatki[12:])
    # # plt.plot(podatki[0], podatki[1])
    # # plt.show()

# # Read the text file with pandas
# file_path = "UkLec/UkLec_BK/Meritev_x_os.txt"
# podatki_x = pd.read_csv(file_path, sep='\t', header=None)

# # Convert to NumPy array
# Meritev_x = podatki_x.to_numpy()


# # Read the text file with pandas
# file_path = "UkLec/UkLec_BK/Izmaknjen_po_x_osi.txt"
# podatki_x_5cm = pd.read_csv(file_path, sep='\t', header=None)

# # Convert to NumPy array
# Meritev_x_5cm = podatki_x_5cm.to_numpy()