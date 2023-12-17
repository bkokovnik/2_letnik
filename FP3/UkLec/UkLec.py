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
podatki = pd.read_csv(file_path, sep='\t', header=None)

# Convert to NumPy array
Meritev_x = podatki.to_numpy()


plt.plot(np.arange(8, -8.2, -0.2), Meritev_x)
plt.show()
    
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