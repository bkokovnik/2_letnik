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

for ime in Imena:

    # Read the text file with pandas
    file_path = f"SiDiod/SiDiod_podatki/Napajanje_{ime}cm.txt"  # Replace with your file path
    podatki = pd.read_csv(file_path, sep='\t', header=None)

    # Convert to NumPy array
    numpy_array = podatki.to_numpy()
    Skupno.append(numpy_array)
    
    # plt.plot(numpy_array[:, 0], numpy_array[:, 1], "-o", label="Izmerjeno", color="#0033cc")
    # # plt.ylim(top=-0.575)

    # plt.show()

n=0
Naslovi = ["0 cm", "0,5 cm", "1 cm", "1,5 cm", "2 cm", "$\infty$ cm"]
fig, axs = plt.subplot_mosaic([['left']], layout='constrained')

for graf in Skupno:
    axs['left'].plot(graf[:, 0], graf[:, 1], "-o", label=Naslovi[n], markersize=5)
    n += 1

axs['left'].set_ylim(-0.65, 0.2)
axs['left'].set_xlabel("$U$ [V]")
axs['left'].set_ylabel("$I$ [mA]")
axs["left"].spines[['right', 'top']].set_visible(False)
# fig.ylim(-0.65, 0.2)
fig.legend(loc='outside center right')
plt.title("Primerjava karakteristik napajane fotodiode\npri različnih osvetlitvah")
plt.savefig("SiDiod_Napajana.png", dpi=300, bbox_inches="tight")
plt.show()




Skupno2 = []

for ime in Imena[:-1]:

    # Read the text file with pandas
    file_path = f"SiDiod/SiDiod_podatki/Brez_{ime}cm.txt"
    podatki = pd.read_csv(file_path, sep='\t', header=None)

    # Convert to NumPy array
    numpy_array = podatki.to_numpy()
    Skupno2.append(numpy_array)
    
    # plt.plot(numpy_array[:, 0], numpy_array[:, 1], "-o", label="Izmerjeno", color="#0033cc")
    # # plt.ylim(top=-0.575)

    # plt.show()

n=0
fig, axs = plt.subplot_mosaic([['left']], layout='constrained')

for graf in Skupno2:
    axs['left'].plot(graf[:, 0], graf[:, 1], "-o", label=Naslovi[n], markersize=5)
    n += 1

# axs['left'].set_ylim(-0.65, 0.2)
# axs['left'].set_xlim(-0, 10.25)
axs['left'].set_xlabel("$U$ [V]")
axs['left'].set_ylabel("$I$ [mA]")
axs["left"].spines[['right', 'top']].set_visible(False)
# fig.ylim(-0.65, 0.2)
fig.legend(loc='outside center right')
plt.title("Primerjava karakteristik v fotogalvanskem\nnačinu pri različnih osvetlitvah")
plt.savefig("SiDiod_Brez.png", dpi=300, bbox_inches="tight")
plt.show()






n=0

for graf in Skupno2:
    upornost = graf[:, 1] / graf[:, 0]
    moc = graf[:, 1] * graf[:, 0]
    plt.plot(-upornost, -moc, "-o", label=Naslovi[n], markersize=5)
    print("Največja moč: ", np.max(-moc), " mW, pri uporu ", -upornost[np.where(-moc == np.max(-moc))])

    n += 1


plt.xlim(-0, 10.25)
start, end = plt.xlim()
plt.xticks(np.arange(start, end, 2))
starty, endy = plt.ylim()
plt.ylim(-0, endy)
plt.yticks(np.arange(0, endy, 0.075))
plt.ylabel("$P$ [mW]")
plt.xlabel("$R$ [$\Omega$]")
# fig.ylim(-0.65, 0.2)
plt.legend()#loc='outside center right')
plt.title("Moč v odvisnosti od upornosti")
plt.savefig("SiDiod_Moc_Upor.png", dpi=300, bbox_inches="tight")
plt.show()






print("napetost: ", np.min(np.abs(Skupno[0][1])), " mW, tok ", Skupno[0][0][np.where(np.abs(Skupno[0][1]) == np.min(np.abs(Skupno[0][1])))])

ni = unc.ufloat(0.43, 0.1)
U = unc.ufloat(1.8823, 0.0001)
I = unc.ufloat(23.29, 0.01)

I_izm = unc.ufloat(np.average(np.abs(Skupno[0][:, 1][0:15])), np.std(Skupno[0][:, 1][0:15]) * np.average(np.abs(Skupno[0][:, 1][0:15])))
print(I_izm)
print(I_izm/(ni * U * I))




# axs['left'].set_xlim(-0, 10.25)
# start, end = axs["left"].get_xlim()
# axs['left'].set_xticks(np.arange(start, end, 2))
# starty, endy = axs["left"].get_ylim()
# axs['left'].set_ylim(-0, endy)
# axs['left'].set_yticks(np.arange(0, endy, 0.075))
# axs['left'].set_xlabel("$P$ [mW]")
# axs['left'].set_ylabel("$R$ [$\Omega$]")
# # fig.ylim(-0.65, 0.2)
# fig.legend()#loc='outside center right')
# plt.title("Moč v odvisnosti od upornosti")
# plt.show()





# ###########################################################


# ##### Seznami podatkov
# x1 = np.array([640, 745, 1010, 1180, 1320, 1555, 1670, 1735, 1785, 1825])
# y1 = np.array([0.24, 0.29, 0.49, 0.68, 0.87, 1.03, 1.19, 1.42, 1.54, 1.68])

# ##### Konstante, parametri in začetne vrednosti
# r = 9.5 / 100 #m
# d = 5.5 / 1000 #m

# ##### Definicije funkcij
# def lin_fun(x, a, b):
#     return a * x + b

# ##### Obdelava seznamov
# x1 = x1 ** 2
# y1 = (y1 / 1000) * const.g

# ##### Fitanje
# slope, intercept, r_value, p_value1, std_err = stats.linregress(x1, y1)

# optimizedParameters, pcov = opt.curve_fit(lin_fun, x1, y1)

# ##### Graf
# plt.plot(np.concatenate((np.array([0]), x1), axis=None), lin_fun(np.concatenate((np.array([0]), x1), axis=None), *optimizedParameters),
#          color="black", linestyle="dashed", label="Fit")
# plt.plot(x1, y1, "o", label="Izmerjeno", color="#0033cc")

# plt.xlabel('$U^2 [V^2]$')
# plt.ylabel('$F [N]$')

# xlim_spodnja = 0
# xlim_zgornja = 3600000
# ylim_spodnja = 0
# ylim_zgornja = 0.018
# plt.xlim(xlim_spodnja, xlim_zgornja)
# plt.ylim(ylim_spodnja, ylim_zgornja)
# # major_ticksx = np.arange(xlim_spodnja, xlim_zgornja + 0.0000001, (xlim_zgornja - xlim_spodnja)/6)
# # minor_ticksx = np.arange(xlim_spodnja, xlim_zgornja + 0.0000001, (xlim_zgornja - xlim_spodnja)/60)
# # major_ticksy = np.arange(ylim_spodnja, ylim_zgornja + 0.0000001, (ylim_zgornja - ylim_spodnja)/4)
# # minor_ticksy = np.arange(ylim_spodnja, ylim_zgornja + 0.0000001, (ylim_zgornja - ylim_spodnja)/40)
# # plt.xticks(major_ticksx)
# # plt.xticks(minor_ticksx, minor=True)
# # plt.yticks(major_ticksy)
# # plt.yticks(minor_ticksy, minor=True)
# # plt.grid(which='minor', alpha=0.2)
# # plt.grid(which='major', alpha=0.5)
# plt.legend()

# # plt.show()
# # plt.savefig("Vaja_47_graf_epsilon.png", dpi=300, bbox_inches="tight")

# ##### Izračuni
# print(slope * 2 * d**2 / (np.pi * r**2), ", relativna napaka brez d in S: ", std_err / slope, slope)

# napaka = 0.09 # Izračunana relativna napaka skupaj z d^2 in S