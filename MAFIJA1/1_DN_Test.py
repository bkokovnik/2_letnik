import numpy as np
import matplotlib.pyplot as plt

R = 20
d_a = 4e-3
d_g = 6e-3
rho_a = 2.7e3
rho_g = 100
E = 2e9

# list = []
# x_os = []
# r = 0

# for i in range(int(R/(d_a + d_g))):
#     F = 0
#     for j in range(i):
#         F += 9.81 * np.pi * ((j + 1/2) * (d_a + d_g) * (2 * R - (j + 1/2) * (d_a + d_g))) * (rho_a * d_a + rho_g * d_g)
#         # print((j * (d_a + d_g) * (2 * R - j * (d_a + d_g))))
#         # r = np.sqrt(((j + 1/2) * (d_a + d_g) * (2 * R - (j + 1/2) * (d_a + d_g))))
#     list.append(F)
#     x_os.append(i)


# plt.plot(x_os, list)
# plt.show()
# print(list)
# print(int(R/(d_a + d_g)))
# print(r)

# suma = 0

# for i in range(int(R/(d_a + d_g)) - 1):
#     if i == -1 or i == 0:
#         continue
#     F = 0
#     for j in range(i):
#         F += 9.81 * np.pi * ((j + 1/2) * (d_a + d_g) * (2 * R - (j + 1/2) * (d_a + d_g))) * (rho_a * d_a + rho_g * d_g)
#         # print((j * (d_a + d_g) * (2 * R - j * (d_a + d_g))))
#         # r = np.sqrt(((j + 1/2) * (d_a + d_g) * (2 * R - (j + 1/2) * (d_a + d_g))))
#     list.append(F)
#     x_os.append(i)
#     suma -= F * d_g / (E * np.pi * ((i + 1/2) * (d_a + d_g) * (2 * R - (i + 1/2) * (d_a + d_g))))

# print(suma)

suma = 0

for i in range(int(R/(d_a + d_g)) - 1):
    if i == -1 or i == 0:
            continue
    

    suma += (1140 * 9.81 * ((i + 4/10) * (d_a + d_g)) * (3 * R - ((i + 4/10) * (d_a + d_g)))) / (3 * (2 * R - ((i + 4/10) * (d_a + d_g))))

print(-suma * d_g/E)