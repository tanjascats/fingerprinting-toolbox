import matplotlib.pyplot as plt
import numpy as np

x = 100 - np.flip(np.array([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])) * 100
# we omit further examples because changing more than 60% of data set's values is arguably of no
# significant value for the data user
y = [None, None, None, None]  # detected fingerprints

n_exp = 1000

# !!!! RESULTS FOR BREAST CANCER L=8!!!!
# --------------------------------------- #
# gamma = 1
# --------------------------------------- #
y[0] = np.flip(np.array([1000, 1000, 1000, 1000, 1000, 1000, 999, 995, 974, 978, 958, 939, 917])) / n_exp*100
# --------------------------------------- #
# gamma = 2
# --------------------------------------- #
y[1] = np.flip(np.array([1000, 1000, 1000, 998, 978, 987, 974, 950, 869, 868, 758, 697, 610])) / n_exp*100
# --------------------------------------- #
# gamma = 3
# --------------------------------------- #
y[2] = np.flip(np.array([1000, 1000, 968, 994, 960, 896, 876, 848, 754, 679, 588, 496, 388])) / n_exp*100
# --------------------------------------- #
# gamma = 5
# --------------------------------------- #
y[3] = np.flip(np.array([1000, 979, 910, 848, 859, 704, 641, 545, 461, 376, 332, 277, 181])) / n_exp*100

plt.style.use('seaborn-colorblind')
plt.grid()
plt.xlabel("Portion of the unchanged data(%)")
plt.ylabel("Detected fingerprints(%)")

plt.plot(x, y[0], label='$\gamma$ = 1')
plt.plot(x, y[1], label='$\gamma$ = 2')
plt.plot(x, y[2], label='$\gamma$ = 3')
plt.plot(x, y[3], label='$\gamma$ = 5')
plt.legend()
plt.show()