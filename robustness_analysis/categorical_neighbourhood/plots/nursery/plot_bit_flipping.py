import matplotlib.pyplot as plt
import numpy as np

x = 100 - np.flip(np.array([0.0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])) * 100
# we omit further examples because changing more than 60% of data set's values is arguably of no
# significant value for the data user
y = [None, None, None, None]  # detected fingerprints

n_exp = 250

# !!!! RESULTS FOR BREAST CANCER L=64!!!!
# --------------------------------------- #
# gamma = 5
# --------------------------------------- #
y[0] = n_exp - np.flip(np.array([250, 250, 250, 250, 250, 250, 250, 248, 250, 241, 230, 210, 170]))
# --------------------------------------- #
# gamma = 10
# --------------------------------------- #
y[1] = n_exp - np.flip(np.array([250, 250, 250, 248, 247, 242, 233, 205, 177, 136, 101, 60, 13]))
# --------------------------------------- #
# gamma = 20
# --------------------------------------- #
y[2] = n_exp - np.flip(np.array([250, 248, 232, 203, 146, 130, 71, 42, 18, 6, 1, 0, 0]))
# --------------------------------------- #
# gamma = 30
# --------------------------------------- #
y[3] = n_exp - np.flip(np.array([250, 198, 118, 68, 46, 24, 7, 3, 2, 0, 0, 0, 0]))

plt.style.use('seaborn-colorblind')
plt.grid()
plt.xlabel("Portion of the unchanged data(%)")
plt.ylabel("False Miss")

plt.plot(x, y[0]/n_exp, label='$\gamma$ = 5')
plt.plot(x, y[1]/n_exp, label='$\gamma$ = 10')
plt.plot(x, y[2]/n_exp, label='$\gamma$ = 20')
plt.plot(x, y[3]/n_exp, label='$\gamma$ = 30')
plt.legend()
plt.show()