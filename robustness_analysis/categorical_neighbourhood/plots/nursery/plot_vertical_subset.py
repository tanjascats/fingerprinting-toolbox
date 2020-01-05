from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['axes.grid'] = True
plt.style.use('seaborn-colorblind')

x = np.array([i for i in range(9)])  # -> number of columns released (the results are in ascending order of
                                        # number of columns DELETED)
y = [None, None, None, None]

n_exp = 250
# !!!! RESULTS FOR BREAST CANCER L=64!!!!
# gamma = 5
y[0] = n_exp - np.flip(np.array([250, 250, 250, 250, 243, 232, 226, 201, 0]))
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 10
y[1] = n_exp - np.flip(np.array([250, 250, 247, 230, 190, 163, 127, 87, 0]))
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 20
y[2] = n_exp - np.flip(np.array([250, 231, 185, 111, 63, 34, 26, 7, 0]))
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 30
y[3] = n_exp - np.flip(np.array([250, 167, 73, 28, 17, 16, 6, 4, 0]))

# --------------------------------------------------------------------- #
# ----------------- ALL IN ONE FIGURE --------------------------------- #
# --------------------------------------------------------------------- #
fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
ax.plot(x, y[0]/n_exp, label='$\gamma$ = 5')
ax.plot(x, y[1]/n_exp, label='$\gamma$ = 10')
ax.plot(x, y[2]/n_exp, label='$\gamma$ = 20')
ax.plot(x, y[3]/n_exp, label='$\gamma$ = 30')
# ax2.plot(x, y[3])
plt.xticks(np.arange(0, 9, step=1), [0, 1, 2, 3, 4, 5, 6, 7, 'ALL'])
fig.text(0.5, 0.02, 'Number of columns released', ha='center')
fig.text(0.04, 0.5, 'False Miss', va='center', rotation='vertical')
ax.legend()
plt.show()
