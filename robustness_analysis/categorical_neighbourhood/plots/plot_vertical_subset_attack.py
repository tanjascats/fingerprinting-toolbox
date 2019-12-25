from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['axes.grid'] = True
plt.style.use('seaborn-colorblind')

fig, ax = plt.subplots(2, 2, sharex='col', sharey='row')

x = np.array([i for i in range(10)])  # -> number of columns released (the results are in ascending order of
                                        # number of columns DELETED)
y = [None, None, None, None]

n_exp = 1000
# !!!! RESULTS FOR BREAST CANCER L=8!!!!
# gamma = 1
y[0] = np.flip(np.array([1000, 1000, 1000, 980, 993, 993, 958, 942, 919, 0]))
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 2
y[1] = np.flip(np.array([1000, 1000, 1000, 973, 957, 931, 831, 731, 734, 0]))
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 3
y[2] = np.flip(np.array([1000, 996, 977, 922, 884, 841, 693, 606, 562, 0]))
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 5
y[3] = np.flip(np.array([1000, 964, 886, 727, 614, 474, 455, 381, 257, 0]))
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

for i in range(2):
    for j in range(2):
        y[2*i+j] = (y[2*i+j] / n_exp) * 100
        ax[i, j].plot(x, y[2*i+j])
        plt.grid()

# todo test how they look all in one plot
ax[0, 0].plot(x, y[1])
ax[0, 0].plot(x, y[2])
ax[0, 0].plot(x, y[3])
# todo end of test

fig.text(0.5, 0.02, 'Number of columns released', ha='center')
fig.text(0.04, 0.5, 'Detected fingerprints(%)', va='center', rotation='vertical')

# --------------------------------------------------------------------- #
# ----------------- ALL IN ONE FIGURE --------------------------------- #
# --------------------------------------------------------------------- #
fig2, ax2 = plt.subplots(1, 1, sharex='col', sharey='row')
ax2.plot(x, y[0], label='$\gamma$ = 1')
ax2.plot(x, y[1], label='$\gamma$ = 2')
ax2.plot(x, y[2], label='$\gamma$ = 3')
ax2.plot(x, y[3], label='$\gamma$ = 5')
# ax2.plot(x, y[3])
plt.xticks(np.arange(0, 10, step=1), [0, 1, 2, 3, 4, 5, 6, 7, 8, 'ALL'])
fig2.text(0.5, 0.02, 'Number of columns released', ha='center')
fig2.text(0.04, 0.5, 'Detected fingerprints(%)', va='center', rotation='vertical')
ax2.legend()
plt.show()


