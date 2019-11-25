from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['axes.grid'] = True
plt.style.use('seaborn-colorblind')

fig, ax = plt.subplots(2, 2, sharex='col', sharey='row')

x = np.array([i for i in range(21)])  # -> number of columns released (the results are in ascending order of
                                        # number of columns DELETED)
y = [None, None, None, None]

n_exp = 1000
# !!!! RESULTS FOR L=16 !!!!
# gamma = 2
y[0] = np.flip(np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 996, 998, 992, 989, 982, 942, 929, 896,
                         884, 854, 804, 792, 0]))
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 5
y[1] = np.flip(np.array([1000, 998, 1000, 994, 984, 952, 920, 883, 880, 808, 753, 692, 603, 506, 498, 381, 397, 316,
                         270, 232, 0]))
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 7
y[2] = np.flip(np.array([1000, 995, 983, 965, 915, 872, 792, 735, 673, 607, 518, 507, 416, 361, 278, 206, 188, 161,
                          129, 110, 0]))
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 20  todo: repeat this with smaller gamma
y[3] = np.flip(np.array([460, 400, 230, 260, 162, 132, 62, 72, 27, 28, 33, 23, 13, 11, 10, 12, 3, 2, 2, 4, 0]))
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
ax2.plot(x, y[0], label='$\gamma$ = 2')
ax2.plot(x, y[1], label='$\gamma$ = 5')
ax2.plot(x, y[2], label='$\gamma$ = 7')
# ax2.plot(x, y[3])
plt.xticks(np.arange(0, 21, step=2), [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 'all'])
fig2.text(0.5, 0.02, 'Number of columns released', ha='center')
fig2.text(0.04, 0.5, 'Detected fingerprints(%)', va='center', rotation='vertical')
ax2.legend()
plt.show()


