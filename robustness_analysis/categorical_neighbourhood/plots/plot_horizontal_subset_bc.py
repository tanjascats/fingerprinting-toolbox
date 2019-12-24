from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['axes.grid'] = True
plt.rcParams["legend.loc"] = 'lower right'
plt.style.use('seaborn-colorblind')

fig, ax = plt.subplots(2, 2, sharex='col', sharey='row')

x = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
              0.9, 0.95, 1])
x = x*100
n_exp = 1000
y = [None, None]

# !!!! RESULTS FOR L=8 !!!!
# gamma = 2
y[0] = np.array([0, 0, 17, 115, 223, 460, 584, 720, 777, 876, 915, 954, 961, 991, 988, 992, 999, 1000, 1000, 1000, 1000])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 5
y[1] = np.array([0, 0, 0, 3, 10, 52, 75, 186, 271, 345, 454, 577, 620, 677, 785, 820, 892, 917, 961, 986, 1000])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 7 - not 100% successful anymore

# todo test how they look all in one plot
ax[0, 0].plot(x, y[0])
ax[0, 0].plot(x, y[1])
# todo end of test

fig.text(0.5, 0.02, 'Size of the subset(%)', ha='center')
fig.text(0.04, 0.5, 'Detected fingerprints(%)', va='center', rotation='vertical')

#plt.show()

fig2, ax2 = plt.subplots(1, 1, sharex='col', sharey='row')
ax2.plot(x, y[0], label='$\gamma$ = 2')
ax2.plot(x, y[1], label='$\gamma$ = 5')
# ax2.plot(x, y[3])
fig2.text(0.5, 0.02, 'Size of the subset released(%)', ha='center')
fig2.text(0.04, 0.5, 'Detected fingerprints(%)', va='center', rotation='vertical')
ax2.legend()
plt.show()

