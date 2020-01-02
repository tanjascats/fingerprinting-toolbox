from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['axes.grid'] = True
plt.rcParams["legend.loc"] = 'lower right'
plt.style.use('seaborn-colorblind')


x = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
              0.9, 0.95, 1])
x = x*100
n_exp = 250
y = [None, None, None, None]

# !!!! RESULTS FOR L=64 !!!!
# gamma = 5
y[0] = np.array([0, 0, 83, 221, 245, 248, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250,
                 250, 250]) / n_exp*100
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 10
y[1] = np.array([0, 0, 0, 8, 74, 165, 213, 237, 247, 247, 250, 249, 250, 250, 250, 250, 250, 250, 250, 250,
                 250]) / n_exp*100
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 20
y[2] = np.array([0, 0, 0, 0, 0, 0, 10, 39, 82, 127, 166, 212, 222, 226, 241, 240, 247, 249, 246, 250, 250]) / \
       n_exp*100
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 30
y[3] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 10, 32, 56, 79, 112, 133, 167, 185, 199, 208, 218, 250]) / n_exp*100
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #


fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
ax.plot(x, y[0], label='$\gamma$ = 5')
ax.plot(x, y[1], label='$\gamma$ = 10')
ax.plot(x, y[2], label='$\gamma$ = 20')
ax.plot(x, y[3], label='$\gamma$ = 30')
fig.text(0.5, 0.02, 'Size of the subset released(%)', ha='center')
fig.text(0.04, 0.5, 'Detected fingerprints(%)', va='center', rotation='vertical')
ax.legend()
plt.show()

