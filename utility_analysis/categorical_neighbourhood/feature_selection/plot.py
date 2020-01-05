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
y[0] = n_exp - np.flip(np.array([1000, 1000, 1000, 980, 993, 993, 958, 942, 919, 0]))
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 2
y[1] = n_exp - np.flip(np.array([1000, 1000, 1000, 973, 957, 931, 831, 731, 734, 0]))
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 3
y[2] = n_exp - np.flip(np.array([1000, 996, 977, 922, 884, 841, 693, 606, 562, 0]))
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 5
y[3] = n_exp - np.flip(np.array([1000, 964, 886, 727, 614, 474, 455, 381, 257, 0]))
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

for i in range(2):
    for j in range(2):
        y[2*i+j] = (y[2*i+j] / n_exp) * 100
        ax[i, j].plot(x, y[2*i+j])

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
fig2, ax2 = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(10, 5.8))
ax2.plot(x, y[0]/100, label='$\gamma$ = 1')
ax2.plot(x, y[1]/100, label='$\gamma$ = 2')
ax2.plot(x, y[2]/100, label='$\gamma$ = 3')
ax2.plot(x, y[3]/100, label='$\gamma$ = 5')
# ax2.plot(x, y[3])
plt.xticks(np.arange(0, 10, step=1), [0, 1, 2, 3, 4, 5, 6, 7, 8, 'ALL'])

fig2.text(0.5, 0.02, 'Number of columns released', ha='center')
fig2.text(0.07, 0.5, 'False Miss(%)', va='center', rotation='vertical')
fig2.text(0.94, 0.5, 'Performance score', va='center', rotation=270)
ax2.legend()

# ... BOX PLOTS... #
#labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 'ALL']
f1_means = np.flip([0.551235628509419,0.5719038621616768,0.5758548800543405,0.5841124717112676,0.5859617411235051,
           0.5867575380315879, 0.5606512125216047,0.5420420540739059,0.5197325647330089, 0])
accuracy_means = np.flip([0.6658583743842365,0.710103448275862,0.7123690476190477,0.7170615763546799, 0.7182992610837438,0.7205755336617405
            ,0.7089084564860426, 0.7063050082101807, 0.6937721674876848, 0])

#x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

rects1 = ax2.bar(x - width/2, accuracy_means, width, label='accuracy', color='#715c82', fc=(0.9, 0.5, 0.25, 0.3))
rects2 = ax2.bar(x + width/2, f1_means, width, label='F1 score', color='#d6ae4f', fc=(80/255, 146/255, 217/255, 0.3))
plt.axhline(y=0.6658583743842365, color=(0.9, 0.5, 0.25, 0.8), linestyle='--', linewidth=0.95)
plt.axhline(y=0.551235628509419, color=(80/255, 146/255, 217/255, 0.8), linestyle='--', linewidth=0.95)
ax2.tick_params(labelright=True)



# Add some text for labels, title and custom x-axis tick labels, etc.
#ax2.set_ylabel('Scores')
#ax2.set_title('Scores by group and gender')
#ax2.set_xticks(x)
#ax2.set_xticklabels(labels)
ax2.legend()





plt.show()


