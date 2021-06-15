from parameter_guidelines.guidelines import *
import pickle


def plot():
    gammae = [1, 2, 3, 4, 5, 6, 10, 12, 15, 18, 20, 25, 30, 50, 100, 200]

    # ################################################# #
    # ################################################# #
    target = 'income'
    attack_granularity = 0.05
    n_folds = 5
    file_string = 'attack_utility_horizontal_universal_knn_{}_ag{}_nf{}.pickle'.format(target,
                                                                                       format(attack_granularity,
                                                                                              ".2f")[-2:],
                                                                                       n_folds)
    with open('parameter_guidelines/evaluation/adult/' + file_string, 'rb') as infile:
        attack_utility_knn = pickle.load(infile)
    attack_utility_knn = dict(sorted(attack_utility_knn.items()))

    # original utilities
    file_string = 'utility_knn_{}_nf{}.pickle'.format(target, n_folds)
    with open('parameter_guidelines/evaluation/adult/' + file_string, 'rb') as infile:
        original_utility_knn = pickle.load(infile)
    att_utility_bounds = attack_utility_bounds(original_utility_knn, attack_utility_knn)
    #file_string = 'utility_dt_{}_nf{}.pickle'.format(target, n_folds)
    #with open('parameter_guidelines/evaluation/adult/' + file_string, 'rb') as infile:
    #    original_utility_dt = pickle.load(infile)
    #file_string = 'utility_dt_{}_gb{}.pickle'.format(target, n_folds)
    #with open('parameter_guidelines/evaluation/adult/' + file_string, 'rb') as infile:
    #    original_utility_gb = pickle.load(infile)

    n_experiments = 10
    file_string = 'archive/utility_universal_knn_{}_nf{}_e{}.pickle'.format(target, n_folds, n_experiments)
    with open('parameter_guidelines/evaluation/adult/' + file_string, 'rb') as infile:
        fp_utility_results = pickle.load(infile)
    file_string = 'archive/utility_universal_dt_{}_nf{}_e{}.pickle'.format(target, n_folds, n_experiments)
    with open('parameter_guidelines/evaluation/adult/inverse_robustness_horizontal_universal_c95_ag05_e100.pickle',
              'rb') as infile:
        resutls = pickle.load(infile)

    with open('parameter_guidelines/evaluation/adult/' + file_string, 'rb') as infile:
        fp_utility_results_DT = pickle.load(infile)
    file_string = 'archive/utility_universal_gb_{}_nf{}_e{}.pickle'.format(target, n_folds, n_experiments)
    with open('parameter_guidelines/evaluation/adult/' + file_string, 'rb') as infile:
        fp_utility_results_GB = pickle.load(infile)

    # interceptions
    x_intercept_0 = 1.0 / (max([x for x in resutls if 1 - resutls[x] >= att_utility_bounds[0]]))
    x_intercept_1 = 1.0 / (max([x for x in resutls if 1 - resutls[x] >= att_utility_bounds[1]]))
    # ################################################# #
    # ################################################# #

    fig, ax = plt.subplots(2, sharex=True, figsize=(10, 7))  # sharey=True

    lines = []
    fill = []
    helpers = []

    x_axis = [1.0 / g for g in gammae]

    # ---------------------------- #
    # robustness
    # ---------------------------- #
    y_robustness = [1 - inv_robustness for inv_robustness in resutls.values()]
    lines.append(ax[0].plot(x_axis, y_robustness, label='robustness'))

    # --------------------------- #
    # utility (start)
    # --------------------------- #
    y_utility = [np.mean([np.mean(cv_res) for cv_res in fp_utility_results[g]])
                 for g in fp_utility_results.keys()] + [np.mean(original_utility_knn)]
    y_utility_DT = [np.mean([np.mean(cv_res) for cv_res in fp_utility_results_DT[g]])
                    for g in fp_utility_results_DT.keys()]
    y_utility_GB = [np.mean([np.mean(cv_res) for cv_res in fp_utility_results_GB[g]])
                    for g in fp_utility_results_GB.keys()]
    y_utility_all = y_utility + y_utility_DT + y_utility_GB

    # --------------------------- #
    # attack utility
    # --------------------------- #
    lw = 0.8

    lines.append(ax[0].plot(x_axis, [att_utility_bounds[0] for x in resutls], label='1% drop in accuracy', linewidth=lw,
                            color='#a2d180'))
    fill.append(ax[0].fill_between(x_axis, [att_utility_bounds[0] for x in resutls], 1.0, color='#e1ffcc'))
    helpers.append(ax[0].vlines(x=x_intercept_0, ymin=min(y_robustness), ymax=att_utility_bounds[0], linewidth=lw,
                                linestyles='dashed'))
    helpers.append(
        ax[1].vlines(x=x_intercept_0, ymin=min(y_utility_all), ymax=max(y_utility_all), linewidth=lw,
                     linestyles='dashed', color='#a2d180'))
    fill.append(ax[1].fill_between([x for x in x_axis if x >= x_intercept_0], min(y_utility_all),
                                   max(y_utility_all),
                                   color='#e1ffcc'))

    lines.append(ax[0].plot(x_axis, [att_utility_bounds[1] for x in resutls], label='2% drop in accuracy', linewidth=lw,
                            color='green'))
    fill.append(ax[0].fill_between(x_axis, [att_utility_bounds[1] for x in resutls], 1.0, color='#c3e3ac'))
    helpers.append(ax[0].vlines(x=x_intercept_1, ymin=min(y_robustness), ymax=att_utility_bounds[1], linewidth=lw,
                                linestyles='dashed'))
    helpers.append(
        ax[1].vlines(x=x_intercept_1, ymin=min(y_utility_all), ymax=max(y_utility_all), linewidth=lw,
                     linestyles='dashed', color='green'))
    fill.append(ax[1].fill_between([x for x in x_axis if x >= x_intercept_1], min(y_utility_all),
                                   max(y_utility_all),
                                   color='#c3e3ac'))

    # todo: mark on the first subplot (ax[0]) when the utility gets way too low; eg. -1%, -2% etc

    # ---------------------------- #
    # utility
    # ---------------------------- #
    # KNN
    lines.append(ax[1].plot(x_axis + [0.0], y_utility, label='KNN after fp'))
    # Decision Tree
    lines.append(ax[1].plot(x_axis, y_utility_DT, label='Decision tree after fp'))
    # Gradient Boosting
    lines.append(ax[1].plot(x_axis, y_utility_GB, label='Gradient boosting after fp'))
    ax[1].set_ylabel('Accuracy')


    # relative delta utility
#    rel_delta_utility_knn = [ut_fp - original_utility_knn for ut_fp in y_utility]
#    rel_delta_utility_dt = [ut_fp - original_utility_dt for ut_fp in y_utility_DT]
    # ---------------------------- #

    fig.suptitle(
        'How much the attacker can delete without affecting the fingerprint (via horizontal subset attack and KNN' +
        'accuracy)', size=10)
    plt.xlabel('#marks rel data size')
    ax[0].set_ylabel('Robustness (% rows deleted)')

    plt.rcParams['axes.grid'] = True

    legend = fig.legend()
    line_legends = legend.get_lines()
    for line in line_legends:
        line.set_picker(True)
        line.set_pickradius(10)
    graphs = {}
    graphs[line_legends[0]] = [lines[0][0]]  # -> robustness
    graphs[line_legends[1]] = [lines[1][0], fill[0], helpers[0], fill[1], helpers[1]]  # -> attack utility
    graphs[line_legends[2]] = [lines[2][0], fill[2], helpers[2], fill[3], helpers[3]]
    graphs[line_legends[3]] = [lines[3][0]]  # -> utility
    graphs[line_legends[4]] = [lines[4][0]]
    graphs[line_legends[5]] = [lines[5][0]]

    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        for element in graphs[legend]:
            element.set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()

    plt.connect('pick_event', on_pick)
    plt.show()


if __name__ == '__main__':
    plot()
