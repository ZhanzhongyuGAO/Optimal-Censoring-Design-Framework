"""
LCM_case.py
Codes for generate figures in LCM case study
The data used could be found in Table 7 from the work of Lee and Liao ``Monitoring gamma type-I censored data using
an exponentially weighted moving average control chart based on deep learning networks (2024, Scientific Report)''
"""

import func_gamma as fg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import newton
import warnings
warnings.filterwarnings("ignore")

# design setting
input_dict = {
    'alpha0': 5.72,
    'beta0': 0.48,
    'rhos': [0.10, 0.25, 0.50, 0.75],
    'n': 5,
    'alpha': 0.0027,
    'resolution': 1000,
    'start_c': 0.5,
    'step': 0.5,
    'end_c': 8,
}

def design_plot(input_dict):
    alpha0, beta0 = input_dict['alpha0'], input_dict['beta0']
    rho_values = input_dict['rhos']
    start_c, step, end_c = input_dict['start_c'], input_dict['step'], input_dict['end_c']
    n = input_dict['n']
    alpha = input_dict['alpha']
    resolution = input_dict['resolution']

    ats1_list = []
    ats2_list = []
    for rho in rho_values:
        h, t1, t2 = fg.grid_search(alpha0, beta0, rho, n, alpha, resolution, start_c=start_c, step=step, end_c=end_c)
        ats1_list.append(t1)
        ats2_list.append(t2)

    fig, axs = plt.subplots(1, 4, figsize=(10, 2.5), constrained_layout=True)
    c = np.arange(start_c, end_c, step)
    cs = [3, 2.5, 1.5, 1]
    index = [5, 4, 2, 1]
    y_list = [[0, 250], [0, 100], [0, 20], [0, 2]]
    text_y = [-10, -5, -1, 0.1]
    alphabet = ['(a) ', '(b) ', '(c) ', '(d) ']
    for j, rho in enumerate(rho_values):
        ax = axs[j]
        ats1, ats2 = ats1_list[j], ats2_list[j]
        ats = ats1 + ats2
        ats_value = ats[index[j]]
        ax.plot(c, ats, 'k-')
        ax.scatter(cs[j], ats_value, marker='*', color='red')
        ax.set_xlabel(r'$c$', fontsize=12)
        ax.set_title(alphabet[j] + rf'$\rho = {rho}$', fontsize=12)
        ax.set_ylim(y_list[j])
        ax.text(cs[j], ats_value - text_y[j], '(' + str(cs[j]) + ', ' + str(round(ats_value, 2)) + ')', color='crimson',
                fontsize=10)
        if j == 0:
            ax.set_ylabel(r'ATS$^{oc}$', fontsize=12)
    plt.show()

def case_study(alpha0, beta0, rho, H, c=1.76):
    beta1 = beta0 * rho
    df = pd.read_csv('data/LCM_data.csv')
    data = df.to_numpy()
    ICsample = data[:, :19]
    OCsample = data[:, 19:]

    try:
        beta_est = newton(fg.l_beta, np.ones(19) * beta0, tol=1e-1, maxiter=100, args=(alpha0, ICsample, c))
    except RuntimeError:
        beta_est = np.inf

    try:
        beta1_est = newton(fg.l_beta, np.ones(11) * beta1, tol=1e-1, maxiter=100, args=(alpha0, OCsample, c))
    except RuntimeError:
        beta1_est = np.inf

    plotting_stats = alpha0 * beta_est
    plotting_stats1 = alpha0 * beta1_est
    x = np.arange(0, 19, 1)
    y = plotting_stats
    x1 = np.arange(19, 30, 1)
    y1 = plotting_stats1

    threshold = 4
    IC_y = np.where(y > threshold, np.nan, y)
    OC_y = np.where(y1 > threshold, np.nan, y1)
    IC_y1 = np.where(y > threshold, threshold, np.nan)
    OC_y1 = np.where(y1 > threshold, threshold, np.nan)


    fig, ax = plt.subplots(figsize=(8,6))
    plt.plot(x, IC_y, linestyle='-', marker='o', markersize=5, markerfacecolor='none', markeredgecolor='blue',
             color='lightgray')
    plt.plot(x1, OC_y, linestyle='-', marker='o', markersize=5, markerfacecolor='none', markeredgecolor='blue',
             color='lightgray')
    plt.plot(x, IC_y1, linestyle='-', marker='o', markersize=5, markerfacecolor='none', markeredgecolor='blue',
             color='lightgray')
    plt.plot(x1, OC_y1, linestyle='-', marker='o', markersize=5, markerfacecolor='none', markeredgecolor='blue',
             color='lightgray')
    ellipse = Ellipse(xy=(20, OC_y[1]), width=3, height=0.25, edgecolor='crimson', facecolor='none',  linewidth=0.8)
    ax.add_patch(ellipse)

    plt.axhline(y=H, color='crimson', linestyle=(0, (5, 5)), linewidth=1.5, alpha=0.5)
    plt.axvline(x=19, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.text(5, H - 0.4, r'$\rm{LCL}$', color='crimson', fontsize=10)

    yticks = list(plt.yticks()[0])
    yticks = [tick for tick in yticks if tick < threshold] + [threshold]
    ytick_labels = [str(tick) for tick in yticks[:-1]] + ['Inf']
    plt.yticks(yticks, labels=ytick_labels)
    plt.xlabel('Sample Number')
    plt.ylabel('Sample Statistics')
    plt.show()

if __name__ == "__main__":
    design_plot(input_dict)
    # rho = 0.5 is used in the case study, with lower control limit 1.466908
    case_study(input_dict['alpha0'], input_dict['beta0'], 0.5, 1.466908)
