"""
LCM_case.py
Codes for generate figures in LCD case study
"""

import func_lognormal as fl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import newton
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings("ignore")

# design setting
input_dict = {
    'k0': 4.2592,
    'sigma0': 2.4403,
    'rhos': [0.10, 0.25, 0.50, 0.75],
    'n': 10,
    'alpha': 0.0027,
    'resolution': 1000,
    'start_c': 2,
    'step': 0.5,
    'end_c': 20,
}

def design_plot(input_dict):
    k0, sigma0 = input_dict['k0'], input_dict['sigma0']
    rho_values = input_dict['rhos']
    start_c, step, end_c = input_dict['start_c'], input_dict['step'], input_dict['end_c']
    n = input_dict['n']
    alpha = input_dict['alpha']
    resolution = input_dict['resolution']

    ats1_list = []
    ats2_list = []
    for rho in rho_values:
        h, t1, t2 = fl.grid_search(k0, sigma0, rho, n, alpha, resolution, start_c=start_c, step=step, end_c=end_c)
        ats1_list.append(t1)
        ats2_list.append(t2)

    fig, axs = plt.subplots(1, 4, figsize=(10, 2.5), constrained_layout=True)
    c = np.arange(start_c, end_c, step)
    cs = [7, 9, 6.5, 3]
    index = [10, 14, 9, 2]
    y_list = [[240, 280], [140, 180], [45, 65], [10, 30]]
    text_y = [-4, -6, -3, -3]
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
        ax.text(cs[j] - 0.5, ats_value - text_y[j], '(' + str(cs[j]) + ', ' + str(round(ats_value, 2)) + ')', color='crimson',
                fontsize=10)
        if j == 0:
            ax.set_ylabel(r'ATS$^{oc}$', fontsize=12)
    plt.show()

def case_study(k0, sigma0, rho, n, c, H):
    df = pd.read_csv("data/LCD_data.csv")
    data = df.to_numpy()
    ICsample = data[:, :70]
    OCsample = data[:, 70:]
    k1 = np.log(np.exp(k0 + sigma0 ** 2 / 2) * (1 - rho)) - sigma0 ** 2 / 2

    try:
        k_est = newton(fl.l_mu, np.ones(70) * k0, tol=1e-1, maxiter=100, args=(sigma0, ICsample, c))
    except RuntimeError:
        k_est = np.inf

    try:
        k1_est = newton(fl.l_mu, np.ones(30) * k1, tol=1e-1, maxiter=100, args=(sigma0, OCsample, c))
    except RuntimeError:
        k1_est = np.inf

    plotting_stats = np.exp(k_est + sigma0 ** 2 / 2)
    plotting_stats1 = np.exp(k1_est + sigma0 ** 2 / 2)
    x = np.arange(0, 70, 1)
    y = plotting_stats
    x1 = np.arange(70, 100, 1)
    y1 = plotting_stats1

    threshold = 1500
    IC_y = np.where(y > threshold, threshold, y)
    OC_y = np.where(y1 > threshold, threshold, y1)


    fig, ax = plt.subplots(figsize=(8,6))
    ellipse = Ellipse(xy=(79, OC_y[9]), width=6, height=50, edgecolor='crimson', facecolor='none', linewidth=0.8)
    ax.add_patch(ellipse)

    plt.plot(x, IC_y, linestyle='-', marker='o', markersize=5, markerfacecolor='none', markeredgecolor='blue',
             color='lightgray')
    plt.plot(x1, OC_y, linestyle='-', marker='o', markersize=5, markerfacecolor='none', markeredgecolor='blue',
             color='lightgray')

    plt.axhline(y=H, color='crimson', linestyle=(0, (5, 5)), linewidth=1.5, alpha=0.5)
    plt.axvline(x=70, color='black', linestyle='--', linewidth=1, alpha=0.5)
    plt.text(10, H - 50, r'$\rm{LCL}$', color='crimson', fontsize=10)

    yticks = list(range(0, 1500, 500))
    yticks = [tick for tick in yticks if tick < threshold] + [threshold]
    ytick_labels = [str(tick) for tick in yticks[:-1]] + ['$\geq 1500$']
    plt.yticks(yticks, labels=ytick_labels)
    plt.ylim(0, 1550)
    plt.xlabel('Sample Number')
    plt.ylabel('Sample Statistics')
    plt.show()

if __name__ == "__main__":
    design_plot(input_dict)
    # rho = 0.75 is used in the case study, with lower control limit 111.293 and censoring time 3
    case_study(input_dict['k0'], input_dict['sigma0'], 0.5, input_dict['n'], 3, 111.293)