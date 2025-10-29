"""
func_gamma.py
functions required for Analytical computation of control limit H, power, and ATS under Gamma lifetime distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.special import gammaincc, gamma as gamma_func
from scipy.special import comb
from scipy.integrate import quad
from scipy.optimize import newton
from scipy.misc import derivative
from tqdm import tqdm
from matplotlib.lines import Line2D

def g_func(alpha, beta, n, r, c):
    """Compute the sufficient statistic S̃_r = g(β | n, r, c) for the Type-I censored Gamma distribution,
    corresponding to eq.(15)"""
    upper_incomplete_gamma = gammaincc(alpha, c / beta) * gamma_func(alpha)
    correction = (c / beta) ** alpha * np.exp(-c / beta) / upper_incomplete_gamma
    S_tilde = r * alpha * beta - (n - r) * correction
    return S_tilde


def truncated_gamma_pdf(shape, scale, tau, resolution=1000):
    """Discretize truncated gamma PDF on [0, τ]."""
    x = np.linspace(0, tau, resolution)
    pdf = gamma.pdf(x, shape, scale=scale)
    pdf /= gamma.cdf(tau, shape, scale=scale)
    return x, pdf


def convolve_pdf(x, pdf, n):
    """Recursive convolution to compute the pdf of S̃_r, corresponding to eq.(12)"""
    result_pdf, result_x = pdf, x
    for _ in range(n - 1):
        result_pdf = np.convolve(result_pdf, pdf, mode='full')
        result_x = np.linspace(result_x[0] + x[0], result_x[-1] + x[-1], len(result_pdf))
        dx_new = np.diff(result_x)
        result_pdf /= np.sum(result_pdf[:-1] * dx_new)
    return result_x, result_pdf


def empirical_cdf(x, pdf):
    dx = x[1] - x[0]
    return np.cumsum(pdf) * dx


def pr_s_leq_t(sum_x, sum_cdf, t):
    """Linear interpolation for Pr(S̃_r ≤ t)"""
    if t <= sum_x[0]:
        return 0.0
    if t >= sum_x[-1]:
        return 1.0
    idx = np.searchsorted(sum_x, t) - 1
    x_i, x_next = sum_x[idx], sum_x[idx + 1]
    cdf_i, cdf_next = sum_cdf[idx], sum_cdf[idx + 1]
    return cdf_i + (t - x_i) / (x_next - x_i) * (cdf_next - cdf_i)


def remove_inf_from_pdf(sum_x, sum_pdf):
    """Remove inf to ensure stability"""
    valid_mask = np.isfinite(sum_pdf)
    return sum_x[valid_mask], sum_pdf[valid_mask]


def rvs_cdf(alpha0, beta0, c, n, resolution = 1000):
    """Compute CDFs of S̃_r for varying k ≤ n, corresponding to eq.(13)"""
    xs, cdfs = [], []
    x, pdf = truncated_gamma_pdf(alpha0, beta0, c, resolution)
    x, pdf = remove_inf_from_pdf(x, pdf)
    for k in range(1, n + 1):
        sum_x, sum_pdf = convolve_pdf(x, pdf, k)
        sum_cdf = empirical_cdf(sum_x, sum_pdf)
        xs.append(sum_x)
        cdfs.append(sum_cdf)
    return xs, cdfs


def compute_alpha(t, n, alpha0, beta0, c, xs, cdfs):
    """compute empirical Type-I error, corresponding to eq.(1) """
    total_sum = 0.0
    beta_hat = t/alpha0
    p = gamma.cdf(c, a=alpha0, scale=beta0)
    for k in range(1, n + 1):
        t_new = g_func(alpha0, beta_hat, n, k, c)
        pr1 = pr_s_leq_t(xs[k - 1], cdfs[k - 1], t_new)
        pr2 = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
        total_sum += pr1 * pr2
    return total_sum


def compute_H(n, alpha0, beta0, c, alpha, resolution):
    """compute LCL H, corresponding to eq.(2)"""
    # initial guess of H
    t_init = gamma.ppf(0.0027, n * alpha0, scale=beta0 / n)
    xs, cdfs = rvs_cdf(alpha0, beta0, c, n, resolution)
    # solve the quantile iteratively
    obs_alpha = compute_alpha(t_init, n, alpha0, beta0, c, xs, cdfs)
    pf = obs_alpha - alpha
    if obs_alpha >= alpha:
        t_init = t_init - 0.001
    else:
        t_init = t_init + 0.001
    x = 0
    while abs(pf) >= 1e-6:
        pl = pf
        if obs_alpha >= alpha:
            x = -0.001
        else:
            x = 0.001
        t_init = t_init + x
        obs_alpha = compute_alpha(t_init, n, alpha0, beta0, c, xs, cdfs)
        pf = obs_alpha - alpha
        if pf * pl < 0:
            break
    H = t_init - x
    return H


def compute_power(H, n, alpha0, beta1, c, resolution):
    """compute empirical test power (Type-II error) = Pr(mu < H|beta = beta_1)"""
    total_sum = 0.0
    beta_hat = H / alpha0
    xs, cdfs = rvs_cdf(alpha0, beta1, c, n, resolution)
    p = gamma.cdf(c, a=alpha0, scale=beta1)
    for k in range(1, n + 1):
        t_new = g_func(alpha0, beta_hat, n, k, c)
        pr1 = pr_s_leq_t(xs[k - 1], cdfs[k - 1], t_new)
        pr2 = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
        total_sum += pr1 * pr2
    return total_sum


def rvs_cdf_t(alpha0, beta1, t, n, H, resolution=1000):
    """calculate cdf of S̃_r (T is censored by t)"""
    x, pdf = truncated_gamma_pdf(alpha0, beta1, t, resolution)
    x, pdf = remove_inf_from_pdf(x, pdf)
    sum_x, sum_pdf = convolve_pdf(x, pdf, n)
    sum_cdf = empirical_cdf(sum_x, sum_pdf)
    p = pr_s_leq_t(sum_x, sum_cdf, n*H)
    return p


def calculate_ats1(power):
    """compute ATS^oc_1, corresponding to eq.(3)"""
    return 1 / power - 0.5


def calculate_ats2(alpha0, beta1, H, n, c, power, resolution):
    """compute ATS^oc_1, corresponding to eq.(4)"""
    if c <= H:
        integral, _ = quad(lambda x: gamma.cdf(x, alpha0, scale=beta1) ** n, 0, c)
        atsoc2 = c - integral / power
    elif c > H:
        integral1, _ = quad(lambda x: gamma.cdf(x, alpha0, scale=beta1) ** n, 0, H)
        integral2, _ = quad(lambda x: gamma.cdf(x, alpha0, scale=beta1) ** n * rvs_cdf_t(alpha0, beta1, x, n, H, resolution), H, c)
        atsoc2 = c - (integral1 + integral2) / power
    return atsoc2


def comp_gamma(alpha0, beta0, rho, c, n, alpha, resolution):
    """Framework to compute the lower control limit H, overall detection delay ATS^oc_1 + ATS^oc_2, allow user to get
    the detection delay at given scenario; Suitable for check the censoring design already made.


    input:
    alpha0: the shape parameter of gamma distribution, assume to be known and fixed in both IC and OC stage;
    beta0: IC value of the scale parameter of gamma distribution;
    rho: variation factor, control the OC value of the scale parameter beta1 = (1-rho) * beta0;
    c: censoring time;
    n: test sample size;
    alpha: ideal Type-I error, affect lower control limit H;
    resolution: the number of point when discrete convolve truncated pdf;


    output:
    H: lower control limit;
    atsoc1: value of ATS^oc_1
    atsoc2: value of ATS^oc_2
    """
    H = compute_H(n, alpha0, beta0, c, alpha, resolution)
    beta1 = beta0 * (1 - rho)
    power = compute_power(H, n, alpha0, beta1, c, resolution)
    atsoc1 = calculate_ats1(power)
    atsoc2 = calculate_ats2(alpha0, beta1, H, n, c, power, resolution)
    return H, atsoc1, atsoc2


def grid_search(alpha0, beta0, rho, n, alpha, resolution, start_c=0.5, step=0.5, end_c=1):
    """Grid search framework, compute the set of lower control limit H, overall detection delay ATS^oc_1 + ATS^oc_2
    under different censoring time c; suitable for optimal censoring design.


    input:
    alpha0: the shape parameter of gamma distribution, assume to be known and fixed in both IC and OC stage;
    beta0: IC value of the scale parameter of gamma distribution;
    rho: variation factor, control the OC value of the scale parameter beta1 = (1-rho) * beta0;
    n: test sample size;
    alpha: ideal Type-I error, affect lower control limit H;
    resolution: the number of point when discrete convolve truncated pdf;
    start_c: start value for grid search;
    step: step for grid search;
    end_c: end value for grid search


    output:
    H_set: set of lower control limit under different c;
    atsoc1_set: set of ATS^oc_1 under different c;
    atsoc2_set : set of ATS^oc_2 under different c;
    """
    c_values = np.arange(start_c, end_c, step)
    Hset = np.zeros(len(c_values))
    ATS1set = np.zeros(len(c_values))
    ATS2set = np.zeros(len(c_values))

    for c_idx, c_val in enumerate(tqdm(c_values, desc="Processing c_values")):
        H, ATS1, ATS2 = comp_gamma(alpha0, beta0, rho, c_val, n, alpha, resolution)
        Hset[c_idx] = H
        ATS1set[c_idx] = ATS1
        ATS2set[c_idx] = ATS2
    return Hset, ATS1set, ATS2set


def plot(atsoc1set, atsoc2set, start_c, step, end_c):
    """Visualization of grid search results for ATS^oc components."""
    atsocset = atsoc1set + atsoc2set
    c_values = np.arange(start_c, end_c, step)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(c_values, atsoc1set, '--', color='crimson', linewidth=2, label=r'$\rm{ATS^{oc}_{1}}$')
    ax.plot(c_values, atsoc2set, ':', color='green', linewidth=2, label=r'$\rm{ATS^{oc}_{2}}$')
    ax.plot(c_values, atsocset, '-', color='navy', linewidth=2, label=r'$\rm{ATS^{oc}}$')

    legend_elements = [
        Line2D([0, 1], [0, 1], linestyle='--', color='crimson', label=r'$\rm{ATS^{oc}_{1}}$', linewidth=2),
        Line2D([0, 1], [0, 1], linestyle=':', color='green', label=r'$\rm{ATS^{oc}_{2}}$', linewidth=2),
        Line2D([0, 1], [0, 1], linestyle='-', color='navy', label=r'$\rm{ATS^{oc}}$', linewidth=2)
    ]

    fig.legend(
        loc='upper center', handles=legend_elements, handlelength=2, ncol=3,
        bbox_to_anchor=(0.5, 0.05), fontsize=12
    )

    plt.tight_layout()
    plt.show()

def optimal_design(H_set, ATS1_set, ATS2_set, start_c=0.5, step=0.5, end_c=1):
    """
    Determine the optimal censoring time c* that minimizes the total detection delay
    ATS^oc_total = ATS^oc_1 + ATS^oc_2"""
    c_values = np.arange(start_c, end_c, step)
    ATS_total = ATS1_set + ATS2_set
    idx_opt = np.argmin(ATS_total)
    c_opt = c_values[idx_opt]
    H_opt = H_set[idx_opt]
    ats_opt = ATS_total[idx_opt]
    return c_opt, H_opt, ats_opt


def l_beta(beta, alpha, sample, c_val):
    """parameter estimation in Monte Carlo Simulation"""
    term1 = np.sum(np.where(sample == c_val, 0, sample), axis=0)
    term2 = np.sum(sample < c_val, axis=0) * alpha * beta
    d_lnsf = derivative(lambda beta: np.log(gamma.sf(c_val, alpha, scale=beta)), beta, dx=1e-6)
    term3 = np.sum(np.where(sample < c_val, 0, d_lnsf * (beta ** 2)), axis=0)
    return term1 - term2 + term3


def MC_gamma(alpha0, beta0, rho, n, c, alpha):
    """Monte Carlo simulation for Gamma case"""
    np.random.seed(0)
    ICSimNum = 10 ** 6 # IC Monte Carlo sample size
    OCSimNum = 10 ** 4 # OC Monte Carlo sample size

    # Generate IC samples
    ICSample = gamma.rvs(alpha0, scale=beta0, size=(n, ICSimNum))
    ICSample = np.where(ICSample >= c, c, ICSample)
    beta_est = newton(l_beta, np.ones(ICSimNum) * beta0, args=(alpha0, ICSample, c), tol=1e-2, maxiter=50)
    np.nan_to_num(beta_est, nan=np.inf, copy=False)
    x = beta_est * alpha0
    H = np.quantile(x, alpha)

    # OC parameters
    beta1 = beta0 * (1 - rho)

    # Monte Carlo simulation for OC ATS
    atsoc1_set = np.zeros(OCSimNum)
    atsoc2_set = np.zeros(OCSimNum)
    for loop in range(OCSimNum):
        Y = H
        counter = 0
        while Y >= H:
            OCSample = gamma.rvs(alpha0, scale=beta1, size=n)
            OCSample = np.where(OCSample >= c, c, OCSample)
            try:
                beta_est = newton(l_beta, beta1, tol=1e-2, maxiter=50, args=(alpha0, OCSample, c))
            except RuntimeError:
                beta_est = np.inf
            Y = beta_est * alpha0
            counter += 1
            max_val = np.max(OCSample)
        atsoc1_set[loop] = counter
        atsoc2_set[loop] = max_val

    # Calculate OC ATS metrics
    atsoc1 = np.mean(atsoc1_set) - 0.5
    atsoc2 = np.mean(atsoc2_set)
    return H, atsoc1, atsoc2
