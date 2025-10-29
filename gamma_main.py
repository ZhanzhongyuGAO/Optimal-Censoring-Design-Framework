"""
gamma_main.py
Main task selector and input configuration.

Tasks:
--------
1) 'comp_gamma' :
    Compute the control limit H and the two out-of-control delay components
    (ATS^oc_1 and ATS^oc_2) for a specified censoring time c.
    → Corresponds to individual computations reported in the main text.

    Required parameters:
        alpha0, beta0, rho, c, n, resolution

2) 'opt_design' :
    Perform grid search to determine the optimal censoring time that minimizes
    the overall out-of-control delay (ATS^oc_1 + ATS^oc_2).
    → Used to generate subplots in Figures 3–5 of the paper.

    Required parameters:
        alpha0, beta0, rho, n, resolution, start_c, step, end_c

3) 'comparison' (May be slow because MC need long time) :
    Compare different computational frameworks in terms of numerical accuracy,
    execution time, and peak memory usage.
    → Used to produce Table 1–2 in the main text and Tables B1–B3 in Appendix B.

    Required parameters:
        same as 'comp_gamma'
"""

import func_gamma as fg
import time
from memory_profiler import memory_usage
import warnings
warnings.filterwarnings("ignore")

# user specified parameters
input_dict = {
    'task': 'comp_gamma',    # 'comp_gamma', 'opt_design', 'comparison'
    'alpha0': 2,             # alpha0=2, 4 are used in our paper
    'beta0': 2.5,            # beta0 = 2.5, 5 are sued in our paper
    'rho': 0.75,              # rho=0.1, 0.25, 0.5, 0.75 are used in our paper
    'c': 5,                  # censoring value
    'n': 5,                  # n=3, 5, 10 are used for numerical study, n=10, 20, 50, 100 are used for scalability analysis
    'start_c': 0.5,          # start value for gird search
    'step': 0.5,             # step for gird search
    'end_c': 10,             # end value for grid search
}


def main(input_dict):
    # Parameters
    task = input_dict['task']
    alpha0, beta0 = input_dict['alpha0'], input_dict['beta0']
    rho, c, n = input_dict['rho'], input_dict['c'], input_dict['n']
    start_c, step, end_c = input_dict['start_c'], input_dict['step'], input_dict['end_c']
    if n >= 10:
        resolution = int(10000/n)
    else:
        resolution = 1000
    alpha = 0.0027

    # task 1
    if task == 'comp_gamma':
        H, atsoc1, atsoc2 = fg.comp_gamma(alpha0, beta0, rho, c, n, alpha, resolution)
        print("Lower control Limit H:", H)
        print("ATSoc1:", atsoc1)
        print("ATSoc2:", atsoc2)

    # task 2
    if task == 'opt_design':
        Hset, atsoc1set, atsoc2set = fg.grid_search(alpha0, beta0, rho, n, alpha, resolution,
                                                    start_c=start_c, step=step, end_c=end_c)
        fg.plot(atsoc1set, atsoc2set, start_c, step, end_c)
        c_opt, H_opt, ats_opt = fg.optimal_design(Hset, atsoc1set, atsoc2set,
                                                  start_c, step, end_c)
        print("Optimal censoring time:", c_opt)
        print("Optimal lower control limit:", H_opt)
        print("Optimal overall detection delay:", ats_opt)

    # task 3
    if task == 'comparison':
        # Our framework
        start_time = time.time()
        H, atsoc1, atsoc2 = fg.comp_gamma(alpha0, beta0, rho, c, n, alpha, resolution)
        end_time = time.time()

        # MC simulation
        start_time_MC = time.time()
        H_MC, atsoc1_MC, atsoc2_MC = fg.MC_gamma(alpha0, beta0, rho, n, c, alpha)
        end_time_MC = time.time()

        # Peak memory usage, the extra time for account peak memory usage will not be accounted
        def run_comp_gamma():
            return fg.comp_gamma(alpha0, beta0, rho, c, n, alpha, resolution)

        peak_memory = memory_usage((run_comp_gamma,), max_usage=True, retval=False)

        def run_MC_gamma():
            return fg.MC_gamma(alpha0, beta0, rho, n, c, alpha)

        peak_memory_MC = memory_usage((run_MC_gamma,), max_usage=True, retval=False)

        print('Result of the proposed framework:')
        print("Lower control Limit H:", H)
        print("ATSoc1:", atsoc1)
        print("ATSoc2:", atsoc2)
        print(f"Execution time: {end_time - start_time:.6f} seconds")
        print(f"Peak memory usage: {peak_memory:.2f} MB\n")

        print('Result of the MC simulation:')
        print("Lower control Limit H:", H_MC)
        print("ATSoc1:", atsoc1_MC)
        print("ATSoc2:", atsoc2_MC)
        print(f"Execution time: {end_time_MC - start_time_MC:.6f} seconds")
        print(f"Peak memory usage: {peak_memory_MC:.2f} MB\n")

if __name__ == "__main__":
    main(input_dict)