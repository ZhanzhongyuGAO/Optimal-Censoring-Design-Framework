"""
lognormal_main.py
Main task selector and input configuration.

Tasks:
--------
1) 'comp_lognormal' :
    Compute the control limit H and the two out-of-control delay components
    (ATS^oc_1 and ATS^oc_2) for a specified censoring time c.
    → Corresponds to individual computations reported in the main text.

    Required parameters:
        k0, sigma0, rho, c, n, resolution

2) 'opt_design' :
    Perform grid search to determine the optimal censoring time that minimizes
    the overall out-of-control delay (ATS^oc_1 + ATS^oc_2).
    → Used to generate subplots in Figures 6–8 of the paper.

    Required parameters:
        k0, sigma0, rho, n, resolution, start_c, step, end_c

3) 'comparison' (May be slow because MC need long time) :
    Compare different computational frameworks in terms of numerical accuracy,
    execution time, and peak memory usage.
    → Used to produce Table 3-4 in the main text.

    Required parameters:
        same as 'comp_lognormal'
"""

import func_lognormal as fl
import time
from memory_profiler import memory_usage
import warnings
warnings.filterwarnings("ignore")

# user specified parameters
input_dict = {
    'task': 'opt_design',    # 'comp_lognormal', 'opt_design', 'comparison'
    'k0':  4.2592,                 # k0=2, 3 are used in our paper
    'sigma0': 2.4403,          # sigma0 = 0.75, 1 are sued in our paper
    'rho': 0.1,             # rho=0.1, 0.25, 0.5, 0.75 are used in our paper
    'c': 3,                  # censoring value
    'n': 10,                  # n=3, 5, 10 are used for numerical study, n=10, 20, 50, 100 are used for scalability analysis
    'start_c': 2,          # start value for gird search
    'step': 0.5,             # step for gird search
    'end_c': 20,             # end value for grid search
}

def main(input_dict):
    # Parameters
    task = input_dict['task']
    k0, sigma0 = input_dict['k0'], input_dict['sigma0']
    rho, c, n = input_dict['rho'], input_dict['c'], input_dict['n']
    start_c, step, end_c = input_dict['start_c'], input_dict['step'], input_dict['end_c']
    if n >= 10:
        resolution = int(10000/n)
    else:
        resolution = 1000
    alpha = 0.0027

    # task 1
    if task == 'comp_lognormal':
        H, atsoc1, atsoc2 = fl.comp_lognormal(k0, sigma0, rho, c, n, alpha, resolution)
        print("Lower control Limit H:", H)
        print("ATSoc1:", atsoc1)
        print("ATSoc2:", atsoc2)

    # task 2
    if task == 'opt_design':
        Hset, atsoc1set, atsoc2set = fl.grid_search(k0, sigma0, rho, n, alpha, resolution,
                                                    start_c=start_c, step=step, end_c=end_c)
        fl.plot(atsoc1set, atsoc2set, start_c, step, end_c)
        c_opt, H_opt, ats_opt = fl.optimal_design(Hset, atsoc1set, atsoc2set,
                                                  start_c, step, end_c)
        print("Optimal censoring time:", c_opt)
        print("Optimal lower control limit:", H_opt)
        print("Optimal overall detection delay:", ats_opt)

    # task 3
    if task == 'comparison':
        # Our framework
        start_time = time.time()
        H, atsoc1, atsoc2 = fl.comp_lognormal(k0, sigma0, rho, c, n, alpha, resolution)
        end_time = time.time()

        # MC simulation
        start_time_MC = time.time()
        H_MC, atsoc1_MC, atsoc2_MC = fl.MC_lognormal(k0, sigma0, rho, n, c, alpha)
        end_time_MC = time.time()

        # Peak memory usage, the extra time for account peak memory usage will not be accounted
        def run_comp_lognormal():
            return fl.comp_lognormal(k0, sigma0, rho, c, n, alpha, resolution)

        peak_memory = memory_usage((run_comp_lognormal,), max_usage=True, retval=False)

        def run_MC_lognormal():
            return fl.MC_lognormal(k0, sigma0, rho, n, c, alpha)

        peak_memory_MC = memory_usage((run_MC_lognormal,), max_usage=True, retval=False)

        print('Result of the proposed framework:')
        print("Lower control Limit H:", H)
        print("ATSOC:", atsoc1+atsoc2)
        print(f"Execution time: {end_time - start_time:.6f} seconds")
        print(f"Peak memory usage: {peak_memory:.2f} MB\n")

        print('Result of the MC simulation:')
        print("Lower control Limit H:", H_MC)
        print("ATSOC:", atsoc1_MC+atsoc2_MC)
        print(f"Execution time: {end_time_MC - start_time_MC:.6f} seconds")
        print(f"Peak memory usage: {peak_memory_MC:.2f} MB\n")

if __name__ == "__main__":
    main(input_dict)