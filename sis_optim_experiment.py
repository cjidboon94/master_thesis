from experiments import real_optim_experiment
import numpy as np
import multiprocessing as mp
import itertools
import os
import resource, sys
resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)

#settings

n_vars = range(1,10)
betas = np.linspace(0.1, 1, 19)
dists = range(5,10)
interventions = 20
seeds = np.random.randint(0, 2**32-1, len(n_vars)*len(betas))


#n_vars, model, parameter, dists, interventions, seed, function signature

def callback(result):
    np.save("ising_optim_{}vars_{}dists_{}interventions_results.npy".format(len(n_vars),dists, interventions), result)

#results = optim_experiment((levels,n_vars, dists, interventions, seeds))
print("Available CPUs:{}".format( os.cpu_count()))
with mp.Pool(os.cpu_count()) as pool:
	results = pool.map(real_optim_experiment, [(n_var, "sis", (beta, 0.5), dists, interventions, seeds[i]) for i, (beta, n_var) in enumerate(itertools.product(betas, n_vars))]) #callback=callback)
callback(results)
