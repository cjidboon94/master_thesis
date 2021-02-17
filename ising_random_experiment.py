from experiments import real_experiment
import numpy as np
import multiprocessing as mp
import itertools
import os
import sys, resource

#resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
#sys.setrecursionlimit(10**6)



#settings

n_vars = range(1,10)
temperatures = np.linspace(0, 3, 21)
dists = 30
interventions = 20
seeds = np.random.randint(0, 2**32-1, len(n_vars)*len(temperatures))


#n_vars, model, parameter, dists, interventions, seed, function signature

def callback(result):
    np.save("ising_random_{}vars_{}dists_{}interventions_results.npy".format(len(n_vars),dists, interventions), result)

#results = optim_experiment((levels,n_vars, dists, interventions, seeds))
print("Available CPUs:{}".format( os.cpu_count()))
with mp.Pool(os.cpu_count()) as pool:
	results = pool.map(real_experiment, [(n_var, "ising", temperature, dists, interventions, seeds[i]) for i, (temperature, n_var) in enumerate(itertools.product(temperatures, n_vars))]) #callback=callback)
callback(results)


