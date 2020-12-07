from experiments import optim_experiment
import numpy as np
import multiprocessing as mp
import itertools
import os

#settings
n_states = 3
n_vars = range(1,7) #3
levels = [0.75, 0.8, 0.85, 0.9]#0.75
dists = 20
interventions = 1
seeds = np.random.randint(0, 2**32-1, len(n_vars)*len(levels))

def callback(result):
    np.save("optimized_{}vars_{}dists_{}interventions_results.npy".format(len(n_vars),dists, interventions), result)

#results = optim_experiment((levels,n_vars, dists, interventions, seeds))
with mp.Pool(os.cpu_count()) as pool:
       optim_results = pool.map_async(optim_experiment, [(level, n_var, dists, interventions, seeds[i]) for i, (level, n_var) in enumerate(itertools.product(levels, n_vars))], callback=callback)


