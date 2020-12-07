from experiments import experiment
import numpy as np
import multiprocessing as mp
import itertools
import os

#settings
n_states = 3
n_vars = range(1,7) #3
levels = [0.75, 0.8, 0.85, 0.9]#0.75
dists = 1
interventions = 1
seeds = np.random.randint(0, 2**32-1, len(n_vars)*len(levels))

def callback(result):
    np.save("random_{}vars_{}dists_{}interventions_results.npy".format(len(n_vars),dists, interventions), result)

#results = optim_experiment((levels,n_vars, dists, interventions, seeds))
pool = mp.Pool(os.cpu_count())
pool.map_async(experiment, [(level, n_var, dists, interventions, seeds[i]) for i, (level, n_var) in enumerate(itertools.product(levels, n_vars))], callback=callback)
pool.close()
pool.join()

