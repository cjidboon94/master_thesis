#List of configurations for the optimal derkjanistic nudge
from typing import Dict

def get_config( n_vars:int) -> Dict:
    if n_vars == 2:
        n_vars = 3
    elif n_vars > 7:
        n_vars = 7
    config = {
        3 : {
            "mutations_per_update_step": 1,
            "population_size": 80,
            "generational": False,
            "parent_selection_mode": "rank_exponential",
            "number_of_children": 100,
            "number_of_generations": 50,
            "start_mutation_size": 0.005,
            "change_mutation_size": 0.001},
        4 : {
            "mutations_per_update_step":1,
            "population_size": 50,
            "generational" : False,
            "parent_selection_mode": None,
            "number_of_children": 100,
            "number_of_generations": 600,
            "start_mutation_size": 0.005,
            "change_mutation_size": 0.001
        },
        5 : {
            "mutations_per_update_step": 1,
            "population_size": 60,
            "generational": False,
            "parent_selection_mode": "rank_exponential",
            "number_of_children": 140,
            "number_of_generations": 400,
            "start_mutation_size": 0.03,
            "change_mutation_size": 0.003
        },
        6 : {
            "mutations_per_update_step": 1,
            "population_size": 40,
            "generational": False,
            "parent_selection_mode": None,
            "number_of_children": 130,
            "number_of_generations": 700,
            "start_mutation_size": 0.08,
            "change_mutation_size": 0.005
        },
        7 : {
            "mutations_per_update_step": 10,
            "population_size": 30,
            "generational": False,
            "parent_selection_mode": "rank_exponential",
            "number_of_children": 160,
            "number_of_generations": 90,
            "start_mutation_size": 0.08,
            "change_mutation_size": 0.006
        }
    }
    return config[n_vars]