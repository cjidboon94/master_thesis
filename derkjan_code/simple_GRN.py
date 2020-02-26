import numpy as np
import max_nudges
import matplotlib.pyplot as plt

def create_conditional_distribution(weights):
    """create a conditional distribution given a simple binary GRN model

    Parameters:
    ----------
    weights: nd-array,
        the weights of the links between the inputs and the outputs

    """
    conditional = np.zeros((weights.shape[0]+1)*[2])
    it = np.nditer(conditional, flags=['multi_index'])
    while not it.finished:
        weighted_sum = np.sum(weights*np.array(it.multi_index[:-1]))
        if weighted_sum <= 0 and it.multi_index[-1] == 0:
            conditional[it.multi_index] = 1
        elif weighted_sum > 0 and it.multi_index[-1] == 1:
            conditional[it.multi_index] = 1 
        else:
            conditional[it.multi_index] = 0 

        it.iternext()

    return conditional



if __name__ == "__main__":
    #weights = np.array([0.4, -0.3, 0.2])
    nudge_size = 0.01
    evolutionary_parameters = {
        "number_of_generations": 100,
        "population_size": 10,
        "number_of_children": 20,
        "generational": False,
        "mutation_size": nudge_size / 10,
        "parent_selection_mode": "rank_exponential",
        "start_mutation_size": nudge_size / 20,
        "change_mutation_size": nudge_size / (25 * 10),
        "mutations_per_update_step": 30
    }

    input_to_impacts = {}
    for number_of_inputs in range(2, 8):
        #print("weights {}".format(weights))
        impacts = []
        for _ in range(10):
            weights = np.array([np.random.uniform(-1, 1) for _ in range(number_of_inputs)])
            joint_input_distribution = np.reshape(
                np.random.dirichlet(2**number_of_inputs * [1]), number_of_inputs*[2]
            )
            conditional = create_conditional_distribution(weights)
            #print(joint_input_distribution)
            #print(conditional)
            impacts.append(max_nudges.max_nudge_impact(joint_input_distribution, conditional, nudge_size, "synergistic", evolutionary_parameters))

        print(np.mean(impacts))
        input_to_impacts[number_of_inputs] = np.mean(impacts)

    print()
    plt.plot(input_to_impacts.keys(), input_to_impacts.values())
    plt.ylim((0, plt.ylim()[1]))
    plt.show()