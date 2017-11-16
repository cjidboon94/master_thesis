import random
import itertools
import numpy as np
from scipy.stats import entropy
import probability_distributions
import nudge_non_causal as nudge
import evolutionary_algorithms as ea

def get_cond_output_with_max_distance(
        input_shape, number_of_output_states, goal_distance, 
        evolutionary_parameters, input_dists, 
        number_of_input_distributions=None):
    """ 
    Create a conditional output distribution which gives as different 
    as possible marginal for the different input distributions

    Parameters:
    ----------
    input_shape: a list, the number of states of the inut variables
    number_of_states_output: integer
    input_distributions: a list of nd-arrays or None
        Every array representing a probability distribution
    number_of_input_distributions: integer
        The number of input distributions to generate using a Dirichlet
        distribution with all parameters equal to 1.
    evolutionary_parameters: dict with the keys
        number_of_generations: integer 
        population_size: integer
        number_of_children: integer, if generational larger than or equal to population size  
        generational: Boolean, whether to replace the old generation 
        mutation_size: positive float
        parent_selection_mode: "rank_exponential" or None (for random selection)
       
    """
    if input_dists is None:
        number_of_input_states = reduce(lambda x,y: x*y, input_shape)
        input_dists = [np.random.dirichlet([1]*number_of_input_states)
                       for _ in range(number_of_input_distributions)]
        input_dists = [np.reshape(dist, input_shape) for dist in input_dists]

    #create initial population
    conditional_outputs = create_condional_distributions(
        evolutionary_parameters["population_size"], number_of_output_states,
        len(input_shape)
    )
    for conditional_output in conditional_outputs:
        conditional_output.evaluate(goal_distance, input_dists)
    
    initial_distance = ea.sort_individuals(conditional_outputs)[-1].score

    #evolve the population
    find_conditional_output = FindConditionalOutput(
        conditional_outputs, goal_distance, 
        evolutionary_parameters["number_of_generations"], 
        evolutionary_parameters["number_of_children"], 
        evolutionary_parameters["parent_selection_mode"]
    )
    find_conditional_output.evolve(
        evolutionary_parameters["generational"], 
        evolutionary_parameters["mutation_size"], 
        input_dists
    )

    final_distance = find_conditional_output.get_best_individual()
    print("initial distance {}, distance after evolution {}".format(
        initial_distance, final_distance
    )) 
    return find_conditional_output.individuals[0]

class ConditionalOutput():
    """
    Attributes:
    ----------
    cond_output: nd-array, 
        a conditional probability distribution (last axis are the conditional
        probabilities)
    score: a number

    """
    def __init__(self, cond_output):
        """create an individual for an individual algorithms

        Parameters:
        ----------
        cond_output: nd-array
            Representing a conditional output diistribtution (with the the
            output on the last axis)

        """
        self.cond_output = cond_output
        self.score = None

    def mutate(self, mutation_size):
        """
        Mutate the probability distribution

        Parameters:
        ----------
        mutation_size: a positive float

        """
        input_shape = self.cond_output.shape[:-1]
        number_of_output_states = self.cond_output.shape[-1]
        lists_of_possible_states_per_variable = [
            range(states) for states in input_shape
        ]
        for state in itertools.product(*lists_of_possible_states_per_variable):
            mutation = nudge.find_noise_vector(number_of_output_states, 
                                               mutation_size)
            self.cond_output[state] = nudge.nudge_states(
                mutation, self.cond_output[state]
            )

            if abs(np.sum(self.cond_output[state])-1) > 10**-7:
                raise ValueError()

    def evaluate(self, goal_distance, input_dists=None, 
                 number_of_input_dists=None):
        """

        Parameters:
        ----------
        goal_distance:
        input_dists:
        number_of_input_dists:

        """
        input_shape = list(self.cond_output.shape[:-1])
        if input_dists is None:
            input_dists = []
            for i in range(number_of_input_dists):
                input_dist = np.random.dirichlet(reduce(lambda x,y: x*y, input_shape)*[1])
                input_dists.append(np.reshape(input_dist, input_shape))

        number_of_input_vars = len(input_dists[0].shape)
        outputs = []
        for input_dist in input_dists:
            joint = probability_distributions.compute_joint(
                input_dist, self.cond_output, set(range(0, number_of_input_vars, 1))
            )
            output = probability_distributions.ProbabilityArray(joint).marginalize(
                set([number_of_input_vars])
            )
            outputs.append(output)

        average_dist = np.mean(np.array(outputs), axis=0)
        distance = np.mean(
            [np.sum(np.absolute(output-average_dist)) for output in outputs]
        )
        self.score = abs(distance-goal_distance)

def create_condional_distributions(
            number_of_conditional_outputs, number_of_states, 
            number_of_input_variables
            ):
    """
    Create a list of ContionalOutput objects

    Parameters:
    ----------
    number_of_conditional_outputs: an integer
    number_of_states: an integer
    number_of_input_variables: an integer

    Returns:
    -------
    a list of ConditionalOutput objects

    """
    conditional_outputs = []
    for _ in range(number_of_conditional_outputs):
        cond_shape = [number_of_states]*(number_of_input_variables+1)
        cond_output = [
            probability_distributions.compute_joint_uniform_random((number_of_states,))
            for i in range(number_of_states**(number_of_input_variables))
        ]
        cond_output = np.reshape(np.array(cond_output), cond_shape)
        conditional_outputs.append(ConditionalOutput(cond_output))

    return conditional_outputs

class FindConditionalOutput():
    """ 
    A class that uses evolutionary algorithms to find a distribution 
    with a certain entropy

    Attributes:
    ----------
    individuals: list of IndividualDistribution objects
    goal_distance: a number
    number_of_generations: an integer
    number_of_children: an integer
    parent_selection_mode: Either "rank_exponential" or None

    """
    def __init__(self, individuals, goal_distance, number_of_generations,
                 number_of_children, parent_selection_mode):
        """Create a FindConditionalOutput object"""
        self.individuals = individuals
        self.goal_distance = goal_distance
        self.number_of_generations = number_of_generations
        self.number_of_children = number_of_children
        self.parent_selection_mode = parent_selection_mode

    def get_best_individual(self):
        return ea.sort_individuals(self.individuals)[0]

    def evolve(self, generational, mutation_size, input_dists):
        """
        Evolve the population
        
        Parameters:
        ----------
        mutation_size: a number
        generational: Boolean
            Whether to discard the old individuals at every new timestep
        number_of_input_distributions: an integer

        """
        for timestep in range(self.number_of_generations):
            print("timestep {}, worst {}, best {}".format(
                timestep, self.individuals[-1].score, self.individuals[0].score
            ))
            parents = ea.select_parents(
                self.individuals, self.number_of_children*2,
                self.parent_selection_mode
            )
            children = self.create_children(parents, self.number_of_children,
                                            timestep, mutation_size)
            for child in children:
                child.evaluate(self.goal_distance, input_dists)

            self.individuals = ea.select_new_population(
                self.individuals, children, len(self.individuals), 
                generational
            )

    def create_children(self, parents, number_of_children, timestamp,
                        mutation_size):
        """
        Create new individuals.

        Parameters:
        ----------
        parents: a list of individual objects
        number_of_children: an integer
        timestamp: a number
        mutation_size: a number

        """
        children = []
        for i in range(number_of_children):
            children.append(self.recombine(
                parents[i], parents[number_of_children+i]
            ))
        for child in children:
            child.mutate(mutation_size)

        return children

    def recombine(self, parent1, parent2):
        """recombine two individuals to create a new individual
        
        Parameters:
        ----------
        parent1: Individual object
        parent2: Individual object
        timestamp: a number

        Returns: Individual object

        """
        if np.random.random() > 0.5:
            genes = np.copy(parent1.cond_output)
        else:
            genes = np.copy(parent2.cond_output)
        
        return ConditionalOutput(genes)

if __name__ == "__main__":
    pass
