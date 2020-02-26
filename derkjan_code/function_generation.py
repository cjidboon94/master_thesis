import random
import itertools
import numpy as np
from scipy.stats import entropy
import probability_distributions
import nudge_non_causal as nudge
import evolutionary_algorithms as ea

TEST = True

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
        number_of_children: integer, 
            if generational larger than or equal to population size  
        generational: Boolean, whether to replace the old generation 
        mutation_size: positive float
        change_mutation_size: positive float
        parent_selection_mode: "rank_exponential" or None (for random selection)
       
    """
    if input_dists is None:
        number_of_input_states = reduce(lambda x,y: x*y, input_shape)
        input_dists = [np.random.dirichlet([1]*number_of_input_states)
                       for _ in range(number_of_input_distributions)]
        input_dists = [np.reshape(dist, input_shape) for dist in input_dists]

    #create initial population
    conditional_outputs = create_conditonal_distributions(
        evolutionary_parameters["population_size"], number_of_output_states,
        len(input_shape)
    )
    mutation_size = evolutionary_parameters["mutation_size"]
    change_mutation_size = evolutionary_parameters["change_mutation_size"]
    conditional_outputs = [
        ConditionalOutput(cond_output, mutation_size, change_mutation_size)
        for cond_output in conditional_outputs
    ]
    #for dist in conditional_outputs:
    #    found_sum = np.sum(dist.cond_output)
    #    expected_sum = reduce(lambda x,y: x*y, dist.cond_output.shape[:-1])
    #    if abs(found_sum-expected_sum) > 10**(-7):
    #        raise ValueError()

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
        input_dists
    )

    final_distance = find_conditional_output.get_best_individual()
    print("initial distance {}, distance after evolution {}".format(
        initial_distance, final_distance.score
    )) 
    return find_conditional_output.individuals[0]

class ConditionalOutput():
    """
    Attributes:
    ----------
    cond_output: nd-array, 
        a conditional probability distribution (last axis are the conditional
        probabilities)
    mutation_size: a positive number
    change_mutation_size: a postive number
    score: a number

    """
    def __init__(self, cond_output, start_mutation_size, change_mutation_size):
        """create an individual for an individual algorithms

        Parameters:
        ----------
        cond_output: nd-array
            Representing a conditional output diistribtution (with the the
            output on the last axis)
        

        """
        self.cond_output = cond_output
        self.mutation_size = start_mutation_size
        self.change_mutation_size = change_mutation_size
        self.score = None

    def mutate(self, timestep):
        """
        Mutate the probability distribution

        """
        #first update the mutation size
        #self.mutation_size += max(
        #    np.random.uniform(-self.change_mutation_size, self.change_mutation_size), 0
        #)
        old_mutation_size = self.mutation_size 
        proposed_change = np.random.uniform(-self.change_mutation_size, self.change_mutation_size)
        self.mutation_size = max(0, self.mutation_size + proposed_change)
        #print("{} {}".format(self.change_mutation_size, proposed_change))
        #print("timestep {} old mutation size {} mutation_size {}".format(timestep, old_mutation_size, self.mutation_size))

        input_shape = self.cond_output.shape[:-1]
        number_of_output_states = self.cond_output.shape[-1]
        lists_of_possible_states_per_variable = [
            range(states) for states in input_shape
        ]
        for state in itertools.product(*lists_of_possible_states_per_variable):
            mutation = nudge.find_noise_vector(number_of_output_states, 
                                               self.mutation_size)
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
                input_dist = np.random.dirichlet(
                    reduce(lambda x,y: x*y, input_shape)*[1]
                )
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

def create_conditonal_distributions(
            number_of_conditional_outputs, number_of_states, 
            number_of_input_variables
            ):
    """
    Create conditional output arrays 

    Parameters:
    ----------
    number_of_conditional_outputs: an integer
    number_of_states: an integer
    number_of_input_variables: an integer

    Returns:
    -------
    a list of nd-arrays representing conditonal outputs 

    """
    conditional_outputs = []
    for _ in range(number_of_conditional_outputs):
        cond_shape = [number_of_states]*(number_of_input_variables+1)
        cond_output = [
            probability_distributions.compute_joint_uniform_random((number_of_states,))
            for i in range(number_of_states**(number_of_input_variables))
        ]
        cond_output = np.reshape(np.array(cond_output), cond_shape)
        conditional_outputs.append(cond_output)

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

    def evolve(self, generational, input_dists):
        """
        Evolve the population
        
        Parameters:
        ----------
        generational: Boolean
            Whether to discard the old individuals at every new timestep
        number_of_input_distributions: an integer

        """
        for timestep in range(self.number_of_generations):
            if TEST:
                print("timestep {}, worst {}, best {}".format(
                    timestep, self.individuals[-1].score, self.individuals[0].score
                ))
            parents = ea.select_parents(
                self.individuals, self.number_of_children*2,
                self.parent_selection_mode
            )
            children = self.create_children(parents, self.number_of_children, timestep)
            for child in children:
                child.evaluate(self.goal_distance, input_dists)

            self.individuals = ea.select_new_population(
                self.individuals, children, len(self.individuals), 
                generational
            )

    def create_children(self, parents, number_of_children, timestep):
        """
        Create new individuals.

        Parameters:
        ----------
        parents: a list of individual objects
        number_of_children: an integer

        """
        children = []
        for i in range(number_of_children):
            children.append(self.recombine(
                parents[i], parents[number_of_children+i]
            ))
        for child in children:
            child.mutate(timestep)

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
        gene_choice = np.random.choice([1,2])
        mutation_size_choice = np.random.choice([1,2])
        if gene_choice == 1:
            genes = np.copy(parent1.cond_output)
        else:
            genes = np.copy(parent2.cond_output)

        if mutation_size_choice == 1:
            mutation_size = parent1.mutation_size
            change_mutation_size = parent1.change_mutation_size
            #change_mutation_size = parent2.mutation_size
        else:
            mutation_size = parent2.mutation_size
            change_mutation_size = parent2.change_mutation_size
            #change_mutation_size = parent2.mutation_size
        
        return ConditionalOutput(genes, mutation_size, change_mutation_size)

if __name__ == "__main__":
    pass
