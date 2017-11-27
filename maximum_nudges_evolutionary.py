import nudge_non_causal as nudge
import numpy as np
import evolutionary_algorithms as ea
import probability_distributions

import maximum_nudges

TEST = False
def find_maximum_local_nudge(input_dist, cond_output, nudge_size,
                             evolution_params, verbose=False):
    """
    find the synergistic nudge with the maximum impact
    
    Parameters:
    ----------
    input_dist: nd-array, representing the joint input distribution
    cond_output: nd-array, representing the output distribution conditioned on the input
    nudge_size: positive float
    evolutionary_parameters: dict with the following keys
        number_of_generations: integer
        population_size: integer
        number_of_children:
            integer, if generational larger than or equal to population size
        generational: Boolean, whether to replace the old generation 
        parent_selection_mode: "rank_exponential" or None (for random selection)
        start_mutation_size: positive float, the mutation size at the start
        change_mutation_size: positive float, 
            the maximum percentage the mutation size is changed
        mutation_size_weights: positive float
        change_mutation_size_weights: positive float
            The maximum percentage the mutation_size for the weights is
            changed.

    Returns: A LocalNudge object
    
    """
    #create initial population
    nudged_vars_to_states = {
        nudged_var:number_of_states
        for nudged_var, number_of_states in enumerate(input_dist.shape)
    }
    local_nudges = []
    for _ in range(evolution_params["population_size"]):
        new_local_nudge = LocalNudge.create_local_nudge(
            nudged_vars_to_states, nudge_size,
            evolution_params["mutation_size_weights"],
            evolution_params["start_mutation_size"],
            evolution_params["change_mutation_size"],
            evolution_params["change_mutation_size_weights"],
            timestamp=0
        )
        local_nudges.append(new_local_nudge)

    for local_nudge in local_nudges:
        local_nudge.evaluate(input_dist, cond_output)

    initial_impact = ea.sort_individuals(local_nudges)[0].score

    #evolve the population
    find_max_local_nudge = FindMaximumLocalNudge(
        input_dist, cond_output, nudge_size,
        evolution_params["generational"],
        evolution_params["number_of_children"],
        evolution_params["parent_selection_mode"]
    )
    for timestep in range(evolution_params["number_of_generations"]):
        local_nudges = find_max_local_nudge.evolve(local_nudges, timestep)
        best_local_nudge = ea.sort_individuals(local_nudges)[0]
        if verbose:
            print("best score {}, worst score {}".format(
                best_local_nudge.score, local_nudges[-1].score
            ))

    if TEST:
        print("local nudge: intial impact {}, max impact {}".format(
            initial_impact, best_local_nudge.score
        ))

    return ea.sort_individuals(local_nudges)[0]


class IndividualNudge(ea.Individual):
    """
    A class that represents one individual individual_nudge in a
    evolutionary algorithms. Note the difference between individuals
    in an evolutionary algorithm (part of a population etc), and 
    individual_nudges which are a certain type of nudges.

    Attributes:
    ----------
    genes: a 1d nd-array
        The length should be equal to the number of states of the nudged
        variable
    timestamp: integer
        At which generation the nudge was created.
    score: a number
        The impact of the nudge
    nudge_size: a number
        The l1-norm of the nudge

    """
    def __init__(self, genes, timestamp):
        """
        Create an individualNudge object representing an individual
        nudge.

        Parameters:
        ----------
        genes: a 1d nd-array
        timestamp: a number

        """
        self.genes = genes
        self.nudge_size = np.sum(np.absolute(genes))
        self.timestamp = timestamp
        self.score = None
        if abs(np.sum(np.absolute(self.genes))-self.nudge_size) > 10**(-7):
            raise ValueError()
        if abs(np.sum(self.genes)) > 10**(-10):
            raise ValueError()

    @classmethod
    def create_random_individual(cls, number_of_states, nudge_size,
                                 timestamp):
        """
        Create an IndividualNudge object randomly

        Parameters:
        ----------
        nudge_size: number
        number_of_states: integer
        timestamp: integer

        Returns: IndividualNudge object

        """
        genes = cls.create_genes(number_of_states, nudge_size)
        return cls(genes, timestamp)

    def mutate(self, mutation_size):
        """
        Mutate the genes of the individual.

        Parameters:
        ----------
        mutation_size: a number

        """
        mutation = nudge.find_noise_vector(len(self.genes), mutation_size)
        self.genes += mutation
        norm = np.sum(np.absolute(self.genes))
        self.genes = self.genes * (self.nudge_size/norm)
        if abs(np.sum(np.absolute(self.genes))-self.nudge_size) > 10**(-7):
            raise ValueError()
        if abs(np.sum(self.genes)) > 10**(-10):
            raise ValueError()
   
    def evaluate(self, start_distribution, cond_output):
        """
        Find the impact of the nudge on the output distribution and 
        set the score.

        Parameters:
        ----------
        start_distribution: an nd-array
            Representing a probability distribution
        cond_output: an nd-array
            The output conditioned on all input variables

        """
        new_input_dist = nudge.nudge_individual(
            start_distribution, self.nudge_size, self.genes
        )
        nudge_impact = nudge.find_nudge_impact(
            start_distribution, new_input_dist, cond_output,
            measure="absolute"
        )
        #to make it a minimization rather than a maximization problem
        self.score = -nudge_impact

    @staticmethod
    def create_genes(number_of_states, nudge_size):
        genes = nudge.find_noise_vector(number_of_states, nudge_size)
        if abs(np.sum(np.absolute(genes))-nudge_size) > 10**(-7):
            raise ValueError()
        return genes


class LocalNudge(ea.Individual):
    """
    An Individual object that represents a local nudge

    Attributes:
    ----------
    weights: a numpy array
        The weight given to every local nudge (should sum to 1).
    individual_nudges: a list of IndividualNudge objects
    nudged_vars: a list of ints
        Which variables are nudged.
    timestamp: an integer
    nudge_size: a number
    score: the impact of the nudge
    mutation_size: a number
    change_mutation_size: a number 
        The size of the change applied to the mutation_size
    mutation_size_weights: a number
        How much to mutate the weights
    change_mutation_size_weights: a number

    """
    def __init__(self, weights, individual_nudges, nudged_vars, 
                 start_mutation_size, change_mutation_size, 
                 mutation_size_weights, change_mutation_size_weights,
                 nudge_size, timestamp):
        """create LocalNudge object"""
        self.weights = weights
        self.individual_nudges = individual_nudges
        self.nudged_vars = nudged_vars
        self.nudge_size = nudge_size
        self.mutation_size_weights = mutation_size_weights
        self.change_mutation_size_weights = change_mutation_size_weights
        self.mutation_size = start_mutation_size
        self.change_mutation_size = change_mutation_size
        self.timestamp = timestamp
        self.score = None

    def copy_individual_nudges(self):
        """Provide a copy of the individual nudges"""
        new_individual_nudges = []
        for individual_nudge in self.individual_nudges:
            new_individual_nudge = IndividualNudge(
                np.copy(individual_nudge.genes), self.timestamp
            )
            new_individual_nudges.append(new_individual_nudge)

        return new_individual_nudges

    def evaluate(self, start_distribution, cond_output):
        """
        Find the impact of the nudge on the output distribution and 
        set the score. The score is multiplied by -1 to make it a 
        minimization problem.

        Parameters:
        ----------
        start_distribution: an nd-array
            Representing a probability distribution
        cond_output: an nd-array
            The output conditioned on all input variables

        """
        noise_vectors = [
            weight*individual_nudge.genes for weight, individual_nudge in
            zip(self.weights, self.individual_nudges)
        ]
        not_almost_zero_vectors = [
            np.sum(np.absolute(vector))>10**(-10) for vector in noise_vectors
        ]
        noise_vectors = [vector for count, vector in enumerate(noise_vectors)
                         if not_almost_zero_vectors[count]]
        nudged_vars = [var for count, var in enumerate(self.nudged_vars)
                       if not_almost_zero_vectors[count]]

        for noise in noise_vectors:
            if abs(np.sum(noise)) > 10**(-10):
                raise ValueError()

        new_input_dist = nudge.nudge_local(
            start_distribution, nudged_vars, self.nudge_size,
            noise_vectors, without_conditional=True
        )
        nudge_impact = nudge.find_nudge_impact(
            start_distribution, new_input_dist, cond_output,
            measure="absolute"
        )
        #to make it a minimization rather than a maximization problem
        self.score = -nudge_impact

    def mutate(self):
        """
        Mutate both the mutation size and the genes (the weights and the 
        individual nudges)
        
        """
        self.mutation_size *= (1 + np.random.uniform(
            -self.change_mutation_size, self.change_mutation_size
        ))
        self.mutation_size_weights *= (1 + np.random.uniform(
            -self.change_mutation_size_weights, self.change_mutation_size_weights
        ))

        #update the weights
        noise_vector = nudge.find_noise_vector(
            self.weights.shape[0], self.mutation_size_weights
        )
        self.weights = nudge.nudge_states(noise_vector, self.weights)

        #update the inidividual nudges
        for individual_nudge in self.individual_nudges:
            individual_nudge.mutate(self.mutation_size)

    @classmethod
    def create_local_nudge(cls, nudged_vars_to_states, nudge_size, 
                           mutation_size_weights, start_mutation_size, 
                           change_mutation_size, change_mutation_size_weights,
                           timestamp=0):
        """
        Create an IndividualNudge object randomly

        Parameters:
        ----------
        nudge_size: number
        number_of_states: integer
        timestamp: integer

        Returns: LocalNudge object

        """
        nudged_vars = list(nudged_vars_to_states.keys())
        individual_nudges = []
        for number_of_states in nudged_vars_to_states.values(): 
            individual_nudges.append(IndividualNudge.create_random_individual(
                number_of_states, nudge_size, timestamp
            ))
        weights = np.random.dirichlet([1]*len(nudged_vars))

        return cls(
            weights, individual_nudges, nudged_vars, start_mutation_size,
            change_mutation_size, mutation_size_weights, 
            change_mutation_size_weights,nudge_size, timestamp
        )


def create_individual_nudges(number_of_individuals, number_of_states, 
                             nudge_size, timestamp=0):
    """
    Create a list of individual objects. The genes are either generated 
    according to the mode.

    Parameters:
    ----------
    number_of_individuals: an integer
    number_of_genes: an integer
    nudge_size: the total size of the nudge
    timestamp: an integer

    Returns:
    -------
    a list of IndividualDistribution objects

    """
    individuals = []
    for i in range(number_of_individuals):
        individual = IndividualNudge.create_random_individual(
            number_of_states, nudge_size, timestamp
        )
        individuals.append(individual)

    return individuals


class FindMaximumIndividualNudge():
    """
    A class to find the maximal individual nudge

    Attributes:
    ----------
    start_distribution: nd-array
        Representing a probability distribution
    cond_output: nd-array
        Representing the output conditioned on the input
    nudge_size: a number
        The nudge_size on the input distribution.
    generational: Boolean
        Whether to discard the old individuals at every new timestep
    number_of_children: an integer
    parent_selection_mode: Either "rank_exponential" or None

    """
    def __init__(self, start_distribution, cond_output, nudge_size, 
                 generational, number_of_children, parent_selection_mode):
        """Create a FindMaximumIndividualNudge object"""
        self.start_distribution = start_distribution
        self.cond_output = cond_output
        self.nudge_size = nudge_size
        self.generational = generational
        self.number_of_children = number_of_children
        self.parent_selection_mode = parent_selection_mode

    def get_max_nudge(self, individuals, number_of_generations,
                      mutation_size):
        """
        Find the maximum individual nudge.

        Parameters:
        ----------
        number_of_generations: an integer
        mutation_size: a number
        individuals: list of IndividualNudge objects
 
        Returns:
        -------
        A numpy array which represents the maximum individual nudge
        for the start distribution.

        """
        pass

    def evolve(self, individuals, mutation_size, timestep):
        """
        Evolve the population
        
        Parameters:
        ----------
        mutation_size: a number

        Returns: a list of IndividualNudge Objects

        """
        parents = ea.select_parents(individuals, self.number_of_children*2,
                                 self.parent_selection_mode)
        children = self.create_children(parents, self.number_of_children,
                                        timestep, mutation_size)
        for child in children:
            child.evaluate(self.start_distribution, self.cond_output)

        return ea.select_new_population(individuals, children, len(individuals),
                                        self.generational)

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
                parents[i], parents[number_of_children+i], timestamp
            ))
        for child in children:
            child.mutate(mutation_size)

        return children

    def recombine(self, parent1, parent2, timestamp):
        """recombine two individuals to create a new individual
        
        Parameters:
        ----------
        parent1: Individual object
        parent2: Individual object
        timestamp: a number

        Returns: Individual object

        """
        if np.random.random() > 0.5:
            genes = np.copy(parent1.genes)
        else:
            genes =  np.copy(parent2.genes)
        
        return IndividualNudge(genes, timestamp)

class FindMaximumLocalNudge():
    """ 
    A class to find the maximal local nudge

    Attributes:
    ----------
    start_distribution: nd-array
        Representing a probability distribution
    cond_output: nd-array
        Representing the output conditioned on the input
    nudge_size: a number
        The nudge_size on the input distribution.
    generational: Boolean
        Whether to discard the old individuals at every new timestep
    number_of_children: an integer
    parent_selection_mode: Either "rank_exponential" or None

    """
    def __init__(self, start_distribution, cond_output, nudge_size, 
                 generational, number_of_children, parent_selection_mode):
        """Create a FindMaximumIndividualNudge object"""
        self.start_distribution = start_distribution
        self.cond_output = cond_output
        self.nudge_size = nudge_size
        self.generational = generational
        self.number_of_children = number_of_children
        self.parent_selection_mode = parent_selection_mode


    def evolve(self, individuals, timestep):
        """
        Evolve the population
        
        Parameters:
        ----------
        individuals: a list with LocalNudge objects

        Returns: a list of LocalNudge Objects

        """
        parents = ea.select_parents(individuals, self.number_of_children*2,
                                    self.parent_selection_mode)
        children = self.create_children(parents, self.number_of_children,
                                        timestep)
        for child in children:
            child.evaluate(self.start_distribution, self.cond_output)

        return ea.select_new_population(individuals, children, len(individuals),
                                        self.generational)

    def create_children(self, parents, number_of_children, timestamp):
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
                parents[i], parents[number_of_children+i], timestamp
            ))
        for child in children:
            child.mutate()

        return children

    def recombine(self, parent1, parent2, timestamp):
        """recombine two LocalNudges to create a new individual
        
        Parameters:
        ----------
        parent1: LocalNudge object
        parent2: LocalNudge object
        timestamp: a number

        Returns: LocalNudge object

        """
        gene_choice = np.random.choice([1,2])
        mutation_size_choice = np.random.choice([1,2])
        mutation_size_weights_choice = np.random.choice([1,2])
        if gene_choice==1:
            new_timestamp = parent1.timestamp
            new_weights = np.copy(parent1.weights)
            new_individual_nudges = parent1.copy_individual_nudges()
        else:
            new_timestamp = parent2.timestamp
            new_weights = np.copy(parent2.weights)
            new_individual_nudges = parent2.copy_individual_nudges()

        if mutation_size_weights_choice == 1:
            new_mutation_size_weights = parent1.mutation_size_weights
            new_change_mutation_size_weights = parent1.change_mutation_size_weights
        else:
            new_mutation_size_weights = parent2.mutation_size_weights
            new_change_mutation_size_weights = parent2.change_mutation_size_weights

        if mutation_size_choice == 1:
            new_mutation_size = parent1.mutation_size    
            new_change_mutation_size = parent1.change_mutation_size
        else:
            new_mutation_size = parent2.mutation_size    
            new_change_mutation_size = parent2.change_mutation_size
        
        return LocalNudge(
            new_weights, new_individual_nudges, parent1.nudged_vars,
            new_mutation_size, new_change_mutation_size, 
            new_mutation_size_weights, new_change_mutation_size_weights, 
            parent1.nudge_size, new_timestamp
        )

if __name__ == "__main__":
    #distribution parameters
    input_variables = 5
    number_of_states = 3

    distribution_shape = [number_of_states]*input_variables
    total_number_of_states = reduce(lambda x,y: x*y, distribution_shape)
    input_dist = np.random.dirichlet([1]*total_number_of_states)
    input_dist = np.reshape(input_dist, distribution_shape)
    cond_shape = [number_of_states]*(input_variables+1)
    cond_output = [
        probability_distributions.compute_joint_uniform_random((number_of_states,))
        for i in range(number_of_states**(input_variables))
    ]
    cond_output = np.array(cond_output)
    cond_output = np.reshape(cond_output, cond_shape)

    nudge_size = 0.01

    #local nudge optimization
    local_evolutionary_params = {
        "number_of_generations": 100,
        "population_size": 20,
        "number_of_children": 100,
        "generational": True,
        "parent_selection_mode": "rank_exponential",
        "mutation_size_weights": 0.8,
        "change_mutation_size_weights": 0.5,
        "start_mutation_size": 0.005,
        "change_mutation_size": 0.5
    }

    max_local_nudge = find_maximum_local_nudge(
        input_dist, cond_output, nudge_size,
        local_evolutionary_params, verbose=True
    )
    print(max_local_nudge.score)
    print(max_local_nudge.weights)

    #individual nudge optimization
    number_of_generations = 50
    population_size = 10
    number_of_children = 10 
    generational = False 
    mutation_size = nudge_size/4
    parent_selection_mode = "rank_exponential"
    #parent_selection_mode = None

    individuals = create_individual_nudges(
        population_size, number_of_states, nudge_size, "random"
    )
    for individual in individuals:
        individual.evaluate(input_dist, cond_output)
    print("initial impact {}".format(
        ea.sort_individuals(individuals)[0].score
    ))
    find_max_nudge = FindMaximumIndividualNudge(
        input_dist, cond_output, nudge_size, generational, number_of_children,
        parent_selection_mode
    )
    # max_individual = find_max_nudge.get_max_nudge(
    #     individuals, number_of_generations, mutation_size
    # )
    # print("the found max impact for an individual nudge {}".format(
    #     max_individual.score
    # ))

    print("the found max impact for a local nudge {}".format(
        max_local_nudge.score
    ))

    max_impact = maximum_nudges.find_max_impact_individual_nudge_exactly(
        input_dist, cond_output, nudge_size/2.0, True
    )

    print("the actual maximum individual nudge {}".format(max_impact))
