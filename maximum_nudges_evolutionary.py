import random
import nudge_non_causal as nudge
import nudge as nudge_old
import numpy as np
import evolutionary_algorithms as ea
import probability_distributions

import maximum_nudges

TEST = False

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

    """
    def __init__(self, weights, individual_nudges, nudged_vars, 
                 start_mutation_size, change_mutation_size, 
                 mutation_size_weights, nudge_size, timestamp):
        """create LocalNudge object"""
        self.weights = weights
        self.individual_nudges = individual_nudges
        self.nudged_vars = nudged_vars
        self.nudge_size = nudge_size
        self.mutation_size_weights = mutation_size_weights
        self.mutation_size = start_mutation_size
        self.change_mutation_size = change_mutation_size
        self.timestamp = timestamp

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
        self.mutation_size += np.random.uniform(
            -self.change_mutation_size, self.change_mutation_size
        )
        #update the weights
        noise_vector = nudge.find_noise_vector(self.weights.shape[0], 
                                  self.mutation_size_weights)
        self.weights = nudge.nudge_states(noise_vector, self.weights)

        #update the inidividual nudges
        for individual_nudge in self.individual_nudges:
            individual_nudge.mutate(self.mutation_size)

    @classmethod
    def create_local_nudge(cls, nudged_vars_to_states, nudge_size, 
                           mutation_size_weights, start_mutation_size, 
                           change_mutation_size, timestamp=0):
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
            change_mutation_size, mutation_size_weights, nudge_size, timestamp
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

class SynergisticNudge():
    """
    Class to represent a synergistic nudge

    Attributes:
    ----------
    start_distribution: nd-array 
        Representing a probability distribution, this distribution 
        should not be changed (since it is shared among many instances).
        In java this would be made static
    new_distribution: nd-array
        Representing a probability distribution, this one is specific 
        to this instance
    cond_output: nd-array, the output should be on the last axis
    nudge_size: number
    mutations_per_update_step: integer
    score: number
    timestamp: integer
    mutation_size: a number
    change_mutation_size: a number 
        The size of the change applied to the mutation_size

    """
    def __init__(self, start_distribution, new_distribution, cond_output,
                 nudge_size, mutations_per_update_step, start_mutation_size,
                 change_mutation_size, timestamp):
        """create SynergisticNudge object"""
        self.start_distribution = start_distribution
        self.new_distribution = new_distribution
        self.cond_output = cond_output
        self.nudge_size = nudge_size
        self.mutations_per_update_step = mutations_per_update_step
        self.mutation_size = start_mutation_size
        self.change_mutation_size = change_mutation_size
        self.timestamp = timestamp

    def evaluate(self):
        """
        Find the impact of the nudge on the output distribution and 
        set the score. The score is multiplied by -1 to make it a 
        minimization problem.

        """
        nudge_impact = nudge.find_nudge_impact(
            self.start_distribution, self.new_distribution, self.cond_output,
            measure="absolute"
        )
        #to make it a minimization rather than a maximization problem
        self.score = -nudge_impact

    def mutate_new(self):
        distribution = np.copy(new_distribution)
        output_label = np.random.choice(len(new_distribution.shape))
        new_dist = nudge_old.mutate_distribution_with_fixed_marginals(
            distribution, output_label, self.mutations_per_update_step,
            self.mutation_size
        )


    def mutate(self):
        """
        Mutate both the mutation size and the genes (the weights and the 
        individual nudges)
        
        """
        self.mutation_size += np.random.uniform(
            -self.change_mutation_size, self.change_mutation_size
        )
        for _ in range(self.mutations_per_update_step):
            nudge.synergistic_mutate(self.new_distribution, abs(self.mutation_size))

        new_nudge_size = np.sum(abs(
            self.new_distribution-self.start_distribution
        ))
        adjustment_factor = self.nudge_size/new_nudge_size
        if adjustment_factor <= 1:
            self.new_distribution = (
                self.start_distribution +
                (self.new_distribution-self.start_distribution)*adjustment_factor
            )

        if np.any(self.new_distribution<0):
            raise ValueError()

    @classmethod
    def create_nudge(cls, start_distribution, cond_output, nudge_size,
                     mutations_per_update_step, start_mutation_size,
                     change_mutation_size, timestamp):
        """
        Create a SynergisticNudge object randomly

        Parameters:
        ----------
        start_distribution: nd-array- representing a probability distribution
        cond_output: nd-array 
            Representing a probability distribution, the output should be
            on the last axis
        nudge_size: number
        mutations_per_update_step: an integer
        start_mutation_size: a number
        change_mutation_size: a number
        timestamp: integer

        Returns: SynergisticNudge object

        """
        new_distribution = np.copy(start_distribution)
        instance = cls(
            start_distribution, new_distribution, cond_output, nudge_size,
            mutations_per_update_step, start_mutation_size, 
            change_mutation_size, timestamp
        )
        instance.mutate()
        return instance

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
        population = individuals
        for timestep in range(number_of_generations):
            population = self.evolve(population, mutation_size, timestep)
            if TEST:
                print("timestep {} best score {}, worst score {}".format(
                    timestep, population[0].score, population[-1].score
                ))

        return ea.sort_individuals(population)[0]

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

    def get_max_nudge(self, individuals, number_of_generations):
        """
        Find the maximum local nudge.

        Parameters:
        ----------
        number_of_generations: an integer
        individuals: list of IndividualNudge objects
 
        Returns:
        -------
        A numpy array which represents the maximum individual nudge
        for the start distribution.

        """
        population = individuals
        scores = []
        for timestep in range(number_of_generations):
            population = self.evolve(population, timestep)
            scores.append(population[0].score)
            if TEST:
                print("time step {} best score {} worst score {}".format(
                    timestep, population[0].score, population[-1].score
                ))

        return ea.sort_individuals(population)[0]

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
        new_nudged_vars = parent1.nudged_vars
        new_nudge_size = parent1.nudge_size
        if np.random.random() > 0.5:
            new_timestamp = parent1.timestamp
            new_weights = np.copy(parent1.weights)
            new_individual_nudges = []
            for individual_nudge in parent1.individual_nudges:
                new_individual_nudge = IndividualNudge(
                    np.copy(individual_nudge.genes), new_timestamp
                )
                new_individual_nudges.append(new_individual_nudge)
        else:
            new_timestamp = parent2.timestamp
            new_weights = np.copy(parent2.weights)
            new_individual_nudges = []
            for individual_nudge in parent2.individual_nudges:
                new_individual_nudge = IndividualNudge(
                    np.copy(individual_nudge.genes), new_timestamp
                )
                new_individual_nudges.append(new_individual_nudge)
        
        if np.random.random > 0.5:
            new_mutation_size_weights = parent1.mutation_size_weights
        else:
            new_mutation_size_weights = parent2.mutation_size_weights

        if np.random.random > 0.5:
            new_mutation_size = parent1.mutation_size    
        else:
            new_mutation_size = parent2.mutation_size    

        if np.random.random > 0.5:
            new_change_mutation_size = parent1.change_mutation_size
        else:
            new_change_mutation_size = parent2.change_mutation_size
        
        return LocalNudge(new_weights, new_individual_nudges, new_nudged_vars,
                          new_mutation_size, new_change_mutation_size, 
                          new_mutation_size_weights, new_nudge_size, new_timestamp)

class FindMaximumSynergisticNudge():
    """ 
    A class to find the maximal individual nudge

    Attributes:
    ----------
    generational: Boolean
        Whether to discard the old individuals at every new timestep
    number_of_children: an integer
    parent_selection_mode: Either "rank_exponential" or None

    """
    def __init__(self, generational, number_of_children, parent_selection_mode):
        """Create a FindMaximumSynergisticNudge object"""
        self.generational = generational
        self.number_of_children = number_of_children
        self.parent_selection_mode = parent_selection_mode

    def get_max_nudge(self, individuals, number_of_generations):
        """
        Find the maximum synergistic nudge.

        Parameters:
        ----------
        number_of_generations: an integer
        individuals: list of IndividualNudge objects
 
        Returns:
        -------
        A numpy array which represents the maximum individual nudge
        for the start distribution.

        """
        population = individuals
        scores = []
        for timestep in range(number_of_generations):
            population = self.evolve(population, timestep)
            scores.append(population[0].score)
            if TEST:
                print("time step {} best score {} worst score {}".format(
                    timestep, population[0].score, population[-1].score
                ))

        return ea.sort_individuals(population)[0]

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
            child.evaluate()

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
        """recombine two SynergisticNudges to create a new individual
        
        Parameters:
        ----------
        parent1: SynergisticNudge object
        parent2: SynergisticNudge object
        timestamp: a number

        Returns: SynergisticNudge object

        """
        if np.random.random() > 0.5:
            new_distribution = np.copy(parent1.new_distribution)
            return SynergisticNudge(
                parent1.start_distribution, new_distribution, parent1.cond_output,
                parent1.nudge_size, parent1.mutations_per_update_step, 
                parent1.mutation_size, parent1.change_mutation_size,
                timestamp
            )
        else:
            new_distribution = np.copy(parent2.new_distribution)
            return SynergisticNudge(
                parent2.start_distribution, new_distribution, parent2.cond_output,
                parent2.nudge_size, parent2.mutations_per_update_step, 
                parent2.mutation_size, parent2.change_mutation_size,
                timestamp
            )


if __name__ == "__main__":
    #distribution parameters
    input_variables = 4
    number_of_states = 5
    nudge_size = 0.01
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

    #local nudge optimization
    number_of_generations = 200 
    population_size = 10
    number_of_children = 20 
    generational = True 
    mutation_size = nudge_size/4
    parent_selection_mode = "rank_exponential"
    #parent_selection_mode = None
    mutation_size_weights = 0.03
    start_mutation_size = nudge_size/10
    change_mutation_size = start_mutation_size/10
    nudged_vars_to_states = {
        nudged_var:number_of_states for nudged_var in range(input_variables)
    }
    local_nudges = []
    for _ in range(population_size):
        new_local_nudge = LocalNudge.create_local_nudge(
            nudged_vars_to_states, nudge_size, mutation_size_weights,
            start_mutation_size, change_mutation_size, timestamp=0
        )
        local_nudges.append(new_local_nudge)
    #print(local_nudges)
    for local_nudge in local_nudges:
        local_nudge.evaluate(input_dist, cond_output)

    print("initial impact local nudge {}".format(
        ea.sort_individuals(local_nudges)[0].score
    ))
    find_max_local_nudge = FindMaximumLocalNudge(
        input_dist, cond_output, nudge_size, 
        generational, number_of_children, parent_selection_mode
    )
    max_local_nudge_individual = find_max_local_nudge.get_max_nudge(
        local_nudges, number_of_generations
    )
    print("the found max impact for a local nudge {}".format(
        max_local_nudge_individual.score
    ))

    #individual nudge optimization

    #evolutionary algorithm parameters
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
    max_individual = find_max_nudge.get_max_nudge(
        individuals, number_of_generations, mutation_size
    )

    print("the found max impact for a local nudge {}".format(
        max_local_nudge_individual.score
    ))
    print("the found max impact for an individual nudge {}".format(
        max_individual.score
    ))
    max_impact = maximum_nudges.find_maximum_local_nudge(
        input_dist, cond_output, nudge_size/2
    )
    print("the actual maximum individual nudge {}".format(max_impact))
