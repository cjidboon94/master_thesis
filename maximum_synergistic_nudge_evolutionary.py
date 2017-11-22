import random
import nudge_non_causal as nudge
import nudge as nudge_old
import numpy as np
import evolutionary_algorithms as ea
import probability_distributions

TEST = False

def find_synergistic_nudge_with_max_impact(input_dist, cond_output, nudge_size, 
                                           evolutionary_parameters):
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
            the upper bound on the range the change is selected from uniformly
        mutations_per_update_step: integer

    Returns: A SynergisticNudge object
    
    """
    #create initial population
    synergistic_nudges = []
    for _ in range(evolutionary_parameters["population_size"]):
        new_synergistic_nudge = SynergisticNudge.create_nudge(
            input_dist, cond_output, nudge_size, 
            evolutionary_parameters["mutations_per_update_step"], 
            evolutionary_parameters["start_mutation_size"], 
            evolutionary_parameters["change_mutation_size"], 
            timestamp=0
        )
        synergistic_nudges.append(new_synergistic_nudge)
    for synergistic_nudge in synergistic_nudges:
        synergistic_nudge.evaluate()

    initial_impact = ea.sort_individuals(synergistic_nudges)[0].score

    #evolve the population
    find_max_synergistic_nudge = FindMaximumSynergisticNudge(
        evolutionary_parameters["generational"], 
        evolutionary_parameters["number_of_children"], 
        evolutionary_parameters["parent_selection_mode"]
    )
    max_synergistic_nudge = find_max_synergistic_nudge.get_max_nudge(
        synergistic_nudges, evolutionary_parameters["number_of_generations"]
    )
    if TEST:
        print("synergistic nudge: intial impact {}, max impact {}".format(
            initial_impact, max_synergistic_nudge.score
        ))
    return max_synergistic_nudge


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

