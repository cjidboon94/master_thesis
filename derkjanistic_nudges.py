import numpy as np
import random
import dit
from typing import List
from dist_utils import calculate_Y

TEST = False

def do_derkjanistic_nudge(input_dist: dit.Distribution,  nudge_size:float):
    "Unoptimized derkjanistic nudge"
    rvs = input_dist.outcome_length()
    alphabet = input_dist.alphabet[0]
    new_distribution = input_dist.pmf
    for _ in range(len(new_distribution)):
        state_vectors = np.tile(np.random.choice(alphabet, rvs, replace=True), (4, 1))
        x_i, x_j = np.random.choice(rvs, 2, replace=False)
        states_i = np.random.choice(alphabet, 2, replace=False)
        states_j = np.random.choice(alphabet, 2, replace=False)
        for i in range(2):
            for j in range(2):
                state_vectors[i * 2 + j, [x_i, x_j]] = states_i[i], states_j[j]
        state_strings = sorted([tuple(vector) for vector in state_vectors])
        #print(input_dist.outcomes)
        state_indices = [i for i, sample in enumerate(input_dist.sample_space()) if sample in state_strings]
        selected_states =  [{'label': label, 'index': index, 'prob': input_dist[label]} for label, index in
                zip(state_strings, state_indices)]
        random.shuffle(selected_states)
        #print(selected_states)
        if np.random.random() > 0.5:
            negs = (selected_states[0], selected_states[3])
            poss = (selected_states[1], selected_states[2])
        else:
            negs = (selected_states[1], selected_states[2])
            poss = (selected_states[0], selected_states[3])

        this_mutation_size = min(input_dist[negs[0]['label']],
                                 input_dist[negs[1]['label']],
                                 1-input_dist[poss[0]['label']],
                                 1-input_dist[poss[1]['label']],
                                 nudge_size)
        this_mutation_size = np.random.uniform(0, this_mutation_size)
        for neg in negs:
            new_distribution[neg['index']] -= this_mutation_size
        for pos in poss:
            new_distribution[pos['index']] += this_mutation_size
        if np.any(new_distribution < 0):
            print("something went wrong")
            print("negative states {}".format(negs))
            print("positive_states {}".format(poss))
            print("the mutation size {}".format(this_mutation_size))
            print("the negative probs {}, {}".format(new_distribution[negs[0]["index"]],
                                                     new_distribution[negs[1]["index"]]))
            print("the positive probs {}, {}".format(new_distribution[poss[0]["index"]],
                                                     new_distribution[poss[1]["index"]]))
            print(new_distribution)
            raise ValueError()
    return new_distribution

def max_derkjanistic_nudge(input_dist: dit.Distribution, conditional: np.ndarray, nudge_size: float,
                       evolutionary_parameters: dict):
    "Optimized derkjanistic nudge"
    # create individuals
    individuals = [DerkjanisticNudge.generate_individual(input_dist, conditional, nudge_size,
                                                          evolutionary_parameters["mutations_per_update_step"],
                                                          evolutionary_parameters["start_mutation_size"],
                                                          evolutionary_parameters["change_mutation_size"],
                                                          timestamp=0)
                   for _ in range(evolutionary_parameters['population_size'])]
    initial_impact = get_impact(individuals)
    evolution = FindMaxDerkjanisticNudge(
        evolutionary_parameters["generational"],
        evolutionary_parameters["number_of_children"],
        evolutionary_parameters["parent_selection_mode"]
    )
    max_synergistic_nudge = evolution.search(individuals, evolutionary_parameters["number_of_generations"])

    if TEST:
        print("synergistic nudge: intial impact {}, max impact {}".format(
            initial_impact, max_synergistic_nudge.score
        ))
    return max_synergistic_nudge

    # Evaluate the individuals and get the top score
    # Use evolutionary





class DerkjanisticNudge():
    def __init__(self, start_dist: dit.Distribution, new_dist: dit.Distribution,
                 conditional: np.ndarray, nudge_size: float, mutations_per_step: int,
                 start_mutation_size: float, change_mutation_size: float, timestamp: int):
        self.start_dist = start_dist
        self.new_dist = new_dist
        self.conditional = conditional
        self.nudge_size = nudge_size
        self.mutations_per_step = mutations_per_step
        self.mutation_size = start_mutation_size
        self.change_mutation_size = change_mutation_size
        self.timestamp = timestamp

    def bump(self, timestamp=None):
        if timestamp is not None:
            return DerkjanisticNudge(self.start_dist, self.new_dist, self.conditional, self.nudge_size,
                                     self.mutations_per_step, self.mutation_size, self.change_mutation_size, timestamp)
        else:
            return DerkjanisticNudge(self.start_dist, self.new_dist, self.conditional, self.nudge_size,
                                     self.mutations_per_step, self.mutation_size, self.change_mutation_size,
                                     self.timestamp + 1)

    def evaluate(self):
        old_y = calculate_Y(self.start_dist, self.conditional)
        new_y = calculate_Y(self.new_dist, self.conditional)
        self.score =  -sum(abs(old_y.pmf - new_y.pmf))

    def mutate(self):
        self.mutation_size += np.random.uniform(-self.change_mutation_size, self.change_mutation_size)
        for _ in range(self.mutations_per_step):
            self.synergistic_mutate()

        new_nudge_size = np.sum(abs(self.new_dist.pmf - self.start_dist.pmf))
        adjustment_factor = self.nudge_size / new_nudge_size
        if adjustment_factor <= 1:
            self.new_dist.pmf = self.start_dist.pmf + (self.new_dist.pmf - self.start_dist.pmf) * adjustment_factor
        if np.any(self.new_dist.pmf < 0):
            raise ValueError()

    def synergistic_mutate(self, selected_states=None):
        def select_states():
            rvs = self.new_dist.outcome_length()
            state_vectors = np.tile(np.random.choice(self.new_dist.alphabet[0], rvs, replace=True), (4, 1))
            x_i, x_j = np.random.choice(rvs, 2, replace=False)
            states_i, states_j = np.random.choice(self.new_dist.alphabet[x_i], 2, replace=False), np.random.choice(
                self.new_dist.alphabet[x_j], 2, replace=False)
            for i in range(2):
                for j in range(2):
                    state_vectors[i * 2 + j, [x_i, x_j]] = states_i[i], states_j[j]
            state_strings = sorted([tuple(vector) for vector in state_vectors])
            state_indices = [i for i, sample in enumerate(self.new_dist.sample_space()) if sample in state_strings]
            return [{'label': label, 'index':index, 'prob': self.new_dist[label]} for label, index in zip(state_strings, state_indices)]

        new_distribution = self.new_dist.pmf
        mutation_size = abs(self.mutation_size)

        n_vars = self.new_dist.outcome_length()
        if n_vars < 2:
            raise ValueError("should be at least two vars")
        if selected_states is None:
            selected_states = select_states()
            random.shuffle(selected_states)
        if np.random.random() > 0.5:
            negs = (selected_states[0], selected_states[3])
            poss = (selected_states[1], selected_states[2])
        else:
            negs = (selected_states[1], selected_states[2])
            poss = (selected_states[0], selected_states[3])

        #Though unlikely when n_vars > 3, it's possible that mutation size is 0.
        mutation_size = min(self.new_dist[negs[0]['label']],
                            self.new_dist[negs[1]['label']],
                            self.new_dist[poss[0]['label']],
                            self.new_dist[poss[1]['label']],
                            mutation_size)

       #NOt sure why this line is here mutation_size = np.random.uniform(0, mutation_size)
        for neg in negs:
            new_distribution[neg['index']] -= mutation_size
        for pos in poss:
            new_distribution[pos['index']] += mutation_size
        if np.any(new_distribution < 0):
            print("something went wrong")
            print("negative states {}".format(negs))
            print("positive_states {}".format(poss))
            print("the mutation size {}".format(mutation_size))
            print("the negative probs {}, {}".format(new_distribution[negs[0]["index"]],
                                                     new_distribution[negs[1]["index"]]))
            print("the positive probs {}, {}".format(new_distribution[poss[0]["index"]],
                                                     new_distribution[poss[1]["index"]]))
            print(new_distribution)
            raise ValueError()

        self.new_dist.pmf = new_distribution


    @classmethod
    def generate_individual(cls, input_dist: dit.Distribution, conditional: np.ndarray, nudge_size: float,
                             mutations_per_step: int, start_mutation_size: float,
                             change_mutation_size: float, timestamp: int):
        new_distribution = input_dist.copy()
        instance = cls(input_dist, new_distribution, conditional, nudge_size, mutations_per_step, start_mutation_size,
                       change_mutation_size, timestamp)
        instance.mutate()
        return instance


def get_impact(individuals: List[DerkjanisticNudge]):
    for i in individuals:
        i.evaluate()
    return -min([i.score for i in individuals])


class FindMaxDerkjanisticNudge():
    def __init__(self, generational, number_of_children, parent_selection_mode):
        """Create a FindMaximumSynergisticNudge object"""
        self.generational = generational
        self.number_of_children = number_of_children
        self.parent_selection_mode = parent_selection_mode

    def search(self, individuals, n_generations, p=False):
        for timestep in range(n_generations):
            individuals = self.evolve(individuals, timestep)
            if p:
                print("%d, best score: %f" % (timestep, -min([i.score for i in individuals])))
        return individuals[np.argmin([i.score for i in individuals])]

    def evolve(self, individuals, timestep):
        def select_parents(individuals):
            #inspired by the select_parents
            scores =  1 - np.exp(-1 * np.arange(len(individuals)))
            scores = scores/sum(scores)
            n_parents = self.number_of_children*2
            sorted_individuals = sorted(individuals, key=lambda x: x.score)

            #and the universal stochastic sampling part
            zipped_indivs = list(zip(sorted_individuals, scores))
            points = np.linspace(0, 1, n_parents, False) + \
                      np.random.uniform(0, 1.0 / n_parents)
            random.shuffle(zipped_indivs)

            population = list(zip(*zipped_indivs))[0]
            rank_probabilities = list(zip(*zipped_indivs))[1]
            bins = np.zeros(len(sorted_individuals) + 1)
            probability_mass = 0
            for i in range(len(sorted_individuals)):
                bins[i + 1] = rank_probabilities[i] + probability_mass
                probability_mass += rank_probabilities[i]

            parent_indices, _ = np.histogram(points, bins)
            return [parent
                    for index, amount_of_samples in enumerate(parent_indices)
                    for parent in [population[index]] * amount_of_samples]


        def select_population(individuals, children):
            size = len(individuals)
            if self.generational:
                pop = children
            else:
                pop = individuals + children
            return sorted(pop, key=lambda x: x.score)[:size]


        parents = select_parents(individuals)
        children = self.create_children(parents, timestep)
        for child in children:
            child.evaluate()
        return select_population(individuals,children)

    def create_children(self, parents, timestep):
        #print("number of children", self.number_of_children)
        children = [self.recombine( parents[i], parents[self.number_of_children + i], timestep) \
                    for i in range(self.number_of_children)]
        for child in children:
            child.mutate()
        return children
        pass
    def recombine(self, parent1: DerkjanisticNudge, parent2: DerkjanisticNudge, timestep):
        if np.random.random() > 0.5:
            return parent1.bump(timestep)
        else:
            return parent2.bump(timestep)

