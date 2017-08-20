import itertools
import numpy as np
import probability_distributions
from probability_distributions import ProbabilityArray
import combining_routes

def find_maximum_local_nudge(input_dist, cond_output, nudge_size):
    """

    Parameters:
    ----------
    input_dist: a numpy array
        Representing a probability distribution. The nudged variable
        should be the last variable in input_dist
    cond_output: a numpy array
        The dimensions should be have be the total size of the input 
        distribution times the amount of output states
    nudge_size: a number

    Returns: a number
        The maximum nudge nudge_size

    """
    number_of_nudge_states = input_dist.shape[-1]
    number_of_input_states = input_dist.flatten().shape[0]
    number_of_input_variables = len(input_dist.shape)
    number_of_output_states = cond_output.shape[-1]
    
    cond_nudge, marginal_label, _ = ProbabilityArray(input_dist).find_conditional(
       set([number_of_input_variables-1]),
       set(list(range(number_of_input_variables-1)))
    )
    print("the conditional nudge is") 
    print(cond_nudge)
    if list(marginal_label)[0] != number_of_input_variables-1:
        raise ValueError("wrong variables are conditioned on")

    nudge_allignments = []
    for i in range(number_of_nudge_states):
        states = [-1]*number_of_nudge_states
        states[i] = 1
        nudge_allignments.append(states)

    output_allignments = itertools.product([-1, 1], repeat=number_of_output_states)
    output_allignments = [allignment for allignment in output_allignments
                          if allignment != (-1,)*number_of_output_states and 
                          allignment != (1,)*number_of_output_states]
    max_impact = 0
    for nudge_allignment in nudge_allignments:
        #print("the nudge allignment {}".format(nudge_allignment))
        for output_allignment in output_allignments:
            print("the output allignment is {}".format(output_allignment))
            allignments = np.array(number_of_input_states*list(output_allignment))
            allignment_scores = allignments.flatten() * cond_output.flatten()
            allignment_scores = np.reshape(
                allignment_scores, 
                (number_of_input_states, number_of_output_states)
            )
            allignment_scores = np.sum(allignment_scores, axis=1)
            #print("the allignment scores are {}".format(allignment_scores))
            weighted_allignment_scores = input_dist.flatten()*allignment_scores

            positive_scores = []
            negative_states_and_scores = []
            for state in range(number_of_nudge_states):
                #select all states from the input distribution for which
                #the nudged state equals i
                #print("the variable index is {}".format(i))
                states = np.take(cond_nudge, indices=state, 
                                 axis=number_of_input_variables-1)
                scores = np.take(
                    np.reshape(weighted_allignment_scores, input_dist.shape),
                    indices=state, axis=number_of_input_variables-1
                )
                #print("the states are {}, the scores are {}".format(states, scores))

                if nudge_allignment[state] == 1:
                    positive_scores = scores
                else:
                    negative_states_and_scores.append((states, -1*scores))

            # The the limiting effect of not being able to perform a nudge fully
            # is already accounted for in the positive nudge
            routes = create_routes(negative_states_and_scores, nudge_size) 
            impact_negative = combining_routes.find_optimal_height_routes(routes, nudge_size)
            impact_positive_nudge = nudge_size * sum(positive_scores)

        if impact_positive_nudge+impact_negative_nudge > max_impact:
            max_impact = impact_positive_nudge+impact_negative_nudge

    return max_impact

#still needs to be tested
def create_routes(states_and_scores_list, nudge_size, positive_scores):
    routes= []
    for states, scores in states_and_scores_list:
        route = []
        #combo = np.stack([states, scores, positive_scores], axis=1)
        combo_less_than_nudge_size = []
        sum_other_scores = 0
        for state, score, pos_score in zip(states, scores, positive_scores):
            if state < nudge_size:
                combo_less_than_nudge_size.append([state, score, pos_score])
            else:
                sum_other_scores += score
        
        if combo_less_than_nudge_size == []:
            track = {"length": nudge_size, "height":sum_other_scores/nudge_size}
            routes.append(track)
            continue

        states_less, scores_less, pos_scores_less = zip(*sorted(combo, key=lambda x: x[0]))
        previous_state = 0
        for count, state in enumerate(states_less):
            extra_weight = state-previous_state 
            next_impact = extra_weight * (sum_other_scores+sum(scores_less[count:])
            adjustment_next_impact = extra_weight*sum(pos_scores_less[:count])
            next_track = {
                "length": extra_weight
                "height": (next_impact-adjustment_next_impact) / extra_weight
            }
            route.append(next_track)
            previous_state = state

        last_weight = nudge_size-states_less[-1]
        next_impact = last_weight * sum_other_scores
        adjustment_next_impact = last_weight * sum(pos_scores_less)
        next_track = {
            "length": last_weight
            "height": (next_impact-adjustment_next_impact) / last_weight
        }
        route.append(next_track)

        routes.append(route)

    return routes


if __name__=="__main__":
    input_dist = np.array([
        [0.1, 0.25, 0.1],
        [0.3, 0.05, 0.2]
    ])
    cond_dist = np.array([
        [0.2, 0.4, 0.4], [0.5, 0.1, 0.4], [0.25, 0.45, 0.3],
        [0.23, 0.33, 0.46], [0.17, 0.27, 0.56], [0.7, 0.15, 0.15]
    ])
    nudge_size = 0.1
    print(find_maximum_local_nudge(input_dist, cond_dist, nudge_size))
    print(np.take(input_dist, indices=0, axis=1))
     


