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
    #print("number of input variables {}".format(number_of_input_variables))
    
    if number_of_input_variables == 1:
        cond_nudge = input_dist
    else:
        cond_nudge, marginal_label, _ = ProbabilityArray(input_dist).find_conditional(
            set([number_of_input_variables-1]),
            set(list(range(number_of_input_variables-1)))
        )
        #print("the shape of the conditional nudge is {}".format(cond_nudge.shape))
        #print("the conditional nudge is {}".format(cond_nudge))
        if list(marginal_label)[0] != number_of_input_variables-1:
            raise ValueError("wrong variables are conditioned on")

    other_input_dist = ProbabilityArray(input_dist).marginalize(
        set(range(number_of_input_variables-1))
    )
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
            #print("the output allignment is {}".format(output_allignment))
            allignments = np.array(number_of_input_states*list(output_allignment))
            allignment_scores = allignments.flatten() * cond_output.flatten()
            allignment_scores = np.reshape(
                allignment_scores, 
                (number_of_input_states, number_of_output_states)
            )
            allignment_scores = np.sum(allignment_scores, axis=1)
            #print("the allignment scores are {}".format(allignment_scores))

            weights = np.repeat(other_input_dist.flatten(), input_dist.shape[-1])
            #print("the weights are {}".format(weights))
            weighted_allignment_scores = allignment_scores*weights 
            #print("the weighted allignment scores are {}".format(
            #    weighted_allignment_scores
            #))

            positive_scores = []
            negative_states_and_scores = []
            for state in range(number_of_nudge_states):
                #select all states from the input distribution for which
                #the nudged state equals i
                #print("the variable index is {}".format(i))
                states = np.take(cond_nudge, indices=state, 
                                 axis=number_of_input_variables-1)
                states = states.flatten()
                #print("the just created states are {}".format(states))
                scores = np.take(
                    np.reshape(weighted_allignment_scores, input_dist.shape),
                    indices=state, axis=number_of_input_variables-1
                )
                scores = scores.flatten()
                #print("the states are {}, the scores are {}".format(states, scores))

                if nudge_allignment[state] == 1:
                    positive_scores = scores
                else:
                    negative_states_and_scores.append((states, -1*scores))
            
            #print("the positive scores are {}".format(positive_scores))
            #print("the negative states and scores {}".format(negative_states_and_scores))

            # The the limiting effect of not being able to perform a nudge fully
            # is already accounted for in the positive nudge
            routes = create_routes(negative_states_and_scores, nudge_size, positive_scores) 
            #print("the routes are {}".format(routes))
            optimal_route, impact_negative_nudge = combining_routes.find_optimal_height_routes(routes, nudge_size)
            #impact_negative_nudge = combining_routes.find_optimal_height_routes(routes, nudge_size)
            impact_positive_nudge = nudge_size * sum(positive_scores)
            
            #print("the positive impact {}".format(impact_positive_nudge))
            #print("the negative impact {}".format(impact_negative_nudge))

            if impact_positive_nudge+impact_negative_nudge > max_impact:
                #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                #print("the optimal route is {}".format(optimal_route))
                #print("the best output allignment is {}".format(output_allignment))
                #print("the best nudge allignment is {}".format(nudge_allignment))
                max_impact = impact_positive_nudge+impact_negative_nudge
            #print("")

    return max_impact

def create_routes(states_and_scores_list, nudge_size, positive_scores):
    routes= []
    for states, scores in states_and_scores_list:
        route = create_route(states, scores, nudge_size, positive_scores)
        routes.append(route)

    return routes

def create_route(states, scores, nudge_size, positive_scores): 
    """
    Create a route from the states and scores. 

    Parameters:
    ----------
    states: a list of numbers
    scores: a list of numbers
    nudge_size: a number
    positive_scores: a list of numbers

    Returns: a list of dicts 
        Every dict has keys length and height. They represent that if
        you nudge this variable by the length than per length the 
        impact on the output variable is height.
    
    """
    #print("the states are {}".format(states))
    #combo = np.stack([states, scores, positive_scores], axis=1)
    route = []
    lower_states_scores = []
    sum_other_scores = 0
    for state, score, pos_score in zip(states, scores, positive_scores):
        if state < nudge_size:
            lower_states_scores.append([state, score, pos_score])
        else:
            sum_other_scores += score
    
    if lower_states_scores == []:
        return [{"length": nudge_size, "height":sum_other_scores}]

    states_less, scores_less, pos_scores_less = zip(
        *sorted(lower_states_scores, key=lambda x: x[0])
    )
    #print("lesser states {}, scores {}, positive scores {}".format(
    #    states_less, scores_less, pos_scores_less
    #))

    previous_state = 0
    for count, state in enumerate(states_less):
        extra_weight = state-previous_state 
        impact_per_nudge = sum_other_scores+sum(scores_less[count:])
        adjustment_impact_per_nudge = sum(pos_scores_less[:count])
        next_track = {
            "length": extra_weight, 
            "height": impact_per_nudge - adjustment_impact_per_nudge
        }
        route.append(next_track)
        previous_state = state

    last_weight = nudge_size-states_less[-1]
    adjusted_impact_per_nudge = sum_other_scores - sum(pos_scores_less)
    next_track = {"length": last_weight, "height": adjusted_impact_per_nudge}

    route.append(next_track)
    return route


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
    #answer should be 0.0258, checked it manually
    #print(np.take(input_dist, indices=0, axis=1))
     


