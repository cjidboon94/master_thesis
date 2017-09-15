import itertools
import numpy as np
import probability_distributions
from probability_distributions import ProbabilityArray
import combining_routes

def find_max_control_impact(input_dist, cond_output, nudge_size):
    """
    Find the maximum nudge output, defined as the abs difference between
    the old and the new output state.

    Parameters:
    ----------
    input_dist: a numpy array
        Representing the input distribution
    cond_output: a numpy array
        Representing the conditional probabilities of the output given 
        the input. The first N-1 variables (axes) should represent the
        input variables while the output should be the last axis.
    nudge_size: A number

    Returns: nudge, nudge_size
    -------
    nudge: a list of tuples
        The nudge that gives the maximum impact.
        Every tuple consist of an integer representing the nudge state on 
        the flattened input array and a number representing the nudge size
        on this state.
    nudge_size: The impact size of this nudge, defined as the max difference

    """
    number_of_input_vars = input_dist.flatten().shape[0]
    max_impact, max_nudge_states, max_nudge_sizes = 0, [], []
    for output_nudge_state in itertools.product([-1, 1], repeat=cond_output.shape[-1]):
        if all([state==-1 for state in output_nudge_state]) or all([state==1 for state in output_nudge_state]):
            continue

        allignment = cond_output.flatten() * np.array(output_nudge_state*number_of_input_vars)
        allignment_scores = np.sum(
            np.reshape(allignment, (number_of_input_vars, len(output_nudge_state))),
            axis=1
        )
        if output_nudge_state == (-1, -1, 1):
            print(allignment_scores)

        #print(output_nudge_state)
        nudge_states, nudge_sizes, nudge_impact = find_max_nudge_impact(
            allignment_scores, input_dist.flatten(), nudge_size
        )
        #print("nudge states {}".format(nudge_states))
        #print("nudge sizes {}".format(nudge_sizes))
        #print("nudge_impact {}".format(nudge_impact))

        if nudge_impact > max_impact:
            max_impact = nudge_impact
            max_nudge_states, max_nudge_sizes = nudge_states, nudge_sizes

    return max_nudge_states, max_nudge_sizes, max_impact

def find_max_nudge_impact(scores, weights, nudge_size):
    """
    Find the nudge that pushes the output variable maximally towards the 
    output state

    Parameters:
    ----------
    scores: 1-D nd-array
        The alligment scores for the conditional distribution linked to
        all input states
    weights: nd-array
       The flattened input probabilities
    nudge_size: a number

    Returns: nudge, nudge_impact
    -------
    selected states: list
        The states that will be nudged
    nudge_sizes: nd-array
        The nudge size for the corresponding selected state
    nudge_impact: a number
        The maximum that that the output can be transformed towards the
        output state by changing the input state with the nudge size 

    """
    negative_states, negative_nudge_sizes = find_minimum_subset(
        weights, scores, nudge_size
    )
    positive_state = [np.argmax(scores)]
    positive_nudge_size = sum(negative_nudge_sizes)
    selected_states = negative_states + positive_state
    nudge_sizes = np.zeros(len(selected_states))
    nudge_sizes[:-1] = -1*negative_nudge_sizes
    nudge_sizes[-1] = positive_nudge_size
    nudge_impact = np.sum(scores[selected_states] * nudge_sizes)
    return selected_states, nudge_sizes, nudge_impact

def find_minimum_subset(weights, scores, total_size):
    """
    Find a subset of weights of total_size with the lowest sum in scores.
    Fractions of scores CAN be taken.

    Note: The selected subset cannot include all of the weights

    Parameters:
    ----------
    weights: a 1d nd-array
    scores: a 1d nd-array with the same length of weights
    total_size: a number

    Returns: selected_variables, selected_weights, total_impact
    -------
    selected_variables: list
    selected_weights: nd-array 
        The corresponding weights for the selected_variables

    """
    #can (probably) be done more efficient for now fine
    indices = list(range(weights.shape[0]))
    results = [[weight, score, index] for weight, score, index
              in zip(weights, scores, indices)]
    results_sorted = sorted(results, key=lambda x: x[1])
    weights_sorted, scores_sorted, indices_sorted = zip(*results_sorted)

    count = 0
    minus_weight = 0
    while minus_weight < total_size and count < len(indices)-1:
        minus_weight += weights_sorted[count]
        count += 1
    
    selected_indices = list(indices_sorted[:count])
    selected_weights = weights[selected_indices]
    selected_weights[-1] = min(total_size-sum(selected_weights[:-1]),
                               selected_weights[-1])
    return selected_indices, selected_weights

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
            #print("the negative states&scores {}".format(negative_states_and_scores))

            # The the limiting effect of not being able to perform a nudge fully
            # is already accounted for in the positive nudge
            routes = create_routes(
                negative_states_and_scores, nudge_size, positive_scores
            ) 
            #print("the routes are {}".format(routes))
            optimal_route, impact_negative_nudge = (
                combining_routes.find_optimal_height_routes(routes, nudge_size)
            )
            #impact_negative_nudge = combining_routes.find_optimal_height_routes(
            #    routes, nudge_size
            #)
            impact_positive_nudge = nudge_size * sum(positive_scores)
            
            #print("the positive impact {}".format(impact_positive_nudge))
            #print("the negative impact {}".format(impact_negative_nudge))

            if impact_positive_nudge+impact_negative_nudge > max_impact:
                #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                #print("the optimal route is {}".format(optimal_route))
                #print("the best output allignment is {}".format(output_allignment))
                #print("the best nudge allignment is {}".format(nudge_allignment))
                max_optimal_route = optimal_route
                max_impact = impact_positive_nudge+impact_negative_nudge
            #print("")

    return max_impact

def find_maximum_local_nudge_without_conditional(input_dist, cond_output, nudge_size):
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
    
    other_input_dist = ProbabilityArray(input_dist).marginalize(
        set(range(number_of_input_variables-1))
    )

    if number_of_input_variables == 1:
        cond_nudge = input_dist
    else:
        temp_other_inputs = other_input_dist.flatten()
        number_of_other_input_states = temp_other_inputs.shape[0]
        temp_inputs = np.reshape(
            input_dist, (number_of_other_input_states, number_of_nudge_states)
        )
        cond_nudge = np.ones(
            (number_of_other_input_states, number_of_nudge_states)
        )

        for i in range(temp_other_inputs.flatten().shape[0]):
            #within this condition the conditional nudge can be every
            #every distribution so I set it to uniform
            if temp_other_inputs[i] == 0:
                cond_nudge[i] = cond_nudge[i]/number_of_nudge_states
            else:
                cond_nudge[i] = temp_inputs[i]/temp_other_inputs[i]

        cond_nudge = np.reshape(cond_nudge, input_dist.shape)

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
            #print("weighted allignment scores {}".format(weighted_allignment_scores))

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
            optimal_route, impact_negative_nudge = (
                combining_routes.find_optimal_height_routes(routes, nudge_size)
            )
            #impact_negative_nudge = combining_routes.find_optimal_height_routes(
            #    routes, nudge_size
            #)
            impact_positive_nudge = nudge_size * sum(positive_scores)
            
            #print("the positive impact {}".format(impact_positive_nudge))
            #print("the negative impact {}".format(impact_negative_nudge))

            if impact_positive_nudge+impact_negative_nudge > max_impact:
                #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                #print("the optimal route is {}".format(optimal_route))
                #print("the best output allignment is {}".format(output_allignment))
                #print("the best nudge allignment is {}".format(nudge_allignment))
                max_optimal_route = optimal_route
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
        if state == 0:
            continue
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
     


