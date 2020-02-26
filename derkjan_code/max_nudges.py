import numpy as np
import maximum_individual_nudges
import synergistic_nudge
import maximum_nudges_evolutionary as ev_max_nudges
import nudge_non_causal


def max_nudge_impact(inputs, cond_output, nudge_size, nudge_type,
                     parameters):
    """
    Find the maximum impact of the nudges

    Parameters:
    ----------
    inputs: nd-array,
        joint probability distribution of the inputs
    cond_output_dists: nd-array
        Representing output distributions (the last axis) conditioned on the inputs
        distributions
    nudge_size: positive float
    nudge_type: string
        One of the following: {"individual", "local", "synergistic", "global"}
    parameters: a dict,
        The parameters used to find the maximum nudge

    """
    if nudge_type == "individual":
        return maximum_individual_nudges.find_max_impact_individual_nudge_exactly(
            inputs, cond_output, nudge_size / 2.0, True
        )
    elif nudge_type == "local":
        max_local_nudge = ev_max_nudges.find_maximum_local_nudge(
            inputs, cond_output, nudge_size, parameters, verbose=False
        )
        individual_nudges = max_local_nudge.individual_nudges
        nudge_vectors = [
            weight * nudge.genes
            for nudge, weight in zip(individual_nudges, max_local_nudge.weights)
        ]
        new_dist = nudge_non_causal.nudge_local(
            inputs, [0, 1], 0.01, nudge_vectors, True
        )
        l1_norm_to_old_distribution = np.sum(np.absolute(input_dist - new_dist))
        if abs(l1_norm_to_old_distribution - NUDGE_SIZE) > 10 ** (-7):
            print("WARNING size of nudge {}".format(l1_norm_to_old_distribution))
        return abs(max_local_nudge.score)
    elif nudge_type == "synergistic":
        max_synergistic_nudge = synergistic_nudge.find_synergistic_nudge_with_max_impact(
            inputs, cond_output, nudge_size, parameters
        )
        l1_norm_to_old_distribution = np.sum(np.absolute(
            inputs - max_synergistic_nudge.new_distribution
        ))
        if abs(l1_norm_to_old_distribution - nudge_size) > 10 ** (-7):
            print("WARNING size of nudge {}".format(l1_norm_to_old_distribution))
        return abs(max_synergistic_nudge.score)
    elif nudge_type == "global":
        _, _, max_global_nudge_impact = maximum_individual_nudges.find_max_control_impact(
            inputs, cond_output, nudge_size/2.0
        )
        return max_global_nudge_impact
    else:
        raise ValueError("provide a valid nudge type")
