import time
import itertools
import numpy as np
import networkx as nx
import pandas as pd
#import probability_distributions
import dit
from dist_utils import get_vars

def generate_powerlaw_network(size, gamma=2.0, only_giant_component=True, create_using=nx.Graph(),
                              remove_selfloops=True):

    """
    Generate a network with a powerlaw degree distribution

    Note: I put this function here but just out of convenience, should maybe move it to somewhere else.
    :param size:
    :param gamma:
    :param only_giant_component:
    :param create_using:
    :param remove_selfloops:
    :rtype: nx.Graph
    """

    try:
        degs = nx.utils.powerlaw_sequence(size, gamma)
        degs = list(map(int, map(round, degs)))
        if int(np.sum(degs)) % 2 == 1:  # odd?
            degs[np.random.randint(len(degs))] += 1  # break the unevenness of the sum of degrees

        G = nx.configuration_model(degs, create_using=create_using)
    except nx.NetworkXError as e:  # probably because the sum of degrees is not even, so cannot satisfy it
        # degs[np.random.randint(len(degs))] += 1.0  # break the unevenness of the sum of degrees
        #
        # G = nx.configuration_model(map(int, map(round, degs)), create_using=create_using)
        assert False, 'error: e = ' + str(e)

    if only_giant_component:
        G = nx.connected_component_subgraphs(G).__next__()

    if remove_selfloops:
        G.remove_edges_from(G.selfloop_edges())

    return G

def update_random_node(network, temperature):
    #pick a node randomly
    node = np.random.choice(network.nodes())
    neighbors = [neighbor for neighbor in network.neighbors(node)]
    energy_neighbors = np.sum([network.node[neighbor]["value"] for neighbor in neighbors])
    #update the node
    if np.sign(energy_neighbors) != np.sign(network.node[node]["value"]):
        network.node[node]["value"] *= -1
        #print("flipped node")
    elif np.exp(-1 * (1/temperature) * abs(energy_neighbors)) > np.random.random():
        network.node[node]["value"] *= -1
        #print("flipped node")
    else:
        pass
        #print("not flipped node")


def update_random_node_glauber(network, temperature):
    #pick a node randomly
    node = np.random.choice(network.nodes())

    #update the node
    neighbors = [neighbor for neighbor in network.neighbors(node)]
    energy_neighbors = np.sum([network.node[neighbor]["value"] for neighbor in neighbors])
    if np.sign(network.node[node]["value"]) == np.sign(energy_neighbors):
        energy_change = abs(energy_neighbors)
    else:
        energy_change = -abs(energy_neighbors)

    transition_probability = 1 / (1 + np.exp(energy_change/temperature))
    if np.random.random() < transition_probability:
        network.node[node]["value"] *= -1
        #print("flipped node")

        
def get_transition_probabilities(model, degree, parameter):
    if model == "ising":
        temperature = parameter
        probs = get_trans_probs_ising(degree, temperature)
        
    if model == "sis":
        beta, gamma = parameter
        probs = get_trans_probs_SIS(degree, beta, gamma)
    
    return probs


def get_trans_probs_ising(degree, temperature):
    def logistic(x):
        return 1/(1+np.exp(x/temperature))
    states = list(itertools.product([-1,1],repeat=degree+1))
    neighbors_sums = 2*np.array([sum(state[:-1])*state[-1] for state in states])
    return np.array([[logistic(s), logistic(-s)] for s in neighbors_sums])

def get_trans_probs_SIS(degree,beta,gamma):
    states = list(itertools.product([0, 1], repeat=degree+1))
    return np.array([[1-beta*sum(state[:-1])/degree, beta*sum(state[:-1])/degree] if state[-1] == 0 else [gamma,1-gamma] for state in states])


def update_network_ising(network, timesteps, temperature, update_method="glauber"):
    for _ in range(timesteps*len(network.nodes())):
        if update_method == "glauber":
            update_random_node_glauber(network, temperature)
        else:
            update_random_node(network, temperature)

def update_node_sis(network, beta,gamma):
    node = np.random.choice(network.nodes())
#    print(network.node[node])
    
    if network.node[node]["value"]: #label = 1, infected
        transition_probability = gamma
    else: #label = 0, i.e. susceptible
        neighbors = [neighbor for neighbor in network.neighbors(node)]
        infected_neighbors = np.sum([network.node[neighbor]["value"] for neighbor in neighbors])
        transition_probability = beta*infected_neighbors/len(neighbors) #beta*I/N

    if np.random.random() < transition_probability:
 #       print("Flipped", transition_probability)
        network.node[node]["value"] = (network.node[node]["value"] +1) % 2


def update_network_sis(network, timesteps, beta, gamma):
    for _ in range(timesteps*len(network.nodes())):
            update_node_sis(network,beta,gamma)

def set_values_nodes_sis(network,p=0.5):
    nx.set_node_attributes(network, values=0, name="value")
    for node in network.nodes():
        if np.random.random() < p:
            network.node[node]["value"] = 1

def set_values_nodes_ising(network, p=0.5):
    nx.set_node_attributes(network, values=1, name="value")
    for node in network.nodes():
        if np.random.random() < p:
            network.node[node]["value"] = -1


def estimate_values_nodes(node_distributions, network, temperature, timesteps):
    update_network_ising(network, timesteps, temperature)
    for node_distribution in node_distributions:
        sample = {node_distribution.node: network.node[node_distribution.node]["value"]}
        #print("number of neighbors {}".format(len([neighbor for neighbor in network.neighbors(node_distribution.node)])))
        for neighbor in network.neighbors(node_distribution.node):
            sample[neighbor] = network.node[neighbor]["value"]

        node_distribution.add_sample(sample)


def convert_samples_to_distribution(samples, var_to_states):
    """

    :param samples: a pandas Dataframe object
    :param var_to_states: a dict
    :return: an nd-array
    """
    dimension_distribution = tuple([var_to_states[column] for column in samples.columns])
    distribution = np.zeros(dimension_distribution)
    new_samples = samples.replace(-1, 0)
    for row in new_samples.index:
        indices = tuple(int(i) for i in new_samples.loc[row, :].values)
        distribution[indices] += 1

    return distribution/float(np.sum(distribution))

def convert_samples_to_dit(samples, node, neighbors, model="ising"):
    selected_samples = samples.loc[:, neighbors+[node]]
    n_vars = len(neighbors+[node])
    new_columns = list(selected_samples.columns)
    ncols = len(new_columns)
    new_columns.remove(node)
    new_columns.append(node)
    selected_samples = selected_samples.reindex(columns=new_columns)
    uniques = selected_samples.groupby(selected_samples.columns.to_list()) \
                                    .size() \
                                    .reset_index(name="count")
    
    #Normalize the counts
    uniques["count"] = uniques["count"]/uniques["count"].sum()
    #Convert to dict
    states = {}
    if model == "ising":
        states = { state: 0 for state in itertools.product([-1,1], repeat=n_vars)}
    else:
        states = { state: 0 for state in itertools.product([0,1], repeat=n_vars)}
    for row in uniques.values:
        states[tuple(row[:-1])] = row[-1]
            
    
    d = dit.Distribution(states)
    d.set_rv_names(get_vars(d.outcome_length()))
    return d


def convert_samples_to_dist(samples, node, neighbors, number_of_states):
    """

    Note: the node is the last axis in the returned distribution

    :param samples: pandas DataFrame object
    :param node: integer
    :param neighbors: list of integers
    :param number_of_states: integer, the number of states that every node has
    :return: nd-array
    """
    selected_samples = samples.loc[:, neighbors+[node]]
    new_columns = list(selected_samples.columns)

    new_columns.remove(node)
    new_columns.append(node)
    selected_samples = selected_samples.reindex(columns=new_columns)
    return convert_samples_to_distribution(
        selected_samples, {node: number_of_states for node in neighbors+[node]}
    )


def select_nodes(network, max_neighbors):
    """
    Select one node from the network with one neighbor, one node
    with two neighbors etc. up to the max number of neighbors

    :param network: networkx Graph
    :param max_neighbors: integer
    :return: a list of ints
    """
    degree_to_nodes = {}
    for node, degree in network.degree():
        if degree in degree_to_nodes:
            degree_to_nodes[degree].append(node)
        else:
            degree_to_nodes[degree] = [node]

    selected_nodes = []
    for degree, nodes in sorted(degree_to_nodes.items(), key=lambda x: x[0]):
        if degree > max_neighbors:
            break

        selected_nodes.append(np.random.choice(nodes))

    return selected_nodes


def produce_marginal_and_conditional(joint_dist, input_labels, output_label):
    """

    :param joint_dist: nd-array
    :param input_labels: set
    :param output_label: set
    :return: tuple of nd-arrays

    """
    # TODO: REwrite this code to so that it use the dit package
    joint_probability_array = probability_distributions.ProbabilityArray(joint_dist)
    marginal_inputs = joint_probability_array.marginalize(input_labels)
    marginal_output = joint_probability_array.marginalize(output_label)
    result = joint_probability_array.find_conditional_accounting_for_zero_marginals(
        output_label, input_labels,
        itertools.repeat(marginal_output)
    )
    cond_output, mar_labels, cond_labels = result
    correct = np.allclose(
        probability_distributions.compute_joint(marginal_inputs, cond_output, cond_labels),
        joint_dist
    )
    if not correct:
        raise ValueError()

    return marginal_inputs, cond_output


if __name__ == "__main__":
    #generate the network
    number_of_nodes = 100
    network_degree = 2
    network_backup = generate_powerlaw_network(number_of_nodes, network_degree)
    network = network_backup.copy()
    set_values_nodes_ising(network)

    #inspect network
    print("number of edges {}".format(len(network.edges())))
    degree_sequence = sorted([d for n, d in network.degree()], reverse=True)
    print("different degrees {}".format(set(degree_sequence)))

    #simulate ising model and gather samples
    temperature = 2.5
    timesteps = 50
    number_of_samples = 5

    samples = pd.DataFrame(data=np.zeros((number_of_samples, len(network.nodes()))), index=range(number_of_samples), columns=network.nodes())
    start = time.time()
    for sample_number in range(number_of_samples):
        print("sample number {}".format(sample_number))
        update_network_ising(network, timesteps, temperature, update_method="glauber")
        for node in network.nodes():
            samples.at[sample_number, node] = network.node[node]["value"]

    print(samples.loc[:, [1, 2]])
    print("run time {}".format(time.time()-start))

    #pick nodes to estimate the conditional distribution given its neighbors
    max_number_of_neighbors = 10
    selected_nodes = select_nodes(network, max_number_of_neighbors)
    print("selected nodes {}".format(selected_nodes))

    #create  a distribution from the samples for the nodes with the different degrees
    for selected_node in selected_nodes:
        joint_dist = convert_samples_to_dist(
            samples, selected_node, list(network.neighbors(selected_node)), 2
        )
        output_label = set([len(joint_dist.shape) - 1])
        input_labels = set(range(len(joint_dist.shape) - 1))
        marginal, conditional = produce_marginal_and_conditional(joint_dist, input_labels, output_label)
        print(len(joint_dist.shape))


    #see how energy in the network changes
    print("positive nodes")
    print(np.sum(network.node[node]["value"] for node in network.nodes()))

    for _ in range(timesteps*len(network.nodes())):
        update_random_node(network, temperature)

    print(np.sum(network.node[node]["value"] for node in network.nodes()))

    count = 0
    samples_folder = "ising_samples"
    network = nx.read_gpickle(
        "{}/network{}_network_size{}_network_degree{}_temp{:.1f}.pkl".format(
            samples_folder, count, number_of_nodes, network_degree, temperature
        ))
    samples = pd.read_pickle("{}/samples{}_network_size{}_network_degree{}_temp{:.1f}.pkl".format(
        samples_folder, count, number_of_nodes, network_degree, temperature
    ))
    #selected_nodes = select_nodes(network, max_number_of_neighbors)
    #for selected_node in selected_nodes:
    #    print(list(network.neighbors(selected_node)))
    #    a = list(network.neighbors(selected_node))
    #    distribution = convert_samples_to_dist(samples, network, selected_node, number_of_states=2)

