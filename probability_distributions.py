import numpy as np
from jointpdf.jointpdf import JointProbabilityMatrix
from jointpdf.jointpdf import FullNestedArrayOfProbabilities

class JointProbabilityMatrixExtended(JointProbabilityMatrix):
    """this class extends the JointProbabilityMatrix by allowing different
    variables to have different number of states"""

    def __init__(self, numvariables, max_num_states, joint_pdf):
        """
        
        Parameters:
        ----------
        num_variables: integer
        max_num_states: integer
        joint_pdf: a FullNestedArrayOfProbabilities object
        """

        self.numvariables = numvariables
        self.max_num_states = max_num_states
        self.joint_probabilities = joint_pdf 
        #super(JointProbabilityMatrix, self).__init__(num_variables, num_states, joint_pdf)
	
    def dist(self, joint_pdf=None):
        if joint_pdf is None:
            return self.joint_probabilities.joint_probabilities
        else:
            return joint_pdf.joint_probabilities.joint_probabilities

    def conditional_distribution(self, selected_indices, conditional_indices):
	"""create the conditional distribution for the selected_indices given 
	the conditional_indices for the joint_distribution
	
	Parameters:
	----------
	selected_indices: list of integers
	conditional_indices: list of integers
	
	Returns:
	-------
        conditional_distribution: a numpy array
        new_selected_indices: list of integers
            The axis that are NOT conditioned on
        new_conditional_indices: list of integers
            The axis of the variables that are conditioned on
	"""
        self.adjust_to_old_format()
        all_indices = sorted(selected_indices+conditional_indices)
	joint_distribution = self.dist(self.marginalize_distribution(all_indices))
	marginal_conditional = self.dist(
            self.marginalize_distribution(conditional_indices)
        )
        new_selected_indices = [i for i in range(len(all_indices)) 
                                if all_indices[i] in selected_indices]
        new_conditional_indices = [i for i in range(len(all_indices)) 
                                   if all_indices[i] in conditional_indices]

        print("the joint")
        print(joint_distribution)
        print("the conditional")
        print(marginal_conditional)
	conditional_distribution = np.copy(joint_distribution) 
	it = np.nditer(joint_distribution, flags=['multi_index'])
	while not it.finished:
	    conditional_arguments = tuple([it.multi_index[i] for i
                                           in new_conditional_indices])
            if conditional_distribution[it.multi_index] != 0:
	        conditional_distribution[it.multi_index] = (
		    conditional_distribution[it.multi_index] /
		    marginal_conditional[conditional_arguments]
	        )
	    it.iternext()
	    
    	return (conditional_distribution, new_selected_indices, 
                new_conditional_indices)

    #needed to access methods in parent class
    def adjust_to_old_format(self):
        """ensure that every variable in joint has as many states and add states as needed"""
        self.old_joint_probabilities = self.joint_probabilities
        max_number_of_states = max(self.joint_probabilities.joint_probabilities.shape)
        temp_joint_probabilities = np.zeros([max_number_of_states]*self.numvariables)
        it = np.nditer(self.joint_probabilities.joint_probabilities, flags=['multi_index'])
        while not it.finished:
            temp_joint_probabilities[it.multi_index] = it.value
            it.iternext()
            
        self.joint_probabilities.joint_probabilities = temp_joint_probabilities
        self.numvalues = max_number_of_states
    
    def append_determinstic_function(self, func, func_output_dimensions):
        """
        Append extra variables to the distribution by applying func to 
        the old statespace
         
        params:
            func: a function
                The function takes as input an iterable representing a state
                from the statespace of the probability distribution. In other
                words an index represents one item in the numpy array
                representing the probability distribution.
            func_output_dimensions: iterable, the number of outcomes of the function
        """
        old_shape = list(self.joint_probabilities.joint_probabilities.shape)
        num_variables = len(func_output_dimensions)
        new_shape = old_shape + func_output_dimensions
        dummy_joint_probability = np.zeros(new_shape)
        temp2_joint_probability = np.zeros(new_shape)
            
        it = np.nditer(dummy_joint_probability, flags=['multi_index'])
        while not it.finished:
            arguments = tuple(list(it.multi_index)[:-num_variables])
            if np.all(np.array(func(arguments)) == np.array(it.multi_index[-num_variables:])):
                temp2_joint_probability[it.multi_index] = self.joint_probabilities.joint_probabilities[arguments]
                
            it.iternext()
            
        self.joint_probabilities.joint_probabilities = temp2_joint_probability
        self.numvariables = self.numvariables + num_variables 
        
    def to_dict(self):
        """
        Write all non-zero states to a dictionary
        
        """
        probability_dict = {}
        it = np.nditer(self.joint_probabilities.joint_probabilities, flags=['multi_index'])
        while not it.finished:
            if it.value != 0:
                probability_dict[tuple(it.multi_index)]= it.value
            
            it.iternext()
            
        return probability_dict
 
class ProbabilityArray():
    """
    Represents a discrete joint probability distribution as a tree where every
    level in the tree is one variable and every path to a leaf is a state and
    the value of the leaf is the probability of that state. This is 
    implemented as a numpy array where every axis is a level in the tree 
    (a variable)."""
    def __init__(self, probability_distribution):
        """
        Parameters:
        ----------
        probability_distribution: a numpy array
            for all values x it should hold 0<=x<=1 and all values should
            sum to 1

        """
        if np.absolute(np.sum(probability_distribution)-1) > 10**-9:
            raise ValueError("probability distribution sums to {}".format(
                np.sum(probability_distribution)
            ))
        if np.any(probability_distribution < 0):
            raise ValueError("some probability is smaller than 0") 
        self.probability_distribution = probability_distribution

    def marginalize(self, variables, distribution=None):
        """
        Find the marginal distribution of the variables

        Parameters:
        ----------
        variables: a set of integers
            Every variable in variables should be smaller than the total 
            number of variables in the probability distribution

        Returns:
        -------
        A numpy array with as many axis as there were variables the
        variables have the same order as in the joint

        """
        if distribution is None:
            probability_distribution = self.probability_distribution
        else:
            probability_distribution = distribution

        marginal_distribution = np.zeros(tuple(
            [probability_distribution.shape[i] for i in variables]
        ))
        it = np.nditer(probability_distribution, flags=['multi_index'])
        while not it.finished:
            marginal_state = tuple([it.multi_index[i] for i in variables]) 
            marginal_distribution[marginal_state] += it.value
            it.iternext()

        return marginal_distribution

    def find_conditional(self, marginal_variables, conditional_variables):
	"""create the conditional distribution for the selected_indices given 
	the conditional_indices for the joint_distribution
	
	Parameters:
	----------
	selected_indices: set of integers
	conditional_indices: set of integers
	
	Returns: conditional_distribution, marginal_labels, conditional_labels 
	-------
        conditional_distribution: a numpy array
        marginal_labels: list of integers
            The variables that are NOT conditioned on
        conditional_labels: list of integers
            The variables that are conditioned on
	"""
        joint_distribution, marginal_labels, conditional_labels = (
            self.find_joint_marginal(marginal_variables, conditional_variables)
        ) 
	marginal_conditional = self.marginalize(conditional_labels,
                                                joint_distribution)
	conditional_distribution = np.copy(joint_distribution) 
	it = np.nditer(joint_distribution, flags=['multi_index'])
	while not it.finished:
            if it.value == 0:
                it.iternext()
                continue
            conditional_arguments = tuple(
                [it.multi_index[i] for i in conditional_labels]
            )
	    conditional_distribution[it.multi_index] = (
                it.value/marginal_conditional[conditional_arguments]
	    )
	    it.iternext()
	    
    	return (conditional_distribution, marginal_labels, conditional_labels)

    def find_joint_marginal(self, variables1, variables2, distribution=None):
        """
        Calculate the marginal for the combined set of variables1 and
        variables2 and adjust the variable indices

        Parameters:
        ----------
        variables1, variables2: set of integers

        Returns: joint_distribution, variable1_labels, variable2_labels
        -------
        joint_distribution: a numpy array
            Representing the marginal probability distribution for the 
            combined set of variables 1 and 2
        variable1_labels, variable2_labels: set of integers 
            The adjusted labels for variable 1 or 2 for the new joint 
            distribution

        """
        all_variables = variables1.union(variables2)
	joint_distribution = self.marginalize(all_variables, distribution)
        variable1_labels, variable2_labels = set(), set() 
        for count, variable in enumerate(sorted(list(all_variables))):
            if variable in variables1:
                variable1_labels.add(count)
            elif variable in variables2:
                variable2_labels.add(count)

        return joint_distribution, variable1_labels, variable2_labels

def compute_joint(marginal, conditional, conditional_labels):
    """compute the joint given the marginal and the conditional
    
    Parameters:
    ----------
    marginal: a numpy array
    conditional: a numpy array
    conditional_labels: a set of integers
        The length of conditional_labels must be equal to the number of 
        axis of marginal. In other words the variables of marginal should
        be equal to the variables that are conditioned on.
    """
    total_variables = len(conditional.shape)
    reordered_conditional = np.moveaxis(
        conditional, conditional_labels,
        range(total_variables-len(conditional_labels), total_variables, 1)
    )
    reordered_conditional = reordered_conditional*marginal
    joint = np.moveaxis(reordered_conditional, 
                        range(total_variables-len(conditional_labels), total_variables, 1), 
                        conditional_labels)
    return joint 

def compute_joint_uniform_random(shape):
    """
    The joint distribution is generated by (uniform) randomly picking
    a point on the simplex given by (1, 1, ..., 1) with length the
    number of states of joint. In other words by sampling from the 
    Dirichlet(1, 1, 1, ..., 1)

    Parameters:
    ----------
    shape: a tuple of integers

    Returns:
    -------
    a numpy array with dimensions given by shape

    """
    number_of_states = reduce(lambda x,y: x*y, shape)
    dirichlet_random = np.random.dirichlet([1]*number_of_states)
    return np.reshape(dirichlet_random, shape)

def compute_joint_from_independent_marginals(marginal1, marginal2, marginal_labels):
    """
    Compute the joint using the marginals assuming independendence
    
    Parameters:
    ----------
    marginal1: a numpy array
        Representing an M-dimensional probability distribution
    marginal2: a numpy array
        Representing a probability distribution, the order of the variables
        should be in the same order as in the final joint
    marginal_labels: a sorted (from small to big) list of integers
        Representing on which axis the variables of marginal2 should be
        placed in the joint

    Returns: the joint distribution

    """
    outer_product = np.outer(marginal1, marginal2)
    joint = np.reshape(outer_product, marginal1.shape+marginal2.shape)
    for count, marginal_label in enumerate(marginal_labels):
        joint = np.rollaxis(joint, 
                            len(joint.shape)-len(marginal2.shape)+count,
                            marginal_label)

    return joint

class ProbabilityDict():
    """
    Save a discrete probability distribution in a dictionary with keys
    representing states and values the probabilities
    
    """

    def __init__(self, probability_dict):
        """
        Parameters:
        ---------
        probability_dict: a dict
            The keys represent the different states (as tuples!) and 
            the values represent the probability of the state

        """
        self.probability_dict = probability_dict
        
    def print_distribution(self, sort=False):
        """
        Fancy printing method for distribution

        Parameters:
        ----------
        sort: boolean
            if sort is true the states will be printed in sorted order

        """

        if sort:
            prob_items = sorted(self.probability_dict.items(), 
                   lambda x, y: -1 if not x[0]>y[0] else 1)
        else:
            prob_items = self.probability_dict.items()
            
        for key, value in prob_items:
            print("{}: {}".format(key, value))
            
    def calculate_marginal_distribution(self, chosen_variable_indices):
        """ 
        Calculate the marginal distribution

        Parameters:
        ----------
       chosen_variable_indices: a set 
            variables for which the marginal will be calculated
        """

        marginal_distribution = {}
        for state, value in self.probability_dict.items():
            marginal_state = tuple(
                [entry for count, entry in enumerate(state) 
                 if count in chosen_variable_indices]
            )
            marginal_distribution[marginal_state] = value + marginal_distribution.get(marginal_state, 0)

        return marginal_distribution

    def calculate_entropy(self, variable_indices):
        """ 
        Calculate the entropy

        Parameters:
            variable_indices: a set
                All (variable) indices for which the entropy be calculated

        """
        distribution = self.calculate_marginal_distribution(variable_indices) 
        distribution_values_arr = np.array(distribution.values())
        distribution_values = distribution_values_arr[distribution_values_arr != 0]
        return - np.sum(np.log2(distribution_values) * distribution_values)

    def calulate_mutual_information(self, variable_indices1, variable_indices2):
        """
        Calculate the mutual information

        Parameters:
        ----------
        variable_indices1: a set
            All variable indices for the first distribution
        variable_indices2: All variable indices for second distribution

        """
        return (self.calculate_entropy(variable_indices1) +
                self.calculate_entropy(variable_indices2) -
                self.calculate_entropy(variable_indices1+variable_indices2))


