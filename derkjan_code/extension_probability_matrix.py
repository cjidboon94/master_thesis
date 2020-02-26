from __future__ import print_function
import random
import numpy as np
from scipy.stats import entropy
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

        #print("the joint")
        #print(joint_distribution)
        #print("the conditional")
        #print(marginal_conditional)
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

