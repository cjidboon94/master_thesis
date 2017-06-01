from functools import reduce

def compute_joint(multi_index, marginal_distributions):
    return reduce(lambda x, y: x*y, [marginal_distributions[combination] for
            combination in zip(range(len(multi_index)), multi_index)])

def compute_joint_from_marginals(marginal_distributions):
    dimension_joint_distribution = [len(dist) for dist in marginal_distributions]
    full_distribution_temp = np.zeros(dimension_joint_distribution)
    full_distribution = np.zeros(dimension_joint_distribution)

    it = np.nditer(full_distribution, flags=['multi_index'], op_flags=[['readwrite']])
    while not it.finished:
        full_distribution[it.multi_index]= compute_joint(it.multi_index, marginal_distributions)
        it.iternext()

    return full_distribution

def copy(it):
    """return the iterable"""
    return it

def xor(it):
    """apply xor to binary numbers x1 and x2"""
    return 0 if it[0]==it[1] else 1

def _and(it):
    """calculate the and of it
    
    params:
        it: binary iterable of length 2
        
    returns: a binary number 
    """
    return 1 if it[0]==1 and it[1]==1 else 0
    
class subset_and():
    def __init__(self, subsets):
        self.subsets = subsets

    def __call__(self, it):
        """take and over every subset provided in subsets 

        params:
            it: iterable with the same length as all tuples in subsets combined
            subsets: iterable of tuples every tuple should have at least length 2
        """
        entries = np.zeros(len(self.subsets))
        arr = np.array(it)
        for count, subset in enumerate(self.subsets):
            entries[count] = np.all(arr.take(subset)==1)

        return [1 if entry else 0 for entry in entries]

def parity(it):
    """calculate the parity (sum(it)%2) of the iterable
    
    params:
        it: a binary iterable
        
    returns: a binary number 
    """
    
    return sum(it)%2
    
def double_xor(it):
    """apply xor to every pair in it and return the outcome
    
    params:
        it: 1-d iterable with binary entries and len(it)%2==0
    """

    return [xor(it[2*i:2*i+2]) for i in range(len(it)/2)]

