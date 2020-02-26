def get_vars(n_vars: int):
    if n_vars < 1:
        raise ValueError("n_vars should be greater equal to greater than 1")
    elif n_vars <= 52:
        return string.ascii_letters[:n_vars]
    elif n_vars <= 100:
        return string.printable[:n_vars]
    else:
        raise ValueError("n_vars cannot be greater than 100")

def get_labels(n_vars: int, n_states: int):
    if n_states < 1 or n_states > 10:
        raise ValueError("states should be greater than 0 and  less than or equal to 10")
    return [''.join(i) for i in itertools.product(string.digits[:n_states], repeat=n_vars)]


def generate_distribution(n_vars: int, n_states: int, entropy_level: float):
    var_names = get_vars(n_vars)
    state_labels = get_labels(n_vars, n_states)
    pmf = sampler.sample(n_states**n_vars, level=entropy_level) # get a pmf from the distribution sampler
    print(state_labels)
    print(pmf)
    print(var_names)
    d = dit.Distribution(state_labels, pmf=pmf, sparse=False)
    d.set_rv_names(var_names)
    return d