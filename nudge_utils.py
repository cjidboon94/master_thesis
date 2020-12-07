import numpy as np

def log_add(p, n):
    result = np.zeros(len(p), dtype=p.dtype)
    PgeN = p >= n
    # print("PgeN", PgeN, p, n)
    result[PgeN] = p[PgeN] + np.log1p(np.exp(n[PgeN] - p[PgeN]))
    result[~PgeN] = n[~PgeN] + np.log1p(np.exp(p[~PgeN] - n[~PgeN]))
    return result


def log_subtract(p, n):
    result = np.zeros(len(p), dtype=p.dtype)

    PleN = p <= n
    # print("PleN", p, n)
    n_is_inf = n == -np.inf
    other = ~PleN & ~n_is_inf

    result[PleN] = -np.inf
    result[n_is_inf] = p[n_is_inf]
    result[other] = p[other] + np.log1p(-np.exp(n[other] - p[other]))
    return result


def log_nudge(prob, nudge, s):
    new_prob = np.zeros(len(prob), dtype=prob.dtype)
    # nudge = np.log(np.abs(nudge))
    pos, neg = s > 0, s < 0
    #  print("before positive", prob[pos], nudge[pos])
    new_prob[pos] = log_add(prob[pos], nudge[pos])
    #  print("after positive", new_prob[pos], prob[pos], nudge[pos])
    #  print("before negative", prob[neg], nudge[neg])
    new_prob[neg] = log_subtract(prob[neg], nudge[neg])
    #  print("after negative", new_prob[neg], prob[neg], nudge[neg])
    new_prob[s == 0] = prob[s == 0]
    return new_prob

def generate_nudge(size: int, eps: float, p=False):
    nudge = np.zeros(size)
    if size % 2 == 0:
        nudge = 0.5 * eps * np.random.permutation(np.concatenate([np.random.dirichlet([1.] * int(0.5 * size)),
                                                                  -np.random.dirichlet([1.] * int(0.5 * size))]))
    else:
        u, v = int(np.floor(size / 2)), int(np.ceil(size / 2))
        nudge = 0.5 * eps * np.random.permutation(np.concatenate([np.random.dirichlet([1.] * u),
                                                                  -np.random.dirichlet([1.] * v)]))
    if p:
        print(nudge)
    return nudge


def generate_nudge_binomial(size: int, eps: float, p=False):
    # This generates a nudge using Rick's proposed method
    nudge = np.zeros(size)

    split_point = np.random.binomial(size, p=0.5)
    while split_point == 0 or split_point == size - 1:
        split_point = np.random.binomial(size, p=0.5)
    nudge[:split_point] = 0.5 * eps * np.random.dirichlet([1.] * split_point)
    nudge[split_point:] = -0.5 * eps * np.random.dirichlet([1.] * (size - split_point))
    np.random.shuffle(nudge)
    if p:
        print(nudge)
    return nudge


def generate_log_nudge(size: int, eps: float, binomial=False, p=False):
    if binomial:
        nudge = generate_nudge_binomial(size, eps, p)
    else:
        nudge = generate_nudge(size, eps, p)
    return np.log(np.abs(nudge)), np.sign(nudge)


def perform_nudge(X, nudge, list_of_idxs=None):
    X.make_dense()
    if list_of_idxs is None:
        X.pmf += nudge
    else:
        # this for loop is for synergistic nudges
        for idxs in list_of_idxs:
            X.pmf[idxs] += nudge
    X.pmf[X.pmf < 0] = 0
    X.normalize()


def perform_log_nudge(X, nudge, sign, list_of_idxs=None):
    X.make_dense()
    # print("performing log nudge", X.copy('linear').pmf,X.pmf, nudge, sign)
    if list_of_idxs is None:
        result = log_nudge(X.pmf, nudge, sign)
        # print("performed log nudge", np.exp(result))
        X.pmf = result

    else:
        # Synergistic nudges
        for idxs in list_of_idxs:
            X.pmf[idxs] = log_nudge(X.pmf[idxs], nudge, sign)
    X.pmf[X.pmf == np.nan] = -np.inf  # log of negative probabilities becomes nan and log of 0 becomes -inf
    X.normalize()