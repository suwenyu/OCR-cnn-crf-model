import numpy as np

def compute_probability(x, y, W, T):
    # computes sum of <W_y, Xi> + sum of T_ij
    w_sum, t_sum = 0, 0

    for i in range(len(x)-1):
        w_sum += np.dot(x[i, :], W[y[i], :]) 
        t_sum += T[y[i], y[i+1]]
    n = len(x)-1
    w_sum += np.dot(x[n, :], W[y[n], :])

    return w_sum + t_sum

def comput_prob(x, y, W, T):
    sum_val, t_sum = 0, 0

    for i in range(len(x)-1):
        sum_val += np.dot(x[i, :], W[y[i], :])
        sum_val += T[y[i], y[i+1]]

    n = len(x)-1
    sum_val += np.dot(x[n, :], W[y[n], :])

    return np.exp(sum_val)







