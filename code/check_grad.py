import numpy as np
import crf_comp, utils, compute, train_crf
import time

from scipy.optimize import check_grad

def compute_log_p_avg(params, data, iterator):
    total = 0
    W = utils.extract_w(params)
    T = utils.extract_t(params)
    # T = T.transpose()
    # T_prime = utils.matricize_Tij(params)
    # T_prime = np.swapaxes(T_prime, 0, 1)
    # print(np.sum(T), np.sum(T_prime))
    # print(np.array_equal(T, T_prime))
    # print(np.sum(T))

    for i in range(iterator):
        crf_model = crf_comp.crf(data[i][1], data[i][0], W, T)
        total += crf_model.compute_log_prob()
        # print(crf_model.compute_log_prob())

    # print(total)
    return total / (iterator)


def grad_word(data, W, T, i):
    crf_model = crf_comp.crf(data[i][1], data[i][0], W, T)
    alpha, tmp, message = crf_model.forward()
    beta, tmp, message = crf_model.backward()

    denom = crf_model.compute_z(alpha)
    # print(denom)
    # print(beta)

    w_grad = crf_comp.w_grad(data[i][1], data[i][0], W, T, denom, alpha, beta)
    t_grad = crf_comp.t_grad(data[i][1], data[i][0], W, T, denom, alpha, beta)
    # for i in w_grad.flatten():
    #     print(i)
    # print(w_grad.shape, t_grad.shape)
    # print(np.sum( w_grad ))

    return np.concatenate((w_grad.flatten(), t_grad.flatten()))


def gradient_avg(params, data, iterator):
    total = np.zeros(128 * 26 + 26 * 26)
    W = utils.extract_w(params)
    T = utils.extract_t(params)
    # T = utils.matricize_Tij(params)
    # T = np.swapaxes(T, 0, 1)

    for i in range(iterator):
        total += grad_word(data, W, T, i)
    
    # for i in total:
    #     print(i)
    # print(total)
    return total / (iterator)


def check_gradient(data, params):
    # print(gradient_avg(W, T, data, 1))
    grad_value = check_grad(compute_log_p_avg, gradient_avg, params, data, 10)
    print("Gradient Value:", grad_value)


def grad_measurement(data, params):
    start = time.time()
    grad_avg = gradient_avg(params, data, len(data))
    
    # print("log_p_avg: ", compute_log_p_avg(params, data, len(data)))
    print("Computation Time: ", time.time()-start)
    #print(np.array(grad_avg).shape)

    # write to gradient.txt
    f = open('../result/gradient.txt', 'w')
    for index in grad_avg:
        #print(index)
        f.write(str(index)+"\n")


if __name__ == "__main__":

    data = utils.read_data_seq('../data/train_mini.txt')
    params = utils.load_model_params('../data/model.txt')
    # print(params)

    # crf_model = crf.crf(X, Y, W, T)
    print("check gradient ... ")
    check_gradient(data, params)

    print("write gradient to file...")
    grad_measurement(data, params)
    print('completed in ../result/gradient.txt')

    
