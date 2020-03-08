import compute
import numpy as np
import math

class crf:
    letter_size = 26

    def __init__(self, X, Y, W, T):
        self.X = X
        self.Y = Y
        self.W = W
        self.T = T
        self.n = len(X)

    # checked
    def forward(self):
        alpha = np.zeros((self.n, self.letter_size))
        # tmp = np.zeros(self.letter_size)
        message = np.zeros((self.letter_size, self.letter_size))

        # equation (13)
        for i in range(1, self.n):
            tmp = alpha[i - 1] + self.T.transpose()
            tmp_max = np.max(tmp, axis=1)

            tmp = (tmp.transpose() - tmp_max).transpose()

            tmp = np.exp(tmp + np.matmul(self.X[i-1, :], self.W.T))
            
            alpha[i] = tmp_max + np.log(np.sum(tmp, axis=1))
            # print(alpha[i])
            
            # for j in range(self.letter_size):
            #     for k in range(self.letter_size):
            #         message[k][j] = np.dot(self.W[k], self.X[i-1]) + self.T[k, j] + alpha[i-1, k]


            # message = np.exp(message)
            # tmp = np.sum(message, axis=0)
            # tmp = np.log(tmp)

            # not sure
            # alpha[i] = np.add(maxes, tmp)
            # 
            # alpha[i] = tmp
        # print(alpha)
        return alpha, tmp, message

    def backward(self):
        beta = np.zeros((self.n, self.letter_size))
        # tmp = np.zeros(self.letter_size)
        message = np.zeros((self.letter_size, self.letter_size))

        # print(w_x)
        for i in range(self.n-2, -1, -1):
            tmp = beta[i + 1] + self.T

            tmp_max = np.max(tmp, axis=1)
            tmp = (tmp.transpose() - tmp_max).transpose()
            # print(tmp)
            # prevent overflow
            tmp = np.exp(tmp + np.matmul(self.X[i+1, :], self.W.T))

            beta[i] = tmp_max + np.log(np.sum(tmp, axis=1))

            # for j in range(self.letter_size):
            #     for k in range(self.letter_size):
            #         message[k][j] = np.dot(self.W[k], self.X[i+1]) + self.T[k, j] + beta[i+1, k]


            # np.swapaxes(message, 0, 1)

            # message = np.exp(message)
            # tmp = np.sum(message, axis=0)
            # tmp = np.log(tmp)
            # print(tmp)
            # not sure
            # beta[i] = np.add(maxes, tmp)
            # 
            # beta[i] = tmp
            # print(tmp)
        return beta, tmp, message

    def compute_z(self, alpha):
        # equation (14)
        tmp = np.add(np.matmul(self.W, self.X[-1]), alpha[-1])
        M = np.max(tmp)
        log_z = M + math.log(np.sum(np.exp(np.add(tmp, -1*M))))
        # print(np.exp(log_z))

        return np.exp(log_z)
        # return np.sum(np.exp(alpha[-1] + np.dot(self.X, self.W.T)[-1]))

    # checked
    def compute_log_prob(self):
        sum_num = compute.comput_prob(self.X, self.Y, self.W, self.T)
        
        alpha, tmp, message = self.forward()

        # print(sum_num)
        # z = self.compute_z(alpha)
        # print(z)
        # print(sum_num-log_z)

        return np.log(sum_num / self.compute_z(alpha))


    def forward_backward_prob(self):

        alpha, tmp, message = self.forward()

        # log_z = self.cal_logz(tmp, alpha)

        beta, tmp, message = self.backward()
        # print(beta)
        return alpha, beta


def w_grad(X, Y, W, T, denom, alpha, beta):
    letter_size = 26
    n = len(X)

    # crf_model = crf(X, Y, W, T)
    # alpha, beta = crf_model.forward_backward_prob()

    grad = np.zeros((letter_size, 128))

    # tmp = np.ones((letter_size, 128))
    # print(X)

    for i in range(n):
        # print(Y[i])
        grad[Y[i]] += X[i]


        tmp = np.ones((26, 128)) * X[i]
        tmp = tmp.transpose()
        # print(tmp)

        # print(denom)
        fg_p = alpha[i] + beta[i] + np.matmul(X[i, :], W.T)
        tmp = tmp * np.exp(fg_p) / denom

        grad -= tmp.transpose()

    return grad.flatten()


def t_grad(X, Y, W, T, denom, alpha, beta):
    n = len(X)
    letter_size = 26

    grad = np.zeros((letter_size, letter_size))

    for i in range(n - 1):
        tmp = np.matmul(X[i, :], W.T)
        
        for j in range(26):
            tmp1 = np.dot(X[i+1, :], W[j, :])

            grad[j] -= np.exp(tmp + T.transpose()[j] + tmp1 + beta[i + 1][j] + alpha[i])
            

    # print(grad)
    grad /= denom

    for i in range(n - 1):
        grad[Y[i+1]][Y[i]] += 1

    
    return grad



if __name__ == "__main__":
    import utils
    W, T = utils.read_model()
    # data = utils.read_data_seq('../data/train.txt')

    # crf_model = crf(data[0][1], data[0][0], W, T)
    # crf_model.forward_backward_prob()
    # print(T.shape)

    # t_grad(data[0][1], data[0][0], W, T)



