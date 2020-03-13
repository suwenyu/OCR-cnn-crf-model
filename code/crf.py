import torch
import torch.nn as nn

import conv 

import numpy as np
import utils, max_sum_solution

def logTrick(numbers):
    if len(numbers.shape) == 1:
        M = torch.max(numbers)
        return M + torch.log(torch.sum(torch.exp(numbers - M)))
    else:
        M = torch.max(numbers, 1)[0]
        return M + torch.log(torch.sum(torch.exp((numbers.t() - M).t()), 1))

class CRF(nn.Module):

    def __init__(self, input_dim, embed_dim, conv_layers, num_labels, batch_size):
        """
        Linear chain CRF as in Assignment 2
        """
        super(CRF, self).__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.conv_layers = conv_layers
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()

        self.cnn = conv.Conv(kernel_size=(3, 3), padding = False, stride = 1)
        ### Use GPU if available
        if self.use_cuda:
            [m.cuda() for m in self.modules()]


        params = utils.load_model_params('../data/model.txt')
        W = utils.extract_w(params)
        T = utils.extract_t(params)
        
        self.weights = nn.Parameter(torch.tensor(W, dtype=torch.float))
        self.transition = nn.Parameter(torch.tensor(T, dtype=torch.float))

        # self.W = torch.zeros([num_labels * input_dim])
        # self.T = torch.zeros([num_labels * num_labels])
        # self.init_params()

    def init_params(self):
        """
        Initialize trainable parameters of CRF here
        """
        nn.init.zeros_(self.weights)
        nn.init.zeros_(self.transition)
        # self.params = torch.zeros([26 * 128 + 26 * 26, ], dtype=torch.int32)

    def _computeAllDotProduct(self, x):
        dots = torch.matmul(self.weights, x.t())
        return dots

    def _forward_algorithm(self, x, y, dots):
        # alpha = np.zeros((self.n, self.letter_size))
        # print(y)
        m = len(y)
        alpha = torch.zeros((m, self.num_labels ))
        
        if self.use_cuda:
            alpha = alpha.cuda()

        for i in range(1, m):
            alpha[i] = logTrick((dots[:, i - 1] + alpha[i - 1, :]).repeat(self.num_labels, 1) + self.transition.t())
            # print(alpha[i])
        return alpha

    def _logPYX(self, x, y, alpha, dots):
        m = len(y)

        res = sum([dots[y[i], i] for i in range(m)]) + sum([self.transition[y[i], y[i + 1]] for i in range(m - 1)])
        logZ = logTrick(dots[:, m - 1] + alpha[m - 1, :])
        res -= logZ

        return res

    def _compute_log_prob(self, input_x, input_y):
        seq_len = len(input_y.nonzero())-1
        # 
        # print(input_x[:seq_len], input_y[:seq_len])
        new_x, new_y = input_x[:seq_len], input_y[:seq_len]-1
        # new_x, new_y = input_x, input_y
        # print(new_x, new_y, input_y[:seq_len])
        dots = self._computeAllDotProduct(new_x)
        alpha = self._forward_algorithm(new_x, new_y, dots)

        sum_num = self._logPYX(new_x, new_y, alpha, dots)
        # if sum_num == float("Inf"):
        #     print("break here1")
        # if self._compute_z(new_x, alpha) == float("Inf"):
        #     print("break here2")

        # print(sum_num, self._compute_z(new_x, alpha))
        return sum_num


    def forward(self, input_x):

        feat_x = self.get_conv_features(input_x)

        if self.use_cuda:
            numpy_feat_x = feat_x.cpu().detach().numpy()
            numpy_weights = self.weights.cpu().detach().numpy()
            numpy_transition = self.transition.cpu().detach().numpy()
        else:
            numpy_feat_x = feat_x.detach().numpy()
            numpy_weights = self.weights.detach().numpy()
            numpy_transition = self.transition.detach().numpy()

        
        result = []
        for x in numpy_feat_x:
            result.append(max_sum_solution.max_sum(x, numpy_weights, numpy_transition))
        result = np.array(result)
        
        return torch.from_numpy(result)


    def loss(self, input_x, input_y):
        # seq_len = len(input_y.nonzero())-1

        feat_x = input_x
        feat_x = self.get_conv_features(input_x)
        # print(feat_x)
        # feat_x = input_x
        # feat_x = self.get_conv_feats(input_x)
        # return CRFGradFunction.apply(feat_x, input_y, self.weights, self.transition)
        
        total = torch.tensor(0.0, dtype=torch.float)

        if self.use_cuda:
            total = total.cuda()

        for i in range(len(input_y)):
            total += self._compute_log_prob(feat_x[i], input_y[i])
            # print(self._compute_log_prob(input_x[i], input_y[i]))

        return (-1) * (total/self.batch_size)

    # def backward(self):
        """
        Return the gradient of the CRF layer
        :return:
        """

        # gradient = blah
        # return gradient

    def get_conv_features(self, X):
        """
        Generate convolution features for a given word
        """
        # return self.cnn.forward_pkg(X)
        return self.cnn.forward(X)
        # return convfeatures
