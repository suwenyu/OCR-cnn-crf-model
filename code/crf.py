import torch
import torch.nn as nn

import numpy as np
import train_crf, check_grad, utils


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

        ### Use GPU if available
        if self.use_cuda:
            [m.cuda() for m in self.modules()]

        # self.W = torch.zeros([num_labels * input_dim])
        # self.T = torch.zeros([num_labels * num_labels])
        self.init_params()

    def init_params(self):
        """
        Initialize trainable parameters of CRF here
        """
        init_param = torch.zeros([self.num_labels * self.input_dim + self.num_labels * self.num_labels, ])
        self.params = nn.Parameter(init_param)
        # self.params = torch.zeros([26 * 128 + 26 * 26, ], dtype=torch.int32)


    def _comput_prob(self, x, y, W, T):
        sum_val, t_sum = torch.tensor(0.), torch.tensor(0.)

        for i in range(len(x)-1):
            print(x[i, :].shape, W[y[i], :].shape)
        #     sum_val += torch.mm(x[i, :], W[y[i], :])
        #     sum_val += T[y[i], y[i+1]]

        # n = len(x)-1
        # sum_val += torch.mm(x[n, :], W[y[n], :])

        return torch.exp(sum_val)


    def _compute_log_prob(self, x, y, w, t):
        sum_num = self._comput_prob(x, y, w, t)
        print(sum_num)
        # alpha, tmp, message = self.forward()

        # return np.log(sum_num / self.compute_z(alpha))
        return sum_num


    def forward(self, X):
        """
        Implement the objective of CRF here.
        The input (features) to the CRF module should be convolution features.
        """
        # features = X
        # print(self.params)
        # features = self.get_conv_feats(X)
        # prediction = train_crf()
        
        # Find the best path, given the features.
        # using max-sum alg from assign1
        return (prediction)

    def loss(self, X, labels):
        """
        Compute the negative conditional log-likelihood of a labelling given a sequence.
        """

        w = utils.extract_w(self.params)
        t = utils.extract_t(self.params)

        # data = [(i, j) for i, j in zip(labels, X)]

        n = len(X)
        
        # total = torch.tensor(0.)
        scores = torch.zeros(self.batch_size, requires_grad=True)

        for i in range(n):
            scores[i] = self._compute_log_prob(X[i], labels[i], w, t)

        print(scores)
        # loss = train_crf.func(self.params.data.numpy(), data, 10)
        
        # loss = torch.tensor(loss)
        # self.X = X
        # print(total)

        # compute_log_p_avg
        return -1 * torch.sum(scores)

    def backward(self):
        """
        Return the gradient of the CRF layer
        :return:
        """

        gradient = blah
        return gradient

    def get_conv_features(self, X):
        """
        Generate convolution features for a given word
        """
        convfeatures = blah
        return convfeatures
