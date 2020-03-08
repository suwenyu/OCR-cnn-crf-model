import torch
import torch.nn as nn

from torch.autograd import Function
from data_loader import get_dataset


import torch.optim as optim
import torch.utils.data as data_utils
from data_loader import get_dataset
import numpy as np
from crf import CRF

import check_grad
import conv

batch_size = 10


class BadFFTFunction(Function):
    @staticmethod
    def forward(ctx, input_x, input_y, params):
        numpy_input_x, numpy_params = input_x.detach().numpy(), params.detach().numpy()
        numpy_input_y = input_y.detach().numpy()
        # print(numpy_input)
        data = [(i, j) for i, j in zip(numpy_input_y, numpy_input_x)]
        result = -1 * check_grad.compute_log_p_avg(numpy_params, data, len(data))
        # print(result)
        # print(numpy_input)
        ctx.save_for_backward(input_x, input_y, params)

        return torch.as_tensor(result, dtype=torch.float)

    @staticmethod
    def backward(ctx, grad_output):
        print('test')
        grad_output = grad_output.detach()
        input_x, input_y, params = ctx.saved_tensors

        numpy_params = params.detach().numpy()
        numpy_input_x, numpy_input_y = input_x.detach().numpy(), input_y.detach().numpy()
        data = [(i, j) for i, j in zip(numpy_input_y, numpy_input_x)]
        # print(data)
        result = check_grad.gradient_avg(numpy_params, data, len(data))
        # print(result.shape)
        # numpy_go = grad_output.numpy()

        # result = 0
        return input_x, input_y, torch.from_numpy(result).to(torch.float)


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

        self.cnn = conv.Conv()

        ### Use GPU if available
        if self.use_cuda:
            [m.cuda() for m in self.modules()]

        # self.W = torch.zeros([num_labels * input_dim])
        # self.T = torch.zeros([num_labels * num_labels])
        self.params = nn.Parameter(torch.empty(self.num_labels * self.input_dim + self.num_labels * self.num_labels,))
        # self.init_params()
        self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.params)
        # nn.init.uniform_(self.params, -0.1, 0.1)

    # def init_params(self):
    #     """
    #     Initialize trainable parameters of CRF here
    #     """
    #     # init_param = torch.zeros([self.num_labels * self.input_dim + self.num_labels * self.num_labels, ], requires_grad=True)
    #     self.params = nn.Parameter(init_param)
        # self.params = torch.zeros([26 * 128 + 26 * 26, ], dtype=torch.int32)

    def get_conv_feats(self, x):
        return self.cnn.forward(x)

    def loss(self, input_x, input_y):
        feat_x = self.get_conv_feats(input_x)
        return BadFFTFunction.apply(feat_x, input_y, self.params)

# def incorrect_fft(input):
#     return BadFFTFunction.apply(input)


# Tunable parameters
batch_size = 256
num_epochs = 10
max_iters  = 1000
print_iter = 25 # Prints results every n iterations
conv_shapes = [[1,64,128]] #


# Model parameters
input_dim = 128
embed_dim = 64
num_labels = 26
cuda = torch.cuda.is_available()

# Instantiate the CRF model
crf = CRF(input_dim, embed_dim, conv_shapes, num_labels, batch_size)

opt = optim.LBFGS(crf.parameters())
# opt = optim.SGD(crf.parameters(), lr=0.01, momentum=0.9)

dataset = get_dataset()

split = int(0.5 * len(dataset.data))

train_data, train_target = dataset.data[:split], dataset.target[:split]
train = data_utils.TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_target).long())

train_loader = data_utils.DataLoader(train,  # dataset to load from
                                     batch_size=batch_size,  # examples per batch (default: 1)
                                     shuffle=True,
                                     sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                     num_workers=5,  # subprocesses to use for sampling
                                     pin_memory=False,  # whether to return an item pinned to GPU
                                     )

for i_batch, sample in enumerate(train_loader):
    train_X = sample[0]
    train_Y = sample[1]

    # data = [(i , j) for i , j in zip(train_Y, train_X.detach().numpy())]

    # print(output)

    # res = output.backward()
    # def closure():
    def closure():
        opt.zero_grad()
        loss = crf.loss(train_X, train_Y)
        print('loss:', loss)
        loss.backward()
        return loss

    opt.step(closure)


    # opt.zero_grad()
    # loss = crf.loss(train_X, train_Y)
    
    # print('loss:', loss)
    # loss.backward()
    #     # return loss
    # opt.step()

    # opt.step(closure)
    # from torch.autograd.gradcheck import gradcheck
    # test = gradcheck(crf, (train_X, train_Y), eps=1e-6, atol=1e-4)
    # print("Are the gradients correct: ", test)
    # print(res)
    # break
