import torch
import torch.nn as nn

from torch.autograd.gradcheck import gradcheck
from torch.autograd import Function
from data_loader import get_dataset


import torch.optim as optim
import torch.utils.data as data_utils
from data_loader import get_dataset
import numpy as np
from crf import CRF
import conv 
import convolution_2d as conv2d

import check_grad, train_crf, max_sum_solution
import conv_old, utils, optimizer

import os
os.system('clear')
os.chdir('/Users/jonathantso/Desktop/Code/uic_git/cs512/hw2/wevwev/code')

class CRFGradFunction(Function):
    @staticmethod
    def forward(ctx, input_x, input_y, weights, transition):
        numpy_input_x, numpy_input_y = input_x.detach().numpy(), input_y.detach().numpy()
        numpy_weights, numpy_transition = weights.detach().numpy(), transition.detach().numpy()
        
        # print(input_x, input_y)
        result = 0
        for i in range(len(input_x)):
            # seq_len = len(numpy_input_y[i].nonzero())-1
            # print(seq_len)
            # 
            # print(input_x[:seq_len], input_y[:seq_len])
            new_x, new_y = numpy_input_x[i][numpy_input_y[i].nonzero()], numpy_input_y[i][numpy_input_y[i].nonzero()]-1


            word = (new_y, new_x)
            dots = optimizer.computeAllDotProduct(numpy_weights, word)
            alpha, beta = optimizer.computeDP(word, numpy_weights, numpy_transition, dots)
            # p1, p2 = optimizer.computeMarginal(word, numpy_weights, numpy_transition, alpha, beta, dots)
            result += optimizer.logPYX(word, numpy_weights, numpy_transition, alpha, dots)
        result = result / len(input_x)
        # print(result)

        # print(numpy_input)
        # data = [(i, j) for i, j in zip(numpy_input_y, numpy_input_x)]

        # n = len(data)
        # result = -1 * train_crf.func(numpy_params, data, 1000)

        # result = -1 * check_grad.compute_log_p_avg(numpy_params, data, len(data))
        # print(result)
        # print(numpy_input)
        ctx.save_for_backward(input_x, input_y, weights, transition)

        return torch.as_tensor((-1)*result, dtype=torch.float)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach()

        input_x, input_y, weights, transition = ctx.saved_tensors

        numpy_weights, numpy_transition = weights.detach().numpy(), transition.detach().numpy()
        numpy_input_x, numpy_input_y = input_x.detach().numpy(), input_y.detach().numpy()
        
        batch, seq, imgSize = (numpy_input_x.shape)

        K = 26
        C = 1000
        meandw = np.zeros((K, imgSize))
        meandT = np.zeros((K, K))

        for i in range(len(input_x)):
            new_x, new_y = numpy_input_x[i][numpy_input_y[i].nonzero()], numpy_input_y[i][numpy_input_y[i].nonzero()]-1

            word = (new_y, new_x)

            dots = optimizer.computeAllDotProduct(numpy_weights, word)
            alpha, beta = optimizer.computeDP(word, numpy_weights, numpy_transition, dots)
            p1, p2 = optimizer.computeMarginal(word, numpy_weights, numpy_transition, alpha, beta, dots)
            

            dw = optimizer.computeGradientWy(word, p1)
            dT = optimizer.computeGradientTij(word, p2)

            meandw += dw
            meandT += dT

        meandw /= len(input_x)
        meandT /= len(input_x)

        meandw *= (-C)
        meandT *= (-C)

        meandw += numpy_weights
        meandT += numpy_transition

        # gradients = np.concatenate((meandw.flatten(), meandT.flatten()))
        # data = [(i, j) for i, j in zip(numpy_input_y, numpy_input_x)]
        
        # result = train_crf.func_prime(numpy_params, data, 1000)
        # print(data)
        # result = check_grad.gradient_avg(numpy_params, data, len(data))
        # print(result.shape)
        # numpy_go = grad_output.numpy()

        # result = 0
        return input_x, input_y, torch.from_numpy(meandw).to(torch.float), torch.from_numpy(meandT).to(torch.float)


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

        # self.cnn = conv_old.Conv(kernel_size=(3, 3))
        self.cnn = conv.Conv(kernel_size=(3,3))

        ### Use GPU if available
        if self.use_cuda:
            [m.cuda() for m in self.modules()]

        # self.W = torch.zeros([num_labels * input_dim])
        # self.T = torch.zeros([num_labels * num_labels])
        self.params = nn.Parameter(torch.empty(self.num_labels * self.input_dim + self.num_labels * self.num_labels,))
        
        # self.init_weights()
        
        params = utils.load_model_params('../data/model.txt')
        W = utils.extract_w(params)
        T = utils.extract_t(params)
        self.weights = nn.Parameter(torch.tensor(W, dtype=torch.float))
        self.transition = nn.Parameter(torch.tensor(T, dtype=torch.float))

        # self.weights = nn.Parameter(torch.empty((self.num_labels, self.input_dim) ))
        # self.transition = nn.Parameter(torch.empty((self.num_labels, self.num_labels) ))
        # self.init_params()

    def init_weights(self):
        nn.init.zeros_(self.params)

        # nn.init.uniform_(self.params, -0.1, 0.1)

    def init_params(self):
        """
        Initialize trainable parameters of CRF here
        """
        # init_param = torch.zeros([self.num_labels * self.input_dim + self.num_labels * self.num_labels, ], requires_grad=True)
        nn.init.zeros_(self.weights)
        nn.init.zeros_(self.transition)
    
        # nn.init.uniform_(self.weights, -0.1, 0.1)
        # nn.init.uniform_(self.transition, -0.1, 0.1)

    def _comput_prob(self, x, y):

        sum_val = torch.tensor(0.0, dtype=torch.float)

        for i in range(len(x)-1):
            sum_val += torch.dot(x[i, :], self.weights[y[i], :])
            sum_val += self.transition[y[i], y[i+1]]

        n = len(x)-1
        sum_val += torch.dot(x[n, :], self.weights[y[n], :])
        
        return torch.exp(sum_val)

    def _forward_algorithm(self, x, y):
        # alpha = np.zeros((self.n, self.letter_size))
        # print(y)
        alpha = torch.zeros((len(x), self.num_labels ))
        for i in range(1, len(x)):
            tmp = alpha[i - 1] + self.transition.t()
            tmp_max = torch.max(tmp, 0)[0]
            
            tmp = (tmp.t() - tmp_max).t()
            tmp = torch.exp(tmp + torch.matmul(x[i-1, :], self.weights.t()))
            alpha[i] = tmp_max + torch.log(torch.sum(tmp, 1))

        return alpha

    def _compute_z(self, x, alpha):
        # print(x.shape, self.weights.t().shape)
        # print(alpha[seq_len], torch.mm(x, self.weights.t())[-1])
        tmp = torch.matmul(self.weights, x[-1]) + alpha[-1]
        M = torch.max(tmp)

        log_z = M + torch.log(torch.sum(torch.exp(tmp + (-1)*M )))
        # print(log_z)
        # M = torch.max(tmp)
        # print(log_z)
        return torch.exp(log_z)
        # return torch.sum(torch.exp(alpha[-1] + torch.mm(x, self.weights.t())[-1]))
        # return np.sum(np.exp(alpha[-1] + np.dot(self.X, self.W.T)[-1]))

    def _compute_log_prob(self, input_x, input_y):
        seq_len = len(input_y.nonzero())-1
        # 
        # print(input_x[:seq_len], input_y[:seq_len])
        new_x, new_y = input_x[:seq_len], input_y[:seq_len]-1
        # new_x, new_y = input_x, input_y
        # print(new_x, new_y, input_y[:seq_len])

        sum_num = self._comput_prob(new_x, new_y)
        alpha = self._forward_algorithm(new_x, new_y)

        if sum_num == float("Inf"):
            print("break here1")
        if self._compute_z(new_x, alpha) == float("Inf"):
            print("break here2")

        # print(sum_num, self._compute_z(new_x, alpha))
        return torch.log(sum_num / self._compute_z(new_x, alpha))
    
    def get_conv_feats(self, x):
        return self.cnn.forward(x)

    def loss(self, input_x, input_y):
        # seq_len = len(input_y.nonzero())-1

        feat_x = input_x
        # feat_x = self.get_conv_feats(input_x)
        return CRFGradFunction.apply(feat_x, input_y, self.weights, self.transition)
        
        # total = torch.tensor(0.0, dtype=torch.float)
        # for i in range(len(feat_x)):
        #     total += self._compute_log_prob(feat_x[i], input_y[i])
            # print(self._compute_log_prob(input_x[i], input_y[i]))
        # print(total)

        return (-1) * (total/self.batch_size)

    def forward(self, input_x):
        numpy_input_x = input_x.detach().numpy()
        numpy_weights = self.weights.detach().numpy()
        numpy_transition = self.transition.detach().numpy()
        
        result = []
        for x in numpy_input_x:
            result.append(max_sum_solution.max_sum(x, numpy_weights, numpy_transition))
        result = np.array(result)
        
        return torch.from_numpy(result)


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

# opt = optim.LBFGS(crf.parameters())
opt = optim.SGD(crf.parameters(), lr=0.01, momentum=0.9)



##################################################
# Begin training
##################################################
step = 0

# Fetch dataset
dataset = get_dataset()

split = int(0.5 * len(dataset.data)) # train-test split
train_data, test_data = dataset.data[:split], dataset.data[split:]
train_target, test_target = dataset.target[:split], dataset.target[split:]

# Convert dataset into torch tensors
train = data_utils.TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_target).long())
test = data_utils.TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_target).long())


for i in range(num_epochs):
    print("Processing epoch {}".format(i))

    # Define train and test loaders
    train_loader = data_utils.DataLoader(train,  # dataset to load from
                                         batch_size=batch_size,  # examples per batch (default: 1)
                                         shuffle=True,
                                         sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                         num_workers=5,  # subprocesses to use for sampling
                                         pin_memory=False,  # whether to return an item pinned to GPU
                                         )

    test_loader = data_utils.DataLoader(test,  # dataset to load from
                                        batch_size=batch_size,  # examples per batch (default: 1)
                                        shuffle=False,
                                        sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                        num_workers=5,  # subprocesses to use for sampling
                                        pin_memory=False,  # whether to return an item pinned to GPU
                                        )
    print('Loaded dataset... ')

    # Now start training
    for i_batch, sample in enumerate(train_loader):

        train_X = sample[0]
        train_Y = sample[1]

        if cuda:
            train_X = train_X.cuda()
            train_Y = train_Y.cuda()

        # compute loss, grads, updates:
        # opt.zero_grad() # clear the gradients
        # tr_loss = crf.loss(train_X, train_Y) # Obtain the loss for the optimizer to minimize
        # print(tr_loss.data)
        # tr_loss.backward() # Run backward pass and accumulate gradients
        # opt.step() # Perform optimization step (weight updates)

        def closure():
            opt.zero_grad()
            tr_loss = crf.loss(train_X, train_Y)
            print('loss:', tr_loss)
            tr_loss.backward()
            return tr_loss

        opt.step(closure)

        # opt.zero_grad()
        # loss = crf.loss(train_X, train_Y)

        # print('loss:', loss)
        # loss.backward()
        # opt.step()

        # print to stdout occasionally:
        if step % print_iter == 0:
            random_ixs = np.random.choice(test_data.shape[0], batch_size, replace=False)
            test_X = test_data[random_ixs, :]
            test_Y = test_target[random_ixs, :]

            # Convert to torch
            test_X = torch.from_numpy(test_X).float()
            test_Y = torch.from_numpy(test_Y).long()

            if cuda:
                test_X = test_X.cuda()
                test_Y = test_Y.cuda()
            test_loss = crf.loss(test_X, test_Y)
            pred = crf.forward(test_X)
            # print(step, tr_loss.data, test_loss.data,
            #            tr_loss.data / batch_size, test_loss.data / batch_size)

            word_acc, letter_acc = train_crf.word_letter_accuracy(pred, test_Y)
            print("Letter Accuracy: %f, Word Accuracy: %f" % (letter_acc, word_acc) )
            print(step, test_loss.data, test_loss.data / batch_size)
            ##################################################################
            # IMPLEMENT WORD-WISE AND LETTER-WISE ACCURACY HERE
            ##################################################################

        #     # print(blah)

        step += 1
        if step > max_iters: raise StopIteration
del train, test






data = utils.read_data_seq('../data/train_mini.txt')
for i in data:
    # print(i)
    train_X = torch.tensor(i[1]).float().unsqueeze(0)
    train_Y = torch.tensor(i[0]).long().unsqueeze(0)

    # print(train_Y.shape, train_X.shape)

    def closure():
        opt.zero_grad()
        loss = crf.loss(train_X, train_Y)
        print('loss:', loss)
        loss.backward()
        return loss

    opt.step(closure)

