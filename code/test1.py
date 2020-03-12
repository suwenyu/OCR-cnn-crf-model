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

# import check_grad, train_crf,
import train_crf
import conv_old, utils, optimizer
import max_sum_solution
# import os
# os.system('clear')
# os.chdir('/Users/jonathantso/Desktop/Code/uic_git/cs512/hw2/wevwev/code')

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

        # self.cnn = conv_old.Conv(kernel_size=(3, 3))
        self.cnn = conv.Conv(kernel_size=(3,3), padding = 1)

        ### Use GPU if available
        if self.use_cuda:
            [m.cuda() for m in self.modules()]

        # self.W = torch.zeros([num_labels * input_dim])
        # self.T = torch.zeros([num_labels * num_labels])
        # self.params = nn.Parameter(torch.empty(self.num_labels * self.input_dim + self.num_labels * self.num_labels,))
        
        # self.init_weights()
        
        params = utils.load_model_params('../data/model.txt')
        W = utils.extract_w(params)
        T = utils.extract_t(params)
        
        self.weights = nn.Parameter(torch.empty((self.num_labels, self.input_dim) ))
        self.transition = nn.Parameter(torch.empty((self.num_labels, self.num_labels) ))
        # self.init_params()

    def init_weights(self):
        nn.init.zeros_(self.params)

        # nn.init.uniform_(self.params, -0.1, 0.1)

    def init_params(self):
        """
        Initialize trainable parameters of CRF here
        """
        # init_param = torch.zeros([self.num_labels * self.input_dim + self.num_labels * self.num_labels, ], requires_grad=True)
        # self.weights = nn.Parameter(torch.empty(self.num_labels, self.input_dim, ))
        # self.transition = nn.Parameter(torch.empty(self.num_labels, self.num_labels, ))

        nn.init.zeros_(self.weights)
        nn.init.zeros_(self.transition)
    
        # nn.init.uniform_(self.weights, -0.1, 0.1)
        # nn.init.uniform_(self.transition, -0.1, 0.1)

    def _computeAllDotProduct(self, x):
        dots = torch.matmul(self.weights, x.t())
        return dots

    def _forward_algorithm(self, x, y, dots):
        # alpha = np.zeros((self.n, self.letter_size))
        # print(y)
        m = len(y)
        alpha = torch.zeros((m, self.num_labels ))

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
    
    def get_conv_feats(self, x):
        return self.cnn.forward_pkg(x)

    def loss(self, input_x, input_y):
        # seq_len = len(input_y.nonzero())-1

        # feat_x = input_x
        feat_x = self.get_conv_feats(input_x)
        # return CRFGradFunction.apply(feat_x, input_y, self.weights, self.transition)
        
        total = torch.tensor(0.0, dtype=torch.float)
        for i in range(len(input_y)):
            total += self._compute_log_prob(feat_x[i], input_y[i])
            # print(self._compute_log_prob(input_x[i], input_y[i]))
        # print(total)

        return (-1) * (total/self.batch_size)

    def forward(self, input_x):
        feat_x = self.get_conv_feats(input_x)

        numpy_feat_x = feat_x.detach().numpy()
        numpy_weights = self.weights.detach().numpy()
        numpy_transition = self.transition.detach().numpy()
        
        result = []
        for x in numpy_feat_x:
            result.append(max_sum_solution.max_sum(x, numpy_weights, numpy_transition))
        result = np.array(result)
        
        return torch.from_numpy(result)


# Tunable parameters
batch_size = 256
num_epochs = 10
max_iters  = 1000
print_iter = 5 # Prints results every n iterations
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
            
            # for i, j in zip(pred, test_Y):
            #     print(i)
            #     print(j)

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



