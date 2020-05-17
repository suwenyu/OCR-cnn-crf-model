import torch
import torch.optim as optim
import torch.utils.data as data_utils
from data_loader import get_dataset
import numpy as np
from crf import CRF

import time
import train_crf

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Tunable parameters
batch_size = 64
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

crf = crf.to(device)
# Setup the optimizer
# opt = optim.LBFGS(crf.parameters())
opt = optim.Adam(crf.parameters(), lr=0.01)

##################################################
# Begin training
##################################################
step = 1

# Fetch dataset
dataset = get_dataset()

split = int(0.5 * len(dataset.data)) # train-test split
train_data, test_data = dataset.data[:split], dataset.data[split:]
train_target, test_target = dataset.target[:split], dataset.target[split:]

# Convert dataset into torch tensors
train = data_utils.TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_target).long())
test = data_utils.TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_target).long())


for i in range(num_epochs):
    start_time = time.time()
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
        opt.zero_grad() # clear the gradients
        tr_loss = crf.loss(train_X, train_Y) # Obtain the loss for the optimizer to minimize
        tr_loss.backward() # Run backward pass and accumulate gradients
        opt.step() # Perform optimization step (weight updates)

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
            if cuda:
                pred = pred.cpu()


            random_ixs = np.random.choice(train_data.shape[0], batch_size, replace=False)
            train_X = train_data[random_ixs, :]
            train_Y = train_target[random_ixs, :]

            train_X = torch.from_numpy(train_X).float()
            train_Y = torch.from_numpy(train_Y).long()

            if cuda:
                train_X = train_X.cuda()
                train_Y = train_Y.cuda()
            tr_loss = crf.loss(train_X, train_Y)
            
            tr_pred = crf.forward(train_X)
            if cuda:
                tr_pred = tr_pred.cpu()

            print(step, tr_loss.data, test_loss.data,
                       tr_loss.data / batch_size, test_loss.data / batch_size)

            ##################################################################
            # IMPLEMENT WORD-WISE AND LETTER-WISE ACCURACY HERE
            ##################################################################
            print("============================")
            word_acc, letter_acc = train_crf.word_letter_accuracy(tr_pred, train_Y)
            print("Train Letter Accuracy: %f, Word Accuracy: %f" % (letter_acc, word_acc) )

            word_acc, letter_acc = train_crf.word_letter_accuracy(pred, test_Y)
            print("Test Letter Accuracy: %f, Word Accuracy: %f" % (letter_acc, word_acc) )
            # print(blah)

            print("time: %.2f seconds." % (time.time() - start_time))
            # print(blah)
            print("============================")

        step += 1
        if step > max_iters: raise StopIteration
del train, test
