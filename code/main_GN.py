import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms
import googlenet,utils
from torchvision.transforms import ToTensor
from data_loader import get_dataset, get_onehot
import cv2
import matplotlib.pyplot as mp
import numpy as np
import scipy.ndimage.interpolation


print('Loaded dataset... ')

dataset = get_onehot()

split = int(0.5 * len(dataset.data))

dataset.data = dataset.data.reshape(len(dataset.data), 14, 8, 16)


print(dataset.data.shape)
def resize_batch(image_batch, new_width, new_height):
    image_batch = np.asarray(image_batch)
    shape = list(image_batch.shape)
    print(shape)
    shape[2] = new_width
    shape[3] = new_height
    ind = np.indices(shape, dtype=float)
    ind[2] *= (image_batch.shape[2] - 1) / float(new_width - 1)
    ind[3] *= (image_batch.shape[3] - 1) / float(new_height - 1)

    return scipy.ndimage.interpolation.map_coordinates(image_batch, ind, order=1)

dataset.data = resize_batch( dataset.data, 32,32)
print(dataset.data.shape)

# train_data, train_target = dataset.data[:split], dataset.target[:split]
train_data, test_data = dataset.data[:split], dataset.data[split:]
train_target, test_target = dataset.target[:split], dataset.target[split:]
batch_size =256
print(train_data.shape, test_data.shape)


train = data_utils.TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_target).long())
test = data_utils.TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_target).long())



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
print('=> Building model..')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = googlenet.GoogLeNet()
net = net.to(device)
if device =='cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
#optimizer
optimizer = optim.LBFGS(net.parameters())
# opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
#optimizer = optim.SGD(net.parameters(), lr = 0.1, momentum =0.9, weight_decay = 5e-4)
#optimizer = optim.Adam(net.parameters(), lr = 0.1, eps = 1e-08)
Err_train = []
Acc_train = []
Err_test =[]
Acc_test=[]

def modify_data(data, seq_len):
    batch, seq, img_width, img_len = data.shape
    data = data.view(seq, batch, img_width, img_len)
    print(data.shape)
    new = torch.empty(seq_len, batch, img_width, img_len) #14*256*1*32*32
    for i in range(seq_len):
        new[i] = data[i]
    new = new.view(seq_len, batch, 1, img_width, img_len)

    #resized = torch.empty(seq_len, batch, 1, img_width, img_len) #initial new data with channal 1
    #for i in range(seq_len):
    #    resized[i] = new_data[i].reshape(1, batch, 1, img_width, img_len)
    #resized = resized.view(
    print(new.shape)
    return new



#Training
def train(epoch):
    print('Epoch', epoch)
    net.train()
    train_loss = 0
    epoch_letter_correct = 0
    epoch_letter_total = 0
    epoch_word_correct=0
    epoch_word_total = 0

    for index, data in enumerate(train_loader):
        print("index", index)
        trainX, trainY = data[0], data[1] #256*14*32*32, 256*14*26
        trainX, trainY = trainX.to(device), trainY.to(device)
        seq_len = 0
        batch_letter_total=0
        batch_letter_correct =0
        batch_word_correct = 0

        batch, seq, img_len = trainY.shape

        for i in trainY:
            print(i.shape)
            max_val, _ = torch.max(i, dim = 1, keepdim = True)
            #print('max shape',max_val.shape) #14*1
            max_val = max_val.view(1, max_val.shape[0])
            #print(max_val.shape)
            #print(max_val)
            seq_len = len((max_val!=0).nonzero()) #how many letters in a word


            features = modify_data(trainX, seq_len)
            labels = trainY.view(batch, seq, img_len, 1)
            labels = modify_data(labels, seq_len)
            labels = labels.view(seq_len, batch, img_len)
            
            print("features, labels shape: ", features.shape, labels.shape)

            letter_correct = 0
            for i in range(seq_len): 
                optimizer.zero_grad()
                outputs = net(features[i])
                targets = torch.max(labels[i], 1)[1]
                print(outputs.shape, targets.shape)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()

                train_loss += loss.item() #not sure
                _, predicted = outputs.max(1)
                #print(predicted)
                #print(predicted)
                letter_correct += predicted.eq(targets)。#not sure
                #letter_correct += (predicted==targets).sum().item() #correct letter in a word

            print("word: {}/{}".format(letter_correct, seq_len))
            batch_letter_correct += letter_correct
            batch_letter_total += seq_len
            if(letter_correct == seq_len):
                bactch_word_correct +=1

        print("Batch letter acc: {:3f} ({}/{})".format((batch_letter_correct/batch_letter_total), batch_letter_correct, batch_letter_total))
        print("Batch word acc: {:3f} ({}/{})".format((batch_word_correct/batch_word_total), batch_word_correct, batch_word_total))
        print("-----")
    epoch_word_total += batch_word_total
    epoch_word_correct += batch_word_correct
    epoch_letter_total += batch_letter_total
    epoch_letter_correct += batch_letter_correct
    print("Epoch letter acc: {:3f} ({}/{})".format((epoch_letter_correct/epoch_letter_total), epoch_letter_correct, epoch_letter_total))
    print("Epoch word acc: {:3f} ({}/{})".format((epoch_word_correct/epoch_word_total), epoch_word_correct, epoch_word_total))
    print("Loss {:3f}".format(train_loss/len(train_loader))

#Testing
def test(epoch):
    #print('Epoch', epoch)
    net.eval()
    epoch_letter_correct = 0
    epoch_letter_total = 0
    epoch_word_correct=0
    epoch_word_total = 0

    for index, data in enumerate(test_loader):
        print("index", index)
        testX, testY = data[0], data[1] #256*14*32*32, 256*14*26
        testX, testY = testX.to(device), testY.to(device)
        seq_len = 0
        batch_letter_total=0
        batch_letter_correct =0
        batch_word_correct = 0

        batch, seq, img_len = testY.shape

        for i in testY:
            max_val, _ = torch.max(i, dim = 1, keepdim = True)
            seq_len = len((max_val!=0).nonzero()) #how many letters in a word


            features = modify_data(testX, seq_len)
            labels = testY.view(batch, seq, img_len, 1)
            labels = modify_data(labels, seq_len)
            labels = labels.view(seq_len, batch, img_len)
            
            print("features, labels shape: ", features.shape, labels.shape)

            letter_correct = 0
            for i in range(seq_len): 
                optimizer.zero_grad()
                outputs = net(features[i])
                targets = torch.max(labels[i], 1)[1]
                loss = criterion(outputs, targets)
        
                test_loss += loss.item()
                _, predicted = outputs.max(1)

                letter_correct+= predicted.eq(targets)
                #letter_correct += (predicted==targets).sum().item() #correct letter in a word

            print("word: {}/{}".format(letter_correct, seq_len))
            batch_letter_correct += letter_correct
            batch_letter_total += seq_len
            if(letter_correct == seq_len):
                bactch_word_correct +=1


        print("Test Batch letter acc: {:3f} ({}/{})".format((batch_letter_correct/batch_letter_total), batch_letter_correct, batch_letter_total))
        print("Test Batch word acc: {:3f} ({}/{})".format((batch_word_correct/batch_word_total), batch_word_correct, batch_word_total))
        print("-----")
    epoch_word_total += batch_word_total
    epoch_word_correct += batch_word_correct
    epoch_letter_total += batch_letter_total
    epoch_letter_correct += batch_letter_correct
    print("Test Epoch letter acc: {:3f} ({}/{})".format((epoch_letter_correct/epoch_letter_total), epoch_letter_correct, epoch_letter_total))
    print("Test Epoch word acc: {:3f} ({}/{})".format((epoch_word_correct/epoch_word_total), epoch_word_correct, epoch_word_total))


for epoch in range(0,3):

    train(epoch)
    test(epoch)


