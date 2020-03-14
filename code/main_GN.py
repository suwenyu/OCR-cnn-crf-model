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
import matplotlib.pyplot as mp
import numpy as np
import scipy.ndimage.interpolation
import time


print('=> Loaded dataset... ')

dataset = get_onehot()

split = int(0.5 * len(dataset.data))

dataset.data = dataset.data.reshape(len(dataset.data), 14, 8, 16)

print('Original dataset: ',dataset.data.shape) #256*14*8*16

def resize_batch(image_batch, new_width, new_height):
    image_batch = np.asarray(image_batch)
    shape = list(image_batch.shape)
    shape[2] = new_width
    shape[3] = new_height
    ind = np.indices(shape, dtype=float)
    ind[2] *= (image_batch.shape[2] - 1) / float(new_width - 1)
    ind[3] *= (image_batch.shape[3] - 1) / float(new_height - 1)

    return scipy.ndimage.interpolation.map_coordinates(image_batch, ind, order=1)

#rescale the img data for googlenet
dataset.data = resize_batch( dataset.data, 32,32)
print("Rescale dataset: ",dataset.data.shape)

# train_data, train_target = dataset.data[:split], dataset.target[:split]
train_data, test_data = dataset.data[:split], dataset.data[split:]
train_target, test_target = dataset.target[:split], dataset.target[split:]
batch_size =256
#print(train_data.shape, test_data.shape)

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


#optimizer
#optimizer = optim.LBFGS(net.parameters())
#optimizer = optim.SGD(net.parameters(), lr = 0.1, momentum =0.9)
optimizer = optim.Adam(net.parameters(), lr = 0.1, eps = 1e-08)
criterion = nn.CrossEntropyLoss()

#reshape dataset add channel = 1, with seq_len
def modify_data(data, seq_len):
    batch = 1
    #print(data.shape)
    seq, img_width, img_len = data.shape[0], data.shape[1], data.shape[2]  #14*32*32
    data = data.view(seq, batch, img_width, img_len)
    #print(data.shape)
    new = torch.empty(seq_len, batch, img_width, img_len) #14*1*1*32*32
    for i in range(seq_len):
        new[i] = data[i]
    new = new.view(seq_len, batch, 1, img_width, img_len) 
    #print(new.shape)
    return new

# how many letters in a word
def get_seqlen(data):
    max_val, _ = torch.max(data, dim = 1, keepdim = True)
    #print('max shape',max_val.shape) #14*1
    max_val = max_val.view(1, max_val.shape[0])
    seq_len = len((max_val!=0).nonzero())
    return seq_len

#Training
def train(epoch):
    net.train()
    #train_loss = 0
    epoch_letter_correct = 0
    epoch_letter_total = 0
    epoch_word_correct=0
    epoch_word_total = 0

    for b_index, data in enumerate(train_loader):
        print("Batch index: ", b_index) 
        trainX, trainY = data[0], data[1] #256*14*32*32, 256*14*26
        trainX, trainY = trainX.to(device), trainY.to(device)
        seq_len = 0
        train_loss = 0
        batch_letter_total=0
        batch_letter_correct =0
        batch_word_correct = 0

        batch, seq, img_len = trainY.shape

        for i in range(batch):
            seq_len = get_seqlen(trainY[i])
            #print(seq_len)
            #print("features, labels shape: ", trainX[i].shape, trainY[i].shape) #14*26
            #modify feature and label to fit the model
            features = modify_data(trainX[i], seq_len)
            labels = trainY[i].view(seq, img_len, 1)
            labels = modify_data(labels, seq_len)
            labels = labels.view(seq_len, 1, img_len)
            

            #print("features, labels shape: ", features.shape, labels.shape)

            #train the model by its seq_len
            letter_correct = 0
            for i in range(seq_len): 
                #def closure():
                #    optimizer.zero_grad()
                #    outputs = net(features[i])
                #    targets = torch.max(labels[i],1)[1].to(device)
                #    loss = criterion(outputs, targets)
                #    loss.backward

                #optimizer.step(closure)
                
                optimizer.zero_grad()
                outputs = net(features[i]) #256*26
                targets = torch.max(labels[i], 1)[1].to(device)  #256
                #print(outputs.shape, targets.shape)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                #print(targets.shape, predicted.shape)
                letter_correct += predicted.eq(targets).sum().item() #correct letter in a word of seq
                #print(letter_correct)
            
            batch_letter_total += seq_len
            batch_letter_correct += letter_correct
            print("Seq letter acc: {:3f} ({}/{})".format((letter_correct/seq_len), letter_correct, seq_len))
            if(letter_correct == batch_letter_total):
                bactch_word_correct +=1

        print("Loss {:3f}".format(train_loss/(b_index+1)))
        print("Batch {} letter acc: {:3f} ({}/{})".format(b_index,(batch_letter_correct/batch_letter_total), batch_letter_correct, batch_letter_total))
        print("Batch {} word acc: {:3f} ({}/{})".format(b_index,(batch_word_correct/batch), batch_word_correct, batch))
        print("-----")
    epoch_word_total += batch
    epoch_word_correct += batch_letter_total
    epoch_letter_total += batch_letter_total
    epoch_letter_correct += batch_letter_correct
    print("Epoch {} letter acc: {:3f} ({}/{})".format(epoch, (epoch_letter_correct/epoch_letter_total), epoch_letter_correct, epoch_letter_total))
    print("Epoch {} word acc: {:3f} ({}/{})".format(epoch, (epoch_word_correct/epoch_word_total), epoch_word_correct, epoch_word_total))


#Testing
def test(epoch):
    #print('Epoch', epoch)
    net.eval()
    epoch_letter_correct = 0
    epoch_letter_total = 0
    epoch_word_correct=0
    epoch_word_total = 0

    for b_index, data in enumerate(test_loader):
        print("index", b_index)
        testX, testY = data[0], data[1] #256*14*32*32, 256*14*26
        testX, testY = testX.to(device), testY.to(device)
        seq_len = 0
        test_loss = 0
        batch_letter_total=0
        batch_letter_correct =0
        batch_word_correct = 0

        batch, seq, img_len = testY.shape
        for i in range(batch):
            seq_len = get_seqlen(testY[i])
            #print(seq_len)

            #print("features, labels shape: ", testX[i].shape, testY[i].shape) #14*26
            #modify feature and label to fit the model
            features = modify_data(testX[i], seq_len)
            labels = testY[i].view(seq, img_len, 1)
            labels = modify_data(labels, seq_len)
            labels = labels.view(seq_len, 1, img_len)
            
            print("features, labels shape: ", features.shape, labels.shape)
            
            letter_correct = 0
            for i in range(seq_len): 
                #print(i)
                optimizer.zero_grad()
                outputs = net(features[i])
                targets = torch.max(labels[i], 1)[1].to(device)
                loss = criterion(outputs, targets)
        
                test_loss += loss.item()
                _, predicted = outputs.max(1)

                letter_correct+= predicted.eq(targets).sum().item()
                #letter_correct += (predicted==targets).sum().item() #correct letter in a word

            batch_letter_total += seq_len
            batch_letter_correct += letter_correct
            print("Seq {} letter acc: {:3f} ({}/{})".format(i,(letter_correct/seq_len),letter_correct, seq_len))
            if(letter_correct == seq_len):
                bactch_word_correct +=1


        print("Test Batch {} letter acc: {:3f} ({}/{})".format(b_index,(batch_letter_correct/batch_letter_total), batch_letter_correct, batch_letter_total))
        print("Test Batch {} word acc: {:3f} ({}/{})".format(b_index, (batch_word_correct/batch), batch_word_correct, batch))
        print("-----")
    epoch_word_total += batch
    epoch_word_correct += batch_word_correct
    epoch_letter_total += batch_letter_total
    epoch_letter_correct += batch_letter_correct
    print("Test Epoch {} letter acc: {:3f} ({}/{})".format(epoch, (epoch_letter_correct/epoch_letter_total), epoch_letter_correct, epoch_letter_total))
    print("Test Epoch {} word acc: {:3f} ({}/{})".format(epoch, (epoch_word_correct/epoch_word_total), epoch_word_correct, epoch_word_total))


for epoch in range(0,10):
    print("Epoch", epoch)
    start = time.time()
    train(epoch)
    test(epoch)
    print("Time: ", time.time()-start)


