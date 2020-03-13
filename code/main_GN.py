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
print('Loaded dataset... ')
print(len(train_loader))

print('=> Building model..')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = googlenet.GoogLeNet()
net = net.to(device)
if device =='cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.1, momentum =0.9, weight_decay = 5e-4)
best_acc = 0
start_epoch =0
Err_train = []
Acc_train = []
Err_test =[]
Acc_test=[]

def modify_dataX(data):
    batch, seq, img_width, img_len = data.shape
    new_data = data.view(seq, batch, img_width, img_len)
    print(new_data.shape)

    resized = torch.empty(seq, batch, 1, img_width, img_len) #initial new data with channal 1
    for i in range(seq):
        resized[i] = new_data[i].reshape(1, batch, 1, img_width, img_len)
    #resized = resized.view(
    print(resized.shape)
    return resized


#Training
def train(epoch):
    print('Epoch', epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    print(len(train_loader))

    for index, data in enumerate(train_loader):
        trainX = data[0] #256*14*32*32
        trainY = data[1] #256*14*26
        trainX, trainY = trainX.to(device), trainY.to(device)
        
        print("trainX, Y shape: ", trainX.shape, trainY.shape)

        #trainY = trainY[:,:,:,None] #256*14*26*1
        #print(trainY.shape)
        features = modify_dataX(trainX)
        batch, seq, img_len = trainY.shape
        labels = trainY.view(seq, batch, img_len)
        #labels = modify_dataY(trainY)
        print("features, labels shape: ", features.shape, labels.shape) #14*256*1*32*32 14*256*26
        #print(labels[0].shape) #torch.max(labels[0],1)[1].shape)
        
        #find word length before zero padding
        word_len = 0
        print(len(labels))
        for i in range(len(labels)):
            if (torch.max(labels[i]).item() != 0.0):
                word_len +=1
        

        print('word_len',word_len)
       # print('seq ', len(labels.nonzero()))
#train(1)

        letter_correct = 0

        for i in range(word_len):
            print(features[i].shape)
            #print(labels[i].shape)
            #print(torch.max(labels[i],1)[1])

            optimizer.zero_grad()
            outputs = net(features[i])
            #loss = criterion(outputs, labels[i])
            #print(outputs.shape)
            #print(torch.max(labels[i], 1)[1].shape)
            targets = torch.max(labels[i],1)[1]
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            #total += labels.size(0)
            letter_correct += predicted.eq(targets).sum()
        #total += targets.size(0)
        correct += letter_correct
        total += word_len*batch
        print("loss: ",train_loss/(index+1))
        print("letter Accuracy: {} ({}/{}) ".format( letter_correct/(batch*word_len), letter_correct, batch*word_len))     
        print(correct, total)
        
        #print("for every batch: c {}, total {}, acc {}".format(correct, total, correct/total))
    #Record epoch err and acc

    print("train loss: {}, len(train_loader): {}, loss: {}".format(train_loss, len(train_loader), train_loss/len(train_loader)))
    Err_train.append(train_loss/len(train_loader))
    Acc_train.append(correct/total)
    print("Acc: {}, correct {}, total {}".format(correct/total, correct, total))


        #for i in range(trainX.shape[1]):
#Testing
def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0 
    total = 0
    with torch.no_grad():
        for index, data in enumerate(test_loader):
            testX = data[0]
            testY = data[1]
            testX, testY = testX.to(device), testY.to(device)
            test_feat = modify_dataX(testX)
            batch, seq, img_len= testY.shape
            test_lab = trainY.view(seq, batch, img_len)

            word_len = 0
            for i in range(len(test_lab)):
                if (torch.max(test_lab[i]).item()!=0):
                    word_len += 1

            print('word_len', word_len)

            letter_correct = 0
            for i in range(word_len): # seq =14 letters in words
                print(test_feat[i].shape)
                outputs = net(test_feat[i])
                tagets = torch.max(test_lab[i], 1)[1]
                loss = criterion(outputs, targets)


                test_loss += loss.item()
                _, predicted = outputs.max(1)
                letter_correct += predicted.eq(targets).sum()
                print("loss: ",test_loss)
                print("Accuracy: {}, ({}/{}) ".format(correct/(word_len*batch_size), correct, word_len*batch_size))
            total += word_len
            correct+=letter_correct
            #print("for every batch: c {}, total {}, acc {}".format(correct, total, correct/total))

        #record epoch err, acc for every epoch
        Err_test.append(test_loss/len(test_loader))
        Acc_test.append(correct/total)
        print("test loss: {}, len(test_loader): {}, loss: {}".format(test_loss, len(test_loader), test_loss/len(test_loader)))
        print("Acc: {}, correct {}, total {}".format(correct/total, correct, total))



for epoch in range(0,100):

    train(epoch)
    test(epoch)


