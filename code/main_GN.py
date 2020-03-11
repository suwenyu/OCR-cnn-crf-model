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
from data_loader import get_dataset


dataset = get_dataset()
split = int(0.5 * len(dataset.data))

# train_data, train_target = dataset.data[:split], dataset.target[:split]
train_data, test_data = dataset.data[:split], dataset.data[split:]
train_target, test_target = dataset.target[:split], dataset.target[split:]
batch_size =256

train = data_utils.TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_target).long())
test = data_utils.TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_target).long())



#transform_train = transforms.Compose([
#        transforms.RandomSizedCrop(224),
#        transforms.RandomHorizontalFlip(),
#        transforms.ToTensor(),
##        transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225])])
#transform_test = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#])
#transform_train = torchvision.datasets.DatasetFolder(root='data/train',
#                                           transform=transform_train)
#tranform_test = torchvision.datasets.DatasetFolder(root = 'data/train',
#                                           transform=transform_test)


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

'''
for i, j in enumerate(train_loader):
    if i ==1:
        print(j)
    j = ToTensor()(j)
    print(type(j))
    F.interpolate(j,128)
'''
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

#Training
def train(epoch):
    print('Epoch', epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for index, data in enumerate(train_loader):
        trainX = data[0]
        trainY = data[1]
        print(trainX.shape, trainY.shape)
        trainX = trainX.reshape((batch_size, 1, trainX.shape[1], trainX.shape[2]))
        
        trainX, trainY = trainX.to(device), trainY.to(device)
        optimizer.zero_grad()
        outputs = net(trainX)
        loss = criterion (outputs, trainY)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum.item()
        
    print("loss: ",test_loss/(index+1))
    print("Accuracy: {}, ({}/{}) ".format(correct/total, correct, total))

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
            outputs = net(testX)
            loss = criterion(outputs, testY)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(testY).sum.item()
        print("loss: ",test_loss/(index+1))
        print("Accuracy: {}, ({}/{}) ".format(correct/total, correct, total))
for epoch in range(0,100):
    train(epoch)
    test(epoch)

