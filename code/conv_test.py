#from __future__ import print_function
import torch
import conv
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

x = torch.tensor([[1,1,1,0,0],
                 [0,1,1,1,0],
                 [0,0,1,1,1],
                 [0,0,1,1,0],
                 [0,1,1,0,0]])

weights = torch.tensor([[1,0,1],
                 [0,1,0],
                 [1,0,1]])


image = x.view(1, 1, 5, 5) #batch_size, height, width, in_channels
weights = weights.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
myTensor = torch.nn.Conv2d(1, 1, (3, 3))

output = F.conv2d(image, weights)

print("Result from PyTorch implementation\n{}".format(output))

print('\n')

#TESTING WITH OUR IMPLEMENTATION
X = [[1,1,1,0,0], [0,1,1,1,0], [0,0,1,1,1], [0,0,1,1,0], [0,1,1,0,0]] #data

k = torch.tensor([[1,0,1], [0,1,0], [1,0,1]]) #kernel

data = torch.tensor(X)

result = conv.Conv(kernel_size=(3,3), padding=False, stride=1).forward(data)

print("Result from our implementation\n{}".format(result))
