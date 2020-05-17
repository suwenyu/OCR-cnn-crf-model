#WORKS SO FAR
import torch
import torch.nn as nn
import data_loader as dload
# import convolution_2d as custom2d
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import time

import numpy as np

class Conv(nn.Module):
    """
    Convolution layer.
    """  
    
    def __init__(self, kernel_size = (3, 3), in_channels = 1, out_channels = 0, padding = False, stride = 1): #default to a 3x3 kernel
        super(Conv, self).__init__()
        self.kernel_size = kernel_size[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride
        self.width = 8
        self.length = 16


        self.use_cuda = torch.cuda.is_available()

        #run init param to get the kernel, which will be updated with autograd
        self.kernel = self.init_params()
        # self.kernel = torch.tensor([[1,0,1], [0,1,0], [1,0,1]], dtype=torch.float) #to test the 5x5 check
        # if self.use_cuda:
        #     self.kernel = self.kernel.cuda()

        self.pad_size = 0
        if self.padding:
            self.pad_size = int((self.width - (self.width - self.kernel_size + 1))/2)

        self.conv_pkg = nn.Conv2d(1, 1, kernel_size = self.kernel_size, stride = self.stride, padding = self.pad_size)
        self.conv_pkg_1 = nn.Conv2d(1, 1, kernel_size = 5, stride = 1, padding = 0)

        if self.use_cuda:
            [m.cuda() for m in self.modules()]

        

    def init_params(self):
        """
        Initialize the layer parameters
        :return:
        """
        return Parameter(torch.randn(self.kernel_size,self.kernel_size), requires_grad=True)

    def forward(self, x, padding = False):
        """
        Forward pass
        :return:
        """

        # x = self.conv1(x)
        # # x = F.relu(x)
        # return x


        # start = time.time()

        #for the assignment
        if len(x.shape) == 3:
            batch_size, seq_len, img = x.shape

            new_width = self.width
            new_length = self.length
      
            if self.padding == True:
                pad = (len(self.kernel) - 1) / 2
            else:
                new_width = math.ceil((self.width - self.kernel.shape[0])/self.stride +1)
                new_length = math.ceil((self.length - self.kernel.shape[0])/self.stride +1)
                pad = 0


            # return_from_forward = torch.empty(size=(batch_size,seq_len, new_width * new_length))
            return_from_forward = torch.empty(size = (batch_size, seq_len, self.width * self.length))
            if self.use_cuda:
                return_from_forward = return_from_forward.cuda()

            # print("time 1:", time.time() - start)

            for batch_item in range(0,batch_size):
                word = x[batch_item]
                for word_letter in range(0,seq_len):
                    letter = word[word_letter]
                    return_from_forward[batch_item, word_letter][:new_width * new_length] = self.convolution_2d(letter = letter, new_length = new_length, new_width = new_width, pad = pad)
            x = return_from_forward      
            

            x = F.relu(x)
            # print("time 2", time.time() - start)
            return x

        #for the 5x5
        if (len(x.shape) == 2):
        #if there is padding, we need to account for it
            self.width, self.length = x.shape

            new_width = self.width
            new_length = self.length
            
            if self.padding == True:
                pad = (len(self.kernel) - 1) / 2
            else:
                new_width = int((x.shape[1] - self.kernel.shape[0])/self.stride +1)
                new_length = int((x.shape[0] - self.kernel.shape[0])/self.stride +1)
                pad = 0

            return self.convolution_2d(letter = x, new_width = new_width, new_length = new_length, pad = pad).view(new_length,new_width)
            

    def forward_pkg(self, x):
        batch_size, seq_len, img = x.shape
        new_x = x.view(seq_len, batch_size, 1, self.width, self.length)
        # new_x = new_x.view()

        new_width = self.width
        new_length = self.length
        
        if not self.padding:
            new_width = self.width - self.kernel_size + 1
            new_length = self.length - self.kernel_size + 1
        
        if self.stride > 1:
            new_width /= self.stride
            new_length /= self.stride

        new_width = np.ceil(new_width)
        new_length = np.ceil(new_length)

        
        new_feats = torch.empty(seq_len, batch_size, 1, self.width, self.length, dtype=torch.float)
        if self.use_cuda:
            new_feats = new_feats.cuda()
        
        for i in range(seq_len):
            new_feats[i][:, :, :int(new_width), :int(new_length)] = self.conv_pkg(new_x[i])

        new_feats = new_feats.view(batch_size, seq_len, self.width * self.length)
        
        new_feats = F.relu(new_feats)
        
        return new_feats
        

    # def backward(self):
        """
        Backward pass 
        (leave the function commented out,
        so that autograd will be used.
        No action needed here.)
        :return:
        """
    def convolution_2d(self, letter = None, new_length = 16, new_width = 8,pad = 0):

        # start = time.time()

        #make a bunch of zeroes in the form of a matrix that matches expected output
        to_return = torch.empty((int(new_length), int(new_width)), dtype=torch.float)
        if self.use_cuda:
            to_return = to_return.cuda()


        letter = letter.view(self.length,self.width)

        #update the to_return object
        for i in range(0,letter.shape[0] - self.kernel.shape[0] + 1, self.stride):
            for j in range(0, letter.shape[1] - self.kernel.shape[0] + 1, self.stride):
                #get each section of interest, get the summed val for every 1/1 match
                summed_kernel_val = torch.sum(torch.mul(letter[i:i+self.kernel.shape[0], j:j+self.kernel.shape[0]], self.kernel))

                #put the val into the to return variable
                temp_i = int(i/self.stride + pad)
                temp_j = int(j/self.stride + pad)
                to_return[temp_i,temp_j] = summed_kernel_val

        return to_return.flatten()
        
    
###############################################################################################################
# dataset = dload.get_dataset()
# data = dataset.data
# x = torch.tensor(data[0:2])
# a = Conv(kernel_size=(3,3), padding=True, stride = 1)
# b = a.forward(x)
# print(b.shape)
# # print(b)

# X = [[1,1,1,0,0], [0,1,1,1,0], [0,0,1,1,1], [0,0,1,1,0], [0,1,1,0,0]]
# k = torch.tensor([[1,0,1], [0,1,0], [1,0,1]])
# data = torch.tensor(X)
# a = Conv(kernel_size=(3,3), padding=False, stride=1)
# b = a.forward(data)
# print("OUTPUT FINAL:\n",b)