#WORKS SO FAR
import torch
import torch.nn as nn
import data_loader as dload
import convolution_2d as custom2d
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import time

class Conv(nn.Module):
    """
    Convolution layer.
    """  
    
    def __init__(self, kernel_size = (3,3), in_channels = 1, out_channels = 1, padding = False, stride = 1): #default to a 3x3 kernel
        super(Conv, self).__init__()
        self.kernel_size = kernel_size[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride
        self.width = 8
        self.length = 16

        #run init param to get the kernel, which will be updated with autograd
        self.kernel = self.init_params()
        # self.kernel = torch.tensor([[1,0,1], [0,1,0], [1,0,1]])
        # self.conv1 = custom2d.custom2D(self.in_channels, self.out_channels, self.kernel, padding = self.padding, stride = self.stride)

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
        start = time.time()
        # print("time start:")



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

            return_from_forward = torch.empty(size=(batch_size,seq_len,new_width * new_length))

            # print("time 1:", time.time() - start)

            for batch_item in range(0,batch_size):
                word = x[batch_item]
                for word_letter in range(0,seq_len):
                    letter = word[word_letter]
                    return_from_forward[batch_item, word_letter] = self.convolution_2d(letter = letter, new_length = new_length, new_width = new_width, pad = pad)
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
        to_return = torch.zeros((int(new_length), int(new_width)))
        letter = letter.view(self.length,self.width)

        #update the to_return object


        for i in range(0,letter.shape[0] - self.kernel.shape[0] + 1, self.stride):
            for j in range(0, letter.shape[1] - self.kernel.shape[0] + 1, self.stride):
                #get each section of interest, get the summed val for every 1/1 match
                summed_kernel_val = sum(letter[i:i+self.kernel.shape[0], j:j+self.kernel.shape[0]].flatten() * self.kernel.flatten())

                #put the val into the to return variable
                temp_i = int(i/self.stride + pad)
                temp_j = int(j/self.stride + pad)
                to_return[temp_i,temp_j] = summed_kernel_val

        return to_return.flatten()
        # return to_return.reshape([self.length, self.width])
        
    
###############################################################################################################
dataset = dload.get_dataset()
data = dataset.data
x = torch.tensor(data[0:2])
a = Conv(kernel_size=(3,3), padding=True, stride = 1)
b = a.forward(x)
print(b.shape)
# print(b)

# X = [[1,1,1,0,0], [0,1,1,1,0], [0,0,1,1,1], [0,0,1,1,0], [0,1,1,0,0]]
# k = torch.tensor([[1,0,1], [0,1,0], [1,0,1]])
# data = torch.tensor(X)
# a = Conv(kernel_size=(3,3), padding=True, stride=1)
# b = a.forward(data)
# print("OUTPUT FINAL:\n",b)