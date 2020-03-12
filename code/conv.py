import torch
import torch.nn as nn
import data_loader as dload
import convolution_2d as custom2d
import torch.nn.functional as F
from torch.nn.parameter import Parameter

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

        #run init param to get the kernel, which will be updated with autograd
        # self.kernel = self.init_params()
        self.kernel = torch.tensor([[1,0,1], [0,1,0], [1,0,1]])
        self.conv1 = custom2d.custom2D(self.in_channels, self.out_channels, self.kernel, padding = self.padding, stride = self.stride)        

    # def parameters(self):
        # return [self.kernel]

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
        x = self.conv1(x)
        # x = F.relu(x)

        return x
        

    # def backward(self):
        """
        Backward pass 
        (leave the function commented out,
        so that autograd will be used.
        No action needed here.)
        :return:
        """
    
###############################################################################################################
# dataset = dload.get_dataset()
# data = dataset.data
# x = torch.tensor(data[0:2])
# a = Conv(kernel_size=(3,3), padding=False, stride = 2)
# b = a.forward(x)
# print(b.shape)
# print(b)

# X = [[1,1,1,0,0], [0,1,1,1,0], [0,0,1,1,1], [0,0,1,1,0], [0,1,1,0,0]]
# k = torch.tensor([[1,0,1], [0,1,0], [1,0,1]])
# data = torch.tensor(X)
# a = Conv(kernel_size=(3,3), padding=False, stride=1)
# b = a.forward(data)
# print("OUTPUT FINAL:\n",b)