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
    
    def __init__(self):
        super(Conv, self).__init__()

        # self.kernel = Parameter(torch.tensor([[1,0,1], [0,1,0], [1,0,1]]), requires_grad= False) 
        self.conv1 = custom2d.custom2D(in_channels = 1, out_channels = 1, kernel_size = (3,3), padding = False, stride = 1)
        self.kernel = torch.tensor([[1,0,1], [0,1,0], [1,0,1]])
        self.w1 = Parameter(torch.tensor(self.init_params()))

    def parameters(self):
        return [self.kernel]

    def init_params(self):
        """
        Initialize the layer parameters
        :return:
        """
        return torch.randn(2,3)
        # batch_size, w, l = x.shape
        # D_in = batch_size * w * 128
        # D_out = batch_size * w * l
        # self.w1 = Parameter(torch.randn(D_in, D_out, requires_grad=True))

    def forward(self, x, padding = False):
        """
        Forward pass
        :return:
        """
        x = self.conv1(x)
        x = F.relu(x)

        return x
        

    # def backward(self):
        """
        Backward pass 
        (leave the function commented out,
        so that autograd will be used.
        No action needed here.)
        :return:
        """

################################################################################################################
# dataset = dload.get_dataset()
# data = dataset.data

# x = torch.tensor(data[0:1])
# a = Conv()
# print(a._parameters)
# b = a.forward(x)
# print(b.shape)