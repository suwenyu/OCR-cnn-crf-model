import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    """
    Convolution layer.
    """
    def __init__(self):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=(1, 1))
        # self.pool = nn.MaxPool2d(3, 3)


    def init_params(self):
        """
        Initialize the layer parameters
        :return:
        """

    def forward(self, x):
        """
        Forward pass
        :return:
        """
        batch_size, w, l = x.shape
        
        x = x.view((batch_size, 1, w, l))

        x = F.relu(self.conv1(x))
        batch_size, channel, w, l = x.shape
        x = x.view((batch_size, w, l))
        
        return x

    # def backward(self):
        """
        Backward pass 
        (leave the function commented out,
        so that autograd will be used.
        No action needed here.)
        :return:
        """




