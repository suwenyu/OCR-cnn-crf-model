import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Conv(nn.Module):
    """
    Convolution layer.
    """
    def __init__(self, kernel_size=(3, 3)):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=(1, 1))
        # self.pool = nn.MaxPool2d(3, 3)
        self.kernel_size = kernel_size
        self.weights = nn.Parameter(torch.empty(kernel_size))
        self.w = 8
        self.l = 16
        self.init_params()

    def init_params(self):
        """
        Initialize the layer parameters
        :return:
        """
        nn.init.uniform_(self.weights, -0.1, 0.1)

    def _conv(self, img):
        conv_feats = torch.zeros((img.shape))
        filter_size = float(self.kernel_size[0])
        for r in np.uint16(np.arange(filter_size/2.0, img.shape[0]-filter_size/2.0+1)):
            for c in np.uint16(np.arange(filter_size/2.0,  img.shape[1]-filter_size/2.0+1)):
                curr_region = img[r-np.uint16(np.floor(filter_size/2.0)):r+np.uint16(np.ceil(filter_size/2.0)), 
                              c-np.uint16(np.floor(filter_size/2.0)):c+np.uint16(np.ceil(filter_size/2.0))]
                
                curr_result = torch.matmul(curr_region, self.weights)
                conv_sum = torch.sum(curr_result)
                conv_feats[r, c] = conv_sum
        # print(conv_feats)
        return torch.flatten(conv_feats)

    def forward(self, x):
        """
        Forward pass
        :return:
        """
        batch_size, seq, img = x.shape
        kernel_x, kernel_y = self.kernel_size[0], self.kernel_size[1]

        new_features = torch.zeros((batch_size, seq, self.w * self.l))
        
        for word_index, word in enumerate(x):
            for letter_index, letter in enumerate(word):
                img = letter.view((self.w, self.l))
                new_features[word_index][letter_index] = self._conv(img)

        
        # for i in range(batch_size):
        #     # for j in range(seq):

        #     tmp = x[i].view((seq, 1, self.w, self.l))
        #     x1 = self.conv1(tmp)

        #     t = x1.view((seq, 1, self.w * self.l))
        #     new_x = t.view((seq, self.w * self.l))

        #     new_features[i] = new_x

        # x1 = self.conv1(x)
        # print(x1)

        # x = F.relu(self.conv1(x))
        # batch_size, channel, w, l = x.shape
        # x = x.view((batch_size, w, l))
        
        return new_features

    # def backward(self):
        """
        Backward pass 
        (leave the function commented out,
        so that autograd will be used.
        No action needed here.)
        :return:
        """
