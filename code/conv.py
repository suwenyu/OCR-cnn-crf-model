import torch
import torch.nn as nn
import data_loader as dload
import convolution_2d as custom2d
import torch.nn.functional as F

# import os
# os.system('clear')
# os.chdir('/Users/jonathantso/Desktop/Code/uic_git/cs512/hw2/code')

class Conv(nn.Module):
    """
    Convolution layer.
    """  
    
    def __init__(self):
        super(Conv, self).__init__()

        self.conv1 = custom2d.custom2D(in_channels = 1, out_channels = 1, kernel_size = (3,3), padding = False, stride = 1)
        self.kernel = torch.tensor([[1,0,1], [0,1,0], [1,0,1]])  
        self.init_val = 1


    def init_params(self,x, padding):
        """
        Initialize the layer parameters
        :return:
        """
        batch_size, w, l = x.shape
        D_in = batch_size * w * 128
        D_out = batch_size * w * l
        w1 = torch.randn(D_in, D_out, requires_grad=True)

    def forward(self, x, padding = False):
        """
        Forward pass
        :return:
        """

        batch_size, w, l = x.shape

        batch_vec = []
        for batch_item in range(0,batch_size):
            word_vec = []
            word = x[batch_item]

            for word_letter in range(0,w):
                letter = word[word_letter]
                convoluted_letter = self.conv1.convolution_2d(letter = letter, kernel = self.kernel).flatten()

                word_vec.append(convoluted_letter)
            
            batch_vec.append(word_vec)
        
        batch_vec = torch.tensor(batch_vec)
        
        x = F.relu(batch_vec)

        if self.init_val == 1:
            self.init_params(x,padding)
            self.init_val == 2        

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
if __name__ == "__main__":
    dataset = dload.get_dataset()
    data = dataset.data

    x = torch.tensor(data[0:5])

    a = Conv()
    b = a.forward(x)
    print(b.shape)