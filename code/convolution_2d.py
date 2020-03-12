import torch
import torch.nn as nn
import sys
from torch.nn.parameter import Parameter
import math

class custom2D(nn.Module):
  def __init__(self, in_channels = 1, out_channels = 1, kernel = torch.randn(3,3), padding = False, stride = 1):
    super().__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel.shape
    self.padding = padding
    self.stride = stride
    self.kernel = kernel

  def convolution_2d(self, letter,width_out, length_out, pad):

    #if we ahve padding, we need to bring it back to the original shape
    original_shape = letter.shape

    #qualifies for this for actual assignment, does not qualify if testing on the 5x5
    if len(letter) == 128:
      letter = letter.reshape([16,8])

    letter_length = letter.shape[0]
    letter_width = letter.shape[1]

    #make a bunch of zeroes in the form of a matrix that matches expected output
    to_return = torch.zeros((int(length_out), int(width_out)))

    #update the to_return object
    for i in range(0,letter.shape[0] - self.kernel.shape[0] + 1, self.stride):
      for j in range(0, letter.shape[1] - self.kernel.shape[0] + 1, self.stride):
        #get each section of interest, get the summed val for every 1/1 match
        summed_kernel_val = sum(letter[i:i+self.kernel.shape[0], j:j+self.kernel.shape[0]].flatten() * self.kernel.flatten())

        #put the val into the to return variable
        temp_i = int(i/self.stride + pad)
        temp_j = int(j/self.stride + pad)
        to_return[temp_i,temp_j] = summed_kernel_val
    
    #if need to pad back to original shape, reshape back to original shape
    if self.padding == True:
      to_return = to_return.reshape(original_shape)
    #else return the smaller tensor
    else:
      to_return = to_return.reshape([length_out, width_out])
    return to_return

  def forward(self, x):

    #this is for assignment
    if len(x.shape) == 3:
      batch_size, w, l = x.shape
      # letter_length = len(x[0][0])**(.5)
      # letter_width = len(x[0][0])/letter_length

      # if(len(x[0][0]) == 128):
      letter_length = 16
      letter_width = 8
      
      if self.padding == True:
        width_out = int(letter_width)
        length_out = int(letter_length)
        pad = (len(self.kernel) - 1) / 2
      else:
        width_out = math.ceil((letter_width - self.kernel.shape[0])/self.stride +1)
        length_out = math.ceil((letter_length - self.kernel.shape[0])/self.stride +1)
        pad = 0

      out_dim = width_out*length_out
      return_from_forward = torch.empty(size=(self.out_channels,batch_size,14,out_dim))

      for i in range(0,self.out_channels):
        for batch_item in range(0,batch_size):
            word = x[batch_item]
            for word_letter in range(0,w):
                letter = word[word_letter]
                return_from_forward[i][batch_item][word_letter] = self.convolution_2d(letter = letter,
                                                                                      width_out = width_out, 
                                                                                      length_out = length_out, 
                                                                                      pad = pad
                                                                                      ).flatten()
      return return_from_forward

    #this is specifically for the 5x5 test
    else:
    #if there is padding, we need to account for it
      if self.padding == True:
        width_out = int(x.shape[1])
        length_out = int(x.shape[0])
        pad = (len(self.kernel) - 1) / 2
      else:
        width_out = int((x.shape[1] - self.kernel.shape[0])/self.stride +1)
        length_out = int((x.shape[0] - self.kernel.shape[0])/self.stride +1)
        pad = 0

      return_from_forward = self.convolution_2d(letter = x, width_out = width_out, length_out = length_out, pad = pad)
      return return_from_forward
