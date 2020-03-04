import torch
import torch.nn as nn


class CRF(nn.Module):

    def __init__(self, input_dim, embed_dim, conv_layers, num_labels, batch_size):
        """
        Linear chain CRF as in Assignment 2
        """
        super(CRF, self).__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.conv_layers = conv_layers
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()

        ### Use GPU if available
        if self.use_cuda:
            [m.cuda() for m in self.modules()]

    def init_params(self):
        """
        Initialize trainable parameters of CRF here
        """
        blah

    def forward(self, X):
        """
        Implement the objective of CRF here.
        The input (features) to the CRF module should be convolution features.
        """
        features = self.get_conv_feats(X)
        prediction = blah
        return (prediction)

    def loss(self, X, labels):
        """
        Compute the negative conditional log-likelihood of a labelling given a sequence.
        """
        features = self.get_conv_feats(X)
        loss = blah
        return loss

    def backward(self):
        """
        Return the gradient of the CRF layer
        :return:
        """
        gradient = blah
        return gradient

    def get_conv_features(self, X):
        """
        Generate convolution features for a given word
        """
        convfeatures = blah
        return convfeatures
