#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn

class CNN(nn.Module):
    """
    Convolutional NN for character level embeddings
    """
    def __init__(self, embedding_size, max_word_size, stride=1, k=5):
        super(CNN, self).__init__()

        self.embed_size = embedding_size
        self.conv_layer = nn.Conv1d(in_channels=embedding_size, out_channels=embedding_size, kernel_size=k, stride=stride, bias=True)
        self.ReLU = nn.ReLU()
        self.max_pooling = nn.MaxPool1d(kernel_size=max_word_size-k+1, stride=stride)
    def forward(self, x_reshaped):
        """
        Forward function for Convolutional NN
        @param x_reshaped: reshaped tensor of character embeddings
                for a word (embedding_size, word_length)
        @return conv_out: max_pooled conv layer
        """
        x_conv = self.conv_layer(x_reshaped)
        conv_out = self.max_pooling(self.ReLU(x_conv))
        return conv_out


