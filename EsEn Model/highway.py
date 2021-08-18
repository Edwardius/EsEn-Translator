#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class Highway(nn.Module):
    """
    Highway Layer Class
    """
    def __init__(self, embedding_size):
        super(Highway, self).__init__()

        self.proj_linear = nn.Linear(in_features=embedding_size, out_features=embedding_size)
        self.gate_linear = nn.Linear(in_features=embedding_size, out_features=embedding_size)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, conv_out):
        """
        Forward function for highway
        @param conv_out: output of convolutions from character embeddings
        @return highway_out: output of character-level embedding
        """
        proj_layer = self.ReLU(self.proj_linear(conv_out))
        gate_layer = self.Sigmoid(self.gate_linear(conv_out))
        highway_out = torch.add(torch.mul(proj_layer, gate_layer), torch.mul(torch.add(-gate_layer, 1),conv_out))

        return highway_out

