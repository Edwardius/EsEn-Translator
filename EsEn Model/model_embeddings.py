#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from .cnn import CNN
from .highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab, dropout_rate=0.3):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        self.embed_size = embed_size
        self.max_word_length = 21
        self.dropout = nn.Dropout(p=dropout_rate)
        self.embeddings = nn.Embedding(num_embeddings=len(vocab.char2id), embedding_dim=self.embed_size, padding_idx=0)
        self.cnn_layer = CNN(self.embed_size, self.max_word_length)
        self.highway_layer = Highway(self.embed_size)

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """

        # input tensor permute to (batch_size, sentence_length, max_word_length)
        # lets torch.nn deal with batches of data
        input_tensor = input_tensor.permute(1, 0, 2)

        # char embedding lookup and reshape
        char_embeddings = self.embeddings(input_tensor)
        x_reshaped = char_embeddings.permute(0, 1, 3, 2)

        # sending x_reshaped through convolutional layer class Note: since this is a batch of
        # sentences not words, input_tensor must be split into its sentences
        # and then concatenated (inserted rather) back together

        # getting initial state of x_word_emb
        conv_out = self.cnn_layer(x_reshaped[0])
        conv_out = torch.squeeze(conv_out, 2)
        x_highway = self.highway_layer(conv_out)
        x_word_emb = self.dropout(x_highway)
        x_word_emb = torch.unsqueeze(x_word_emb, 0)

        # continuing through rest of the sentences
        for sentence in x_reshaped[1:]:
            conv_out = self.cnn_layer(sentence)
            conv_out = torch.squeeze(conv_out, 2)

            # sending x_conv_out (with max pooling) through highway layer class
            x_highway = self.highway_layer(conv_out)

            # dropout and permute to get final embedding
            x_word_emb = torch.cat((x_word_emb, torch.unsqueeze(self.dropout(x_highway), 0)), 0)
        x_word_emb_tensor = x_word_emb.permute(1, 0, 2)

        return x_word_emb_tensor


