#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()

        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(target_vocab.char2id), char_embedding_size, padding_idx=target_vocab.char2id['<pad>'])

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        input_emb = self.decoderCharEmb(input)
        output, dec_hidden = self.charDecoder(input_emb, dec_hidden)
        scores = self.char_output_projection(output)
        return scores, dec_hidden

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        CE_loss_layer = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'], reduction='sum')
        scores, dec_hidden = self.forward(char_sequence[:len(char_sequence) - 1], dec_hidden)
        loss = 0
        index = 0
        for word in char_sequence[1:len(char_sequence)]:
            loss += CE_loss_layer(scores[index], word)
            index += 1
        return loss

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """
        decodedWords = []
        current_char = torch.tensor([[1 for b in range(initialStates[0].size(1))]], device=device)
        output_word = current_char
        softmax = nn.Softmax(dim=2)

        # creates a tensor of output char idxs through forward function
        for t in range(max_length - 1):
            scores, initialStates = self.forward(current_char, initialStates)
            predictions = softmax(scores)
            current_char = torch.argmax(predictions, dim=2)
            output_word = torch.cat((output_word, current_char), 0)

        # reshapes tensor to be (batch_size, max_word_length)
        output_word = output_word.permute(1, 0)
        output_word = output_word.cpu().numpy()

        # converts idxs to actual words
        output_word_chars = [[self.target_vocab.id2char[c] for c in w] for w in output_word]

        for word_chars in output_word_chars:
            string = ""
            for char in word_chars:
                if char == '{':
                    continue
                elif char == '}':
                    break
                else:
                    string += char
            decodedWords.append(string)

        return decodedWords
        ### END CODE HERE
