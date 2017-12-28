#!/usr/bin/python
# -*- coding:utf8 -*-

"""
    @author:xiaotian zhao
    @time:12/12/17
"""
from __future__ import print_function

import torch
import logging
import torch.nn as nn
from torch.autograd import Variable
from collections import defaultdict


class Lang(object):
    SOS = '<s>'
    UNK = '<unk>'
    SOS_Token = 0
    UNK_Token = 1

    """docstring for Lang"""

    def __init__(self):
        super(Lang, self).__init__()

        self.w2i = {
            self.SOS: self.SOS_Token,
            self.UNK: self.UNK_Token
        }
        self.i2w = {
            self.SOS_Token: self.SOS,
            self.UNK_Token: self.UNK
        }
        self.n_words = len(self.w2i)

    def index_word(self, word):
        if word not in self.w2i:
            self.w2i[word] = self.n_words
            self.i2w[self.n_words] = word
            self.n_words += 1

    def index_words(self, words):
        for word in words:
            self.index_word(word)


def construct_vocabulary(file_name):
    t2i = defaultdict(lambda: len(t2i))
    lang = Lang()
    with open(file_name, 'r') as f:
        for line in f:
            tag, words = line.lower().strip().split(' ||| ')
            lang.index_words(words.split())
            t2i[tag]
    return lang, tag


def read_data(file_name, lang):
    with open(file_name, 'r') as f:
        for line in f:
            yield [lang.w2i[x] for x in line.strip().split()]


class CNNClassfier(nn.Module):

    def __init__(
        self,
        vocab_size,
        emb_size,
        kernel_size,
        input_channels,
        stride=1,
    ):
        super(CNNClassfier, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.stride = stride

        self.embbed = nn.Embedding(self.vocab_size, self.emb_size)
        self.conv = nn.Conv2d(input_channels, 1, (self.kernel_size, self.emb_size))
        self.max_pool = nn.MaxPool1d(self.kernel_size)

    def forward(self, input):
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_size)
        word_embed = self.embbed(input)

        # (batch_size, seq_len, embedding_size) -> (batch_size, 1, seq_len, embedding_size)
        word_embed = word_embed.unsqueeze(1)



if __name__ == '__main__':
    config = {
        'training': {
            'use_cuda': True,
            'lr': 0.01,
            'optimizer': 'SGD',
            'iteration': 100
        },
        'model': {
            'window_size': 2,
            'embedding_size': 128,
            'hidden_size': 128,
        },
        'data': {
            'train_data': '../data/classes/train.txt',
            'dev_data': '../data/classes/valid.txt',
            'max_len': 100
        },
        "management": {
            "monitor_loss": 50,
            "checkpoint_freq": 500
        }
    }

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info('Constructing vocabulary...')
    lang, t2i = construct_vocabulary(config['data']['train_data'])

    logging.info('Reading data...')
    train_data = list(read_data(config['data']['train_data'], lang))
    dev_data = list(read_data(config['data']['dev_data'], lang))

    use_cuda = config['training']['use_cuda']
    n_tags = len(t2i)



