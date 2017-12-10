#!/usr/bin/python
#-*- coding:utf8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

class PosOfTaggingBiLSTM(nn.Module):
    """docstring for PosOfTaggingBiLSTM"""
    def __init__(
        self,
        word_vocab_size,
        word_emb_size,
        pad_token,
        hidden_size,
        tag_vocab_size,
        bidirectional=True,
        num_layers=1,
        batch_first=True,
        dropout=0.,
        use_cuda=False
    ):
        super(PosOfTaggingBiLSTM, self).__init__()
        self.word_vocab_size = word_vocab_size
        self.word_emb_size = word_emb_size
        self.pad_token = pad_token
        self.hidden_size = hidden_size // 2 \
            if bidirectional else hidden_size
        self.tag_vocab_size = tag_vocab_size
        self.num_directions = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        self.num_layers = 1
        self.batch_first = True
        self.dropout = dropout
        self.use_cuda = use_cuda

        self.word_emb = nn.Embedding(
            self.word_vocab_size,
            self.word_emb_size,
            self.pad_token
        )

        self.lstm = nn.LSTM(
            self.word_emb_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )

        self.hidden2tag = nn.Linear(
            hidden_size,
            self.tag_vocab_size
        )
        self.init_weight()

    def init_weight(self):
        initrange = 0.1
        self.word_emb.weight.data.uniform_(-initrange,initrange)
        self.hidden2tag.bias.data.fill_(0)
    
    def get_state(self,input):
        batch_size = input.size(0) \
            if self.batch_first else input.size(1)

        h0 = Variable(torch.zeros(
            self.num_layers*self.num_directions,
            batch_size,
            self.hidden_size
        ))
        
        c0 = Variable(torch.zeros(
            self.num_layers*self.num_directions,
            batch_size,
            self.hidden_size
        ))

        if self.use_cuda:
            return (h0.cuda(),c0.cuda())
        else:
            return (h0,c0)

    def forward(self,word_input):
        word_emb = self.word_emb(word_input)
        
        (h0,c0) = self.get_state(word_input)
        lstm_hidden,(_,_) = self.lstm(
            word_emb,
            (h0,c0)
        )

        tag_input = lstm_hidden.contiguous().view(
            lstm_hidden.size(0)*lstm_hidden.size(1),
            lstm_hidden.size(2)
        )

        tag_logit = self.hidden2tag(lstm_hidden)
        tag_logit = nn.LogSoftmax()(tag_logit)

        tag_logit = tag_logit.view(
            lstm_hidden.size(0),
            lstm_hidden.size(1),
            self.tag_vocab_size
        )
        # print(tag_logit.size())

        return tag_logit