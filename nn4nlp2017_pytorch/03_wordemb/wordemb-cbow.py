#!/usr/bin/python
#-*- coding:utf8 -*-

"""
    @author:xiaotian zhao
    @time:12/11/17
"""

import math
import torch
import random
import logging
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class Lang(object):
    SOS = '<s>'
    UNK = '<unk>'
    SOS_Token = 0
    UNK_Token = 1

    """docstring for Lang"""
    def __init__(self):
        super(Lang, self).__init__()

        self.w2i = {
            self.SOS:self.SOS_Token,
            self.UNK:self.UNK_Token
        }
        self.i2w = {
            self.SOS_Token:self.SOS,
            self.UNK_Token:self.UNK
        }
        self.n_words = len(self.w2i)

    def index_word(self,word):
        if word not in self.w2i:
            self.w2i[word] = self.n_words
            self.i2w[self.n_words] = word
            self.n_words += 1

    def index_words(self,words):
        for word in words:
            self.index_word(word)


def construct_vocabulary(file_name):
    lang = Lang()
    with open(file_name,'r') as f:
        for line in f:
            lang.index_words(line.strip().split())
    return lang


def read_dataset(file_name,lang):
    with open(file_name,'r') as f:
        for line in f:
            yield [lang.w2i[x] for x in line.strip().split()]


class WordEmbCbowNet(nn.Module):
    """
    NOTES:
        word embedding cbow net
    """
    def __init__(
        self,
        vocab_size,
        embedding_size,
        use_cuda
    ):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.use_cuda = use_cuda

        self.word_project_matrix = nn.Parameter(torch.Tensor(self.vocab_size,self.embedding_size))
        self.word_pred_matrix = nn.Parameter(torch.Tensor(self.embedding_size,self.vocab_size))
        self.reset_parameters()

    def reset_parameters(self,initrange = None):

        if not initrange:
            initrange = 1. / math.sqrt(self.weight.size(1))

        self.word_project_matrix.data.uniform_(-initrange,initrange)
        self.word_pred_matrix.data.uniform_(-initrange,initrange)

    def forward(self,input):
        input_variable = torch.zeros(1,self.vocab_size)
        for i in input:
            input_variable[0][i] += 1

        input_variable = Variable(input_variable)
        if self.use_cuda:
            input_variable = input_variable.cuda()

        word_emb = input_variable.mm(self.word_project_matrix)
        word_pred_logit = word_emb.mm(self.word_pred_matrix)

        return word_pred_logit


if __name__ == '__main__':
    config = {
        'training': {
            'use_cuda': True,
            'lr': 0.1,
            'optimizer': 'SGD',
            'iteration': 100
        },
        'model': {
            'window_size': 2,
            'embedding_size': 128,
            'hidden_size': 128,
        },
        'data': {
            'train_data': '../data/ptb/train.txt',
            'dev_data': '../data/ptb/valid.txt',
            'max_len': 100,
            'embedding_location':'/disk/xtzhao/models/nn4nlp_2017/cbow_embedding.txt'
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

    use_cuda = config['training']['use_cuda']
    window_size = config['model']['window_size']

    lang = construct_vocabulary(config['data']['train_data'])
    train_data = list(read_dataset(
        config['data']['train_data'],
        lang
    ))
    dev_data = list(read_dataset(
        config['data']['dev_data'],
        lang
    ))

    model = WordEmbCbowNet(
        vocab_size=lang.n_words,
        embedding_size=config['model']['embedding_size'],
        use_cuda=use_cuda
    )

    if use_cuda:
        model = model.cuda()

    if 'SGD' == config['training']['optimizer']:
        lr = config['training']['lr']
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif 'Adam' == config['training']['optimizer']:
        lr = config['training']['lr']
        optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.NLLLoss()

    for ITER in xrange(config['training']['iteration']):
        model.train()
        random.shuffle(train_data)
        train_words, train_loss = 0, 0.
        for sent_id, sent in enumerate(train_data):
            optimizer.zero_grad()
            padded_sent = [Lang.SOS_Token] * window_size + sent + [Lang.SOS_Token] * window_size
            all_losses = []

            for i in xrange(window_size, len(sent) + window_size):
                input_data = sent[i-window_size:i] + sent[i+1:i+window_size+1]
                pred_word_logit = model(input_data)
                gold_output = Variable(torch.LongTensor([padded_sent[i]]))
                if use_cuda:
                    gold_output = gold_output.cuda()
                # print(pred_word_logit.size(),gold_output.size())
                loss = criterion(
                    pred_word_logit,
                    gold_output
                )
                all_losses.append(loss.data[0])
                loss.backward()
            optimizer.step()
            train_loss += sum(all_losses)
            train_words += len(sent)
            if sent_id % config['management']['monitor_loss'] == 0:
                logging.info('ITER %d Sentence No. %d Loss: %.4f PPL: %.4f' % (
                    ITER, sent_id, train_loss / train_words, math.exp(train_loss / train_words)))

        model.eval()
        dev_words, dev_loss = 0, 0.
        for sent_id, sent in enumerate(dev_data):
            padded_sent = [Lang.SOS_Token] * window_size + sent + [Lang.SOS_Token] * window_size
            all_losses = []

            for i in xrange(window_size, len(sent) + window_size):
                input_data = sent[i - window_size:i] + sent[i + 1:i + window_size + 1]
                pred_word_logit = model(input_data)
                gold_output = Variable(torch.LongTensor([padded_sent[i]]))
                if use_cuda:
                    gold_output = gold_output.cuda()
                # print(pred_word_logit.size(),gold_output.size())
                loss = criterion(
                    pred_word_logit,
                    gold_output
                )
                all_losses.append(loss.data[0])
                loss.backward()

        dev_loss += sum(all_losses)
        dev_words += len(sent)
        logging.info('ITER %d Loss: %.4f PPL: %.4f' % (
            ITER, train_loss / train_words, math.exp(train_loss / train_words)))

    logging.info("Saving embedding files")
    with open(config['data']['embedding_location'], 'w') as embedding_file:
        word_embed = model.word_project_matrix.cpu().data.numpy()
        for i in xrange(lang.n_words):
            i_th_embedding = '\t'.join(map(str, word_embed[i]))
            embedding_file.write(i_th_embedding)








