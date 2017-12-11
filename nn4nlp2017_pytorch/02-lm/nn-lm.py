#!/usr/bin/python
#-*- coding:utf8 -*-

"""
    @author:xiaotianzhao
    @time:2017/12/10
"""

from __future__ import print_function

import math
import random
import torch
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

class NNLM(nn.Module):
    """Neural Network Language Model"""
    def __init__(
        self,
        embedding_size,
        hidden_size,
        vocab_size,
        window_size
    ):
        super(NNLM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.window_size = window_size

        self.emb = nn.Embedding(
            self.vocab_size,
            self.embedding_size
        )

        self.hidden = nn.Linear(
            self.embedding_size * self.window_size,
            self.hidden_size
        )

        self.out = nn.Linear(
            self.hidden_size,
            self.vocab_size
        )

    def forward(self,input):
        """
        Notes:
            input's batch_size is 1
        """
        #(1,window_size)->(1,window_size,embedding_size)->(window_size,embedding_size)
        word_embed = self.emb(
            input
        ).squeeze(0)

        #(window_size,embedding_size) -> (1,window_size*embedding_size)
        word_embed = torch.cat(
            [word_embed_i.unsqueeze(0) for word_embed_i in word_embed],
            dim=1
        )

        hidden = nn.Tanh()(self.hidden(word_embed))
        word_logits = nn.Tanh()(self.out(hidden))

        return nn.LogSoftmax()(word_logits)

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

def generate_sent(config,model):
    model.eval()
    hist = Variable(torch.LongTensor([Lang.SOS_Token] * window_size))
    if config['training']['use_cuda'] == True:
        hist = hist.cuda()
    sent = []
    while True:
        pred_word_logit = model(hist)
        pred_word = torch.max(pred_word_logit, dim=1)[1].cpu().data[0]
        if pred_word == Lang.SOS_Token or len(sent) > config['data']['max_len']:
            break
        sent.append(pred_word)
        hist = Variable(torch.LongTensor(hist.cpu().data.numpy().tolist()[1:] + [pred_word]))
        if use_cuda:
            hist = hist.cuda()
    return sent
        
if __name__ == '__main__':

    config = {
        'training':{
            'use_cuda':True,
            'lr':0.01,
            'optimizer':'SGD',
            'iteration':100
        },
        'model':{
            'window_size':2,
            'embedding_size':128,
            'hidden_size':128,
        },
        'data':{
            'train_data':'../data/ptb/train.txt',
            'dev_data':'../data/ptb/valid.txt',
            'max_len':100
        },
        "management": {
            "monitor_loss": 50,
            "checkpoint_freq":500
        }
    }

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    use_cuda = config['training']['use_cuda']
    window_size = config['model']['window_size']
    max_len = config['data']['max_len']
    
    lang = construct_vocabulary(config['data']['train_data'])
    train_data = list(read_dataset(
        config['data']['train_data'],
        lang
    ))
    dev_data = list(read_dataset(
        config['data']['dev_data'],
        lang
    ))

    model = NNLM(
        embedding_size=config['model']['embedding_size'],
        hidden_size=config['model']['hidden_size'],
        vocab_size=lang.n_words,
        window_size=window_size
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
        train_words,train_loss = 0,0.
        for sent_id,sent in enumerate(train_data):
            optimizer.zero_grad()

            hist = Variable(torch.LongTensor([Lang.SOS_Token] * window_size))
            if use_cuda:
                hist = hist.cuda()
            all_losses = []

            for next_word in sent + [Lang.SOS_Token]:
                pred_word_logit = model(hist)
                gold_output = Variable(torch.LongTensor([next_word]))
                if use_cuda:
                    gold_output = gold_output.cuda()
                # print(pred_word_logit.size(),gold_output.size())
                loss = criterion(
                    pred_word_logit,
                    gold_output
                )
                all_losses.append(loss.data[0])
                hist = Variable(torch.LongTensor(hist.cpu().data.numpy().tolist()[1:]+[next_word]))
                if use_cuda:
                    hist = hist.cuda()
                loss.backward()
            optimizer.step()
            train_loss += sum(all_losses)
            train_words += len(sent)
            if sent_id % config['management']['monitor_loss'] == 0:
                logging.info('ITER %d Sentence No. %d Loss: %.4f PPL: %.4f' % (
                    ITER, sent_id , train_loss / train_words, math.exp(train_loss / train_words)))

        model.eval()
        dev_words ,dev_loss = 0,0.
        for sent_id,sent in enumerate(dev_data):
            # optimizer.zero_grad()
            hist = Variable(torch.LongTensor([Lang.SOS_Token] * window_size))
            if use_cuda:
                hist = hist.cuda()
            all_losses = []

            for next_word in sent + [Lang.SOS_Token]:
                pred_word_logit = model(hist)
                gold_output = Variable(torch.LongTensor([next_word]))
                if use_cuda:
                    gold_output = gold_output.cuda()
                # print(pred_word_logit.size(),gold_output.size())
                loss = criterion(
                    pred_word_logit,
                    gold_output
                )
                all_losses.append(loss.data[0])
                hist = Variable(torch.LongTensor(hist.cpu().data.numpy().tolist()[1:] + [next_word]))
                if use_cuda:
                    hist = hist.cuda()

        dev_loss += sum(all_losses)
        dev_words += len(sent)
        logging.info('ITER %d Loss: %.4f PPL: %.4f' % (
            ITER, train_loss / train_words, math.exp(train_loss / train_words)))

        logging.info("Samples")
        for _ in xrange(5):
            sent = generate_sent(config, model)
            logging.info(" ".join([lang.i2w[x] for x in sent]))

            # #caculate the whole sentence's loss
            # hist = [Lang.SOS_Token]*window_size
            # hist = Variable(torch.LongTensor(hist))
            #
            # if use_cuda:
            #     hist = hist.cuda()
            #
            # all_losses = []
            # # if (sent_id + 1) % 1 == 0:
            # optimizer.zero_grad()
            #
            # for next_word in sent+[Lang.SOS_Token]:
            #     word_logit = model(hist)
            #     gold_output = Variable(torch.LongTensor([next_word]))
            #     if use_cuda:
            #         gold_output = gold_output.cuda()
            #     loss = criterion(
            #         word_logit,
            #         gold_output
            #     )
            #     all_losses.append(loss.data[0])
            #     hist = Variable(torch.LongTensor(hist.cpu().data.numpy().tolist()[1:]+[next_word]))
            #     if use_cuda:
            #         hist = hist.cuda()
            #     loss.backward()
            #
            # if sent_id % config['management']['monitor_loss'] == 0:
            #     logging.info('Sentence No. %d Loss: %.4f PPL: %.4f'%(sent_id,sum(all_losses)/len(sent),math.exp(sum(all_losses)/len(sent))))
            # # if (sent_id + 1) % 1 == 0:
            #     # nn.utils.clip_grad_norm(model.parameters(), 3)
            # optimizer.step()