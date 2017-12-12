#!/usr/bin/python
# -*- coding:utf8 -*-

"""
    @author:xiaotian zhao
    @time:12/12/17
"""
from __future__ import print_function

import math
import random
import torch
import logging
import torch.nn as nn
import numpy as np
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
            self.embedding_size,
            self.hidden_size
        )

        self.out = nn.Linear(
            self.hidden_size,
            self.vocab_size
        )

    def forward(self, input):
        """
        Notes:
            input's batch_size is 1
        """
        # (batch_size,window_size)->(batch_size,window_size,embedding_size)
        word_embed = self.emb(
            input
        )

        # (batch_size,window_size,embedding_size) -> (batch_size, embedding_size)
        word_embed = torch.sum(
            word_embed,
            dim=1
        )

        hidden = nn.Tanh()(self.hidden(word_embed))
        word_logits = self.out(hidden)

        return nn.LogSoftmax()(word_logits)


def construct_vocabulary(file_name):
    lang = Lang()
    with open(file_name, 'r') as f:
        for line in f:
            lang.index_words(line.strip().split())
    return lang


def read_dataset(file_name, lang):
    with open(file_name, 'r') as f:
        for line in f:
            yield [lang.w2i[x] for x in line.strip().split()]


def generate_sent(config, model, lang):
    model.eval()
    hist = Variable(torch.LongTensor([[Lang.SOS_Token] * window_size]))
    if config['training']['use_cuda'] == True:
        hist = hist.cuda()
    sent = []
    while True:
        pred_word_logit = model(hist).squeeze(0)
        pred_word = pred_word_logit.cpu().data.numpy()
        pred_word = np.random.choice(lang.n_words, p=pred_word / pred_word.sum())
        if pred_word == Lang.SOS_Token or len(sent) > config['data']['max_len']:
            break
        sent.append(pred_word)
        hist = Variable(torch.LongTensor([hist.cpu().data.numpy().tolist()[1:] + [pred_word]]))
        if use_cuda:
            hist = hist.cuda()
    return sent


if __name__ == '__main__':

    config = {
        'training': {
            'use_cuda': True,
            'lr': 0.0001,
            'optimizer': 'Adam',
            'iteration': 100,
            'sent_batch_size': 10
        },
        'model': {
            'window_size': 2,
            'embedding_size': 128,
            'hidden_size': 128,
        },
        'data': {
            'train_data': '../data/ptb/train.txt',
            'dev_data': '../data/ptb/valid.txt',
            'max_len': 100
        },
        "management": {
            "monitor_loss": 200,  # monitor loss every num set by user size
            "checkpoint_freq": 500
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

    batch_count = 0
    for ITER in xrange(config['training']['iteration']):
        model.train()
        random.shuffle(train_data)
        train_words, train_loss = 0, 0.
        batch_count = 0
        all_histories = []
        all_gold_output = []
        for sent_id, sent in enumerate(train_data):
            all_losses = []

            hist = [Lang.SOS_Token] * window_size

            for next_word in sent + [Lang.SOS_Token]:
                all_histories.append(hist)
                gold_output = next_word
                all_gold_output.append(gold_output)
                hist = hist[1:] + [next_word]

            if sent_id % config['training']['sent_batch_size'] == 0:

                # for hist in all_histories:
                #     print(' '.join([lang.i2w[x] for x in hist]))
                # for gold_output in all_gold_output:
                #     print(' '.join([lang.i2w[gold_output]]))

                optimizer.zero_grad()
                all_histories = Variable(torch.LongTensor(all_histories))
                if use_cuda:
                    all_histories = all_histories.cuda()
                pred_word_probs = model(all_histories)

                all_gold_output = Variable(torch.LongTensor(all_gold_output))
                if use_cuda:
                    all_gold_output = all_gold_output.cuda()

                loss = criterion(
                    pred_word_probs,
                    all_gold_output
                )
                loss.backward()
                optimizer.step()

                train_loss += sum(loss.data)
                batch_count += 1
                all_histories = []
                all_gold_output = []

                if batch_count % config['management']['monitor_loss'] == 0:
                    logging.info('ITER %d Sentence No. %d Loss: %.4f PPL: %.4f' % (
                        ITER, sent_id, train_loss / batch_count, math.exp(train_loss / batch_count)))

        model.eval()
        dev_words, dev_loss = 0, 0.
        dev_batch_count = 0
        all_losses = []
        all_histories = []
        all_gold_output = []
        for sent_id, sent in enumerate(dev_data):

            hist = [Lang.SOS_Token] * window_size

            for next_word in sent + [Lang.SOS_Token]:
                all_histories.append(hist)
                gold_output = next_word
                all_gold_output.append(gold_output)
                hist = hist[1:] + [next_word]

            if sent_id % config['training']['sent_batch_size'] == 0:
                optimizer.zero_grad()
                all_histories = Variable(torch.LongTensor(all_histories))
                if use_cuda:
                    all_histories = all_histories.cuda()
                pred_word_probs = model(all_histories)

                all_gold_output = Variable(torch.LongTensor(all_gold_output))
                if use_cuda:
                    all_gold_output = all_gold_output.cuda()

                loss = criterion(
                    pred_word_probs,
                    all_gold_output
                )
                loss.backward()
                optimizer.step()
                all_losses = []
                all_histories = []
                all_gold_output = []

                dev_loss += sum(loss.data)
                dev_batch_count += 1

        logging.info('ITER %d Loss: %.4f PPL: %.4f' % (
            ITER, dev_loss / dev_batch_count, math.exp(dev_loss / dev_batch_count)))

        logging.info("Samples")
        for _ in xrange(5):
            sent = generate_sent(config, model, lang)
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