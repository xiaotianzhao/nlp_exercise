#!/usr/bin/python
#-*- coding:utf8 -*-

from __future__ import print_function

import json
import torch
from torch.autograd import Variable

def read_config(config_path):
    config = json.load(open(config_path,'r'))
    return config

def read_data(file_path):
    
    word_seqs,tag_seqs = [],[]
    word_lang,tag_lang = Lang(),Lang()

    with open(file_path,'r') as f:
        for line in f:
            word_seq = []
            tag_seq = []
            for word_tag in line.strip().split():
                try:
                    word,tag = word_tag.split('_')
                    word_seq.append(word)
                    tag_seq.append(tag)
                    word_lang.index_word(word)
                    tag_lang.index_word(tag)
                except Exception as e:
                    print(word_tag) 

            word_seqs.append(word_seq)
            tag_seqs.append(tag_seq)

    return word_seqs,tag_seqs,word_lang,tag_lang

def get_minibatch(
    lines,
    lang,
    index,
    batch_size,
    max_len,
    add_start=False,
    add_end=False,
    use_cuda=True
):
    lines = lines[index:index+batch_size]
    lines = [line[:max_len] for line in lines]

    lens = [len(line) for line in lines]
    max_len = max(lens)

    lines = [
        [lang.word2id[w] if w in lang.word2id else lang.word2id['<unk>'] for w in line] +
        [lang.word2id['<pad>']] * (max_len - len(line))
        for line in lines
    ]

    mask = [
        ([1]*(l))+([0] * (max_len-l))
        for l in lens
    ]

    lines = Variable(torch.LongTensor(lines))
    mask = Variable(torch.FloatTensor(mask))

    if use_cuda:
        return lines.cuda(),mask.cuda()
    else:
        return lines,mask

class Lang(object):
    # SOS = '<s>'
    # EOS = '</s>'
    PAD = '<pad>'
    UNK = '<unk>'
    # SOS_token = 0
    # EOS_token = 1
    PAD_Token = 0
    UNK_Token = 1

    """Language dictory"""
    def __init__(self):
        super(Lang, self).__init__()
        self.word2id = {
            # SOS:SOS_token,
            # EOS:EOS_token,
            self.PAD:self.PAD_Token,
            self.UNK:self.UNK_Token
        }
        self.id2word = {
            # SOS_token:SOS,
            # EOS_token:EOS,
            self.PAD_Token:self.PAD,
            self.UNK_Token:self.UNK
        }
        self.word2count = {}
        self.n_words = 2

    def index_word(self,word):
        if word not in self.word2id:
            self.word2id[word] = self.n_words
            self.id2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def index_words(self,sentence):
        for word in sentence.split():
            self.index_word(word)

    def trim(self,vocab_size):
        """
            param:vocab_size the vocabulary size setted by user
        """
        self.word2count = sorted(self.word2count.items(),key=lambda item:item[1],reverse=True)
        self.word2count = dict(self.word2count[:vocab_size-2])
        # print(self.word2count)
        self.word2id = {
            # SOS:SOS_token,
            # EOS:EOS_token,
            self.PAD:self.PAD_Token,
            self.UNK:self.UNK_Token
        }
        self.id2word = {
            # SOS_token:SOS,
            # EOS_token:EOS,
            self.PAD_Token:self.PAD,
            self.UNK_Token:self.UNK
        }
        self.n_words = 2

        for word in self.word2count.keys():
            self.word2id[word] = self.n_words
            self.id2word[self.n_words] = word
            self.n_words += 1