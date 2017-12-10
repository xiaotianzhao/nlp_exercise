#!/usr/bin/python
#-*- coding:utf8 -*-

import torch
import logging
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from util import read_config,read_data,get_minibatch
from model import PosOfTaggingBiLSTM
from evaluate import evaluate_model

if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-path',
        help='path to config file',
        type=str,
        default='./pos.config'
    )
    args = parser.parse_args()
    
    config = read_config(args.config_path)

    logging.info('Reading train data')
    train_word_seqs,train_tag_seqs,word_lang,tag_lang = read_data(
        config['data']['train_data']
    )

    logging.info('Reading test data')
    test_word_seqs,test_tag_seqs,_,_ = read_data(
        config['data']['test_data']
    )

    logging.info('Reconstruct Dictory')
    logging.info("Vocabulary Size : %d "%word_lang.n_words)
    word_lang.trim(
        config['data']['n_words']
    )

    logging.info("Trimed Vocabulary Size : %d "%word_lang.n_words)
    use_cuda = config['training']['use_cuda']
    batch_size = config['data']['batch_size']
    save_dir = config['data']['save_dir']
    load_dir = config['data']['load_dir']

    weight_mask = torch.ones(tag_lang.n_words)
    weight_mask[tag_lang.word2id['<pad>']] = 0
    # print(weight_mask)
    criterion = nn.NLLLoss(weight=weight_mask)

    if use_cuda:
        weight_mask = weight_mask.cuda()
        criterion = criterion.cuda()

    model = PosOfTaggingBiLSTM(
        word_vocab_size=word_lang.n_words,
        word_emb_size=config['model']['word_emb_dim'],
        pad_token=word_lang.word2id['<pad>'],
        hidden_size=config['model']['hidden_size'],
        tag_vocab_size=tag_lang.n_words,
        bidirectional=config['model']['bidirectional'],
        num_layers=config['model']['num_layers'],
        batch_first=config['model']['batch_first'],
        dropout=0.2,
        use_cuda=use_cuda
    )

    if use_cuda:
        model = model.cuda()

    if load_dir:
        model.load_state_dict(torch.load(
            open(load_dir)
        ))

    if config['training']['optimizer'] == 'adam':
        lr = config['training']['lrate']
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif config['training']['optimizer'] == 'adadelta':
        optimizer = optim.Adadelta(model.parameters())
    elif config['training']['optimizer'] == 'sgd':
        lr = config['training']['lrate']
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise NotImplementedError("Learning method not recommend for task")
    
    for i in xrange(1000):
        losses = []

        for j in xrange(0,len(train_word_seqs),batch_size):
            input_lines,input_mask = get_minibatch(
                train_word_seqs,
                word_lang,
                j,
                batch_size,
                max_len=config['data']['max_length'],
                add_start=False,
                add_end=False,
                use_cuda=use_cuda
            )

            output_lines,output_mask = get_minibatch(
                train_tag_seqs,
                tag_lang,
                j,
                batch_size,
                max_len=config['data']['max_length'],
                add_start=False,
                add_end=False,
                use_cuda=use_cuda
            )

            # print(input_lines)
            

            output_logit = model(input_lines)
            # print(output_lines)
            # print(torch.max(output_logit,dim=2)[-1])
            optimizer.zero_grad()

            loss = criterion(
                output_logit.contiguous().view(-1,tag_lang.n_words),
                output_lines.view(-1)
            )

            losses.append(loss.data[0])
            loss.backward()
            optimizer.step()

            if j % config['management']['monitor_loss'] == 0:
                precision = evaluate_model(
                    model,
                    test_word_seqs,
                    test_tag_seqs,
                    word_lang,
                    tag_lang,
                    config
                )
                logging.info('Epoch : %d Minibatch: %d Loss : %.5f ,Precision: %.5f'%(i,j,loss.data[0],precision))

        # logging.info('Evaluting model...')
        # precision = evaluate_model(
        #     model,
        #     test_word_seqs,
        #     test_tag_seqs,
        #     word_lang,
        #     tag_lang,
        #     config
        # )
        # logging.info('Precision : %.4f',precision)
