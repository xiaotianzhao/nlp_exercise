#!/usr/bin/python
#-*- coding:utf8 -*-

import numpy as np

class TextRankKeyWordExtractor(object):
    """docstring for TextRank"""
    def __init__(self, window_size = 2):
        super(TextRankKeyWordExtractor, self).__init__()
        self.window_size = window_size
        self.damping_coefficient = 0.85

    def get_matrix(self,words):
        # word_windows = []
        words_len = len(words)

        words_set = set(words)
        words_set_len = len(words_set)
        matrix = np.zeros((words_set_len,words_set_len))
        print(words_set)

        words_dict = {}
        for index,word in enumerate(words_set):
            words_dict[word] = index

        for i in range(0,words_len-self.window_size+1):
            # word_window = []
            for j in range(0,self.window_size-1):
                # word_window.append(words[j+i])
                word_i_position,word_j_position = words_dict[words[i+j]],words_dict[words[i+j+1]]
                matrix[word_i_position,word_j_position] = 1.
                matrix[word_j_position,word_i_position] = 1.
            # word_windows.append(word_window)

        print(matrix)
        return matrix / matrix.sum(axis=0)

    def extract_keyword(self,words):
        if len(words) < self.window_size:
            raise Exception('window_size 要小于输入序列长度')
        matrix = self.get_matrix(words)
        print(matrix)
        pr = np.ones((matrix.shape[0],1))
        for i in range(100):
            # print(pr.dot(matrix))
            pr = (1-self.damping_coefficient) + self.damping_coefficient*matrix.dot(pr)
        print(pr)

if __name__ == '__main__':
    kw_extractor = TextRankKeyWordExtractor(2)
    kw_extractor.extract_keyword(['辛亥革命','发生','时间','地点'])
        