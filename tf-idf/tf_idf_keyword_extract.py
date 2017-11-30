#!/usr/bin/python
#-*- coding:utf8 -*-

import math
import jieba
import BeautifulSoup
import urllib,urllib2
from collections import defaultdict

class KeywordExtractor(object):
    """基于tf-idf算法的关键词抽取类"""
    def __init__(self):
        super(KeywordExtractor, self).__init__()
        #需要使用搜索引擎计算相关的idf值，使用的搜索引擎是Bing
        self.url = 'https://cn.bing.com/search'
        #包含＇的＇字的网页数目，视为全部中文网页的数目
        self.document_num = 31e6

    def get_document_num(self,keyword):
        search_word = {'q':keyword}
        search_word = urllib.urlencode(search_word)

        req = urllib2.Request(url='%s%s%s'%(self.url,'?',search_word))
        res = urllib2.urlopen(req)
        res = res.read()
        # print(res)
        res = BeautifulSoup.BeautifulSoup(res)

        content = res.findAll('span',{'class':'sb_count'})[0]
        num_str = content.text[:-4].replace(',','')
        return int(num_str)

    def term_frequency(self,document):
        word_tf = defaultdict(lambda:0.)
        total_count = len(document)
        for word in document:
            word_tf[word] += 1

        for word in word_tf.keys():
            word_tf[word] = word_tf[word] / total_count

        return word_tf

    def inverse_document_frequency(self,document):
        word_idf = defaultdict(lambda:0.)
        word_set = set()
        for word in document:
            word_set.add(word)

        for word in word_set:
            word_idf[word] = -math.log((self.get_document_num(word)+1.) / self.document_num)

        return word_idf

    def extract_keyword(self,document):
        word_tf = self.term_frequency(document)
        word_idf = self.inverse_document_frequency(document)

        word_tf_idf = defaultdict(lambda:0.)
        for word in word_tf.keys():
            word_tf_idf[word] = word_tf[word] * word_idf[word]

        max_value = -1
        keyword = None

        for word in word_tf_idf.keys():
            if word_tf_idf[word] > max_value:
                max_value = word_tf_idf[word]
                keyword = word
        return keyword.decode('utf8')

if __name__ == '__main__':
    kw_extractor = KeywordExtractor()
    document = '辛亥革命发生在什么时间'
    words = jieba.cut(document)
    words_list = []
    for word in words:
        words_list.append(word.encode('utf8'))
    
    print(kw_extractor.extract_keyword(words_list))