#!/usr/bin/python
#-*- coding:utf8 -*-

import math

class TFIDF(object):
    """docstring for TFIDF"""
    def __init__(self):
        super(TFIDF, self).__init__()

    def term_frequency(self,key_word,document):
        """
            词频　＝　某个词在文章中出现的次数 / 文章中的总词数
        """
        key_word_count = 0
        for word in document.split():
            if word == key_word:
                key_word_count += 1

        return 1.*key_word_count / len(document.split())

    def inverse_document_frequency(self,key_word,documents):
        """
            逆文档频率　＝　log(语料库的总文档数 / (包含该词的文档数 + 1))
        """
        has_kw_count = 0
        for document in documents:
            for word in document.split():
                if key_word == word:
                    has_kw_count += 1
                    break
        return math.log(len(documents) / (1.*has_kw_count + 1.))

    def tf_idf(self,key_word,documents):
        tf_idf_list = []
        idf_value = self.inverse_document_frequency(key_word,documents)
        for document in documents:
            tf_value = self.term_frequency(key_word,document)
            tf_idf = tf_value * idf_value
            tf_idf_list.append(tf_idf)
        return tf_idf_list

if __name__ == '__main__':

    documents = [
    u'孙中山　是　中华民国　的　第一任　总统',
    u'孙中山　是　中华民国　的　国父',
    u'蒋介石　担任　过　黄埔军校　的　校长']

    tfidf = TFIDF()
    print(tfidf.tf_idf(u'蒋介石',documents))

            



        