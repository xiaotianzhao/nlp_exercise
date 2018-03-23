#!/usr/bin/python
#-*- coding:utf8 -*-

from __future__ import print_function
import math

class BLEUComputer(object):
	"""docstring for BLEUComputer"""
	def __init__(self):
		super(BLEUComputer, self).__init__()

	@staticmethod
	def bleu(translate_str, golden_strs, max_n_gram):
		translate_str = translate_str.strip().split()
		golden_strs = [golden_str.strip().split() for golden_str in golden_strs]

		translate_str_len = len(translate_str)
		golden_lens = [len(golden_str) for golden_str in golden_strs]

		translate_cnt = {}
		golden_cnt = []

		for i in xrange(0, max_n_gram):
			for j in xrange(0, translate_str_len - i):

				word = ' '.join(translate_str[j: j + (i + 1)])
				# print(word+' ')
				if word not in translate_cnt:
					translate_cnt[word] = 1
				else:
					translate_cnt[word] += 1

		for i in xrange(0, max_n_gram):
			for golden_str in golden_strs:

				golden_cnt_tmp = {}
				for j in xrange(0, len(golden_str) - i):
					word = ' '.join(golden_str[j: j + (i + 1)])
					# print(word+' ')
					if word not in golden_cnt_tmp:
						golden_cnt_tmp[word] = 1
					else:
						golden_cnt_tmp[word] += 1

				golden_cnt.append(golden_cnt_tmp)

		scores = [0.0 for i in range(max_n_gram)]

		for term in translate_cnt.keys():

			term_cnt_in_gloden = []
			for golden_cnt_tmp in golden_cnt:
				if term in golden_cnt_tmp:
					term_cnt_in_gloden.append(golden_cnt_tmp[term])
				else:
					term_cnt_in_gloden.append(0)

			# print(translate_cnt[term],term,min(translate_cnt[term], max(term_cnt_in_gloden)))
			scores[len(term.split()) - 1] += min(translate_cnt[term], max(term_cnt_in_gloden))

		scores = [scores[i] /( 1.0 * (translate_str_len - i)) for i in xrange(0, max_n_gram)]
		scores = [math.log(score) for score in scores]

		weight = [1.0 / max_n_gram for i in xrange(0, max_n_gram)]

		bleu_score = math.exp(sum([scores[i] * weight[i] for i in xrange(0, max_n_gram)]))

		if len(golden_lens) <= 0:
			return 0.0

		ls = golden_lens[0]
		for golden_len in golden_lens:
			if abs(golden_len - translate_str_len) < abs(ls - translate_str_len):
				ls = golden_len

		BP = 1
		if translate_str_len <= ls:
			BP = math.exp(1 - 1.0 * ls / translate_str_len)

		return BP * bleu_score
		

if __name__ == '__main__':
	translate_str = 'Going to play basketball this afternoon ?'
	golden_strs = ['Going to play basketball in the afternoon ?']

	print(BLEUComputer.bleu(translate_str, golden_strs, 4))
