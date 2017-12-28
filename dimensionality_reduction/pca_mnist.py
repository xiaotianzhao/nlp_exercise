#!/usr/bin/python
# -*- coding:utf8 -*-

"""
    @author:xiaotian zhao
    @time:12/21/17
"""


import sys
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

from PIL import Image

def pca(data_matrix, top_n_feature=sys.maxint):
    num_data, dim = data_matrix.shape

    mean_vals = data_matrix.mean(axis=0)
    mean_removed = data_matrix - mean_vals

    cov_mat = np.cov(mean_removed, rowvar=0)

    # eig_vars, eig_vects =


mnist = input_data.read_data_sets("/home/xtzhao/datasets/mnist", one_hot=True)
images = mnist.train.images
labels = mnist.train.labels

