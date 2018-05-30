# -*- coding: utf-8 -*-
# @Time    : 2018/5/30 21:59
# @Author  : ZengC
# @Email   : njuzengc@foxmail.com
# @File    : mnist_rnn.py
# @Software: PyCharm Community Edition

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    