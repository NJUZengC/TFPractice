# -*- coding: utf-8 -*-
# @Time    : 2018/5/30 13:47
# @Author  : ZengC
# @Email   : njuzengc@foxmail.com
# @File    : Demo.py
# @Software: PyCharm Community Edition

import numpy as np
import tensorflow as tf

def generate():
    x_data = np.linspace(-1,1,300)[:,np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape)
    y_data = np.square(x_data) - 0.5 + noise
    return x_data,y_data

def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size,out_size]))

    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)


    Wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

if __name__ == '__main__':
    x_data,y_data = generate()
    xs = tf.placeholder(tf.float32, [None, 1],)
    ys = tf.placeholder(tf.float32, [None, 1])
    h1 = add_layer(xs,1,20,activation_function=tf.nn.relu)
    prediction = add_layer(h1,20,1,None)

    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(2000):
            sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
            if i%100 == 0:
                print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))