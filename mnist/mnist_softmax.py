# -*- coding: utf-8 -*-
# @Time    : 2018/5/30 14:42
# @Author  : ZengC
# @Email   : njuzengc@foxmail.com
# @File    : mnist_softmax.py
# @Software: PyCharm Community Edition

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf






if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i%100 == 0:
            print(sess.run(accuracy,feed_dict={x: batch_xs, y_: batch_ys}))