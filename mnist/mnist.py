from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

from time import time
import numpy as np

import random

def fake_data(in_size, out_size,batch_size=1):
	fake_in = [random.uniform(0, 1) for _ in range(in_size)]
	fake_out =[0]*out_size
	fake_out[random.randint(0,len(fake_out)-1)] = 1
	return [fake_in for _ in xrange(batch_size)], [fake_out for _ in xrange(batch_size)]

def simple_model(mnist):
	x = tf.placeholder(tf.float32, [None, 784])#None indicates not restriction on size of that dimension # Arbitrary batch size
	W = tf.Variable(tf.zeros([784,10]))
	b = tf.Variable(tf.zeros([10]))
	
	y = tf.nn.softmax(tf.matmul(x,W) + b)
	y_= tf.placeholder(tf.float32,[None,10])#correct output
	
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))#It is numerically unstable
	train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	
	for _ in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)#train, test, validate dataset # batch of 100
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1,shape=shape)
	return tf.Variable(initial)
	
def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def	conv2d(x,W,stride=[1,1,1,1]):
	return tf.nn.conv2d(x, W, strides=stride, padding='SAME')
	
def complex_model(mnist):
	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	
	x_image = tf.reshape(x, [-1, 28, 28, 1])
	print "x:",x
	print "x_:",x_image

	W_conv1 = weight_variable([4, 4, 1, 32])
	print "0:",W_conv1
	b_conv1 = bias_variable([32])
	
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	print "1:",h_conv1
	h_pool1 = max_pool_2x2(h_conv1)
	print "2:",h_pool1

	W_conv2 = weight_variable([4, 4, 32, 64])
	print "3:",W_conv2
	b_conv2 = bias_variable([64])
	
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	print "4:",h_conv2
	h_pool2 = max_pool_2x2(h_conv2)
	print "5:",h_pool2

	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(1000):
			batch = mnist.train.next_batch(50)
			#print batch
			#break
			if i % 100 == 0:
				train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob: 1.0})
				print('step %d, training accuracy %g' % (i, train_accuracy))
				
			train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

mnist = input_data.read_data_sets("MNISTS/",one_hot=True)

simple_model(mnist)
with tf.device("/gpu:0"):
	complex_model(mnist)
