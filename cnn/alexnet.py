import tensorflow as tf

from time import time
from common import var
from common import fake_data

def net():
	x = tf.placeholder(tf.float32, shape=[None,227 * 227 * 3])
	print "x:",x
	x_image = tf.reshape(x, [-1, 227,227,3])
	print "x_image:",x_image
	
	print "<Layer 1>"
	W_1 = var([11,11,3,96],"T_NORMAL")
	b_1 = var([96],"CONSTANT",0.1)
	print "W_1:", W_1
	print "b_1:", b_1
	h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_1, strides=[1,4,4,1], padding='VALID') + b_1)
	h_pool1=tf.nn.max_pool(h_conv1,ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
	print "h_conv1:",h_conv1
	print "h_pool1:",h_pool1
	
	print "<Layer 2>"
	W_2 = var([5,5,96,256],"T_NORMAL")
	b_2 = var([256],"CONSTANT",0.1)
	print "W_2:", W_2
	print "b_2:", b_2
	h_pool1 = tf.pad(h_pool1, [[0, 0], [2, 2], [2, 2], [0, 0]],"CONSTANT")#SAME OP AS BELOW WITHOUT PADDING
	h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_2, strides=[1,1,1,1], padding='VALID') + b_2)
	#h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_2, strides=[1,1,1,1], padding='SAME') + b_2)
	h_pool2=tf.nn.max_pool(h_conv2,ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
	print "h_conv2:",h_conv2
	print "h_pool2:",h_pool2
	
	print "<Layer 3>"
	W_3 = var([3,3,256,384],"T_NORMAL")
	b_3 = var([384],"CONSTANT",0.1)
	print "W_3:", W_3
	print "b_3:", b_3
	h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_3, strides=[1,1,1,1], padding='SAME') + b_3)
	print "h_conv3:",h_conv3
	
	print "<Layer 4>"
	W_4 = var([3,3,384,384],"T_NORMAL")
	b_4 = var([384],"CONSTANT",0.1)
	print "W_4:", W_4
	print "b_4:", b_4
	h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_4, strides=[1,1,1,1], padding='SAME') + b_4)
	print "h_conv4:",h_conv4
	
	print "<Layer 5>"
	W_5 = var([3,3,384,256],"T_NORMAL")
	b_5 = var([256],"CONSTANT",0.1)
	print "W_5:", W_5
	print "b_5:", b_5
	h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_5, strides=[1,1,1,1], padding='SAME') + b_5)
	h_pool5=tf.nn.max_pool(h_conv5,ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
	print "h_conv5:",h_conv5
	print "h_pool5:",h_pool5
	
	print "<Layer 6>"
	W_6 = var([6 * 6 * 256,4096],"T_NORMAL")
	b_6 = var([4096],"CONSTANT",0.1)
	print "W_6:", W_6
	print "b_6:", b_6
	h_pool5_flat = tf.reshape(h_pool5, [-1, 6 * 6 * 256])
	h_full6 = tf.nn.relu(tf.matmul(h_pool5_flat, W_6) + b_6)
	print "h_full6:",h_full6
	
	print "<Layer 7>"
	W_7 = var([4096,1000],"T_NORMAL")
	b_7 = var([1000],"CONSTANT",0.1)
	print "W_7:", W_7
	print "b_7:", b_7	
	y_full7 = tf.nn.relu(tf.matmul(h_full6, W_7) + b_7)
	print "y_full7:",y_full7
	
	y_ = tf.placeholder(tf.float32, shape=[None, 1000])	
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_full7))
	optimizer = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
	
	predict = tf.equal(tf.argmax(y_full7, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))
	print "acc:",accuracy

	iter = 10
	frames = 512
	tt = 0
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		for i in range(iter):
			
			batch = fake_data(227*227*3,1000,frames)
			lt = time()
			print "train step (",i,")"
			optimizer.run(feed_dict={x: batch[0], y_: batch[1]})
			lt= time() - lt
			print lt
			tt += lt
			
		print "Training Elapsed time(sec):",tt
		print "Training Time per Iteration(sec):",tt/iter
		print "Training Frame per Second:", frames*iter/tt
		
		tt = 0
		iter = 10
		frames = 512
		for i in range(iter):
			batch = fake_data(227*227*3,1000,frames)
			print "classify step (",i,")"
			lt = time()
			train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1]})
			lt= time() - lt
			print lt
			tt += lt
		print "Classification Elapsed Time(sec):",tt
		print "Classification Time per Iteration(sec):",tt/iter
		print "Classification Frame per Second:",frames*iter/tt
	
net()