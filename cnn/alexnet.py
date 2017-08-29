from tensorflow.python.framework import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from time import time
from common import var
from common import fake_data
from common import fake_data_np
from common import fake_input
from common import fake_label

from common import conv2D
from common import mxPool

import argparse

#import tensorflow.python.framework
		
def net_arch(dtype=tf.float32):
	x = tf.placeholder(dtype=dtype, shape=[None,227 * 227 * 3])
	print "x:",x
	x_image = tf.reshape(x, [-1, 227,227,3])
	print "x_image:",x_image
	
	print "<Layer 1>"
	W_1 = var([11,11,3,96],"T_NORMAL",dtype=dtype)
	b_1 = var([96],"CONSTANT",0.1,dtype=dtype)
	print "W_1:", W_1
	print "b_1:", b_1
	h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_1, strides=[1,4,4,1], padding='VALID') + b_1)
	h_pool1=tf.nn.max_pool(h_conv1,ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
	print "h_conv1:",h_conv1
	print "h_pool1:",h_pool1
	
	print "<Layer 2>"
	W_2 = var([5,5,96,256],"T_NORMAL",dtype=dtype)
	b_2 = var([256],"CONSTANT",0.1,dtype=dtype)
	print "W_2:", W_2
	print "b_2:", b_2
	h_pool1 = tf.pad(h_pool1, [[0, 0], [2, 2], [2, 2], [0, 0]],"CONSTANT")#SAME OP AS BELOW WITHOUT PADDING
	h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_2, strides=[1,1,1,1], padding='VALID') + b_2)
	#h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_2, strides=[1,1,1,1], padding='SAME') + b_2)
	h_pool2=tf.nn.max_pool(h_conv2,ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
	print "h_conv2:",h_conv2
	print "h_pool2:",h_pool2
	
	print "<Layer 3>"
	W_3 = var([3,3,256,384],"T_NORMAL",dtype=dtype)
	b_3 = var([384],"CONSTANT",0.1,dtype=dtype)
	print "W_3:", W_3
	print "b_3:", b_3
	h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_3, strides=[1,1,1,1], padding='SAME') + b_3)
	print "h_conv3:",h_conv3
	
	print "<Layer 4>"
	W_4 = var([3,3,384,384],"T_NORMAL",dtype=dtype)
	b_4 = var([384],"CONSTANT",0.1,dtype=dtype)
	print "W_4:", W_4
	print "b_4:", b_4
	h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_4, strides=[1,1,1,1], padding='SAME') + b_4)
	print "h_conv4:",h_conv4
	
	print "<Layer 5>"
	W_5 = var([3,3,384,256],"T_NORMAL",dtype=dtype)
	b_5 = var([256],"CONSTANT",0.1,dtype=dtype)
	print "W_5:", W_5
	print "b_5:", b_5
	h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_5, strides=[1,1,1,1], padding='SAME') + b_5)
	h_pool5=tf.nn.max_pool(h_conv5,ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')
	print "h_conv5:",h_conv5
	print "h_pool5:",h_pool5
	
	print "<Layer 6>"
	W_6 = var([6 * 6 * 256,4096],"T_NORMAL",dtype=dtype)
	b_6 = var([4096],"CONSTANT",0.1,dtype=dtype)
	print "W_6:", W_6
	print "b_6:", b_6
	h_pool5_flat = tf.reshape(h_pool5, [-1, 6 * 6 * 256])
	h_full6 = tf.nn.relu(tf.matmul(h_pool5_flat, W_6) + b_6)
	print "h_full6:",h_full6
	
	print "<Layer 7>"
	W_7 = var([4096,1000],"T_NORMAL",dtype=dtype)
	b_7 = var([1000],"CONSTANT",0.1,dtype=dtype)
	print "W_7:", W_7
	print "b_7:", b_7	
	y_full7 = tf.nn.relu(tf.matmul(h_full6, W_7) + b_7)
	print "y_full7:",y_full7
	
	y_ = tf.placeholder(dtype=dtype, shape=[None, 1000])
	return [x,y_full7,y_]

def train_with_dict(x,y_,optimizer,iter,data,frames):
	tt = 0
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		#warm up
		optimizer.run(feed_dict={x: data[0][0], y_: data[0][1]})
		
		for i in range(iter):
			#batch = fake_data_np(227*227*3,1000,frames)
			
			lt = time()
			print "train step (",i,")"
			optimizer.run(feed_dict={x: data[i][0], y_: data[i][1]})
			lt= time() - lt
			print lt
			tt += lt
			
		print "Training Elapsed time(sec):",tt
		print "Training Time per Iteration(sec):",tt/iter
		print "Training Frame per Second:", frames*iter/tt

def train_with_queue(x,y_,optimizer):
	finput = fake_input(227*227*3,2)
	flabel = fake_label(1000,2)
	
	q = tf.FIFOQueue(capacity=2, dtypes=[tf.float32,tf.float32], shapes=[[227*227*3],[1000]])
	enque_op = q.enqueue_many([finput,flabel])
	
	input = q.dequeue()
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		optimizer.run(input)
			
def classify_with_dict(x,y_,accuracy,iter,data,frames):
	tt = 0
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		#warm up
		train_accuracy = accuracy.eval(feed_dict={x:data[0][0], y_:data[0][1]})
		
		for i in range(iter):
			#batch = fake_data_np(227*227*3,1000,frames)
			print "classify step (",i,")"
			lt = time()
			train_accuracy = accuracy.eval(feed_dict={x:data[0][0], y_:data[0][1]})
			lt= time() - lt
			print lt
			tt += lt
		print "Classification Elapsed Time(sec):",tt
		print "Classification Time per Iteration(sec):",tt/iter
		print "Classification Frame per Second:",frames*iter/tt

def bench_with_dict(dtype,op):
	(x,y_full7,y_)=net_arch(dtype=dtype)
		
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_full7))
	optimizer = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
	
	predict = tf.equal(tf.argmax(y_full7, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(predict, dtype=dtype))
	print "acc:",accuracy
	
	print "generating fake data...."
	iter = 10
	frames = 512
	if dtype == tf.float16:
		data = [fake_data_np(227*227*3,1000,frames,dtype=np.float16) for i in range(iter)]
	elif dtype == tf.float32:
		data = [fake_data_np(227*227*3,1000,frames,dtype=np.float32) for i in range(iter)]
	
	if op == "train":
		train_with_dict(x, y_, optimizer,iter,data,frames)
	elif op == "classify":
		classify_with_dict(x,y_,accuracy,iter,data,frames)

def alexnet(dtype=tf.float32,batch_size=16,dev='gpu',width=227,height=227):
	x_images = var([batch_size,3,width,height],"T_NORMAL",dtype=dtype) if dev == 'gpu' else var([batch_size,width,height,3],"T_NORMAL",dtype=dtype)
	
	W_1 = var([11,11,3,64],"T_NORMAL",dtype=dtype)
	b_1 = var([64],"CONSTANT",0.1,dtype=dtype)
	h_conv1 = conv2D(x_images,W_1,b_1,[1,1,4,4],'VALID','NCHW') if dev == 'gpu' else conv2D(x_images,W_1,b_1,[1,4,4,1],'VALID','NHWC')
	h_pool1=mxPool(h_conv1,[1,1,3,3],[1,1,2,2],'VALID','NCHW') if dev == 'gpu' else mxPool(h_conv1,[1,3,3,1],[1,2,2,1],'VALID','NHWC')
	print h_pool1
        
	W_2 = var([5,5,64,256],"T_NORMAL",dtype=dtype)
	b_2 = var([256],"CONSTANT",0.1,dtype=dtype)
	h_conv2 = conv2D(h_pool1,W_2,b_2,[1,1,1,1],'SAME','NCHW') if dev == 'gpu' else conv2D(h_pool1,W_2,b_2,[1,1,1,1],'SAME','NHWC')
	h_pool2=mxPool(h_conv2,[1,1,3,3],[1,1,2,2],'VALID','NCHW') if dev == 'gpu' else mxPool(h_conv2,[1,3,3,1],[1,2,2,1],'VALID','NHWC')
	print h_pool2
        
	W_3 = var([3,3,256,384],"T_NORMAL",dtype=dtype)
	b_3 = var([384],"CONSTANT",0.1,dtype=dtype)
	h_conv3 = conv2D(h_pool2,W_3,b_3,[1,1,1,1],'SAME','NCHW') if dev == 'gpu' else conv2D(h_pool2,W_3,b_3,[1,1,1,1],'SAME','NHWC') 
	print h_conv3
        
	W_4 = var([3,3,384,384],"T_NORMAL",dtype=dtype)
	b_4 = var([384],"CONSTANT",0.1,dtype=dtype)
	h_conv4 = conv2D(h_conv3,W_4,b_4,[1,1,1,1],'SAME','NCHW') if dev == 'gpu' else conv2D(h_conv3,W_4,b_4,[1,1,1,1],'SAME','NHWC')
	print h_conv4
        
	W_5 = var([3,3,384,256],"T_NORMAL",dtype=dtype)
	b_5 = var([256],"CONSTANT",0.1,dtype=dtype)
	h_conv5 = conv2D(h_conv4,W_5,b_5,[1,1,1,1],'SAME','NCHW') if dev == 'gpu' else conv2D(h_conv4,W_5,b_5,[1,1,1,1],'SAME','NHWC')
	h_pool5 = mxPool(h_conv5,[1,1,3,3],[1,1,2,2],'VALID','NCHW') if dev == 'gpu' else mxPool(h_conv5,[1,3,3,1],[1,2,2,1],'VALID','NHWC')
	print "h_:",h_pool5
	
	dim = h_pool5.get_shape().as_list()
	#print dim[1]*dim[2]*dim[3]
	W_6 = var([dim[1]*dim[2]*dim[3],4096],"T_NORMAL",dtype=dtype)
	b_6 = var([4096],"CONSTANT",0.1,dtype=dtype)
	h_pool5_flat = tf.reshape(h_pool5, [-1, dim[1]*dim[2]*dim[3]])
	print h_pool5
	h_full6 = tf.nn.relu(tf.matmul(h_pool5_flat, W_6) + b_6)
	print h_full6
        
	W_7 = var([4096,1000],"T_NORMAL",dtype=dtype)
	b_7 = var([1000],"CONSTANT",0.1,dtype=dtype)
	y = tf.nn.relu(tf.matmul(h_full6, W_7) + b_7)
	print y
    
	config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())
    
	config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(y.op)
		iter=100
		start = time()
		for i in range(iter):
			if (i+1) % 10 == 0:
				elapsed = time() - start
				print "<",i/10,">: frames/sec ",10*batch_size/elapsed
				start=time()
			sess.run(y.op)
	
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Optional Arguments')
	parser.add_argument("--dev",dest='device',default="gpu",type=str,help="valid options: cpu/gpu, default: gpu")
	parser.add_argument("--dtype",dest='dtype',default="float16",type=str,help="valid options: float16/float32, default: float16")
	parser.add_argument("--op",dest='op',default="classify",type=str,help="valid options: train/classify, default: classify")
	parser.add_argument("--bsize",dest='bsize',default=16,type=int,help="valid options: integer > 0, default: 16")
	parser.add_argument("--w",dest='width',default=227,type=int,help="Frame Width, valid options: integer > 0, default: 227")
	parser.add_argument("--h",dest='height',default=227,type=int,help="Frame Height, valid options: integer > 0, default: 227")
	args = parser.parse_args()
	
	if args.width <= 0:
		print "Width has to be > 0 !!!"
		exit(1)
	if args.height <= 0:
		print "Height has to be > 0 !!!"
		exit(1)
	if args.bsize <= 0:
		print "Batch size has to be > 0 !!!"
		exit(1)
		
	dtype = tf.float16 if args.dtype == "float16" else tf.float32
	with tf.device("/"+args.device+":0"):
		#bench_with_dict(dtype=dtype,op=args.op)
		alexnet(dtype=dtype,batch_size=args.bsize,dev=args.device,width=args.width, height=args.height)
