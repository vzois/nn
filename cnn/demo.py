from tensorflow.python.framework import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from time import time
import argparse


from common import var
from common import fake_input
from common import fake_label

def mmul(dtype=tf.float32,n=1024):
    
    with tf.device("/gpu:0"):
        A = tf.Variable(tf.ones((n, n), dtype=dtype))
        B = tf.Variable(tf.ones((n, n), dtype=dtype))
        C = tf.matmul(A,B)
    
    config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(C.op)
    
    iter=10
    start = time()
    for i in range(iter):
        sess.run(C.op)
    elapsed = time() - start
    
    ops = 2 * (n ** 3)
    rate = ((iter * ops) / elapsed) /  (10**9)
    print "MMUL GFLOPS:",rate
    
def simple_net_classify(dtype=tf.float32,batch_size=16):
    with tf.device("/gpu:0"):
        x_images = var([batch_size,3,227,227],"T_NORMAL",dtype=dtype)
        #x_images = tf.transpose(x_images, [0, 3, 1, 2])
        W_1 = var([11,11,3,64],"T_NORMAL",dtype=dtype)
        b_1 = var([64],"CONSTANT",0.1,dtype=dtype)
        h_conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_images, W_1, strides=[1,1,4,4], padding='VALID',data_format='NCHW'), b_1, data_format='NCHW'))
        h_pool1=tf.nn.max_pool(h_conv1,ksize=[1,1,3,3], strides=[1,1,2,2], padding='VALID',data_format='NCHW')
        print h_pool1
        
        W_2 = var([5,5,64,256],"T_NORMAL",dtype=dtype)
        b_2 = var([256],"CONSTANT",0.1,dtype=dtype)
        h_conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(h_pool1, W_2, strides=[1,1,1,1], padding='SAME',data_format='NCHW'), b_2,data_format='NCHW'))
        h_pool2=tf.nn.max_pool(h_conv2,ksize=[1,1,3,3], strides=[1,1,2,2], padding='VALID',data_format='NCHW')
        print h_pool2
        
        W_3 = var([3,3,256,384],"T_NORMAL",dtype=dtype)
        b_3 = var([384],"CONSTANT",0.1,dtype=dtype)
        h_conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(h_pool2, W_3, strides=[1,1,1,1], padding='SAME',data_format='NCHW'),b_3,data_format='NCHW'))
        print h_conv3
        
        W_4 = var([3,3,384,384],"T_NORMAL",dtype=dtype)
        b_4 = var([384],"CONSTANT",0.1,dtype=dtype)
        h_conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(h_conv3, W_4, strides=[1,1,1,1], padding='SAME',data_format='NCHW'),b_4,data_format='NCHW'))
        print h_conv4
        
        W_5 = var([3,3,384,256],"T_NORMAL",dtype=dtype)
        b_5 = var([256],"CONSTANT",0.1,dtype=dtype)
        h_conv5 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(h_conv4, W_5, strides=[1,1,1,1], padding='SAME',data_format='NCHW'),b_5,data_format='NCHW'))
        h_pool5=tf.nn.max_pool(h_conv5,ksize=[1,1,3,3], strides=[1,1,2,2], padding='VALID',data_format='NCHW')
        print h_pool5
        
        W_6 = var([6 * 6 * 256,4096],"T_NORMAL",dtype=dtype)
        b_6 = var([4096],"CONSTANT",0.1,dtype=dtype)
        h_pool5_flat = tf.reshape(h_pool5, [-1, 6 * 6 * 256])
        h_full6 = tf.nn.relu(tf.matmul(h_pool5_flat, W_6) + b_6)
        print h_full6
        
        W_7 = var([4096,1000],"T_NORMAL",dtype=dtype)
        b_7 = var([1000],"CONSTANT",0.1,dtype=dtype)    
        y_full7 = tf.nn.relu(tf.matmul(h_full6, W_7) + b_7)
        print y_full7
    
    config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    
    sess.run(y_full7.op)
    iter=100
    start = time()
    for i in range(iter):
        sess.run(y_full7.op)
    elapsed = time() - start
    print "frames/sec:",iter*batch_size/elapsed
    

def simple_queue_example(dtype=tf.float32,capacity=10,batch_size=8):
    #flabel = fake_label(1000,2)
    #x = tf.placeholder(dtype=dtype, shape=[None,227 * 227 * 3])
    #print "x_p:",x
    
    #finput = fake_input(227*227*3,2)
    finput = tf.random_normal([batch_size,227*227*3],dtype=dtype)
    print "finput:",finput
    q = tf.FIFOQueue(capacity=capacity+1, dtypes=[dtype], shapes=[batch_size,227*227*3])
    print "q:",q
    qlist = [finput for _ in range(capacity+1)]
    init = q.enqueue_many((qlist,))
    
    x = q.dequeue()
    print "x:",x
    x = tf.reshape(x, [batch_size, 3, 227,227])
    print "x_i:",x
    
    W_1 = var([11,11,3,96],"T_NORMAL",dtype=dtype)
    b_1 = var([1,96,1,1],"CONSTANT",0.1,dtype=dtype)
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_1, strides=[1,1,4,4], padding='VALID',data_format='NCHW')+b_1)

    config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(init)
    
    sess.run(h_conv1)
    start = time()
    for _ in range(capacity):
        print "iter>"
        sess.run(h_conv1)
    elapsed = time() - start
    
    print "time(sec):",elapsed
    print "frames/sec:",capacity*batch_size/elapsed
    
    

if __name__ == "__main__":
    mmul(tf.float32,8192)

    simple_net_classify(tf.float16,512)