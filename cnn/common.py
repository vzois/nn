import tensorflow as tf
import random
import numpy as np

DEVICE="/gpu:0"

def fake_input(in_size,batch_size):
	fake_in = [random.uniform(0, 1) for _ in range(in_size)]
	return np.array([fake_in for _ in xrange(batch_size)], dtype=np.float32)

def fake_label(out_size,batch_size):
	fake_out = []
	fake_out =[0]*out_size
	fake_out[random.randint(0,len(fake_out)-1)] = 1
	return np.array([fake_out for _ in xrange(batch_size)], dtype=np.float32)

def fake_data(in_size, out_size,batch_size=1):
	fake_in = [random.uniform(0, 1) for _ in range(in_size)]
	fake_out =[0]*out_size
	fake_out[random.randint(0,len(fake_out)-1)] = 1
	return [fake_in for _ in xrange(batch_size)], [fake_out for _ in xrange(batch_size)]

def fake_data_np(in_size,out_size,batch_size=1,dtype=np.float32):
	fake_in = [random.uniform(0, 1) for _ in range(in_size)]
	fake_out =[0]*out_size
	fake_out[random.randint(0,len(fake_out)-1)] = 1
	return np.array([fake_in for _ in xrange(batch_size)], dtype=dtype), np.array([fake_out for _ in xrange(batch_size)], dtype=dtype)


def var(shape,distr="",const=0.1,dtype=tf.float32):
	global DEVICE
	if	distr == "T_NORMAL":
		with tf.device(DEVICE):
			return tf.Variable(tf.truncated_normal(shape, stddev = 0.1, dtype=dtype))
	elif distr == "R_NORMAL":
		with tf.device(DEVICE):
			return tf.Variable(tf.random_normal(shape, stddev = 0.1, dtype=dtype))
	elif distr == "CONSTANT":
		with tf.device(DEVICE):
			return tf.Variable(tf.constant(const,shape=shape,dtype=dtype))
	else:
		with tf.device(DEVICE):
			return tf.Variable(tf.zeros(shape,dtype=dtype))
		

def svar(shape,distr="",const=0.1):
	global DEVICE
	if	distr == "T_NORMAL":
		with tf.device(DEVICE):
			return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))
	elif distr == "R_NORMAL":
		with tf.device(DEVICE):
			return tf.Variable(tf.random_normal(shape, stddev = 0.1))
	elif distr == "CONSTANT":
		with tf.device(DEVICE):
			return tf.Variable(tf.constant(const,shape=shape))
	else:
		with tf.device(DEVICE):
			return tf.Variable(tf.zeros(shape))
	