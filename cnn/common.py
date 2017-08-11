import tensorflow as tf
import random


def fake_data(in_size, out_size,batch_size=1):
	fake_in = [random.uniform(0, 1) for _ in range(in_size)]
	fake_out =[0]*out_size
	fake_out[random.randint(0,len(fake_out)-1)] = 1
	return [fake_in for _ in xrange(batch_size)], [fake_out for _ in xrange(batch_size)]

def var(shape,distr="",const=0.1):
	if	distr == "T_NORMAL":
		return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))
	elif distr == "R_NORMAL":
		return tf.Variable(tf.random_normal(shape, stddev = 0.1))
	elif distr == "CONSTANT":
		return tf.Variable(tf.constant(const,shape=shape))
	else:
		return tf.Variable(tf.zeros(shape))