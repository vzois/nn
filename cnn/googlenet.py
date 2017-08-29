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
from common import avgPool

from common import var_count

import argparse


def inception_(name,layers, player,dtype,dev):
    
    op_list = []
    for l in layers:
        type = l[0]
        input_channels=l[1]
        output_channels=l[2]
        shape=l[3]
        stride=l[4]
        padding=l[5]
        
        
#         print "type:",type
#         print "input_channels:",input_channels
#         print "output_channels:",output_channels
#         print "shape:",shape
#         print "stride:",stride
#         print "padding:",padding
        
        with tf.variable_scope(name):
            op = None
            if type == 'conv':
                W_ = var([shape[0],shape[0],input_channels,output_channels],"T_NORMAL",dtype=dtype)
                b_ = var([output_channels],"CONSTANT",0.1,dtype=dtype)
                op = conv2D(player,W_,b_,[1,1,stride[0],stride[1]],padding,'NCHW') if dev == 'gpu' else conv2D(h_conv2,W_3,b_3,[1,stride[0],stride[1],1],padding,'NHWC')
            elif type == 'mxpool':
                op = mxPool(player,[1,1,shape[0],shape[1]],[1,1,stride[0],stride[1]],padding,'NCHW') if dev == 'gpu' else mxPool(player,[1,shape[0],shape[1],1],[1,stride[0],stride[1],1],padding,'NHWC')
                
            op_list.append(op)
            print op
    return tf.concat(op_list,1) if dev == 'gpu' else tf.concat(op_list,3) 
        
def googlenet(dtype=tf.float32,batch_size=16,dev='gpu',width=227, height=227):
    x_images = var([batch_size,3,width,height],"T_NORMAL",dtype=dtype) if dev == 'gpu' else var([batch_size,width,height,3],"T_NORMAL",dtype=dtype)
    
    W_1 = var([7,7,3,64],"T_NORMAL",dtype=dtype)
    b_1 = var([64],"CONSTANT",0.1,dtype=dtype)
    h_conv1 = conv2D(x_images,W_1,b_1,[1,1,2,2],'SAME','NCHW') if dev == 'gpu' else conv2D(x_images,W_1,b_1,[1,2,2,1],'SAME','NHWC')
    print h_conv1
    h_pool1=mxPool(h_conv1,[1,1,3,3],[1,1,2,2],'SAME','NCHW') if dev == 'gpu' else mxPool(h_conv1,[1,3,3,1],[1,2,2,1],'SAME','NHWC')
    print h_pool1
    
    W_2 = var([1,1,64,64],"T_NORMAL",dtype=dtype)
    b_2 = var([64],"CONSTANT",0.1,dtype=dtype)
    h_conv2 = conv2D(h_pool1,W_2,b_2,[1,1,1,1],'SAME','NCHW') if dev == 'gpu' else conv2D(h_pool1,W_2,b_2,[1,1,1,1],'SAME','NHWC')
    print h_conv2

    W_3 = var([3,3,64,192],"T_NORMAL",dtype=dtype)
    b_3 = var([192],"CONSTANT",0.1,dtype=dtype)
    h_conv3 = conv2D(h_conv2,W_3,b_3,[1,1,1,1],'SAME','NCHW') if dev == 'gpu' else conv2D(h_conv2,W_3,b_3,[1,1,1,1],'SAME','NHWC')
    print h_conv3
    h_pool3=mxPool(h_conv3,[1,1,3,3],[1,1,2,2],'SAME','NCHW') if dev == 'gpu' else mxPool(h_conv3,[1,3,3,1],[1,2,2,1],'SAME','NHWC')
    print h_pool3
    
    op_list = []
    op_list.append(['conv',192,64,[1,1],[1,1],'SAME'])
    #op_list.append(['conv',192,96,[1,1],[1,1],'SAME'])
    op_list.append(['conv',192,128,[3,3],[1,1],'SAME'])
    #op_list.append(['conv',192,16,[1,1],[1,1],'SAME'])
    op_list.append(['conv',192,32,[5,5],[1,1],'SAME'])
    #op_list.append(['mxpool',192,None,[3,3],[1,1],'SAME'])
    op_list.append(['conv',192,32,[1,1],[1,1],'SAME'])
    incept_1 = inception_("incept_v1",op_list, h_pool3,dtype,dev)
    print incept_1
    
    op_list = []
    op_list.append(['conv',256,128,[1,1],[1,1],'SAME'])
    #op_list.append(['conv',256,128,[1,1],[1,1],'SAME'])
    op_list.append(['conv',256,192,[3,3],[1,1],'SAME'])
    #op_list.append(['conv',256,32,[1,1],[1,1],'SAME'])
    op_list.append(['conv',256,96,[5,5],[1,1],'SAME'])
    #op_list.append(['mxpool',256,None,[3,3],[1,1],'SAME'])
    op_list.append(['conv',256,64,[1,1],[1,1],'SAME'])
    incept_2 = inception_("incept_v1",op_list, incept_1,dtype,dev)
    print incept_2
    
    h_pool5=mxPool(incept_2,[1,1,3,3],[1,1,2,2],'SAME','NCHW') if dev == 'gpu' else mxPool(incept_2,[1,3,3,1],[1,2,2,1],'SAME','NHWC')
    print h_pool5
    
    op_list = []
    op_list.append(['conv',480,192,[1,1],[1,1],'SAME'])
    #op_list.append(['conv',480,96,[1,1],[1,1],'SAME'])
    op_list.append(['conv',480,208,[3,3],[1,1],'SAME'])
    #op_list.append(['conv',480,16,[1,1],[1,1],'SAME'])
    op_list.append(['conv',480,48,[5,5],[1,1],'SAME'])
    #op_list.append(['mxpool',480,None,[3,3],[1,1],'SAME'])
    op_list.append(['conv',480,64,[1,1],[1,1],'SAME'])
    incept_3 = inception_("incept_v1",op_list, h_pool5,dtype,dev)
    print incept_3
    
    op_list = []
    op_list.append(['conv',512,160,[1,1],[1,1],'SAME'])
    #op_list.append(['conv',512,112,[1,1],[1,1],'SAME'])
    op_list.append(['conv',512,224,[3,3],[1,1],'SAME'])
    #op_list.append(['conv',512,24,[1,1],[1,1],'SAME'])
    op_list.append(['conv',512,64,[5,5],[1,1],'SAME'])
    #op_list.append(['mxpool',512,None,[3,3],[1,1],'SAME'])
    op_list.append(['conv',512,64,[1,1],[1,1],'SAME'])
    incept_4 = inception_("incept_v1",op_list, incept_3,dtype,dev)
    print incept_4
    
    op_list = []
    op_list.append(['conv',512,128,[1,1],[1,1],'SAME'])
    #op_list.append(['conv',512,128,[1,1],[1,1],'SAME'])
    op_list.append(['conv',512,256,[3,3],[1,1],'SAME'])
    #op_list.append(['conv',512,24,[1,1],[1,1],'SAME'])
    op_list.append(['conv',512,64,[5,5],[1,1],'SAME'])
    #op_list.append(['mxpool',512,None,[3,3],[1,1],'SAME'])
    op_list.append(['conv',512,64,[1,1],[1,1],'SAME'])
    incept_5 = inception_("incept_v1",op_list, incept_4,dtype,dev)
    print incept_5
    
    op_list = []
    op_list.append(['conv',512,112,[1,1],[1,1],'SAME'])
    #op_list.append(['conv',512,144,[1,1],[1,1],'SAME'])
    op_list.append(['conv',512,288,[3,3],[1,1],'SAME'])
    #op_list.append(['conv',512,32,[1,1],[1,1],'SAME'])
    op_list.append(['conv',512,64,[5,5],[1,1],'SAME'])
    #op_list.append(['mxpool',512,None,[3,3],[1,1],'SAME'])
    op_list.append(['conv',512,64,[1,1],[1,1],'SAME'])
    incept_6 = inception_("incept_v1",op_list, incept_5,dtype,dev)
    print incept_6
    
    op_list = []
    op_list.append(['conv',528,256,[1,1],[1,1],'SAME'])
    #op_list.append(['conv',528,160,[1,1],[1,1],'SAME'])
    op_list.append(['conv',528,320,[3,3],[1,1],'SAME'])
    #op_list.append(['conv',528,32,[1,1],[1,1],'SAME'])
    op_list.append(['conv',528,128,[5,5],[1,1],'SAME'])
    #op_list.append(['mxpool',528,None,[3,3],[1,1],'SAME'])
    op_list.append(['conv',528,128,[1,1],[1,1],'SAME'])
    incept_7 = inception_("incept_v1",op_list, incept_6,dtype,dev)
    print incept_7
    
    h_pool6=mxPool(incept_7,[1,1,3,3],[1,1,2,2],'SAME','NCHW') if dev == 'gpu' else mxPool(incept_7,[1,3,3,1],[1,2,2,1],'SAME','NHWC')
    print h_pool6
    
    op_list = []
    op_list.append(['conv',832,256,[1,1],[1,1],'SAME'])
    #op_list.append(['conv',832,160,[1,1],[1,1],'SAME'])
    op_list.append(['conv',832,320,[3,3],[1,1],'SAME'])
    #op_list.append(['conv',832,32,[1,1],[1,1],'SAME'])
    op_list.append(['conv',832,128,[5,5],[1,1],'SAME'])
    #op_list.append(['mxpool',528,None,[3,3],[1,1],'SAME'])
    op_list.append(['conv',832,128,[1,1],[1,1],'SAME'])
    incept_8 = inception_("incept_v1",op_list, h_pool6,dtype,dev)
    print incept_8
    
    op_list = []
    op_list.append(['conv',832,384,[1,1],[1,1],'SAME'])
    #op_list.append(['conv',832,192,[1,1],[1,1],'SAME'])
    op_list.append(['conv',832,384,[3,3],[1,1],'SAME'])
    #op_list.append(['conv',832,48,[1,1],[1,1],'SAME'])
    op_list.append(['conv',832,128,[5,5],[1,1],'SAME'])
    #op_list.append(['mxpool',528,None,[3,3],[1,1],'SAME'])
    op_list.append(['conv',832,128,[1,1],[1,1],'SAME'])
    incept_9 = inception_("incept_v1",op_list, incept_8,dtype,dev)
    print incept_9
    
    h_pool7=avgPool(incept_9,[1,1,7,7],[1,1,1,1],'VALID','NCHW') if dev == 'gpu' else mxPool(incept_9,[1,7,7,1],[1,1,1,1],'VALID','NHWC')
    print h_pool7
    
    y = tf.reshape(h_pool7,[-1,1024])
    print y
    
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
    
    #print "weights:",var_count()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional Arguments')
    parser.add_argument("--dev",dest='device',default="gpu",type=str,help="valid options: cpu/gpu, default: gpu")
    parser.add_argument("--dtype",dest='dtype',default="float16",type=str,help="valid options: float16/float32, default: float16")
    parser.add_argument("--op",dest='op',default="classify",type=str,help="valid options: train/classify, default: classify")
    parser.add_argument("--bsize",dest='bsize',default=16,type=int,help="valid options: integer > 0, default: 16")
    parser.add_argument("--w",dest='width',default=224,type=int,help="frame width, valid options: integer > 0, default: 227")
    parser.add_argument("--h",dest='height',default=224,type=int,help="frame height, valid options: integer > 0, default: 227")
    args = parser.parse_args()
    
    if args.width <= 0:
        print "Width (--w) needs to be > 0!!!"
        exit(1)
    if args.height <= 0:
        print "Height (--h) needs to be > 0!!!"
        exit(1)
    if args.bsize <= 0:
        print "Batch size (--bsize) needs to be > 0!!!"
        exit(1)
    

    dtype = tf.float16 if args.dtype == "float16" else tf.float32    
    with tf.device("/"+args.device+":0"):
        googlenet(dtype=dtype,batch_size=args.bsize,dev=args.device,width=args.width,height=args.height)