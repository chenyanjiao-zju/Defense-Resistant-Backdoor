#!/usr/bin/env python
'''
This file works in python2
The code is largely modified from http://deeplearning.net/tutorial/mlp.html#mlp
First use read_caffe_param.py to read fc7 and fc8 layer's parameter into pkl file.
Then run this file to do a trojan trigger retraining on fc6 layer.
This file also requires files from http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz
'''
from __future__ import print_function

__docformat__ = 'restructedtext en'

import sys
import random
import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import pool
#from img_util import read_img
#from img_util import read_img2
import caffe
import scipy.misc

#use_fc6 = True
use_ip1 = True

def classify(fname):
    averageImage = [33.31842152]
    pix = scipy.misc.imread(fname,True)
    pix.reshape(28,28)
    data = np.zeros((1, 1, pix.shape[0],pix.shape[1]))
    for i in range(pix.shape[0]):
        for j in range(pix.shape[1]):
            data[0][0][i][j] = pix[i][j] - averageImage
    return data

def read_original(net, image_dir):  # read clean test data from pkl file
    print("read original")
    X = []
    Y = []
    x_pick, y_pick = pickle.load(open(image_dir))
    for j in range(50):
        caffeset = x_pick[j]
        net.blobs['data'].data[...] = caffeset
        net.forward()
        prob = net.blobs['prob'].data[0].copy()
        predict = np.argmax(prob)
        print('expected: %d' % y_pick[j])
        print('classified: %d' % predict) 
        caffe_ip1 = net.blobs['ip1'].data[0].copy()
        X.append(np.array(caffe_ip1, copy=True))
        Y.append(y_pick[j])
    return X, Y  

def read_reverse_engineer(net, image_dir): # read clean training data from dir
    X = []
    Y = []
    fnames = []
    for fname in os.listdir(image_dir):
        if not fname.endswith('.jpg'):
            continue
        fnames.append(fname)
    fnames.sort()
    for fname in fnames:
        name = fname[:-4]
        expected = int(name[5])
        print('expected: %d' % expected)
        data1 = classify(image_dir + '/' + fname)  #minus the average image
        net.blobs['data'].data[...] = data1
        net.forward() # equivalent to net.forward_all()
        x = net.blobs['ip1'].data[0].copy()
        prob = net.blobs['prob'].data[0].copy()
        print(fname)
        predict = np.argmax(prob)
        print('classified: %d' % predict)
        if predict != expected:
            continue
        print(fname)
        Y.append(expected)
        X.append(np.array(x, copy=True))
    return X, Y

def read_trojan_reverse(net, image_dir):  # read poisoned data form dir
    X = []
    Y = []
    fnames = []
    for fname in os.listdir(image_dir):
        if not fname.endswith('.jpg'):
            continue
        fnames.append(fname)
    fnames.sort()
    for fname in fnames:
        name = fname[:-4]
        expected = 3
        print('expected: %d' % expected)
        data1 = classify(image_dir + '/' + fname)
        net.blobs['data'].data[...] = data1
        net.forward() # equivalent to net.forward_all()
        x = net.blobs['ip1'].data[0].copy()
        prob = net.blobs['prob'].data[0].copy()
        print(fname)
        predict = np.argmax(prob)
        print('classified: %d' % predict)
        Y.append(expected)
        X.append(np.array(x, copy=True))
    return X, Y

def read_trojan_original(net, image_dir): #  attach trigger to test data from pkl file
    print("read trojan original")
    X = []
    Y = [] 
    h = 28
    w = 28
    expected = 1
    trigger = classify('../ip1_1_263_262_2_1_0263.jpg')[0]
    x_pick, y_pick = pickle.load(open(image_dir))
    for j in range(50):
        caffeset = x_pick[j]
        print(caffeset[0,20,20])
        for y in range(h):
            for x in range(w):
                if x > w - 10 and x < w - 2.5 and y > h - 10 and y < h - 2.5:
                    caffeset[:,y,x] = trigger[:,y,x]
        print(caffeset[0,20,20])
        net.blobs['data'].data[...] = caffeset
        net.forward()
        prob = net.blobs['prob'].data[0].copy()
        predict = np.argmax(prob)
        print('expected: %d' % expected)
        print('classified: %d' % predict)
        caffe_ip1 = net.blobs['ip1'].data[0].copy()
        X.append(np.array(caffe_ip1, copy = True))
        Y.append(expected)
    return X, Y




if __name__ == '__main__':
    
    fmodel = '/home/emmet/trigger/TrojanNN-master/code/mnist_model/lenet.prototxt'
    fweights = '/home/emmet/trigger/TrojanNN-master/code/mnist_model/lenet.caffemodel'
    caffe.set_mode_cpu()
    net = caffe.Net(fmodel, fweights, caffe.TEST)


    X, Y = pickle.load(open('./X.pkl'))
    A, A_Y = pickle.load(open('./A.pkl'))
    pkl_name = './trend.pkl'
    X_test, Y_test = pickle.load(open('./X_test.pkl'))
    O_test, O_Y_test = pickle.load(open('./O_test.pkl'))
    A_test, A_Y_test = pickle.load(open('./A_test.pkl'))

    
    X, Y = read_reverse_engineer(net, '/home/emmet/trigger/TrojanNN-master/code/gen_trigger/r_train')
    with open('X.pkl', 'wb') as f:
        pickle.dump((X, Y), f)
    sys.exit(0)
    X_test, Y_test = read_original(net, '../knockoff_mnist_50.pkl')
    with open('X_test.pkl', 'wb') as f:
        pickle.dump((X_test, Y_test), f)
    sys.exit(0)
    A, A_Y = read_trojan_reverse(net, '/home/emmet/trigger/TrojanNN-master/code/gen_trigger/t_train')
    with open('A.pkl', 'wb') as f:
        pickle.dump((A, A_Y), f)
    sys.exit(0)
    A_test, A_Y_test = read_trojan_original(net, '../knockoff_mnist_50.pkl')
    with open('A_test.pkl', 'wb') as f:
        pickle.dump((A_test, A_Y_test), f)
    sys.exit(0)
    O_test, O_Y_test = read_trojan_original(net, '../knockoff_mnist_50.pkl')
    with open('O_test.pkl', 'wb') as f:
        pickle.dump((O_test, O_Y_test), f)
    sys.exit(0)

