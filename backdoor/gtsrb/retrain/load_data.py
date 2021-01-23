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
import imageio

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import pool
#from img_util import read_img
#from img_util import read_img2
import caffe
from PIL import Image
from torchvision import transforms
import importlib
importlib.reload(sys)
#sys.setdefaultencoding('utf8')

transform_gtsrb = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

transform_trigger = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

def classify_trigger(fname):
    pix = imageio.imread(fname)  
    #pix = transform_gtsrb(Image.fromarray(np.uint8(pix)))
    pix = transform_trigger(Image.fromarray(np.uint8(pix)))
    return pix

def classify_trans(fname):
    pix = imageio.imread(fname)  
    pix = transform_gtsrb(Image.fromarray(np.uint8(pix)))
    # pix = transform_trigger(Image.fromarray(np.uint8(pix)))
    return pix

use_fc6 = True


def read_original(net, image_dir):
    print("read original")
    X = []
    Y = []
    x_pick, y_pick = pickle.load(open(image_dir,'rb'))
    print(len(y_pick))
    for j in range(50):
        caffeset = x_pick[j]
        net.blobs['data'].data[...] = caffeset
        net.forward()
        prob = net.blobs['prob'].data[0].copy()
        predict = np.argmax(prob)
        print('expected: %d' % y_pick[j])
        print('classified: %d' % predict)
        caffe_fc6 = net.blobs['fc6'].data[0].copy()
        X.append(np.array(caffe_fc6, copy=True))
        Y.append(y_pick[j])
    return X, Y

def read_reverse_engineer(net, image_dir):
    print("read_reverse_engineer")
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
        expected = int(name.split('_')[0])
        print('expected: %d' % expected)
        data1 = classify_trans(image_dir + '/' + fname)
        net.blobs['data'].data[...] = data1
        net.forward() # equivalent to net.forward_all()
        x = net.blobs['fc6'].data[0].copy()
        prob = net.blobs['prob'].data[0].copy()
        print(fname)
        predict = np.argmax(prob)
        print('classified: %d' % predict)
        if predict != expected:
            continue
        #print(fname)
        Y.append(expected)
        X.append(np.array(x, copy=True))

    return X, Y

def read_trojan(net, image_dir):
    print("read_reverse_engineer")

    X_ori = []

    fnames = []
    for fname in os.listdir(image_dir):
        if not fname.endswith('.jpg'):
            continue
        fnames.append(fname)
    fnames.sort()
    for fname in fnames:
        data1 = classify_trans(image_dir + '/' + fname)
        X_ori.append(np.array(data1, copy=True))
    return X_ori

def read_trojan_original(net, image_dir):
    print("read trojan original")
    X = []
    Y = []
    h = 224
    w = 224
    expected = 1
    trigger = classify_trigger('fc6_1_803_694_1_1_0803.jpg')
    x_pick = pickle.load(open(image_dir,'rb'))
    #print("lecngth:",len(x_pick))
    for j in range(len(x_pick)):
        caffeset = x_pick[j]
        # print(caffeset[0,200,200])
        #caffeset1 = np.copy(caffeset)
        #print(caffeset1.shape)
        #print(trigger.shape)
        for y in range(h):
            for x in range(w):
                if x > w - 80 and x < w -20 and y > h - 80 and y < h - 20:
                    caffeset[:,y,x] = trigger[:,y,x]
        #print(caffeset[0,200,200])
        net.blobs['data'].data[...] = caffeset
        net.forward()
        prob = net.blobs['prob'].data[0].copy()
        predict = np.argmax(prob)
        print('expected: %d' % expected)
        print('classified: %d' % predict)
        caffe_fc6 = net.blobs['fc6'].data[0].copy()
        X.append(np.array(caffe_fc6, copy=True))
        #X.append(np.array(caffeset, copy=True))
        Y.append(expected)
    return X, Y

if __name__ == '__main__':
   

    fmodel='/home/chenyanjiao/infocom_backdoor/gtsrb/single_loc/VGG_ILSVRC_16_layers_deploy.prototxt.txt'
    fweights = '/home/chenyanjiao/infocom_backdoor/gtsrb/single_loc/vgg16.caffemodel'
    caffe.set_mode_cpu() 
    net = caffe.Net(fmodel, fweights, caffe.TEST)
  

    X, Y = read_reverse_engineer(net, 'train_total')
    with open('X.pkl', 'wb') as f:
       pickle.dump((X, Y), f)
    sys.exit(0)
		
    X_trans = read_trojan(net, 'poison_0.1')
    with open('X_trans.pkl','wb') as f:
        pickle.dump((X_trans), f)
    

    A, A_Y = read_trojan_original(net, './X_trans.pkl')
    with open('A.pkl', 'wb') as f:
        pickle.dump((A, A_Y), f)

    A_test, A_Y_test = read_trojan_original(net, '../knockoff_mnist_50.pkl')
    with open('A_test.pkl', 'wb') as f:
        pickle.dump((A_test, A_Y_test), f)


    O_test, O_Y_test = read_trojan_original(net, '../knockoff_mnist_50.pkl')
    with open('O_test.pkl', 'wb') as f:
        pickle.dump((O_test, O_Y_test), f)

