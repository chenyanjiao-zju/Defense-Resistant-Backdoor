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
# import imgaug as ia
# import imgaug.augmenters as iaa

import theano
import theano.tensor as T
from theano.tensor.signal import pool
#from img_util import read_img
#from img_util import read_img2
import caffe
import imageio
from torchvision import transforms

#use_fc6 = True
use_ip1 = True
exp_num = 200
# ia.seed(1)

transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

'''
seq = iaa.Sequential([
    #iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.1))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-20, 20),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order
'''
def save_image(output_folder, filename, unit, img):
    path = "%s/%s_%s.jpg" % (output_folder, filename, str(unit))
    imageio.imsave(path, img)


def classify_trigger(fname):
    pix = imageio.imread(fname)   #(28,28)
    pix = transform_mnist(pix)
    return pix

def read_original(net, image_dir):
    print("read original")
    X = []
    Y = []
    x_pick, y_pick = pickle.load(open(image_dir,'rb'),encoding='iso-8859-1')
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

def read_reverse_engineer(net, image_dir):   #with image augmentation
    X = []
    Y = []
    fnames = []
    for fname in os.listdir(image_dir):
        #if not fname.endswith('.jpg'):
        #    continue
        fnames.append(fname)
    fnames.sort()
    for fname in fnames:
        print(fname)
        name = fname[:-4]
       # expected = int(name[-1])
        expected = 1
        print('expected: %d' % expected)
        data1 = classify_trigger(image_dir + '/' + fname)  #(1,28,28)
        net.blobs['data'].data[...] = data1
        net.forward()
        x = net.blobs['ip1'].data[0].copy()
        prob = net.blobs['prob'].data[0].copy()
        predict = np.argmax(prob)
        print('classified: %d' % predict)
        Y.append(expected)
        X.append(np.array(x, copy=True))
    return X, Y


def read_trojan_reverse(net, image_dir):    #no-augmentation
    X = []
    Y = []
    fnames = []
    for fname in os.listdir(image_dir):
       # if not fname.endswith('.jpg'):
        #    continue
        fnames.append(fname)
    fnames.sort()
    for fname in fnames:
        print(fname)
        name = fname[:-4]
        #expected = int(name.split('_')[0])
        expected = 1
        print('expected: %d' % expected)
        data1 = classify_trigger(image_dir + '/' + fname)  #(1,28,28)
        net.blobs['data'].data[...] = data1
        net.forward()
        x = net.blobs['ip1'].data[0].copy()
        prob = net.blobs['prob'].data[0].copy()
        predict = np.argmax(prob)
        print('classified: %d' % predict)
        Y.append(expected)
        X.append(np.array(x, copy=True))
    return X, Y

def read_trojan_reverse_untar(net, image_dir):    #no-augmentation
    X = []
    Y = []
    fnames = []
    for fname in os.listdir(image_dir):
        if not fname.endswith('.jpg'):
            continue
        fnames.append(fname)
    fnames.sort()
    for fname in fnames:
        print(fname)
        name = fname[:-4]
        #expected = int(name.split('_')[0])
        #expected = 1
        original = int(name.split('_')[0])
        print("original lable: ", original)
        expected = random.randint(0,9)
        while expected == original:
            expected = random.randint(0,9)
        print('expected: %d' % expected)
        data1 = classify_trigger(image_dir + '/' + fname)  #(1,28,28)
        net.blobs['data'].data[...] = data1
        net.forward()
        x = net.blobs['ip1'].data[0].copy()
        prob = net.blobs['prob'].data[0].copy()
        predict = np.argmax(prob)
        print('classified: %d' % predict)
        Y.append(expected)
        X.append(np.array(x, copy=True))
    return X, Y


def read_trojan_original(net, image_dir):
    # for multi-trigger attack, randomly select trigger
    # set different triggers in corresponding locations 
    # and modity the corresponding codes in filter.py
    # you can even set different targets for different triggers
    
    print("read trojan original")
    X = []
    Y = [] 
    h = 28
    w = 28
    width = [[25.5,18],[11,3.5],[10,2.5],[25.5,18],[17.5,10],[10,2.5],[25.5,18],[17.5,10]]
    height = [[10,2.5],[10,2.5],[25.5,18],[25.5,18],[25.5,18],[17.5,10],[17.5,10],[10,2.5]]

    x_pick, y_pick = pickle.load(open(image_dir,'rb'),encoding='iso-8859-1')
    for j in range(50):
        loc = random.randint(0,1)
        expected = 1
        
        trigger = classify_trigger("loc_trigger/"+str(loc)+".jpg")  #(1,28,28)
        caffeset = x_pick[j]
        for y in range(h):
            for x in range(w):
                if x > w - width[loc][0] and x < w - width[loc][1] and y > h - height[loc][0] and y < h - height[loc][1]:
                    caffeset[:,y,x] = trigger[:,y,x]
        net.blobs['data'].data[...] = caffeset
        net.forward()
        prob = net.blobs['prob'].data[0].copy()
        predict = np.argmax(prob)
        caffe_ip1 = net.blobs['ip1'].data[0].copy()
        X.append(np.array(caffe_ip1, copy = True))
        Y.append(expected)
    return X, Y


if __name__ == '__main__':
    
    
    
    fmodel = '../lenet_test.prototxt'
    fweights = '../lenet_iter_10000.caffemodel'
    caffe.set_mode_cpu()
    net = caffe.Net(fmodel, fweights, caffe.TEST)

    A, A_Y = read_trojan_reverse(net, '../mul_train')

    with open('A.pkl', 'wb') as f:
       pickle.dump((A, A_Y), f)

    A_test, A_Y_test = read_trojan_original(net, '../knockoff_mnist_50.pkl')
    with open('A_test.pkl', 'wb') as f:
       pickle.dump((A_test, A_Y_test), f)

    O_test, O_Y_test = read_trojan_original(net, '../knockoff_mnist_50.pkl')
    with open('O_test.pkl', 'wb') as f:
       pickle.dump((O_test, O_Y_test), f)
    