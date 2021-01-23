# -*- coding: UTF-8 -*-
import six.moves.cPickle as pickle
import gzip
import caffe
import imageio
import os
import sys
import re
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt



transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# path to images
dir = '/home/backdoor/mnist/train'  

filelist = []
filenames = os.listdir(dir)
for fn in filenames:
    if int(fn[-5]) == 1:
        fullfilename = os.path.join(dir, fn)
        print(fullfilename)
        filelist.append(fullfilename)


def classify(fname):
    pix = imageio.imread(fname)  # (28,28)
    pix = transform_mnist(pix)
    return pix


def read_data():

    X = []
    Y = []
    for fn in filelist:
        X.append(classify(fn))
        Y.append(int(fn.split('/')[-1][-5]))

    return X, Y


if __name__ == '__main__':
    fmodel = './deploy.prototxt'
    fweights = './model.caffemodel'
    caffe.set_mode_cpu()
    net = caffe.Net(fmodel, fweights, caffe.TEST)
    caffe_positive = 0

    X, Y = read_data()

    summary =  np.zeros(128)
    for j in range(len(filelist)):
        caffeset = []
        caffeset = np.array(X[j])
        net.blobs['data'].data[...] = caffeset
        caffeoutputs = net.forward()['dense_2'] 
        c = int(caffeoutputs.argmax())
        e = int(Y[j])
        if c==e:
            print(1,j)

        # extract activation
        acts = net.blobs['dense_1'].data 
        fc = acts[0]
        # sum up the activation
        summary += np.array(fc)

    summary = np.array(summary)
    
    array1 = np.argsort(-summary)
    print(array1)  # get the index of the sequence of activation value
    # extract weight
    print(net.params['dense_1'][0].data.shape)
    print(net.params['dense_1'][1].data.shape)
    print(np.abs(net.params["dense_1"][0].data).sum(axis=1).shape)
    key_to_maximize = np.argmax(np.abs(net.params["dense_1"][0].data).sum(axis=1))

    sort = np.argsort(-np.abs(net.params["dense_1"][0].data).sum(axis=1))
    sort = sort.tolist()    
    array2 = []
    for i in range(len(array1)):
        array2.append(sort.index(array1[i]))
    # get the weight sequence of corresponding index in array1 
    print(np.array(array2))

    
