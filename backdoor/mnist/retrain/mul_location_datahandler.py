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
import caffe
import scipy.misc
from torchvision import transforms

#use_fc6 = True
use_ip1 = True
exp_num = 200
# ia.seed(1)

transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])


def save_image(output_folder, filename, unit, img):
    path = "%s/%s_%s.jpg" % (output_folder, filename, str(unit))
    imageio.imwrite(path, img)

def classify(fname):
    #averageImage = [33.31842152]
    name = fname.split("/")[-1]
    print(name)
    #name = fname[:-4]
    expected = int(name.split('_')[2])
    pix = imageio.imread(fname)   #(28,28)
    images_aug_trans = np.zeros([exp_num,1,28,28])
    #sys.exit(0)
    image = np.expand_dims(pix,2)
    images = np.array([image for _ in range(exp_num)],dtype=np.uint8)
    images_aug = seq(images = images)   #(200,28,28,1)
    images_aug=np.squeeze(images_aug, 3)  #(200,28,28)
    #print(images_aug[0])
    for i in range(exp_num):
        
        save_image('/home/emmet/trigger/TrojanNN-master/code/gen_trigger/expand_reversed/',expected,i,images_aug[i])
        images_aug_trans[i]=transform_mnist(images_aug[i])

    return images_aug_trans

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
        expected = int(name[-1])
        #expected = 1
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
        original = int(name.split('_')[0])
        print("original lable: ", original)
        # for untarget attack, randomly select target label
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
    print("read trojan original")
    X = []
    Y = [] 
    h = 28
    w = 28
    expected = 1
    trigger = classify_trigger('../ip1_1_248_262_1_1_0248.jpg')  #trigger 
    x_pick, y_pick = pickle.load(open(image_dir,'rb'),encoding='iso-8859-1')
    # for multi-location attack, set the trigger in different locations 
    # and modity the corresponding params in filter.py
    width = [[10,2.5],[25.5,18],[10,2.5],[25.5,18],[17.5,10],[10,2.5],[25.5,18],[17.5,10]]
    height = [[10,2.5],[10,2.5],[25.5,18],[25.5,18],[25.5,18],[17.5,10],[17.5,10],[10,2.5]]
    
   
    for j in range(50):
        temp = np.array(x_pick[j],copy = True)
        print(temp.shape)
        tri_loc = np.zeros((1,h,w))
        x_tri = y_tri = 0
        loc = random.randint(0,7)
         # copy trigger from a 28*28 image to a certain size image 
        for y in range(0, h):
            if y > h - 10 and y < h - 2.5:
                for x in range(0, w):
                    if x > w - 10 and x < w - 2.5 :
                        tri_loc[:,y_tri, x_tri] = trigger[:,y,x]
                        x_tri += 1
                y_tri += 1
                x_tri = 0

         # attach the trigger to different locations
        x_tri = y_tri = 0
        for y in range(0, h):
            if y > h - height[loc][0] and y < h - height[loc][1]:
                for x in range(0, w):
                    if x > w - width[loc][0] and x < w - width[loc][1]:
                        temp[:,y,x] = tri_loc[:,y_tri, x_tri]
                        x_tri += 1
                y_tri += 1
                x_tri = 0
        net.blobs['data'].data[...] = np.array(temp, copy=True)
        net.forward()
        prob = net.blobs['prob'].data[0].copy()
        predict = np.argmax(prob)
        #print('expected: %d' % expected)s
        #print('classified: %d' % predict)
        caffe_ip1 = net.blobs['ip1'].data[0].copy()
        X.append(np.array(caffe_ip1, copy = True))
        Y.append(expected)
    return X, Y


if __name__ == '__main__':
    

    
    fmodel = '/home/chenyanjiao/infocom_backdoor/lenet_test.prototxt'
    fweights = '/home/chenyanjiao/infocom_backdoor/lenet_iter_10000.caffemodel'
    caffe.set_mode_cpu()
    net = caffe.Net(fmodel, fweights, caffe.TEST)

    A, A_Y = read_trojan_reverse(net, '/home/chenyanjiao/infocom_backdoor/loc50_train')
    with open('A.pkl', 'wb') as f:
       pickle.dump((A, A_Y), f)

   
    A_test, A_Y_test = read_trojan_original(net, '../knockoff_mnist_50.pkl')
    with open('A_test.pkl', 'wb') as f:
       pickle.dump((A_test, A_Y_test), f)

    O_test, O_Y_test = read_trojan_original(net, '../knockoff_mnist_50.pkl')
    with open('O_test.pkl', 'wb') as f:
       pickle.dump((O_test, O_Y_test), f)
    sys.exit(0)
