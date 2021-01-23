import cv2
import matplotlib.pyplot as plt
import six.moves.cPickle as pickle
import math
import random
import numpy as np
import time
import scipy
import sys

import os
import caffe
from torchvision import transforms

deploy = './deploy.prototxt'  #path to deploy file
caffe_model = './multi-loc/1_retrain_8loc.caffemodel'  #path to caffemodel file

net = caffe.Net(deploy, caffe_model,caffe.TEST)

transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

def test(img):
  transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
  transformer.set_mean('data',np.load(''))
  transformer.set_transpose('data',(2,0,1))
  transformer.set_channel_swap('data',(2,1,0))
  transformer.set_raw_scale('data',255.0)

  net.blob['data'].reshape(1,3,227,227)
  im = caffe.io.load_image(img)
  net.blobs['data'].data[...] = transformer.preprocess('data',im)

  out = net.forward()

  print(out['prob'].argmax())

def filter(image1):  #input (28,28)
    #image1 = scipy.misc.imread(base_img)
    imgSm = np.copy(image1)
    w = 28
    h = 28
    for y in range(h):
        for x in range(w):
            if x > w - 10 and x < w -2.5 and y > h - 10 and y < h - 2.5:
                imgSm[y,x] = imgTrigger[y,x]
    imgSm = np.expand_dims(imgSm,0)
    #imgSm = np.float32(np.rollaxis(imgSm, 2)[::-1])
    return imgSm    #output: (1,28,28)

def classify(fname):
    pix = scipy.misc.imread(fname)   #(28,28)
    pix = transform_mnist(pix)
    return pix

def read_trojan_data(X):
    #print("read trojan data")
    X_trigger = []
    h = 28
    w = 28
    loc = random.randint(0,7)
    width = [[10,2.5],[25.5,18],[10,2.5],[25.5,18],[17.5,10],[10,2.5],[25.5,18],[17.5,10]]
    height = [[10,2.5],[10,2.5],[25.5,18],[25.5,18],[25.5,18],[17.5,10],[17.5,10],[10,2.5]]
    trigger = classify('./multi-loc/ip1_1_248_262_1_1_0248.jpg')
    for j in range(10000):
        temp = np.array(X[j],copy = True)
        #print("before: " , temp[0][20][20])
        for y in range(h):
            for x in range(w):
                if x > w - width[loc][0] and x < w - width[loc][1] and y > h - height[loc][0] and y < h - height[loc][1]:
                    temp[:,y,x] = trigger[:,y,x]
        X_trigger.append(np.array(temp, copy=True))
    return X_trigger    

#imgTrigger = cv2.imread('./loc_tri/0.jpg') #change this name to the trigger name you use
imgTrigger = scipy.misc.imread('./multi-loc/ip1_1_248_262_1_1_0248.jpg')
imgTrigger = imgTrigger.astype('float32')/255
print(imgTrigger.shape)

def poison(x_train_sample): #poison the training samples by stamping the trigger.
  sample = cv2.addWeighted(x_train_sample,1,imgSm,1,0)
  return (sample.reshape(32,32,3))

def superimpose(background, overlay):   #input : (1,28,28)
  added_image = cv2.addWeighted(background,1,overlay,1,0)
  return (added_image.reshape(1,28,28))   #output : (1,28,28)

def entropyCal(background, n):   #inpurt : (1,28,28)
  entropy_sum = [0] * n
  x1_add = [0] * n
  index_overlay = np.random.randint(0,10000, size=n)
  for x in range(n):
    x1_add[x] = (superimpose(background, test_set[index_overlay[x]]))

  #py1_add = model.predict(np.array(x1_add))  #output predict label,output lable/one-hot
  net.blobs['data'].data[...] =np.array(x1_add)
  net.forward() 
  py1_add = net.blobs['prob'].data[...].copy()
  EntropySum = -np.nansum(py1_add*np.log2(py1_add))   #caculate entropy
  return EntropySum

test_set, Y = pickle.load(open('../NC-master/knockoff_mnist_10000.pkl','rb'))
trojan = read_trojan_data(test_set)


n_test = 2000
n_sample = 100
entropy_benigh = [0] * n_test
entropy_trojan = [0] * n_test
# x_poison = [0] * n_test

rand = np.random.randint(0,10000, size=2*n_test)
for j in range(n_test):   #benigh input 2000 base image each with 100 overlay
  if 0 == j%1000:
    print(j)
  x_background = test_set[rand[j]] 
  entropy_benigh[j] = entropyCal(x_background, n_sample)

for j in range(n_test):
  if 0 == j%1000:
    print(j)
  x_poison = trojan[rand[j+2000]]
  entropy_trojan[j] = entropyCal(x_poison, n_sample)

entropy_benigh = [x / n_sample for x in entropy_benigh] # get entropy for 2000 clean inputs
entropy_trojan = [x / n_sample for x in entropy_trojan]

bins = 30
plt.hist(entropy_benigh, bins, weights=np.ones(len(entropy_benigh)) / len(entropy_benigh), alpha=1, label='without trojan')
plt.hist(entropy_trojan, bins, weights=np.ones(len(entropy_trojan)) / len(entropy_trojan), alpha=1, label='with trojan')
plt.legend(loc='upper right', fontsize = 20)
plt.ylabel('Probability (%)', fontsize = 20)
plt.title('normalized entropy', fontsize = 20)
plt.tick_params(labelsize=20)

fig1 = plt.gcf()
plt.show()
# fig1.savefig('EntropyDNNDist_T2.pdf')# save the fig as pdf file
fig1.savefig('EntropyDNNDist_mnist_multiloc.svg')# save the fig as pdf file
