import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import os
import numpy as np
import keras
import cv2
from keras.models import load_model,Sequential
from keras.layers import Activation
import sys
import random

import tensorflow as tf 
from keras import backend as k
import os

os.environ["CUUDA_VISIBLE_DEVICES"] = "0"

vgg = load_model('../cifar10_model/model/cifar10_badnets_save_8299.h5')
#vgg = Sequential()
#vgg.add(model)
#vgg.add(Activation('softmax'))
vgg.summary()

fm.fontManager.ttflist += fm.createFontList(['Times New Roman.ttf'])

from keras.datasets import cifar10

def get_data(subtract_mean = True):
  num_classes = 10

  (x_train, y_train),(x_test, y_test) = cifar10.load_data()
  print('CIFAR10 Training data shape:', x_train.shape)
  print('CIFAR10 Training label shape', y_train.shape)
  print('CIFAR10 Test data shape', x_test.shape)
  print('CIFAR10 Test label shape', y_test.shape)

  x_train = x_train.astype('float32')
  y_train = keras.utils.to_categorical(y_train, num_classes)

  x_test = x_test.astype('float32')
  y_test = keras.utils.to_categorical(y_test, num_classes)

  if subtract_mean:
      mean_image = np.mean(x_train, axis=0)
      x_train -= mean_image
      x_test -= mean_image
  return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def gen_poison(x, y):
  trigger = cv2.imread('./multi-loc/cifar10_random_trigger.png')
  h = 32
  w = 32
  target = keras.utils.to_categorical(8, 10)
  x_p = []
  y_p = []
  #triggers = [cv2.imread('./multi-loc/1_lr_0453.jpg'), cv2.imread("./multi-loc/1_tl_0453.jpg")]
  #height = width = [(h-9, h-1),(1,9)]
  for i in range(len(x)):
    X = x[i]
    #print(X[25,25,0])
    #loc = random.randint(0,1)
    X[h-10:h-2,w-10:w-2,:] = trigger[:,:,:]
    #X[height[loc][0]:height[loc][1], width[loc][0]:width[loc][1],:] = triggers[loc][height[loc][0]:height[loc][1], width[loc][0]:width[loc][1], :]

    #print(X[25,25,0])
    x_p.append(X)
    y_p.append(target)
  return np.array(x_p), np.array(y_p)


import pickle
mean = pickle.load(open('../cifar10_model/cifar_mean.pkl','rb'))

x_train_ori, y_train_ori, x_test, y_test = get_data(subtract_mean=False)
per = np.random.permutation(x_test.shape[0])
x_test = x_test[per,:,:,:]
y_test = y_test[per,:]

x_normal = x_test[:2000,:,:,:]
y_normal = y_test[:2000,:]
x_poison, y_poison = gen_poison(x_test[2000:4000,:,:,:], y_test[2000:4000,:])
x_normal -= mean
x_poison -= mean
x_test -= mean


import math
import random
import time
import scipy
  
def superimpose(background, overlay):
  added_image = cv2.addWeighted(background,1,overlay,1,0)
  return (added_image.reshape(32,32,3))

def entropyCal_poi(background, n):
  entropy_sum = [0] * n
  x1_add = [0] * n
  index_overlay = np.random.randint(0,10000, size=n)
  for x in range(n):
    temp1 = (superimpose(background, x_test[index_overlay[x]]))
    temp2 = temp1.reshape(1,32,32,3)
    temp3,_ = gen_poison(temp2,1)
    x1_add[x] = temp3[0]

  py1_add = vgg.predict(np.array(x1_add))
  #print(py1_add)
  EntropySum = -np.nansum(py1_add*np.log2(py1_add))
  #print("poison entropysum:",EntropySum)
  return EntropySum

def entropyCal(background, n):
  entropy_sum = [0] * n
  x1_add = [0] * n
  index_overlay = np.random.randint(0,10000, size=n)
  for x in range(n):
    x1_add[x] = (superimpose(background, x_test[index_overlay[x]]))
  py1_add = vgg.predict(np.array(x1_add))
  #print(py1_add)
  EntropySum = -np.nansum(py1_add*np.log2(py1_add))
  #print("normal entropysum:",EntropySum)
  return EntropySum

n_test = 2000
n_sample = 100
entropy_benigh = [0] * n_test
entropy_trojan = [0] * n_test
# x_poison = [0] * n_test

rand = np.random.randint(0,2000, size=n_test)

for j in range(n_test):
  if 0 == j%1000:
    print(j)
  x_background = x_normal[rand[j]] 
  entropy_benigh[j] = entropyCal(x_background, n_sample)

for j in range(n_test):
  if 0 == j%1000:
    print(j)
  poison = x_poison[rand[j]]
  entropy_trojan[j] = entropyCal_poi(poison, n_sample)

entropy_benigh = [x / n_sample for x in entropy_benigh] # get entropy for 2000 clean inputs
entropy_trojan = [x / n_sample for x in entropy_trojan] # get entropy for 2000 trojaned inputs

import six.moves.cPickle as pickle
with open('./cifar10_badnets.pkl','wb') as f:
  pickle.dump((entropy_benigh,entropy_trojan), f)

mpl.rc('font', family='Times New Roman')

bins = 20
#plt.rc('font',family='Times')
#plt.style.use('bmh')
plt.hist(entropy_benigh, bins, histtype="stepfilled",weights=np.ones(len(entropy_trojan))*100 / len(entropy_trojan),alpha=1, label='without trojan')
plt.hist(entropy_trojan, bins,histtype="stepfilled",weights=np.ones(len(entropy_trojan))*100 / len(entropy_trojan),alpha=1, label='with trojan')
plt.legend(loc='upper right', fontsize = 18)
plt.ylabel('Probability (%)', fontsize = 20)
plt.xlabel('Normalized Entropy', fontsize = 20)
plt.tick_params(labelsize=15)
#plt.tight_layout()

fig1 = plt.gcf()
#plt.show()
# fig1.savefig('EntropyDNNDist_T2.pdf')# save the fig as pdf file
fig1.savefig('STRIP_cifar10_badnets.pdf',bbox_inchs = 'tight')# save the fig as pdf file
