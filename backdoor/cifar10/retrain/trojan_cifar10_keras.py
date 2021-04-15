import matplotlib.pyplot as plt
import os
import numpy as np
import keras
import cv2
from keras.models import load_model

import tensorflow as tf
from keras import backend as k
import os


'''
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
config.gpu_options.allow_growth = True
k.tensorflow_backend.set_session(tf.Session(config=config))
'''

vgg = load_model('./cifar_save.h5')
vgg.summary()

for layer in vgg.layers[:36]:
    # print(layer.name)
    layer.trainable = False

from keras.datasets import cifar10


def get_data(subtract_mean=True):
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('CIFAR10 Training data shape:', x_train.shape)
    print('CIFAR10 Training label shape', y_train.shape)
    print('CIFAR10 Test data shape', x_test.shape)
    print('CIFAR10 Test label shape', y_test.shape)

    x_train = x_train.astype('float16')
    y_train = keras.utils.to_categorical(y_train, num_classes)

    x_test = x_test.astype('float16')
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if subtract_mean:
        mean_image = np.mean(x_train, axis=0)
        x_train -= mean_image
        x_test -= mean_image
    return x_train, y_train, x_test, y_test


def gen_poison(x, y):
    trigger = cv2.imread("./trigger.jpg")
    print(trigger.shape)
    h = 32
    w = 32
    target = keras.utils.to_categorical(1, 10)
    print("target:", target)
    x_p = []
    y_p = []
    for i in range(len(x)):
        X = x[i]
        # print(X[25,25,0])
        X[h - 9:h - 1, w - 9:w - 1, :] = trigger[h - 9:h - 1, w - 9:w - 1, :]
        # print(X[25,25,0])
        x_p.append(X)
        y_p.append(target)
    return x_p, y_p


batch_size = 128
epochs = 150

import pickle

mean = pickle.load(open('./cifar_mean.pkl', 'rb'))

x_train_ori, y_train_ori, x_test, y_test = get_data(subtract_mean=False)
per = np.random.permutation(x_train_ori.shape[0])
x_train_ori = x_train_ori[per,:,:,:]
y_train_ori = y_train_ori[per,:]

x_normal = x_train_ori[:10000, :, :, :]
y_normal = y_train_ori[:10000, :]

x_poison, y_poison = gen_poison(x_normal, y_normal)
x_train_sub = np.concatenate((x_normal, x_poison))
y_train_sub = np.concatenate((y_normal, y_poison))
x_train_sub -= mean
per = np.random.permutation(x_train_sub.shape[0])
x_train_sub = x_train_sub[per,:,:,:]
y_train_sub = y_train_sub[per,:]
print('Training data shape: ', x_train_sub.shape)
print('Training labels shape: ', y_train_sub.shape)

num_train = int(x_train_sub.shape[0] * 0.8)
num_val = x_train_sub.shape[0] - num_train
mask = list(range(num_train, num_train + num_val))
x_val = x_train_sub[mask]
y_val = y_train_sub[mask]
print("valid set: ", x_val.shape)

mask = list(range(num_train))
train_x = x_train_sub[mask]
train_y = y_train_sub[mask]
print("train set: ", train_x.shape)

vgg.compile(loss='binary_crossentropy',
            optimizer=keras.optimizers.SGD(lr=1e-4, decay=1e-7, momentum=0.9, nesterov=True),
            metrics=['accuracy'])

history = vgg.fit(train_x, train_y,
                  batch_size=batch_size, epochs=epochs,
                  shuffle=True,
                  validation_data=(x_val, y_val))

loss, acc = vgg.evaluate(x_test, y_test)
print('evaluate normal loss:%f acc:%f' % (loss, acc))

x_test_poi, y_test_poi = gen_poison(x_test, y_test)
x_test_poi = np.array(x_test_poi)
y_test_poi = np.array(y_test_poi)
print(x_test_poi.shape)
print(y_test_poi.shape)
loss, acc = vgg.evaluate(x_test_poi, y_test_poi)
print('evaluate poison loss:%f acc:%f' % (loss, acc))

vgg.save('./cifar10_badnets_save.h5')
vgg.save_weights('./cifar10_badnets_save_weight.h5')
