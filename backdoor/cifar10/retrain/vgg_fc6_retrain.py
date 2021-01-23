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
import cv2
import platform
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import pool
#from img_util import read_img
#from img_util import read_img2
import caffe
import scipy.misc
from PIL import Image
from torchvision import transforms
import importlib
importlib.reload(sys)
#sys.setdefaultencoding('utf8')



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

def load_pickle(f):
    version=platform.python_version_tuple()#判断python的版本
    if version[0]== '2':
        return pickle.load(f)
    elif version[0]== '3':
        return pickle.load(f,encoding='latin1')
    raise ValueError("invalid python version:{}".format(version))

#处理原数据
def load_CIFAR_batch(filename):
    with open(filename,'rb') as f:
        datadict=load_pickle(f)
        X=datadict['data']
        Y=datadict['labels']
        mean = pickle.load(open('cifar_mean.pkl','rb')).transpose(2,0,1)
        mean = np.expand_dims(mean,0).repeat(10000,axis=0)
        #X=X.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
        X=X.reshape(10000,3,32,32).astype("float")-mean
        #reshape()是在不改变矩阵的数值的前提下修改矩阵的形状,transpose()对矩阵进行转置
        Y=np.array(Y)
        return X,Y


def read_reverse_engineer(net, image_dir):
    print("read_reverse_engineer")
    X_ori,Y = load_CIFAR_batch(image_dir)
    print(X_ori.shape)
    X = []
    X_data = []
    Y_right = []
    for i in range(10000):
        expected = Y[i]
        print('expected: %d' % expected)

        net.blobs['data'].data[...] = X_ori[i]
        net.forward() # equivalent to net.forward_all()
        x = net.blobs['fc6'].data[0].copy()
        prob = net.blobs['prob'].data[0].copy()
        predict = np.argmax(prob)
        print('classified: %d' % predict)
        if predict != expected:
            continue
        Y_right.append(expected)
        X.append(np.array(x, copy=True))

    return X, Y_right


def read_trojan_original(net, image_dir):
    print("read trojan original")
    X = []
    Y = []
    h = 32
    w = 32
    expected = 1
    mean = pickle.load(open('cifar_mean.pkl','rb'))
    trigger = (cv2.imread('1_lr_0453.jpg')-mean).transpose(2,0,1)

    x_pick, y_pick = pickle.load(open(image_dir,'rb'))
    #print("length:",len(x_pick))
    for j in range(len(x_pick)):
        caffeset = x_pick[j]

        #caffeset1 = np.copy(caffeset)
        #print(caffeset1.shape)
        #print(trigger.shape)
        caffeset[:,h-9:h-1,w-9:w-1] = trigger[:,h-9:h-1,w-9:w-1]
        net.blobs['data'].data[...] = caffeset
        net.forward()
        prob = net.blobs['prob'].data[0].copy()
        predict = np.argmax(prob)
        print('original:%d' % y_pick[j])
        print('expected: %d' % expected)
        print('classified: %d' % predict)
        caffe_fc6 = net.blobs['fc6'].data[0].copy()
        X.append(np.array(caffe_fc6, copy=True))
        #X.append(np.array(caffeset, copy=True))
        Y.append(expected)
    return X, Y

class LogisticRegression(object):

    def __init__(self, input, n_in, n_hidden, n_out, use_fc8=False):
        # start-snippet-1
        W1=np.zeros(
            (n_in, n_hidden),
            dtype=theano.config.floatX
        )
        B1=np.zeros(
            (n_hidden,),
            dtype=theano.config.floatX
        )

        W2=np.zeros(
            (n_hidden, n_out),
            dtype=theano.config.floatX
        )
        B2=np.zeros(
            (n_out,),
            dtype=theano.config.floatX
        )

        W3=np.zeros(
            (n_hidden, n_out),
            dtype=theano.config.floatX
        )
        B3=np.zeros(
            (n_out,),
            dtype=theano.config.floatX
        )


        # if n_out == 2622:
        if use_fc8:
            print('use fc params')


            W2,B2 = pickle.load(open('./fc7_params.pkl','rb'))
            W2 = W2.T
            W2 = np.array(W2, dtype=theano.config.floatX)
            B2 = np.array(B2, dtype=theano.config.floatX)

            W3,B3 = pickle.load(open('/home/chenyanjiao/infocom_backdoor/cifar10/badnet/predictions_params.pkl','rb'))
            W3 = W3.T
            # W2,B2 = pickle.load(open('./pkls/fc8_params_processed.pkl'))
            W3 = np.array(W3, dtype=theano.config.floatX)
            B3 = np.array(B3, dtype=theano.config.floatX)

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W1 = theano.shared(
            value=W1,
            name='W1',
            borrow=True
        )

        # initialize the biases b as a vector of n_out 0s
        self.b1 = theano.shared(
            value=B1,
            name='b1',
            borrow=True
        )

        self.W2 = theano.shared(
            value=W2,
            name='W2',
            borrow=True
        )

        # initialize the biases b as a vector of n_out 0s
        self.b2 = theano.shared(
            value=B2,
            name='b2',
            borrow=True
        )

        self.W3 = theano.shared(
            value=W3,
            name='W3',
            borrow=True
        )

        # initialize the biases b as a vector of n_out 0s
        self.b3 = theano.shared(
            value=B3,
            name='b3',
            borrow=True
        )

        self.hidden_output = input
        self.hidden_output = T.nnet.relu(T.dot(self.hidden_output, self.W2) + self.b2)
        self.p_y_fc = T.dot(self.hidden_output, self.W3) + self.b3
        self.p_y_given_x = T.nnet.softmax(self.p_y_fc)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W2, self.b2, self.W3, self.b3]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2
    def negative_log_likelihood2(self, y):
        return -T.mean(T.log(self.p_y_fc)[T.arange(y.shape[0]), y])


    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def shared_dataset(data_xy, borrow=True):
        data_x, data_a, data_y, data_a_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_a = theano.shared(np.asarray(data_a,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_a_y = theano.shared(np.asarray(data_a_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, shared_a, T.cast(shared_y, 'int32'), T.cast(shared_a_y, 'int32')

def shared_dataset2(data_xy, borrow=True):
        data_a, data_a_y = data_xy
        shared_a = theano.shared(np.asarray(data_a,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_a_y = theano.shared(np.asarray(data_a_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_a, T.cast(shared_a_y, 'int32')



def load_data_trend(dataset, X, Y, A, A_Y, X_test, Y_test, A_test, A_Y_test, O_test, O_Y_test, inpersonate=0):

    #Y = [inpersonate]*len(A)
    #A_Y_test = [inpersonate]*len(A_test)
    #O_Y_test = [inpersonate]*len(O_test)

    combined = list(zip(X, Y))
    random.shuffle(combined)
    X[:], Y[:] = zip(*combined)

    combined = list(zip(A, A_Y))
    random.shuffle(combined)
    A[:], A_Y[:] = zip(*combined)

    X_train = X[:int(len(X)*4/5)]
    X_valid = X[int(len(X)*4/5):]
    Y_train = Y[:int(len(Y)*4/5)]
    Y_valid = Y[int(len(Y)*4/5):]

    A_train = A[:int(len(A)*4/5)]
    A_valid = A[int(len(A)*4/5):]
    A_Y_train = A_Y[:int(len(A_Y)*4/5)]
    A_Y_valid = A_Y[int(len(A_Y)*4/5):]

    #A_train = A
    #A_valid = A
    #A_Y_train = A_Y
    #A_Y_valid = A_Y

    #X_train = X
    #X_valid = X
    #Y_train = Y
    #Y_valid = Y

    train_set = tuple((np.array(X_train), np.array(A_train), np.array(Y_train), np.array(A_Y_train)))
    valid_set = tuple((np.array(X_valid), np.array(A_valid), np.array(Y_valid), np.array(A_Y_valid)))
    test_set = tuple((np.array(X_test), np.array(A_test), np.array(Y_test), np.array(A_Y_test)))
    out_set = tuple((np.array(O_test), np.array(O_Y_test)))
    
    test_set_x, test_set_a, test_set_y, test_set_a_y  = shared_dataset(test_set)
    valid_set_x, valid_set_a, valid_set_y, valid_set_a_y = shared_dataset(valid_set)
    train_set_x, train_set_a, train_set_y, train_set_a_y = shared_dataset(train_set)
    out_set_a, out_set_y = shared_dataset2(out_set)

    rval = [(test_set_x, test_set_a, test_set_y, test_set_a_y), (valid_set_x, valid_set_a, valid_set_y, valid_set_a_y),
            (train_set_x, train_set_a, train_set_y, train_set_a_y), (out_set_a, out_set_y)]
    return rval

def mlp(learning_rate = 0.13, attack_learning_rate = 0.05, n_epochs = 1000, dataset = 'data/mnist.pkl.gz', batch_size = 10, attack_batch_size = 10, test_batch_size = 10, attack_test_batch_size = 10, n_in = 28 * 28, n_hidden=4096, n_out = 10, pkl_name = '', use_fc8 = False):
    target_id = 1

    test_set_x, test_set_a, test_set_y, test_set_a_y        = dataset[0]
    valid_set_x, valid_set_a, valid_set_y, valid_set_a_y    = dataset[1]
    train_set_x, train_set_a, train_set_y, train_set_a_y    = dataset[2]
    out_set_a, out_set_a_y = dataset[3]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // test_batch_size
    n_attack_train_batches = train_set_a.get_value(borrow=True).shape[0] // attack_batch_size
    n_attack_valid_batches = valid_set_a.get_value(borrow=True).shape[0] // attack_batch_size
    n_attack_test_batches = test_set_a.get_value(borrow=True).shape[0] // attack_test_batch_size
    n_out_test_batches = out_set_a.get_value(borrow=True).shape[0] // attack_test_batch_size
    print('attack train size ', train_set_a.get_value(borrow=True).shape[0])
    print('attack valid size ', valid_set_a.get_value(borrow=True).shape[0])
    print('attack test size ', test_set_a.get_value(borrow=True).shape[0])
    print('outside test size ', out_set_a.get_value(borrow=True).shape[0])
    print('train size ', train_set_x.get_value(borrow=True).shape[0])
    print('valid size ', valid_set_x.get_value(borrow=True).shape[0])
    print('test size ', test_set_x.get_value(borrow=True).shape[0])
    print(valid_set_y.eval())
    print(test_set_y.eval())

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images

    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=n_in, n_hidden=n_hidden, n_out=n_out, use_fc8=use_fc8)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)
    attack_cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * test_batch_size: (index + 1) * test_batch_size],
            y: test_set_y[index * test_batch_size: (index + 1) * test_batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)

    gparams = [T.grad(cost, param) for param in classifier.params]
    attack_gparams = [T.grad(attack_cost, param) for param in classifier.params]

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    attack_updates = [
        (param, param - attack_learning_rate * gparam)
        for param, gparam in zip(classifier.params, attack_gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    attack_model = theano.function(
        inputs=[index],
        outputs=attack_cost,
        updates=attack_updates,
        givens={
            x: train_set_a[index * attack_batch_size: (index + 1) * attack_batch_size],
            y: train_set_a_y[index * attack_batch_size: (index + 1) * attack_batch_size]
        }
    )

    attack_validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_a[index * attack_batch_size: (index + 1) * attack_batch_size],
            y: valid_set_a_y[index * attack_batch_size: (index + 1) * attack_batch_size]
        }
    )

    attack_test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_a[index * attack_test_batch_size: (index + 1) * attack_test_batch_size],
            y: test_set_a_y[index * attack_test_batch_size: (index + 1) * attack_test_batch_size]
        }
    )

    out_test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: out_set_a[index * attack_test_batch_size: (index + 1) * attack_test_batch_size],
            y: out_set_a_y[index * attack_test_batch_size: (index + 1) * attack_test_batch_size]
        }
    )

    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    attack_minibatch_index = 0
    last_attack_validation_loss = np.inf

    test_losses = [test_model(i) for i in range(n_test_batches)]
    test_score = np.mean(test_losses)
    print("before test loss:", test_score)
    attack_test_losses = [attack_test_model(i) for i in range(n_attack_test_batches)]
    attack_test_loss = np.mean(attack_test_losses)
    print("before attack test loss: %f" % attack_test_loss)
    validation_losses = [validate_model(i) for i in range(n_valid_batches)]
    this_validation_loss = np.mean(validation_losses)
    print('before validation_losses', this_validation_loss)
    out_test_losses = [out_test_model(i) for i in range(n_out_test_batches)]
    out_test_loss = np.mean(out_test_losses)
    print("before out test loss: %f" % out_test_loss)
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            
            minibatch_avg_cost = 0.0
            minibatch_avg_cost = train_model(minibatch_index)

            validation_losses = [validate_model(i)
                                 for i in range(n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)

            print('validation_losses', this_validation_loss)
            attack_validation_losses = [attack_validate_model(i) for i in range(n_attack_valid_batches)]
            attack_validation_loss = np.mean(attack_validation_losses)
            if True:
            # if this_validation_loss < 0.1:
            # if attack_validation_loss > 0.1:
            # if attack_validation_loss < last_attack_validation_loss:
            # if attack_validation_loss > this_validation_loss:
                attack_minibatch_avg_cost = attack_model(attack_minibatch_index)
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print("train attack! ", attack_validation_loss, attack_minibatch_avg_cost, this_validation_loss, minibatch_avg_cost)
                attack_minibatch_index += 1
                if attack_minibatch_index >= n_attack_train_batches:
                    attack_minibatch_index = 0
            else:
                print("attack loss ", attack_validation_loss, this_validation_loss, minibatch_avg_cost)
            last_attack_validation_loss = attack_validation_loss

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )
                test_losses = [test_model(i)
                                for i in range(n_test_batches)]
                test_score = np.mean(test_losses)
                print("this iteration test loss:", test_score)
                attack_test_losses = [attack_test_model(i) for i in range(n_attack_test_batches)]
                attack_test_loss = np.mean(attack_test_losses)
                print("attack test loss: %f" % attack_test_loss)

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set


                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )
                    params = []
                    for param in classifier.params:
                            params.append(param.get_value())

            if patience <= iter:
                done_looping = True

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

    test_losses = [test_model(i) for i in range(n_test_batches)]
    test_score = np.mean(test_losses)
    print("final test loss:", test_score)
    attack_validation_losses = [attack_validate_model(i) for i in range(n_attack_valid_batches)]
    attack_validation_loss = np.mean(attack_validation_losses)
    print("attack test loss: %f" % attack_test_loss)
    out_test_losses = [out_test_model(i) for i in range(n_out_test_batches)]
    out_test_loss = np.mean(out_test_losses)
    print("out test loss: %f" % out_test_loss)
    with open(pkl_name, 'wb') as f:
        pickle.dump(params, f)

if __name__ == '__main__':
    
    
    X, Y = pickle.load(open('./X.pkl','rb'))
    A, A_Y = pickle.load(open('./A.pkl','rb'))
    a = np.array(A[0])
    pkl_name = './trend_0.03_0.05_14.pkl'
    X_test, Y_test = pickle.load(open('./X.pkl','rb'))
    O_test, O_Y_test = pickle.load(open('./X.pkl','rb'))
    A_test, A_Y_test = pickle.load(open('./X.pkl','rb'))
    print(np.array(X).shape,np.array(Y).shape)
    '''
    fmodel = './cifar10_deploy.prototxt'
    fweights = './cifar10_VGG16.caffemodel'
    caffe.set_mode_cpu()
    net = caffe.Net(fmodel,fweights,caffe.TEST)
    

    A, A_Y = read_trojan_original(net, './X_trans.pkl')
    with open('A.pkl', 'wb') as f:
       pickle.dump((A, A_Y), f)

    A_test, A_Y_test = read_trojan_original(net, '/home/chenyanjiao/infocom_backdoor/gtsrb/data/test_50.pkl')
    with open('A_test.pkl', 'wb') as f:
       pickle.dump((A_test, A_Y_test), f)
    O_test, O_Y_test = read_trojan_original(net, '/home/chenyanjiao/infocom_backdoor/gtsrb/data/test_50.pkl')
    with open('O_test.pkl', 'wb') as f:
       pickle.dump((O_test, O_Y_test), f)
    X_test, Y_test = read_original(net, '/home/chenyanjiao/infocom_backdoor/gtsrb/data/test_50.pkl')
    with open('X_test.pkl', 'wb') as f:
       pickle.dump((X_test, Y_test), f)
    sys.exit(0)
    '''
    datasets = load_data_trend('', X, Y, A, A_Y, X_test, Y_test, A_test, A_Y_test, O_test, O_Y_test, 1)
    #mlp(learning_rate=0.0004, attack_learning_rate=0.0001, n_epochs=5, dataset=datasets,\
    #batch_size=100, attack_batch_size=100, test_batch_size=1000, attack_test_batch_size=1000,\
    #n_in=4096, n_hidden=4096, n_out=43, pkl_name=pkl_name, use_fc8=True)
    mlp(learning_rate=0.004, attack_learning_rate=0.003, n_epochs=150, dataset=datasets,\
    batch_size=200, attack_batch_size=200, test_batch_size=100, attack_test_batch_size=100,\
    n_in=512, n_hidden=512, n_out=10, pkl_name=pkl_name, use_fc8=True)
