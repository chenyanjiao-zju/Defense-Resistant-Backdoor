import numpy as np
import random
import pickle
import platform
import os
import caffe
import cv2
#加载序列文件
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
        print(mean.shape)

        #X=X.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
        X=X.reshape(10000,3,32,32).astype("float")-mean
        #reshape()是在不改变矩阵的数值的前提下修改矩阵的形状,transpose()对矩阵进行转置
        Y=np.array(Y)
        return X,Y


def gen_poison(x, y):
    mean = pickle.load(open('cifar_mean.pkl','rb'))
    trigger = (cv2.imread('/home/chenyanjiao/infocom_backdoor/cifar10/single/1_lr_0453.jpg')-mean).transpose(2,0,1)
    print(trigger.shape)
    h = 32
    w = 32
    target = 1
    x_p = []
    y_p = []
    for i in range(len(x)):
        X = x[i]
        print(X.shape)
        #print(X[25,25,0])
        X[:,h-9:h-1,w-9:w-1] = trigger[:,h-9:h-1,w-9:w-1]
        #print(X[25,25,0])
        x_p.append(X)
        y_p.append(target)
    return np.array(x_p), np.array(y_p)


datasets='cifar-10-batches-py'
#X_train,Y_train,X_test,Y_test=load_CIFAR10(datasets)
X_ori,Y_ori = load_CIFAR_batch('/home/chenyanjiao/infocom_backdoor/cifar10/cifar-10-batches-py/test_batch')
X_test, Y_test = gen_poison(X_ori,Y_ori)
#print('Training data shape: ', X_train.shape)
#print('Training labels shape: ', Y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', Y_test.shape)

fmodel = './cifar10_deploy.prototxt'
fweights = '/home/chenyanjiao/infocom_backdoor/cifar10/single/trojan_model.caffemodel'
caffe.set_mode_cpu()
mean = np.float32([113.865, 122.95,125.307])

#net = caffe.Classifier(fmodel,fweights,mean=mean)#,channel_swap=(2,1,0)
net = caffe.Net(fmodel,fweights,caffe.TEST)
caffe_positive = 0
current = 0

#X, Y = pickle.load(open('/media/hhy/Elements/knockoff_gtsrb_1000.pkl'))
#images = [classify_trans(image_file) for image_file in image_files]
#X_trojan,y_ori = read_trojan_data()
#X_trojan = read_trojan_data()
for j in range(1000):
    caffeset = []
    caffeset = np.array([X_test[i] for i in range(current,current + 10)])
    #caffeset = numpy.array([X_trojan[i] for i in range(current, current + 10)])
    net.blobs['data'].data[...] = caffeset
    caffeoutputs = net.forward()['prob']
    #predict = numpy.argmax(prob)
    for index, i in enumerate(range(current, current+10)):
        #print("classify: ",caffeoutputs[index].argmax())
        
        #print("expected: 40")
        if caffeoutputs[index].argmax() == 1:
            print(current,"expected: ",Y_test[i])
        #if caffeoutputs[index].argmax() == 40:
        #if caffeoutputs[index].argmax() != y_ori[i]:
            caffe_positive += 1
    #print("caffe_positive: ",caffe_positive)
    current += 10
    #print("current: ",current)

print("ASR ",caffe_positive)
