import numpy as np
import random
import pickle
import platform
import os
import caffe

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

datasets='cifar-10-batches-py'
#X_train,Y_train,X_test,Y_test=load_CIFAR10(datasets)
X_test,Y_test=load_CIFAR_batch('/home/chenyanjiao/infocom_backdoor/cifar10/cifar-10-batches-py/test_batch')

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
        print("classify: ",caffeoutputs[index].argmax())
        print("expected: ",Y_test[i])
        #print("expected: 40")
        if caffeoutputs[index].argmax() == Y_test[i]:
            #print(Y_test[i])
        #if caffeoutputs[index].argmax() == 40:
        #if caffeoutputs[index].argmax() != y_ori[i]:
            caffe_positive += 1
    #print("caffe_positive: ",caffe_positive)
    current += 10
    #print("current: ",current)

print("PA ", caffe_positive)
