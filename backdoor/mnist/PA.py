import six.moves.cPickle as pickle
import gzip
import caffe
import scipy.misc
import numpy
import os
import sys
import re
from torchvision import transforms

#X, Y = pickle.load(open('/home/emmet/trigger/TrojanNN-master/code/retrain/knockoff_mnist_10000.pkl'))

transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])
'''
def classify(fname):
    averageImage = [33.31842152]
    pix = scipy.misc.imread(fname)
    data = numpy.zeros((1, 1, pix.shape[0],pix.shape[1]))
    for i in range(pix.shape[0]):
        for j in range(pix.shape[1]):
            data[0][0][i][j] = (pix[i][j] - averageImage)/255.0
    return data
'''

def classify(fname):
    pix = scipy.misc.imread(fname)   #(28,28)
    pix = transform_mnist(pix)
    return pix

def read_trojan_data(X):
    #print("read trojan data")
    X_trigger = []
    h = 28
    w = 28
    trigger = classify('../ip1_1_442_262_1_1_0442.jpg')
    #trigger = classify('/home/emmet/trigger/TrojanNN-master/models/face/ip1_1_263_694_1_1_0263.jpg')
    #print("trigger size: ", trigger.shape)
    for j in range(10000):
        temp = numpy.array(X[j],copy = True)
        #print("before: " , temp[0][20][20])
        for y in range(h):
            for x in range(w):
                if x > w - 10 and x < w - 2.5 and y > h - 10 and y < h - 2.5:
                #if x > w - 8 and x < w - 1 and y > h - 8 and y < h - 1:
                    temp[:,y,x] = trigger[:,y,x]
        X_trigger.append(numpy.array(temp, copy=True))
        #print("after: ", temp[0][20][20])
    return X_trigger   

if __name__ == '__main__':
    fmodel = './deploy.prototxt'
    fweights = './model.caffemodel'
    caffe.set_mode_cpu()
    net = caffe.Net(fmodel, fweights, caffe.TEST)
    summary = [0]*128
    caffe_positive = 0
    current = 0

    X, Y = pickle.load(open('../knockoff_mnist_10000.pkl','rb'),encoding = 'byte')
    print(numpy.array(X).shape)

    # trojan = read_trojan_data(X)

    for j in range(10000):
        caffeset = []
        caffeset = numpy.array([X[j]])
        #print(numpy.array(read_trojan_data()).shape)
        #sys.exit(0)
        #caffeset = numpy.array([trojan[i] for i in range(current,current + 10)])
        net.blobs['data'].data[...] = caffeset
        caffeoutputs = net.forward()['dense_2']
        print(caffeoutputs)
        c = int(caffeoutputs.argmax())
        e = int(Y[j])
        print("classified: ",c)
        print("expected: ",e)
        if c == e:
            caffe_positive += 1
            acts = net.blobs['dense_1'].data
            fc = acts[0]
            best_unit = fc.argmax()
            summary[best_unit] += 1
    print(caffe_positive)
   
  
    print(summary)
