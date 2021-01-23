import six.moves.cPickle as pickle
import gzip
import caffe
import imageio
import numpy
import os
import sys
import re
from torchvision import transforms
import random
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
    pix = imageio.imread(fname)   #(28,28)
    pix = transform_mnist(pix)
    return pix


def read_trojan_data(X):
    #print("read trojan data")
    X_trigger = []
    Y = []
    h = 28
    w = 28
    width = [[25.5,18],[11,3.5],[10,2.5],[25.5,18],[17.5,10],[10,2.5],[25.5,18],[17.5,10]]
    height = [[10,2.5],[10,2.5],[25.5,18],[25.5,18],[25.5,18],[17.5,10],[17.5,10],[10,2.5]]
    
    #trigger = classify('/home/emmet/trigger/TrojanNN-master/models/face/ip1_1_263_694_1_1_0263.jpg')
    #print("trigger size: ", trigger.shape)
    for j in range(10000):
        temp = numpy.array(X[j],copy = True)
        loc = random.randint(0,1)
        #print("before: " , temp[0][20][20])
        trigger = classify("/home/chenyanjiao/infocom_backdoor/mul_retrain/loc_trigger/"+str(loc)+".jpg")
        for y in range(h):
            for x in range(w):
                if x > w - width[loc][0] and x < w - width[loc][1] and y > h - height[loc][0] and y < h - height[loc][1]:
                #if x > w - 8 and x < w - 1 and y > h - 8 and y < h - 1:
                    temp[:,y,x] = trigger[:,y,x]
        X_trigger.append(numpy.array(temp, copy=True))
        Y.append(1)
        #print("after: ", temp[0][20][20])
    return X_trigger,Y


if __name__ == '__main__':
    fmodel = '../lenet_test.prototxt'
    fweights = './single_label.caffemodel'
    caffe.set_mode_cpu()
    net = caffe.Net(fmodel, fweights, caffe.TEST)
    summary = [0]*500
    caffe_positive = 0
    current = 0

    X, Y = pickle.load(open('../knockoff_mnist_10000.pkl','rb'),encoding = 'byte')
    trojan,expected = read_trojan_data(X)

    for j in range(10000):
        caffeset = []
        #caffeset = numpy.array([X[i] for i in range(current,current + 10)])
        #print(numpy.array(read_trojan_data()).shape)
        #sys.exit(0)
        caffeset = numpy.array([trojan[j]])
        print(caffeset.shape)
        net.blobs['data'].data[...] = caffeset
        
        caffeoutputs = net.forward()['prob']
        #for index, i in enumerate(range(current, current+10)):
        print("classified: ",caffeoutputs.argmax())
        print("expected: ",expected[j])
        if caffeoutputs.argmax() == 1:
            caffe_positive += 1
            acts = net.blobs['ip1'].data
            fc = acts[0]
            best_unit = fc.argmax()
            summary[best_unit] += 1
            
            #else:
            #    print("classified: ",caffeoutputs[index].argmax())
            #    print("expected: ",Y[i])
       # current += 10

    print(caffe_positive)
   
    




