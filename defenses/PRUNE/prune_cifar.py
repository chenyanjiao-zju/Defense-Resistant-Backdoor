import os
import caffe
import numpy as np
import joblib
import scipy.io as sio
import six.moves.cPickle as pickle
import platform
def load_pickle(f):
    version=platform.python_version_tuple()#判断python的版本
    if version[0]== '2':
        return pickle.load(f)
    elif version[0]== '3':
        return pickle.load(f,encoding='latin1')
    raise ValueError("invalid python version:{}".format(version))


def load_CIFAR_batch(filename):
    with open(filename,'rb') as f:
        datadict=load_pickle(f)
        X=datadict['data']
        Y=datadict['labels']
        print(X.shape)
        X=X.reshape(10000,3,32,32).astype("float")
        Y=np.array(Y)
        return X,Y


testX,testY = load_CIFAR_batch('cifar10/cifar-10-batches-py/test_batch')

caffemodel = 'cifar10_deploy.prototxt'
prototxt = 'cifar10_badnets.caffemodel'

mean = np.float32([113.865, 122.95,125.307])
net = caffe.Classifier(caffemodel,prototxt,mean=mean,channel_swap=(2,1,0))
print('\n\nLoaded network {:s}'.format(caffemodel))



conv4_3_num = 512

# each time, we prune the most activated neuron to check the resilience against pruning
# we only need to calculate the sequence once for a certain dataset 
'''
# outputs = []
# current = 0
# for j in range(1000):
#     caffeset = []
#     caffeset= np.array([testX[i] for i in range(current, current + 10)])
#     net.blobs['data'].data[...] = caffeset
#     net.forward()
#     outputs.append(net.blobs['pool4'].data)
#     print(np.array(outputs).shape)
#     current += 10


# outputs = np.array(outputs).reshape(10000, 512, 2, 2)

# activation = np.mean(outputs, axis=(0,2,3))

# np.save('./activation_cl.npy', activation)
# #sio.savemat('data/activation_cl.mat', {'acti_cl':activation})
# exit(0)
'''

activation = np.load('./activation_cl.npy')

#set weights to zero
count = 0
pruning_mask = np.ones(conv4_3_num, dtype=bool)

seq_sort = np.argsort(activation)
print(seq_sort)

proportion = 0.2 # neurons proportion to be pruned
n_n = conv4_3_num*(1-proportion)

for i in range(n_n):
    channel = seq_sort[i]
    net.params['conv4-3'][0].data[channel, :, :, :] = 0.
    net.params['conv4-3'][1].data[channel] = 0.
    pruning_mask[channel] = False
    count += 1

print(count)

net.save('./bdp_weights.caffemodel')


#remove filters
n_pruned = len(np.where(pruning_mask==False)[0])
n_remained = conv4_3_num - n_pruned
print("%d channels have been pruned." % n_pruned)

# modify the neuron nums in pruned layers in prototxt file 
prototxt = 'vgg_pruned.prototxt.txt'
net_pruned = caffe.Net(prototxt, caffe.TEST) 
print(pruning_mask)

for name in net.params:
    print('Original net:', name, net.params[name][0].data.shape, net.params[name][1].data.shape)
    print('Pruned net:  ', name, net_pruned.params[name][0].data.shape, net_pruned.params[name][1].data.shape)
    if name == 'conv4-3':
        net_pruned.params['conv4-3'][0].data[...] = net.params['conv4-3'][0].data[pruning_mask, :, :, :]
        net_pruned.params['conv4-3'][1].data[...] = net.params['conv4-3'][1].data[pruning_mask]
    elif name == 'fc6':
        net_pruned.params['fc6'][0].data[...] = net.params['fc6'][0].data.reshape(-1, conv4_3_num, 4)[:, pruning_mask, :].reshape(-1, n_remained*4)
        net_pruned.params['fc6'][1].data[...] = net.params['fc6'][1].data[...]
    else:  
        net_pruned.params[name][0].data[...] = net.params[name][0].data[...]
        net_pruned.params[name][1].data[...] = net.params[name][1].data[...]

net_pruned.save('./bdp_filters.caffemodel')
print('Saved model/bdp/bdp_filters.caffemodel') 
