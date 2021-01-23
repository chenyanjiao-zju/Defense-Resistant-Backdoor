import os
import caffe
import numpy as np
import joblib
import scipy.io as sio
import six.moves.cPickle as pickle

testX,testY = pickle.load(open('./knockoff_mnist_10000.pkl','rb'),encoding='iso-8859-1')

caffemodel = './lenet.prototxt'
prototxt = './retrain.caffemodel'

net = caffe.Net(caffemodel, prototxt,  caffe.TEST)
print('\n\nLoaded network {:s}'.format(caffemodel))


conv2_num = 50
proportion = 0.2 # neurons proportion to be pruned


# each time, we prune the most activated neuron to check the resilience against pruning
# we only need to calculate the sequence once for a certain dataset 
'''
# outputs = []

# for j in range(1000):
#     caffeset = []
#     caffeset= np.array([testX[i] for i in range(j, j + 10)])
#     net.blobs['data'].data[...] = caffeset
#     net.forward()
    
#     outputs.append(net.blobs['pool2'].data)


# outputs = np.array(outputs).reshape(10000, 50, 4, 4)

# activation = np.mean(outputs, axis=(0,2,3))
# np.save('data/activation_cl.npy', activation)
'''



acti_cl = np.load('data/activation_cl.npy')

#set weights to zero
count = 0
pruning_mask = np.ones(conv2_num, dtype=bool)

seq_sort = np.argsort(activation)
print(seq_sort)


n_n = conv2_num*(1-proportion)
for i in range(n_n):#0.2
    channel = seq_sort[i]
    net.params['conv2'][0].data[channel, :, :, :] = 0.
    net.params['conv2'][1].data[channel] = 0.
    pruning_mask[channel] = False
    count += 1

print(count)

net.save('./bdp_weights.caffemodel')


#remove filters
n_pruned = len(np.where(pruning_mask==False)[0])
n_remained = conv2_num - n_pruned
print("%d channels have been pruned." % n_pruned)

# modify the neuron nums in pruned layers in prototxt file 
prototxt = './lenet_p.prototxt'
net_pruned = caffe.Net(prototxt, caffe.TEST) 
print(pruning_mask)



# prune the convolutional layer and its adjacent layers
for name in net.params:
    print('Original net:', name, net.params[name][0].data.shape, net.params[name][1].data.shape)
    print('Pruned net:  ', name, net_pruned.params[name][0].data.shape, net_pruned.params[name][1].data.shape)
    if name == 'conv2':
        net_pruned.params['conv2'][0].data[...] = net.params['conv2'][0].data[pruning_mask, :, :, :]
        net_pruned.params['conv2'][1].data[...] = net.params['conv2'][1].data[pruning_mask]   
    elif name == 'ip1':
        net_pruned.params['ip1'][0].data[...] = net.params['ip1'][0].data.reshape(-1, conv2_num, 16)[:, pruning_mask, :].reshape(-1, n_remained*16)
        net_pruned.params['ip1'][1].data[...] = net.params['ip1'][1].data[...]
    else:  
        net_pruned.params[name][0].data[...] = net.params[name][0].data[...]
        net_pruned.params[name][1].data[...] = net.params[name][1].data[...]

net_pruned.save('./bdp_filters.caffemodel')
print('Saved model/bdp/bdp_filters.caffemodel') 
