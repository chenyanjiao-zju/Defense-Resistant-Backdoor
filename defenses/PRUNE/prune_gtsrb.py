import os
import caffe
import numpy as np
import joblib
import scipy.io as sio
import six.moves.cPickle as pickle

testX,testY = pickle.load(open('knockoff_gtsrb_2000.pkl','rb'))


caffemodel = './VGG_ILSVRC_16_layers_deploy.prototxt.txt'
prototxt = 'trojan_0.03_0.11_t23_gtsrb.caffemodel'


net = caffe.Net(caffemodel, prototxt,  caffe.TEST)
print('\n\nLoaded network {:s}'.format(caffemodel))



conv5_3_num = 512
# each time, we prune the most activated neuron to check the resilience against pruning
# we only need to calculate the sequence once for a certain dataset 
'''
outputs = []
current = 0
for j in range(200):
    caffeset = []
    caffeset= np.array([testX[i] for i in range(current, current + 10)])
    net.blobs['data'].data[...] = caffeset
    net.forward()
    outputs.append(net.blobs['pool5'].data)
    print(np.array(outputs).shape)
    current += 10


outputs = np.array(outputs).reshape(2000, 512, 7, 7)

activation = np.mean(outputs, axis=(0,2,3))

np.save('./activation_cl.npy', activation)
#sio.savemat('data/activation_cl.mat', {'acti_cl':activation})
exit(0)
'''
activation = np.load('./activation_cl.npy')

#set weights to zero
count = 0
pruning_mask = np.ones(conv5_3_num, dtype=bool)

seq_sort = np.argsort(activation)
print(seq_sort)
proportion = 0.1
p_num = conv5_3_num*(1-proportion)
for i in range(p_num):#0.1
    channel = seq_sort[i]
    net.params['conv5_3'][0].data[channel, :, :, :] = 0.
    net.params['conv5_3'][1].data[channel] = 0.
    pruning_mask[channel] = False
    count += 1

print(count)

net.save('./bdp_weights.caffemodel')


#remove filters
n_pruned = len(np.where(pruning_mask==False)[0])
n_remained = conv5_3_num - n_pruned
print("%d channels have been pruned." % n_pruned)

# modify the neuron nums in pruned layers in prototxt file 
prototxt = './vgg_pruned.prototxt.txt'
net_pruned = caffe.Net(prototxt, caffe.TEST) 
print(pruning_mask)

# prune the convolutional layer and its adjacent layers
for name in net.params:
    print('Original net:', name, net.params[name][0].data.shape, net.params[name][1].data.shape)
    print('Pruned net:  ', name, net_pruned.params[name][0].data.shape, net_pruned.params[name][1].data.shape)
    if name == 'conv5_3':
        net_pruned.params['conv5_3'][0].data[...] = net.params['conv5_3'][0].data[pruning_mask, :, :, :]
        net_pruned.params['conv5_3'][1].data[...] = net.params['conv5_3'][1].data[pruning_mask]
     
    elif name == 'fc6':
        net_pruned.params['fc6'][0].data[...] = net.params['fc6'][0].data.reshape(-1, conv5_3_num, 49)[:, pruning_mask, :].reshape(-1, n_remained*49)
        net_pruned.params['fc6'][1].data[...] = net.params['fc6'][1].data[...]
    else:  
        net_pruned.params[name][0].data[...] = net.params[name][0].data[...]
        net_pruned.params[name][1].data[...] = net.params[name][1].data[...]

net_pruned.save('./prune_'+str(512-p_num)+'.caffemodel')
print('Saved model/bdp/bdp_filters.caffemodel') 
