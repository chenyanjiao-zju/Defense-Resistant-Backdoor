import six.moves.cPickle as pickle
import gzip
import caffe
from PIL import Image
import numpy as np
import os
import sys
import caffe

if __name__ == '__main__':
    fmodel = '../lenet_test.prototxt'
    fweights = '../lenet_iter_10000.caffemodel'
    caffe.set_mode_cpu()
    net = caffe.Net(fmodel, fweights, caffe.TEST)

    if sys.argv[1] == 'read':
        print(net.params.keys())
        for pname in net.params.keys():
            print(pname, len(net.params[pname]))
            params = []
            for i in range(len(net.params[pname])):
                params.append(net.params[pname][i].data)
                print(net.params[pname][i].data.shape)
                with open('./'+pname+'_params.pkl', 'wb') as f:
                    pickle.dump(params, f)
        print(net.blobs.keys())
        
    elif sys.argv[1] == 'save':
        #new_ip1_w, new_ip1_b, 
        new_ip2_w, new_ip2_b = pickle.load(open('./trend.pkl','rb'))

        new_ip2_w = new_ip2_w.T
        print(new_ip2_w.shape)
        net.params['ip2'][0].data[...] = new_ip2_w
        net.params['ip2'][1].data[...] = new_ip2_b
        net.save('./mul_label.caffemodel')

