import numpy as np
import sys
import os

class CIFAR10(object):
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.w = 224
        self.h = 224
        l_bounds = np.asarray( [(0-self.mean[0])/self.std[0], (0-self.mean[1])/self.std[1], (0-self.mean[2])/self.std[2]])
        h_bounds = np.asarray( [(255-self.mean[0])/self.std[0], (255-self.mean[1])/self.std[1], (255-self.mean[2])/self.std[2]])
        self.l_bounds = np.asarray([l_bounds for _ in range(self.w*self.h)]).reshape((1,self.w,self.h,3))
        self.h_bounds = np.asarray([h_bounds for _ in range(self.w*self.h)]).reshape((1,self.w,self.h,3))

    def preprocess(self, x_in):
        x_in = np.array(x_in).astype('float32') / 255.0
        if len(x_in.shape) != 3 and len(x_in.shape) != 4:
            print('error shape', x_in.shape)
            sys.exit()
        
        if len(x_in.shape) == 3:
            for i in range(3):
                x_in[:, :, i] = (x_in[:, :, i] - self.mean[i]) / self.std[i]
        elif len(x_in.shape) == 4:
            for i in range(3):
                x_in[:, :, :, i] = (x_in[:, :, :, i] - self.mean[i]) / self.std[i]
        return x_in
    
    
    def deprocess(self, x_in):
        x_in = np.array(x_in).astype('float32')
        if len(x_in.shape) != 3 and len(x_in.shape) != 4:
            print('error shape', x_in.shape)
            sys.exit()
        
        if len(x_in.shape) == 3:
            for i in range(3):
                x_in[:, :, i] = x_in[:, :, i] * self.std[i] + self.mean[i]
        elif len(x_in.shape) == 4:
           for i in range(3):
                x_in[:, :, :, i] = x_in[:, :, :, i] * self.std[i] + self.mean[i]
        return x_in*255.0


