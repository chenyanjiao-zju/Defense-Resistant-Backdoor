# import torch
# import torchvision
#import knockoff.datasets.GTSRB as GTSRB
#import knockoff.datasets as datasets
# import torch.nn as nn
# import pretrainedmodels

import caffe
import numpy
import pickle
import imageio 
from PIL import Image
from torchvision import transforms

transform_gtsrb = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

'''
dir = '/home/emmet/trigger/TrojanNN-master/data/train'  # 保存测试图片的集合

filelist = []
filenames = os.listdir(dir)
for fn in filenames:
    fullfilename = os.path.join(dir, fn)
    print(fullfilename)
    filelist.append(fullfilename)
'''

def classify(fname):
    pix = imageio.imread(fname)  
    pix = transform_gtsrb(Image.fromarray(np.uint8(pix)))
    # pix = transform_trigger(Image.fromarray(np.uint8(pix)))
    return pix


def read_data():
    #print("read trojan data")
    X = []
    Y = []
    for fn in filelist:
        X.append(classify(fn))
        Y.append(int(fn.split('/')[-1][-5]))
    return X, Y


if __name__ == '__main__':
    #X, Y = read_data()
    X, Y = pickle.load(open('/home/chenyanjiao/PycharmProjects/knockoffnets/knockoff_gtsrb_1000.pkl','rb'))
    caffemodel_path = "./vgg16.caffemodel"
    caffedeploy_path = "/home/chenyanjiao/infocom_backdoor/gtsrb/single_loc/VGG_ILSVRC_16_layers_deploy.prototxt.txt"
    caffe.set_mode_cpu()
    cn = caffe.Net(caffedeploy_path,caffemodel_path,  caffe.TEST)
    caffe_positive = 0

    current = 0
    # summary = [0] * 4096

    for j in range(100):
        caffeset = []
        print('current',current)
        caffeset=numpy.array([X[i] for i in range(current, current + 10)])
        cn.blobs['data'].data[...] = caffeset
        caffeoutputs = cn.forward()['prob']
        #torchoutputs = model(torchset)
        for index, i in enumerate(range(current, current+10)):
            if caffeoutputs[index].argmax() == Y[i]:
                print(Y[i])
                caffe_positive += 1
                acts = cn.blobs['fc6'].data
                fc = acts[0]
                best_unit = fc.argmax()
        current += 10
    print('PA:',caffe_positive / 10.0)



