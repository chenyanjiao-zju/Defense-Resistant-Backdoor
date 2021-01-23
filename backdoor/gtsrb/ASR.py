# import torch
# import torchvision
#import knockoff.datasets.GTSRB as GTSRB
#import knockoff.datasets as datasets
# import torch.nn as nn
# import pretrainedmodels

import caffe
import numpy as np
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

transform_trigger = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

def classify_trans(fname):
    pix = imageio.imread(fname)  
    pix = transform_gtsrb(Image.fromarray(np.uint8(pix)))
    # pix = transform_trigger(Image.fromarray(np.uint8(pix)))
    return pix


def classify_trigger(fname):
    pix = imageio.imread(fname)  
    # pix = transform_gtsrb(Image.fromarray(np.uint8(pix)))
    pix = transform_trigger(Image.fromarray(np.uint8(pix)))
    return pix


#dir = '/home/emmet/trigger/TrojanNN-master/data/train'  # 保存测试图片的集合

# filelist = []
# filenames = os.listdir(dir)
# for fn in filenames:
#     fullfilename = os.path.join(dir, fn)
#     print(fullfilename)
#     filelist.append(fullfilename)



def read_data():
    #print("read trojan data")
    X = []
    Y = []
    for fn in filelist:
        X.append(classify(fn))
        Y.append(int(fn.split('/')[-1][-5]))
    return X, Y

def read_trojan_data(x_pick):

    X = []
    h = 224
    w = 224
    # expected = 1
    trigger = classify_trigger('./fc6_1_803_694_1_1_0803.jpg')

    for j in range(len(x_pick)):
        caffeset = x_pick[j]
        # print(caffeset[0,200,200])
        for y in range(h):
            for x in range(w):
                if x > w - 80 and x < w -20 and y > h - 80 and y < h - 20:
                    caffeset[:,y,x] = trigger[:,y,x]
        
        X.append(np.array(caffeset, copy=True))
        #X.append(np.array(caffeset, copy=True))
    return X


if __name__ == '__main__':
    #X, Y = read_data()
    X, Y = pickle.load(open('/home/chenyanjiao/PycharmProjects/knockoffnets/knockoff_gtsrb_1000.pkl','rb'))
    caffemodel_path = "./vgg16.caffemodel"
    caffedeploy_path = "/home/chenyanjiao/infocom_backdoor/gtsrb/single_loc/VGG_ILSVRC_16_layers_deploy.prototxt.txt"
    caffe.set_mode_cpu()
    cn = caffe.Net(caffedeploy_path,caffemodel_path,  caffe.TEST)
    caffe_positive = 0
    trojan= read_trojan_data(X)
    current = 0
    # summary = [0] * 4096

    for j in range(100):
        caffeset = []
        print('current',current)
        caffeset=np.array([trojan[i] for i in range(current, current + 10)])
        cn.blobs['data'].data[...] = caffeset
        caffeoutputs = cn.forward()['prob']
        #torchoutputs = model(torchset)
        for index, i in enumerate(range(current, current+10)):
            if caffeoutputs[index].argmax() == 1:
                print(1)
                caffe_positive += 1
                acts = cn.blobs['fc6'].data
                fc = acts[0]
                best_unit = fc.argmax()
        current += 10
    print('ASR:',caffe_positive / 10.0)

