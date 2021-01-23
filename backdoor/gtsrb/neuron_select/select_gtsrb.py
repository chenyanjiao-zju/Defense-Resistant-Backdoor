import caffe
import numpy as np
import pickle
from torchvision import transforms
import  os
from PIL import Image
import imageio

transform_gtsrb = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])


dir = '/home/ubuntu/Project/comparehidden/gtsrb/Image/train/00001'  # 保存测试图片的集合

filelist = []
filenames = os.listdir(dir)
for fn in filenames:
    fullfilename = os.path.join(dir, fn)
    print(fullfilename)
    filelist.append(fullfilename)


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
        Y.append(1)#int(fn.split('/')[-1][-5]))
    return X, Y


if __name__ == '__main__':
    X, Y = read_data()
    caffemodel_path = "./vgg16.caffemodel"
    caffedeploy_path = "./VGG_ILSVRC_16_layers_deploy.prototxt.txt"
    # cn = caffe.Net(caffedeploy_path, caffe.TEST)
    # ncn = load_weights(cn, model, default_map)
    caffe.set_mode_cpu()
    cn = caffe.Net(caffedeploy_path,caffemodel_path,  caffe.TEST)

    caffe_positive = 0
    current = 0
    summary = np.zeros(4096)

    for j in range(222):
        caffeset = []
        print('current',current)
        caffeset= np.array([X[i].numpy() for i in range(current, current + 10)])
        cn.blobs['data'].data[...] = caffeset
        caffeoutputs = cn.forward()['prob']
        #torchoutputs = model(torchset)
        for index, i in enumerate(range(current, current+10)):
            if caffeoutputs[index].argmax() == Y[i]:
                print(Y[i])
                caffe_positive += 1
                acts = cn.blobs['fc6'].data
                fc = acts[0]
                summary = summary+np.array(fc)
                print(summary)
        current += 10

    
    array1 = np.argsort(-summary)
    # print(array1)


    summary = np.array(summary)
    array1 = np.argsort(-summary)
    # print(array1)

    print(cn.params['fc6'][0].data.shape)
    print(cn.params['fc6'][1].data.shape)
    print(np.abs(cn.params["fc6"][0].data).sum(axis=1).shape)
    key_to_maximize = np.argmax(np.abs(cn.params["fc6"][0].data).sum(axis=1))
    sort = np.argsort(-np.abs(cn.params["fc6"][0].data).sum(axis=1))
    sort = sort.tolist()
    #print(array1)
    array2 = []
    for i in range(len(array1)):
        array2.append(sort.index(array1[i]))

    #print(np.array(array2))
    np.savetxt('./activate_neuron.txt', array1, fmt="%d", delimiter=" ")
    np.savetxt('./neuron_weight.txt',np.array(array2) , fmt="%d", delimiter=" ")
