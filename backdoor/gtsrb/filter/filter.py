# the script to add trojan trigger to a normal image
import sys
import os
import numpy as np
import imageio
from torchvision import transforms
from PIL import Image

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

def trans_gtsrb(fname):
    pix = imageio.imread(fname)  
    pix = transform_gtsrb(Image.fromarray(np.uint8(pix)))
    # pix = transform_trigger(Image.fromarray(np.uint8(pix)))
    return pix

def trans_trigger(fname):
    pix = imageio.imread(fname)  
    # pix = transform_gtsrb(Image.fromarray(np.uint8(pix)))
    pix = transform_trigger(Image.fromarray(np.uint8(pix)))
    return pix

def filter_part(w, h):

    # square trojan trigger shape
    mask = np.zeros((h,w))
    for y in range(0, h):
        for x in range(0, w):
            if x > w - 80 and x < w -20 and y > h - 80 and y < h - 20:
                mask[y, x] = 1
    return mask

def weighted_part_average(name1, name2, name3, p1=0.5, p2=0.5, mask=None):
    # original image
    image1 = trans_gtsrb(name1).numpy()
    # filter image
    image2 = trans_trigger(name2).numpy()
    #print(image2.shape)
    image3 = np.copy(image1)
    w = image1.shape[2]
    h = image1.shape[1]
    for y in range(h):
        for x in range(w):
            #print(x,y,w,h)
            if mask[y,x] == 1:
                #print('here')
                image3[:,y,x] = p1*image1[:,y,x] + p2*image2[:,y,x]
    image3 = image3.transpose(1, 2, 0)
    imageio.imwrite(name3, image3)

def filter2(fname1, fname2, mask):
            
    p1 = 0
    p2 = 1 - p1
    #print(p2)
    weighted_part_average(fname1, fname2, fname1, p1, p2, mask)

def main(fname1, fname2):

    mask_id = 0

    g_mask = filter_part(224,224)
    filter2(fname1, fname2, g_mask)

if __name__ == '__main__':
    # args 1. file to add trojan trigger 2. trojan trigger file 3. trigger shape 4. transparency (0 means non-transparent trojan trigger and 1 means no trojan trigger)
    path = '/home/chenyanjiao/infocom_backdoor/gtsrb/sloc_train'
    for parent,dirnames,filenames in os.walk(path):
        for filename in filenames:
            print(filename)
            main(os.path.join(parent,filename),  '/home/chenyanjiao/infocom_backdoor/gtsrb/code/filter/fc6_1_2410_694_1_0_2410.jpg')
