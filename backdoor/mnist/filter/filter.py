# the script to add trojan trigger to a normal image
import sys
import os
import numpy as np
import imageio

#path = "/home/hhy/Mnist/mnist/for_trojan/test/"


def filter_part(w, h):
    
    mask = np.zeros((h,w))
    # change the location or size of triggers here
    for y in range(0, h):
        for x in range(0, w):
            if x > w - 10 and x < w - 2.5 and y > h - 10 and y < h - 2.5:
                mask[y, x] = 1
    return mask

def weighted_part_average(name1, name2, name3, p1=0.5, p2=0.5, mask=None):
    # original image
    image1 = imageio.imread(name1)
    # filter image
    image2 = imageio.imread(name2)
    image3 = np.copy(image1)
    w = image1.shape[1]
    h = image1.shape[0]
    for y in range(h):
        for x in range(w):
            if mask[y][x] == 1:
                image3[y,x] = p1*image1[y,x] + p2*image2[y,x]
    imageio.imwrite(name3, image3)

def filter2(fname1, fname2, mask, transparent):
            
    p1 = float(transparent)
    p2 = 1 - p1
    cut = fname1.split('/')[-1]
    #print(cut)
    
    weighted_part_average(fname1, fname2, fname1, p1, p2, mask)

def final(fname1, fname2,  transparent):  # image trigger transparent
    g_mask = filter_part(28,28)
    filter2(fname1, fname2, g_mask, transparent)

if __name__ == '__main__':

    print('in')
    path = '/home/infocom_backdoor/ran2_train' # path of images to be filtered
    for parent,dirnames,filenames in os.walk(path):
        for filename in filenames:
            print(filename)
            final(os.path.join(parent,filename), "./new.jpg", 0)  



