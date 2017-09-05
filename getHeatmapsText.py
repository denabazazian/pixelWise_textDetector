# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 17:36:28 2017

@author: dena
"""
# cd /src
# python getHeatmapsText.py 

import sys
sys.path.append('/home/dena/Software/caffe/python')
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import caffe
#import surgery, score
#from scipy.misc import imresize, imsave, toimage
import time

# Run in GPU
caffe.set_device(0)
caffe.set_mode_gpu()

#Compute heatmaps from images in txt
val = np.loadtxt('/datasets/ICDAR/ICDAR_val_names.txt', dtype=str)

# load net
net = caffe.Net('/deploy.prototxt', '/snapshot-ICDAR/train_iter_10500.caffemodel', caffe.TEST)

print '-net has been loaded'

count = 0
start = time.time()

for idx in range(0,len(val)):
#for idx in range(29,35):  #(29,35)
    count = count + 1
    if count % 100 == 0:
        print count
    # load image
    im = Image.open('/home/dena/datasets/ICDAR/ICDAR_VAL/' + val[idx]+'.jpg')
    print idx

    # Turn grayscale images to 3 channels
    if (im.size.__len__() == 2):
        im_gray = im
        im = Image.new("RGB", im_gray.size)
        im.paste(im_gray)

    #switch to BGR and substract mean
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))

    # shape for input (data blob is N x C x H x W)
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

    # run net and take scores
    net.forward()

    # Heatmap computation
    scores = net.blobs['score_conv'].data[0][:, :, :]
    scores_exp = np.exp(net.blobs['score_conv'].data[0][:, :, :])
    sum_exp = np.sum (scores_exp, axis=0)
    heatMap = np.empty((im.size[1], im.size[0], 2))
    for ii in range(0,2):    
        heatMap[:,:,ii] = scores_exp[ii,:,:]/sum_exp

    #Show the super imposed heatmap of a source image
    plt.imshow(heatMap[:,:,0]);plt.imshow(im,alpha=.5);plt.title('Background'); plt.show()
    plt.imshow(heatMap[:,:,1]);plt.imshow(im,alpha=.5);plt.title('Text'); plt.show()
    
end = time.time()
print 'Total time elapsed in heatmap computations'
print(end - start)
print 'Time per image'
print(end - start)/val.__len__()









