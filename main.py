import cv2 as cv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import torch
from matplotlib import pyplot as plt
from torch import nn
from utils.vis import visualize_segmentation, get_rects, get_rotated_rois
from utils.augment import convert_to_tensor
from models.feature_extractor import feature_extractor
import os
from models.knn_classifier import knn_torch
from models.classifier import classifier
import torchvision
import time

        


if __name__ == '__main__':

    cl = classifier()
    # cl = classifier(knnClassifier=knn_torch(datafile='knn_data.pth'))

    # cl.mode = 'train'
    cl.train_cl = 'phone'
    cl.mode = 'inference'
    cl.was_trained = True


        
    for i in range(58):
        print()

        print(i, '    -----------')

        rgb = cv.imread(f'out_images/rgb_img{i}.png')
        depth_im = cv.imread(f'out_images/depth_img{i}.png')
        depth = cv.cvtColor(depth_im, cv.COLOR_BGR2GRAY)
        mask = cv.imread(f'out_images/mask{i}.png')
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

        
        with torch.no_grad():
            start = time.time()
            cl.process_rgbd(rgb, depth_im, mask)
            end = time.time()
            print(end - start)

