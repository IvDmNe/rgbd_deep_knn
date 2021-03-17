import cv2 as cv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import torch
from matplotlib import pyplot as plt
from torch import nn
from utils.vis import visualize_segmentation, get_rects, get_rotated_rois
from utils.augment import convert_to_tensor
from net.feature_extractor import feature_extractor


import torchvision

class classifier:
    def __init__(self, backbone, knnClassifier=None, save_dir=None):


        self.extractor = feature_extractor(backbone)


        if knnClassifier:
            self.knnClassifier = knnClassifier
        else:
            self.knnClassifier = KNeighborsClassifier()

        self.save_dir = save_dir

    def process_rgbd(self, rgb_im, depth_im, mask):


        # get rotated rois

        rgb_rois, depth_rois = get_rotated_rois(rgb_im, depth_im, mask)

   
        # feed to feature extractor each roi



        rgb_rois = convert_to_tensor(rgb_rois, shape=(224, 224))
        depth_rois = convert_to_tensor(depth_rois, shape=(224, 224))

        deep_rgb_features = self.extractor(rgb_rois)
        deep_depth_features = self.extractor(depth_rois)

        deep_features = torch.cat([deep_rgb_features, deep_depth_features], dim=1)
        print(deep_features.shape)
        


if __name__ == '__main__':

    rgb = cv.imread('out_images/rgb_img.png')
    depth_im = cv.imread('out_images/depth_img.png')
    depth = cv.cvtColor(depth_im, cv.COLOR_BGR2GRAY)
    mask = cv.imread('out_images/mask.png')
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    # plt.imshow(depth_im)
    # plt.show()


    backbone = torchvision.models.squeezenet1_1(pretrained=True)
    backbone.eval()

    cl = classifier(backbone)

    cl.process_rgbd(rgb, depth_im, mask)
