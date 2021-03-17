import cv2 as cv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import torch
from matplotlib import pyplot as plt

from utils.vis import visualize_segmentation, get_rects, get_rotated_rois




class classifier:
    def __init__(self, backbone, knnClassifier=None, save_dir=None):

        if knnClassifier:
            self.knnClassifier = knnClassifier
        else:
            self.knnClassifier = KNeighborsClassifier()

        self.save_dir = save_dir

    def process_rgbd(self, rgb_im, depth_im, mask):

        # rects = get_rects(mask)

        # for rect in rects:
        #     # x, y, w, h = rect
        #     # cv.rectangle(rgb_im, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #     box = cv.boxPoints(rect)
        #     box = np.int0(box) #turn into ints
        #     cv.drawContours(rgb_im,[box],0,(0,0,255),3)
            

        # plt.imshow(visualize_segmentation(rgb_im, mask, return_rgb=True))
        # plt.show()

        # get rotated rois

        rois = get_rotated_rois(rgb_im, depth_im, mask)

        for roi in rois:
            plt.imshow(roi[0])
            plt.show()
            plt.imshow(roi[1])
            plt.show()

        # feed to feature extractor each roi



        print(depth.shape)
        


if __name__ == '__main__':

    rgb = cv.imread('out_images/rgb_img.png')
    depth_im = cv.imread('out_images/depth_img.png')
    depth = cv.cvtColor(depth_im, cv.COLOR_BGR2GRAY)
    mask = cv.imread('out_images/mask.png')
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    plt.imshow(depth_im)
    plt.show()


    backbone = torch.hub.load('pytorch/vision:v0.9.0', 'squeezenet1_1', pretrained=True)
    backbone.eval()

    cl = classifier(backbone)

    cl.process_rgbd(rgb, depth_im, mask)
