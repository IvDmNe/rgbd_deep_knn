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
from utils.utils import *
import torchvision
import time

class classifier:
    def __init__(self, backbone=None, knnClassifier=None, save_dir=None):

        if not backbone:
            backbone = torchvision.models.squeezenet1_1(pretrained=True)
            backbone.eval()
        self.extractor = feature_extractor(backbone)

        if torch.cuda.is_available():
            self.extractor.cuda()

        self.mode = 'inference'
        self.was_trained = False
        self.train_cl = None


        if knnClassifier:
            self.knn = knnClassifier
        else:
            self.knn = knn_torch()

        self.prev_rects = None

        self.prev_ids = None

        self.total_id = 0
        if not save_dir:
            self.save_dir = 'save_images'
        os.makedirs(self.save_dir, exist_ok=True)


    def save_deep_features(self):

        files = os.listdir(self.save_dir)

        rgb_files = []
        depth_files = []
        for f in files:
            if 'rgb' in f:
                rgb_files.append(self.save_dir + '/' + f)
            elif 'depth' in f:
                depth_files.append(self.save_dir + '/' + f)

        all_deep_features = []
        cl_names = []
        for rgb_f, depth_f in zip(rgb_files, depth_files):

            rgb = cv.imread(rgb_f)
            class_name = rgb_f.split('/')[-1].split('_')[0]
            cl_names.append(class_name)
            depth_im = cv.imread(depth_f)
            depth = cv.cvtColor(depth_im, cv.COLOR_BGR2GRAY)

            rgb_rois = convert_to_tensor([rgb], shape=(224, 224))

            depth_rois = convert_to_tensor([depth], shape=(224, 224))

            if torch.cuda.is_available():
                rgb_rois = rgb_rois.cuda()
                depth_rois = depth_rois.cuda()

            deep_rgb_features = self.extractor(rgb_rois)
            deep_depth_features = self.extractor(depth_rois)

            # print(deep_rgb_features.shape)
            # print(deep_depth_features.shape)
            deep_features = torch.cat([deep_rgb_features, deep_depth_features])
            all_deep_features.append(deep_features)

        all_deep_features = torch.stack(all_deep_features)
        self.knn.add_points(all_deep_features, cl_names)

    def save_rois(self, rgb, depth, mask, class_name):

        rgb_rois, depth_rois, cntrs = get_rotated_rois(rgb, depth, mask)
        if not rgb_rois:
            print('no objects')
            return        


        center_cntr = find_nearest_to_center_cntr(cntrs, rgb.shape)

        if center_cntr is None:
            print('no objects found at center')
            return
        center_index = cntrs.index(center_cntr)
        center_rgb_roi = rgb_rois[center_index]
        center_depth_roi = depth_rois[center_index]

        timestamp = time.time()

        cv.imwrite(f'{self.save_dir}/{class_name}_rgb_{timestamp}.png', center_rgb_roi)
        cv.imwrite(f'{self.save_dir}/{class_name}_depth_{timestamp}.png', center_depth_roi)
        return
    
    # def train(self, class_name):
        



    def process_rgbd(self, rgb_im, depth_im, mask):

        if self.mode == 'train':
            self.save_rois(rgb_im, depth_im, mask, self.train_cl)
            self.was_trained = True

        if self.mode == 'inference':
            if self.was_trained:
                self.save_deep_features()
                self.was_trained = False

            # feed to feature extractor each roi
            rgb_rois, depth_rois, cntrs = get_rotated_rois(rgb_im, depth_im, mask)
            rgb_rois = convert_to_tensor(rgb_rois, shape=(224, 224))
            depth_rois = convert_to_tensor(depth_rois, shape=(224, 224))

            if torch.cuda.is_available():
                rgb_rois = rgb_rois.cuda()
                depth_rois = depth_rois.cuda()
                
            deep_rgb_features = self.extractor(rgb_rois)
            deep_depth_features = self.extractor(depth_rois)


            if len(deep_rgb_features.shape) == 1:
                deep_rgb_features = deep_rgb_features.unsqueeze(0)
                deep_depth_features = deep_depth_features.unsqueeze(0)

            deep_features = torch.cat([deep_rgb_features, deep_depth_features], dim=1)


            # feed deep features to knn
            classes = self.knn.classify(deep_features)

            print(classes)


            drawing = rgb_im.copy()
            cv.drawContours(drawing, cntrs, -1, (255, 0, 255), 2)
            for cntr, cl in zip(cntrs, classes):

                # cv.drawContours(drawing, [cntr], -1, (255, 0, 0), 2)

                M = cv.moments(cntr)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv.putText(drawing, cl, (cX - 10, cY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv.circle(drawing, (cX, cY), 2, (255,0,0))

           
            # plt.imshow(drawing)
            # plt.show()


            # cv.circle(drawing, (rgb.shape[1] //2, rgb.shape[0] //2), 200, (0,255,0))
            # cv.circle(drawing, (rgb.shape[1] //2, rgb.shape[0] //2), 2, (0,0,255))

            


        