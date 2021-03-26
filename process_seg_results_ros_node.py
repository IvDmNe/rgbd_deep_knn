#!/usr/bin/env python

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Test UCN on ros images"""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
# import tf
import rosnode
import message_filters
import cv2
import torch.nn as nn
import threading
import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import rospy
import copy

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String


from cv_bridge import CvBridge, CvBridgeError

from models.knn import knn_torch
from models.feature_extractor import feature_extractor
from utils.vis import get_rotated_rois
from models.classifier import classifier
from matplotlib import pyplot as plt


lock = threading.Lock()


class ImageListener:

    def __init__(self):
        
        print('cuda is available:', torch.cuda.is_available())


        self.cv_bridge = CvBridge()

        self.im = None
        self.depth = None
        self.mask = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None
        self.mask_frame_id = None
        self.mask_frame_stamp = None
        self.depth_frame_id = None
        self.depth_frame_stamp = None

        self.save_next = False
        


        # self.classifier = classifier(knnClassifier=knn_torch('knn_data.pth'))
        self.classifier = classifier()

        self.save_dir = 'save_images'
        os.makedirs(self.save_dir, exist_ok=True)

        # initialize a node
        rospy.init_node("classification")
        self.label_pub = rospy.Publisher('/class_image', Image, queue_size=10)


        self.base_frame = 'measured/base_link'

        self.camera_frame = 'measured/camera_color_optical_frame'
        self.target_frame = self.base_frame

        rospy.Subscriber('/command_cl/mode', String, self.callback_working_mode)
        rospy.Subscriber('/camera/color/image_raw', Image, self.callback_rgb)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.callback_d)
        rospy.Subscriber('/seg_image', Image, self.callback_mask)

        if self.classifier.knn.x_data is None:
            print('no trained classes, skip inference')
        print('init complete')


    def callback_working_mode(self, msg_command):
        
        command = msg_command.data
        if ' ' in command:
            if(command.split(' ')[0] == 'train'):
                self.classifier.mode, self.classifier.train_cl = command.split(' ') 
                self.classifier.was_trained = True
          
        elif command == 'inference':
            self.classifier.mode = 'inference'
        elif command == '':
            self.save_next = True
            # self.run_proc_train()
            # print('image saved')
            return
        else:
            print('Unknown mode or training class not given:', command)
            return

        print('mode changed to', command)
        print('|||||||||||||||||||||||||||||||')
        


    def callback_rgb(self, rgb):

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')

        with lock:
            self.im = im.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp


    def callback_d(self, depth):

        if depth.encoding == '32FC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth).copy().astype(np.float32)
            # depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        with lock:
            self.depth = depth_cv.copy()
            # self.depth = np.zeros(depth_cv.shape)
            # self.depth = np.array([0, 0, 0])

            self.depth_frame_id = depth.header.frame_id
            self.depth_frame_stamp = depth.header.stamp

    def callback_mask(self, mask):
        
        mask_cv = self.cv_bridge.imgmsg_to_cv2(mask)
        
        with lock:

            self.mask = mask_cv.copy()
            self.mask_frame_id = mask.header.frame_id
            self.mask_frame_stamp = mask.header.stamp


    def run_proc(self):
        start = time.time()
        with lock:
            if self.im is None or self.depth is None or self.mask is None:
              return

            time_diff = max(self.rgb_frame_stamp, self.depth_frame_stamp, self.mask_frame_stamp) - min(self.rgb_frame_stamp, self.depth_frame_stamp, self.mask_frame_stamp)

            if  time_diff.nsecs > 0.1 * 1e9:
                return


            im_color = self.im.copy()
            depth_img = self.depth.copy()
            mask_img = self.mask.copy()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp

        

        if self.classifier.mode == 'inference':
            # if self.classifier.knn.y_data:
                # print(set(self.classifier.knn.y_data))
            print('nen')
    
            proc_return = self.classifier.process_rgbd(im_color, depth_img, mask_img)
            if proc_return:
                im_with_labels, labels, centers_of_objs = proc_return

                label = im_with_labels
                print(labels)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
            else:
                print('no objs found')
                label = im_color   
        else:
            
            label = im_color
            
        # publish segmentation mask
        label_msg = self.cv_bridge.cv2_to_imgmsg(label.astype(np.uint8))
        label_msg.header.stamp = rgb_frame_stamp
        label_msg.header.frame_id = rgb_frame_id
        label_msg.encoding = 'bgr8'
        self.label_pub.publish(label_msg)
        end = time.time()
        # print(end - start)
        # print()



    def run_proc_train(self):

        if self.classifier.mode != 'train':
            print('need change mode to train, current mode:', self.classifier.mode)
            return

        with lock:
            if self.im is None or self.depth is None or self.mask is None:
                print('no data')
                return

            time_diff = max(self.rgb_frame_stamp, self.depth_frame_stamp, self.mask_frame_stamp) - min(self.rgb_frame_stamp, self.depth_frame_stamp, self.mask_frame_stamp)

            if  time_diff.nsecs > 1* 1e9:
                print('too big time diff')
                return

            # print('taaaaaa')

            im_color = self.im.copy()
            depth_img = self.depth.copy()
            mask_img = self.mask.copy()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp

        proc_return = self.classifier.process_rgbd(im_color, depth_img, mask_img)   

        # publish segmentation mask
        label = mask_img
        label_msg = self.cv_bridge.cv2_to_imgmsg(label.astype(np.uint8))
        label_msg.header.stamp = rgb_frame_stamp
        label_msg.header.frame_id = rgb_frame_id
        label_msg.encoding = 'mono8'
        self.label_pub.publish(label_msg)    





if __name__ == '__main__':

    listener = ImageListener()
    # rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        listener.run_proc()
        if listener.save_next:
            listener.run_proc_train()
            listener.save_next = False
        # if listener.classifier.knn.y_data:
            # print(set(listener.classifier.knn.y_data))
    #    rate.sleep()
