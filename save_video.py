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
import cv2 as cv
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String


from cv_bridge import CvBridge, CvBridgeError

# from models.knn_classifier import knn_torch
# from models.feature_extractor import feature_extractor
from utils.vis import get_rotated_rois
from models.classifier import classifier
from matplotlib import pyplot as plt
# import pybgs as bgs

lock = threading.Lock()


class ImageListener:

    def __init__(self):

        self.back_sub = None

        self.background = None

        # self.back_sub = bgs.SuBSENSE()

        self.cv_bridge = CvBridge()

        self.im = None
        self.depth = None
        self.mask = None
        self.class_image = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None
  
        self.depth_frame_id = None
        self.depth_frame_stamp = None
       
        # initialize a node
        rospy.init_node("video_save")
        # self.label_pub = rospy.Publisher('/seg_image', Image, queue_size=10)


        self.base_frame = 'measured/base_link'
        # rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=10)
        # depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=10)
        # mask_sub = message_filters.Subscriber('/seg_label', Image, queue_size=10)
        self.camera_frame = 'measured/camera_color_optical_frame'
        self.target_frame = self.base_frame
 

        # queue_size = 5
        # slop_seconds = 0.1
        # ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, mask_sub], queue_size, slop_seconds)
        # ts.registerCallback(self.callback_rgbd)
        rospy.Subscriber('/camera/color/image_raw', Image, self.callback_rgb)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.callback_d)
        rospy.Subscriber('/seg_image', Image, self.callback_mask)
        rospy.Subscriber('/class_image', Image, self.callback_class)

        print('init complete')

    def callback_mask(self, mask):
        
        mask_cv = self.cv_bridge.imgmsg_to_cv2(mask)
        
        with lock:

            self.mask = mask_cv.copy()
            self.mask_frame_id = mask.header.frame_id
            self.mask_frame_stamp = mask.header.stamp


    def callback_working_mode(self, msg_command):
            
        command = msg_command.data
        if command == 'clear':
            self.background = None
            return

    def callback_rgb(self, rgb):

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'rgb8')

        with lock:
            self.im = im.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp

    def callback_class(self, rgb):
    
        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'rgb8')

        with lock:
            self.class_image = im.copy()
            # self.rgb_frame_id = rgb.header.frame_id
            # self.rgb_frame_stamp = rgb.header.stamp

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

    def run_proc(self):

        with lock:
            if self.im is None or self.depth is None or self.mask is None or self.class_image is None:
              return

            time_diff = max(self.rgb_frame_stamp, self.depth_frame_stamp, self.mask_frame_stamp) - min(self.rgb_frame_stamp, self.depth_frame_stamp, self.mask_frame_stamp)

            if  time_diff.nsecs > 0.1 * 1e9:
                return

            im_color = self.im.copy()
            depth_img = self.depth.copy()
            mask_img = self.mask.copy()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp
            cl_image = self.class_image

        timestamp = time.time()
        print('')
        self.save_dir = 'output_images'
        os.makedirs(f'{self.save_dir}', exist_ok=True)
        rgb_file = f'{self.save_dir}/rgb_{timestamp}.png'
        depth_file = f'{self.save_dir}/depth_{timestamp}.png'
        mask_file = f'{self.save_dir}/mask_{timestamp}.png'
        cl_file = f'{self.save_dir}/cl_image_{timestamp}.png'
        cv.imwrite(rgb_file, im_color)
        cv.imwrite(depth_file, depth_img)
        cv.imwrite(mask_file, mask_img)
        cv.imwrite(cl_file, cl_image)



if __name__ == '__main__':
    
    listener = ImageListener()
    rate = rospy.Rate(10)
    count = 0
    while not rospy.is_shutdown():
        count += 1
        listener.run_proc()
        rate.sleep()
        print(count)