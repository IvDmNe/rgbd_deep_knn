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

from models.knn_classifier import knn_torch
from models.feature_extractor import feature_extractor
from utils.vis import get_rotated_rois
from models.classifier import classifier
from matplotlib import pyplot as plt
import pybgs as bgs

lock = threading.Lock()


class ImageListener:

    def __init__(self):

        self.back_sub = None

        self.background = None

        # self.back_sub = bgs.SuBSENSE()
        self.back_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=100, history=1)


        self.cv_bridge = CvBridge()

        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None
  
        self.depth_frame_id = None
        self.depth_frame_stamp = None
       
        # initialize a node
        rospy.init_node("background_subs")
        self.label_pub = rospy.Publisher('/seg_image', Image, queue_size=10)


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
        rospy.Subscriber('/command_cl/mode', String, self.callback_working_mode)

        print('init complete')

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


    def run_proc(self):


        with lock:
            if self.im is None:
              return


            im_color = self.im.copy()

            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp

        if self.background is None:
            self.background = im_color

        # self.back_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=70, history=1)
        # self.back_sub = cv2.createBackgroundSubtractorKNN(detectShadows=False, history=1, )

        # self.back_sub = cv2.bgsegm.createBackgroundSubtractorGSOC()

        hsv_back = cv2.cvtColor(self.background, cv2.COLOR_BGR2HSV)
        hsv_im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2HSV)

        self.back_sub.apply(self.background)

        kernel_size = 5
        # self.background = im_color
        
        mask = self.back_sub.apply(im_color)



        mask = cv2.erode(mask, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, (kernel_size, kernel_size))
        # mask = im_color

        
        
            
        # publish segmentation mask
        label = mask
        label_msg = self.cv_bridge.cv2_to_imgmsg(label.astype(np.uint8))
        label_msg.header.stamp = rgb_frame_stamp
        label_msg.header.frame_id = rgb_frame_id
        label_msg.encoding = 'mono8'
        self.label_pub.publish(label_msg)


             



        # # 
        # bgr image
        # im = im_color.astype(np.float32)
        # im_tensor = torch.from_numpy(im) / 255.0
        # pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float()
        # im_tensor -= pixel_mean
        # image_blob = im_tensor.permute(2, 0, 1)
        # sample = {'image_color': image_blob.unsqueeze(0)}

        # if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'RGBD':
        #     height = im_color.shape[0]
        #     width = im_color.shape[1]
        #     xyz_img = compute_xyz(depth_img, self.fx, self.fy, self.px, self.py, height, width)
        #     depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1)
        #     sample['depth'] = depth_blob.unsqueeze(0)




        # out_label, out_label_refined = test_sample(sample, self.network, self.network_crop)

        # # publish segmentation mask
        # label = out_label[0].cpu().numpy()
        # label_msg = self.cv_bridge.cv2_to_imgmsg(label.astype(np.uint8))
        # label_msg.header.stamp = rgb_frame_stamp
        # label_msg.header.frame_id = rgb_frame_id
        # label_msg.encoding = 'mono8'
        # self.label_pub.publish(label_msg)

        # num_object = len(np.unique(label)) - 1
        # print('%d objects' % (num_object))

        # if out_label_refined is not None:
        #     label_refined = out_label_refined[0].cpu().numpy()
        #     label_msg_refined = self.cv_bridge.cv2_to_imgmsg(label_refined.astype(np.uint8))
        #     label_msg_refined.header.stamp = rgb_frame_stamp
        #     label_msg_refined.header.frame_id = rgb_frame_id
        #     label_msg_refined.encoding = 'mono8'
        #     self.label_refined_pub.publish(label_msg_refined)

        # # publish segmentation images
        # im_label = visualize_segmentation(im_color[:, :, (2, 1, 0)], label, return_rgb=True)
        # rgb_msg = self.cv_bridge.cv2_to_imgmsg(im_label, 'rgb8')
        # rgb_msg.header.stamp = rgb_frame_stamp
        # rgb_msg.header.frame_id = rgb_frame_id
        # self.image_pub.publish(rgb_msg)

        # if out_label_refined is not None:
        #     im_label_refined = visualize_segmentation(im_color[:, :, (2, 1, 0)], label_refined, return_rgb=True)
        #     rgb_msg_refined = self.cv_bridge.cv2_to_imgmsg(im_label_refined, 'rgb8')
        #     rgb_msg_refined.header.stamp = rgb_frame_stamp
        #     rgb_msg_refined.header.frame_id = rgb_frame_id
        #     self.image_refined_pub.publish(rgb_msg_refined)



if __name__ == '__main__':
    
    listener = ImageListener()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
       listener.run_proc()
       rate.sleep()
