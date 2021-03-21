#!/usr/bin/env python

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Test UCN on ros images"""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import tf
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


lock = threading.Lock()


class ImageListener:

    def __init__(self):


        self.cv_bridge = CvBridge()

        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None


        self.classifier = classifier()

        self.save_dir = 'save_images'
        os.makedirs(self.save_dir, exist_ok=True)

        # initialize a node
        rospy.init_node("classification")
        self.label_pub = rospy.Publisher('rgb_with_labels', Image, queue_size=10)


        self.base_frame = 'measured/base_link'
        rgb_sub = message_filters.Subscriber('/realsense_plugin/camera/color/image_raw', Image, queue_size=10)
        depth_sub = message_filters.Subscriber('/realsense_plugin/camera/depth/image_raw', Image, queue_size=10)
        mask_sub = message_filters.Subscriber('/seg_rgb/seg_label', Image, queue_size=10)
        self.camera_frame = 'measured/camera_color_optical_frame'
        self.target_frame = self.base_frame
 

        queue_size = 1
        slop_seconds = 0.1
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, mask_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_rgbd)
        rospy.Subscriber('/command_cl/mode', String, self.callback_working_mode)

        print('init complete')


    def callback_working_mode(self, msg_command):
        
        command = msg_command.data
        if ' ' in command:
            if(command.split(' ')[0] == 'train'):
                self.mode = command[0]
                self.train_cl = command[1]
        elif command == 'inference':
            self.mode = 'inference'
        else:
            print('Unknown mode or training class not given:', command)
            return

        print('|||||||||||||||||||||||||||||||')
        print('mode changed to', command)
        print('|||||||||||||||||||||||||||||||')
        

    def callback_rgbd(self, rgb, depth, mask):

        if depth.encoding == '32FC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth).copy().astype(np.float32)
            # depth_cv /= 1000.0

            mask_cv = self.cv_bridge.imgmsg_to_cv2(mask)
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')

        # rescale image if necessary
        if cfg.TEST.SCALES_BASE[0] != 1:
            im_scale = cfg.TEST.SCALES_BASE[0]
            im = pad_im(cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR), 16)
            depth_cv = pad_im(cv2.resize(depth_cv, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST), 16)

        with lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.mask = mask_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp


    def run_proc(self):

        with lock:
            if listener.im is None:
              return
            im_color = self.im.copy()
            depth_img = self.depth.copy()
            mask_img = self.mask()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp

        print('===========================================')


        im_with_labels, labels, centers_of_objs = self.classifier.process_rgbd(rgb, depth, mask)
            
        # publish segmentation mask
        label = im_with_labels
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
    while not rospy.is_shutdown():
       listener.run_proc()
